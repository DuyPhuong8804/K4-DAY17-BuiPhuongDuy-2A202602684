"""Tests for the memory layer.

These tests use a scripted fake LLM, so they verify the *plumbing*: what reaches User.md,
what reaches the prompt, what leaks across threads, when compaction fires. They cannot
judge how well a real model extracts facts; that is what `python src/benchmark.py` measures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import (
    COLUMNS,
    format_rows,
    heuristic_quality,
    load_conversations,
    recall_points,
    run_agent_benchmark,
    run_suite,
)
from config import LabConfig
from memory_store import (
    EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
    CompactMemoryManager,
    UserProfileStore,
    parse_profile_update,
)
from model_provider import LLMUnavailableError, ProviderConfig, invoke_text, normalize_provider

REPO_ROOT = Path(__file__).resolve().parent.parent
LONG_TEXT = "Hôm nay mình đọc một bài rất dài về kiến trúc memory cho agent. " * 12
SUMMARY_MARKER = SUMMARY_PROMPT.split("{")[0]


class Reply:
    def __init__(self, content) -> None:
        self.content = content


class ScriptedLLM:
    """Fake chat model. Extraction answers come from `facts` (latest message -> JSON dict);
    summaries return a fixed short text; normal replies either echo the system prompt
    (so a test can see what memory the prompt carried) or just say "ok"."""

    def __init__(self, facts: dict[str, dict] | None = None, echo_system: bool = False) -> None:
        self.facts = facts or {}
        self.echo_system = echo_system
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        system = messages[0]["content"]
        if system == EXTRACTION_PROMPT:
            latest = messages[-1]["content"].split("LATEST MESSAGE:\n", 1)[1]
            return Reply(json.dumps(self.facts.get(latest, {"updates": {}, "remove": []})))
        if system.startswith(SUMMARY_MARKER):
            return Reply("- tóm tắt ngắn")
        return Reply(system if self.echo_system else "ok")


def fact(key: str, value: str, confidence: float = 0.95) -> dict:
    return {"updates": {key: {"value": value, "confidence": confidence}}, "remove": []}


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state lives in tmp_path and compaction triggers fast."""

    provider = ProviderConfig(provider="openai", model_name="test-model", temperature=0.0)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=REPO_ROOT / "data",
        state_dir=state_dir,
        compact_threshold_tokens=300,
        compact_keep_messages=4,
        model=provider,
        judge_model=provider,
        live=False,
        profile_min_confidence=0.7,
    )


# --- User.md ---------------------------------------------------------------
def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.file_size("u1") == 0
    assert store.read_text("u1").startswith("# User Profile")  # default, no file yet

    path = store.write_text("u1", "# User Profile\n\n- name: An\n- city: Hà Nội\n")
    assert path.exists() and path.name == "User.md"
    assert store.file_size("u1") == path.stat().st_size > 0

    assert store.edit_text("u1", "Hà Nội", "Huế") is True
    assert store.facts("u1") == {"name": "An", "city": "Huế"}
    assert store.edit_text("u1", "đoạn không tồn tại", "x") is False

    # upsert replaces a fact in place instead of appending a contradicting line
    assert store.upsert_fact("u1", "city", "Đà Nẵng") is True
    assert store.upsert_fact("u1", "city", "Đà Nẵng") is False
    assert store.read_text("u1").count("- city:") == 1
    assert store.upsert_fact("u1", "Fav Drink", "trà") is True  # any key, sanitized
    assert store.facts("u1")["fav_drink"] == "trà"

    assert store.remove_fact("u1", "city") is True
    assert "city" not in store.facts("u1")
    assert store.remove_fact("u1", "city") is False


def test_user_id_cannot_escape_profile_dir(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../../etc/passwd")
    assert (tmp_path / "profiles").resolve() in path.resolve().parents


# --- Fact extraction parsing ----------------------------------------------
def test_parse_profile_update_applies_confidence_threshold() -> None:
    raw = json.dumps(
        {
            "updates": {
                "city": {"value": "Huế", "confidence": 0.9},
                "job": {"value": "chắc là kỹ sư", "confidence": 0.4},
                "Fav Drink": {"value": "trà", "confidence": 0.7},
                "empty": {"value": "  ", "confidence": 1.0},
            },
            "remove": ["Old Key"],
        }
    )
    update = parse_profile_update(raw, min_confidence=0.7)
    assert update.updates == {"city": "Huế", "fav_drink": "trà"}
    assert update.skipped == {"job": 0.4}
    assert update.remove == ["old_key"]
    assert parse_profile_update(raw, min_confidence=0.3).updates["job"] == "chắc là kỹ sư"


def test_parse_accepts_unwrapped_values_and_joins_lists() -> None:
    raw = '{"updates": {"likes": ["Python", "RAG"], "city": "Huế", "pet": {"value": ["a", "b"], "confidence": 0.9}}}'
    update = parse_profile_update(raw, min_confidence=0.7)
    assert update.updates == {"likes": "Python, RAG", "city": "Huế", "pet": "a, b"}


@pytest.mark.parametrize(
    "raw",
    ["", "không phải json", "{", '["list"]', '{"updates": "sai kiểu"}', '{"updates": {"k": {"value": " "}}}'],
)
def test_malformed_model_output_writes_nothing(raw: str) -> None:
    update = parse_profile_update(raw, min_confidence=0.7)
    assert update.updates == {} and update.remove == []


def test_json_inside_code_fence_is_parsed() -> None:
    raw = '```json\n{"updates": {"city": {"value": "Huế", "confidence": 0.9}}, "remove": []}\n```'
    assert parse_profile_update(raw, 0.7).updates == {"city": "Huế"}


# --- Compact memory --------------------------------------------------------
def test_compact_trigger() -> None:
    manager = CompactMemoryManager(threshold_tokens=300, keep_messages=4)

    for i in range(8):
        manager.append("t1", "user", f"{i}: {LONG_TEXT}")
        manager.append("t1", "assistant", "ok")

    context = manager.context("t1")
    assert manager.compaction_count("t1") > 0
    assert len(context["messages"]) <= 5  # only the recent window stays verbatim
    assert context["summary"]  # older turns were folded into a summary
    assert manager.compaction_count("other-thread") == 0


def test_compact_passes_previous_summary_and_older_messages_to_summarizer() -> None:
    seen: list[tuple[str, list[dict[str, str]]]] = []

    def summarizer(previous: str, older: list[dict[str, str]]) -> str:
        seen.append((previous, older))
        return f"summary-{len(seen)}"

    manager = CompactMemoryManager(threshold_tokens=300, keep_messages=2, summarizer=summarizer)
    for i in range(6):
        manager.append("t1", "user", f"m{i}: {LONG_TEXT}")

    assert len(seen) >= 2
    assert seen[0][0] == ""  # first compaction has no previous summary
    assert seen[1][0] == "summary-1"  # later ones receive the rolling summary
    assert manager.context("t1")["summary"] == f"summary-{len(seen)}"
    kept = manager.context("t1")["messages"]
    assert all(m["content"] not in {o["content"] for o in seen[-1][1]} for m in kept)


def test_short_thread_does_not_compact() -> None:
    manager = CompactMemoryManager(threshold_tokens=300, keep_messages=4)
    manager.append("t1", "user", "Chào bạn")
    manager.append("t1", "assistant", "Chào")
    assert manager.compaction_count("t1") == 0
    assert manager.context("t1")["summary"] == ""


# --- Cross-session recall (plumbing) --------------------------------------
def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    teach = "Mình tên là An, mình thích trà đào."
    llm = ScriptedLLM({teach: {"updates": {"name": {"value": "An", "confidence": 0.95},
                                          "fav_drink": {"value": "trà đào", "confidence": 0.9}},
                               "remove": []}}, echo_system=True)
    advanced = AdvancedAgent(config, llm=llm)
    baseline = BaselineAgent(config, llm=llm)

    advanced.reply("u1", "session-1", teach)
    baseline.reply("u1", "session-1", teach)

    advanced_answer = advanced.reply("u1", "session-2", "Mình tên gì?")["reply"]
    baseline_answer = baseline.reply("u1", "session-2", "Mình tên gì?")["reply"]

    assert recall_points(advanced_answer, ["An", "trà đào"]) == 1.0  # profile reached the new thread's prompt
    assert recall_points(baseline_answer, ["An", "trà đào"]) == 0.0


def test_baseline_sends_only_the_current_thread(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    baseline = BaselineAgent(make_config(tmp_path), llm=llm)
    baseline.reply("u1", "t1", "tin nhắn A")
    baseline.reply("u1", "t1", "tin nhắn B")
    baseline.reply("u1", "t2", "tin nhắn C")

    contents = lambda call: [m["content"] for m in call]  # noqa: E731
    second, third = llm.calls[1], llm.calls[2]
    assert "tin nhắn A" in contents(second)  # same thread: short-term memory
    assert "tin nhắn A" not in contents(third) and "tin nhắn B" not in contents(third)  # new thread: nothing


def test_advanced_memory_survives_restart(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    msg = "Mình tên là An."
    script = {msg: fact("name", "An")}
    AdvancedAgent(config, llm=ScriptedLLM(script)).reply("u1", "t1", msg)

    restarted = AdvancedAgent(config, llm=ScriptedLLM(echo_system=True))  # new process, same state dir
    assert "An" in restarted.reply("u1", "t2", "Mình tên gì?")["reply"]


def test_users_do_not_share_memory(tmp_path: Path) -> None:
    msg = "Mình tên là Alice."
    advanced = AdvancedAgent(make_config(tmp_path), llm=ScriptedLLM({msg: fact("name", "Alice")}, echo_system=True))
    advanced.reply("alice", "t1", msg)
    assert "Alice" not in advanced.reply("bob", "t2", "Mình tên gì?")["reply"]


# --- Correction / confidence ----------------------------------------------
def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    first, second = "Mình ở Đà Nẵng.", "Đính chính: mình đang ở Huế."
    llm = ScriptedLLM({first: fact("city", "Đà Nẵng"), second: fact("city", "Huế")})
    advanced = AdvancedAgent(make_config(tmp_path), llm=llm)
    advanced.reply("u1", "t1", first)
    advanced.reply("u1", "t2", second)

    assert advanced.profile_store.facts("u1") == {"city": "Huế"}
    assert "Đà Nẵng" not in advanced.profile_store.read_text("u1")  # no stale fact kept


def test_extraction_prompt_shows_current_profile(tmp_path: Path) -> None:
    first, second = "Mình ở Đà Nẵng.", "Đính chính: mình đang ở Huế."
    llm = ScriptedLLM({first: fact("city", "Đà Nẵng")})
    advanced = AdvancedAgent(make_config(tmp_path), llm=llm)
    advanced.reply("u1", "t1", first)
    advanced.reply("u1", "t1", second)

    extraction_calls = [c for c in llm.calls if c[0]["content"] == EXTRACTION_PROMPT]
    assert "- city: Đà Nẵng" not in extraction_calls[0][-1]["content"]
    assert "- city: Đà Nẵng" in extraction_calls[1][-1]["content"]  # so the model can reuse the key


def test_low_confidence_fact_is_not_written(tmp_path: Path) -> None:
    msg = "Có lẽ mình sẽ chuyển sang nghề khác."
    advanced = AdvancedAgent(make_config(tmp_path), llm=ScriptedLLM({msg: fact("job", "nghề khác", 0.3)}))
    result = advanced.reply("u1", "t1", msg)

    assert result["profile_updates"] == {}
    assert advanced.profile_store.file_size("u1") == 0  # nothing written, file never created


def test_model_can_remove_a_fact(tmp_path: Path) -> None:
    add, drop = "Mình nuôi mèo.", "Mình không còn nuôi mèo nữa."
    llm = ScriptedLLM({add: fact("pet", "mèo"), drop: {"updates": {}, "remove": ["pet"]}})
    advanced = AdvancedAgent(make_config(tmp_path), llm=llm)
    advanced.reply("u1", "t1", add)
    advanced.reply("u1", "t1", drop)
    assert advanced.profile_store.facts("u1") == {}


# --- Prompt load -----------------------------------------------------------
def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, llm=ScriptedLLM())
    baseline = BaselineAgent(config, llm=ScriptedLLM())

    for i in range(12):
        message = f"Lượt {i}: {LONG_TEXT}"
        advanced.reply("u1", "long", message)
        baseline.reply("u1", "long", message)

    assert advanced.compaction_count("long") > 0
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")


def test_summary_reaches_the_prompt_after_compaction(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    advanced = AdvancedAgent(make_config(tmp_path), llm=llm)
    for i in range(8):
        advanced.reply("u1", "long", f"Lượt {i}: {LONG_TEXT}")

    assert advanced.compaction_count("long") > 0
    reply_prompts = [
        c for c in llm.calls if c[0]["content"] != EXTRACTION_PROMPT and not c[0]["content"].startswith(SUMMARY_MARKER)
    ]
    assert "- tóm tắt ngắn" in reply_prompts[-1][0]["content"]  # summary is in the system prompt
    assert "- tóm tắt ngắn" not in reply_prompts[0][0]["content"]  # but not before any compaction
    assert len(reply_prompts[-1]) <= 1 + 4 + 1  # system + kept window (+ current turn), not the full thread


def test_short_thread_costs_advanced_more_than_baseline(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    msgs = ["Mình tên là An.", "Mình ở Huế."]
    advanced = AdvancedAgent(config, llm=ScriptedLLM({msgs[0]: fact("name", "An"), msgs[1]: fact("city", "Huế")}))
    baseline = BaselineAgent(config, llm=ScriptedLLM())
    for message in msgs:
        advanced.reply("u1", "short", message)
        baseline.reply("u1", "short", message)

    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")  # User.md overhead


def test_memory_upkeep_cost_is_tracked_separately(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, llm=ScriptedLLM())
    for i in range(8):
        advanced.reply("u1", "long", f"Lượt {i}: {LONG_TEXT}")
    assert advanced.overhead_tokens > 0  # extraction + summary calls
    assert advanced.token_usage("long") == 8 * 1  # replies are "ok": 1 token each, upkeep not mixed in


# --- LLM requirement -------------------------------------------------------
def test_agents_require_an_llm(tmp_path: Path) -> None:
    config = make_config(tmp_path)  # live=False and no llm injected
    with pytest.raises(LLMUnavailableError):
        BaselineAgent(config)
    with pytest.raises(LLMUnavailableError):
        AdvancedAgent(config)


# --- Benchmark plumbing ----------------------------------------------------
def test_benchmark_run_reports_every_column(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    teach = "Mình tên là An."
    conversations = [
        {
            "id": "c1",
            "user_id": "u1",
            "turns": [teach, f"Một đoạn dài: {LONG_TEXT}", f"Thêm: {LONG_TEXT}", f"Nữa: {LONG_TEXT}"],
            "recall_questions": [{"question": "Mình tên gì?", "expected_contains": ["An"]}],
        }
    ]
    llm = ScriptedLLM({teach: fact("name", "An")}, echo_system=True)
    base = run_agent_benchmark("Baseline", BaselineAgent(config, llm=llm), conversations, config)
    adv = run_agent_benchmark("Advanced", AdvancedAgent(config, llm=llm), conversations, config)

    assert (base.recall_score, adv.recall_score) == (0.0, 1.0)
    assert base.memory_growth_bytes == 0 < adv.memory_growth_bytes
    assert base.compactions == 0 < adv.compactions
    assert adv.overhead_tokens > 0

    table = format_rows([base, adv])
    assert all(column in table for column in COLUMNS)


def test_benchmark_suites_run_on_real_datasets(tmp_path: Path) -> None:
    """Dataset-agnostic: with a model that learns nothing, we only check the suites run,
    state is isolated, and compaction fires on the long dataset."""

    config = make_config(tmp_path)
    standard = run_suite("standard", REPO_ROOT / "data" / "conversations.json", config, ScriptedLLM())
    stress = run_suite("stress", REPO_ROOT / "data" / "advanced_long_context.json", config, ScriptedLLM())

    assert standard[1].compactions == 0 and stress[1].compactions > 0
    assert stress[1].prompt_tokens_processed < stress[0].prompt_tokens_processed
    assert (config.state_dir / "benchmark" / "stress").is_dir()


def test_recall_points_and_quality() -> None:
    assert recall_points("Tên bạn là DũngCT, uống cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên bạn là dũngct", ["DũngCT", "cà phê sữa đá"]) == 0.5  # case-insensitive
    assert recall_points("Mình không biết", ["DũngCT"]) == 0.0
    assert heuristic_quality("DũngCT", ["DũngCT"]) == 1.0
    assert heuristic_quality("không biết", ["DũngCT"]) < 0.5


def test_datasets_load() -> None:
    assert len(load_conversations(REPO_ROOT / "data" / "conversations.json")) == 10
    assert len(load_conversations(REPO_ROOT / "data" / "advanced_long_context.json")) == 1


# --- Providers -------------------------------------------------------------
def test_normalize_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Google") == "gemini"
    assert normalize_provider("open-router") == "openrouter"
    assert normalize_provider("") == "openai"
    with pytest.raises(ValueError):
        normalize_provider("nope")


def test_invoke_text_flattens_content_blocks() -> None:
    class Blocks:
        def invoke(self, messages):
            return Reply([{"type": "text", "text": "xin "}, {"type": "text", "text": "chào"}])

    assert invoke_text(Blocks(), [{"role": "user", "content": "hi"}]) == "xin chào"
