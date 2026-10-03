from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from model_provider import build_chat_model, invoke_text


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    overhead_tokens: int = 0  # memory-upkeep LLM calls (fact extraction, summaries); not a table column


COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _hits(answer: str, expected: list[str]) -> int:
    haystack = _norm(answer)
    return sum(1 for item in expected if _norm(item) in haystack)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 when every expected fact appears, 0.5 when some do, 0 when none do."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality in [0, 1]: 80% fact coverage, 20% concision (<= 300 chars)."""

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    concise = 1.0 if len(answer) <= 300 else 300 / len(answer)
    return round(0.8 * coverage + 0.2 * concise, 4)


def judge_quality(judge, answer: str, question: str, expected: list[str]) -> float:
    """Ask the judge LLM for a 0-1 score; fall back to the heuristic if it misbehaves."""

    prompt = (
        "Chấm câu trả lời từ 0 đến 1 (chỉ trả về một số).\n"
        f"Câu hỏi: {question}\nCác ý bắt buộc: {', '.join(expected)}\nTrả lời: {answer}"
    )
    try:
        return max(0.0, min(1.0, float(invoke_text(judge, [{"role": "user", "content": prompt}]))))
    except Exception:
        return heuristic_quality(answer, expected)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    `judge` is an optional LLM for response quality; without it the heuristic is used.
    Each conversation is fed turn by turn in its own thread. Recall questions are then
    asked in FRESH threads, so only persistent memory can answer them.
    """

    size_of = getattr(agent, "memory_file_size", None)
    sizes_before: dict[str, int] = {}
    main_threads: list[str] = []
    all_threads: list[str] = []
    recalls: list[float] = []
    qualities: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        if size_of and user_id not in sizes_before:
            sizes_before[user_id] = size_of(user_id)

        main_thread = f"{conv['id']}-main"
        main_threads.append(main_thread)
        all_threads.append(main_thread)
        for turn in conv["turns"]:
            agent.reply(user_id, main_thread, turn)

        for index, item in enumerate(conv.get("recall_questions", [])):
            thread = f"{conv['id']}-recall-{index}"
            all_threads.append(thread)
            answer = agent.reply(user_id, thread, item["question"])["reply"]
            expected = item["expected_contains"]
            recalls.append(recall_points(answer, expected))
            if judge is not None:
                qualities.append(judge_quality(judge, answer, item["question"], expected))
            else:
                qualities.append(heuristic_quality(answer, expected))

    growth = sum(size_of(uid) - before for uid, before in sizes_before.items()) if size_of else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in all_threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in all_threads),
        recall_score=round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 4) if qualities else 0.0,
        memory_growth_bytes=growth,
        compactions=sum(agent.compaction_count(t) for t in main_threads),
        overhead_tokens=getattr(agent, "overhead_tokens", 0),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        text = tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        text = "\n".join(lines)

    if len(rows) == 2 and rows[0].prompt_tokens_processed:
        base, adv = rows
        delta = (adv.prompt_tokens_processed - base.prompt_tokens_processed) / base.prompt_tokens_processed
        text += (
            f"\n\nPrompt tokens: {adv.agent_name} vs {base.agent_name} = {delta:+.1%}"
            f" | Recall: {adv.recall_score:.2f} vs {base.recall_score:.2f}"
        )
        if adv.overhead_tokens:
            text += (
                f"\nNot in the table: {adv.agent_name} also spent ~{adv.overhead_tokens} tokens on "
                "fact-extraction and summary calls."
            )
    return text


def run_suite(name: str, dataset: Path, config, llm=None, judge=None) -> list[BenchmarkRow]:
    """Run baseline and advanced on one dataset with a clean, isolated state dir.

    `llm` / `judge` default to the models in `config`; pass fakes to test the plumbing.
    """

    state_dir = config.state_dir / "benchmark" / name
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)
    suite_config = replace(config, state_dir=state_dir)

    conversations = load_conversations(dataset)
    baseline = BaselineAgent(suite_config, llm=llm)
    advanced = AdvancedAgent(suite_config, llm=llm)
    return [
        run_agent_benchmark("Baseline", baseline, conversations, suite_config, judge),
        run_agent_benchmark("Advanced", advanced, conversations, suite_config, judge),
    ]


def main() -> None:
    """Run the standard benchmark and the long-context stress benchmark."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese output on Windows consoles

    config = load_config(Path(__file__).resolve().parent.parent)
    if not config.live:
        sys.exit(
            "The benchmark needs a real LLM: both agents call it for replies, and Advanced also "
            "for fact extraction and summaries.\nSet LLM_PROVIDER (+ API key / OLLAMA_BASE_URL) in "
            ".env or the environment, then rerun `python src/benchmark.py`."
        )
    llm = build_chat_model(config.model)
    judge = build_chat_model(config.judge_model)

    suites = [
        ("Standard Benchmark", "standard", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", "stress", config.data_dir / "advanced_long_context.json"),
    ]
    for title, name, dataset in suites:
        rows = run_suite(name, dataset, config, llm, judge)
        print(f"\n## {title}\n")
        print(format_rows(rows))
    print()


if __name__ == "__main__":
    main()
