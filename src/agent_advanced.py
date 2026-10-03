from __future__ import annotations

from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    llm_summarizer,
)
from model_provider import LLMUnavailableError, build_chat_model, invoke_text

SYSTEM_PROMPT = "Bạn là trợ lý hữu ích. Trả lời ngắn gọn bằng ngôn ngữ của người dùng."


class AdvancedAgent:
    """Agent B with three memory layers.

    1. Short-term: recent messages of the current thread (CompactMemoryManager).
    2. Persistent: `User.md`, one file per user, shared by every thread. After each user
       message the LLM decides which stable facts to add, correct or remove.
    3. Compact: older messages of a long thread folded into an LLM-written summary.

    Besides the reply call, each turn makes one extraction call, and compaction makes one
    summary call. Those are tracked in `overhead_tokens` so the extra cost stays visible.
    """

    def __init__(self, config: LabConfig | None = None, llm: Any | None = None) -> None:
        self.config = config or load_config()
        self.llm = llm or self._maybe_build_llm()
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
            summarizer=llm_summarizer(self._aux_call, max_lines=10),
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.overhead_tokens = 0

    def _maybe_build_llm(self):
        if not self.config.live:
            raise LLMUnavailableError(
                "No LLM configured. Set LLM_PROVIDER and its API key (see .env), or pass `llm=`."
            )
        return build_chat_model(self.config.model)

    def _aux_call(self, messages: list[dict[str, str]]) -> str:
        """LLM call for memory upkeep (extraction, summaries); its cost goes to overhead."""

        self.overhead_tokens += sum(estimate_tokens(m["content"]) for m in messages)
        out = invoke_text(self.llm, messages)
        self.overhead_tokens += estimate_tokens(out)
        return out

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Persistent memory: extract, then write User.md (a correction overwrites the old fact).
        update = self._update_profile(user_id, message)
        # Short-term memory; compaction may trigger inside append().
        self.compact_memory.append(thread_id, "user", message)

        prompt = self._build_prompt(user_id, thread_id)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        reply = invoke_text(self.llm, prompt)

        self.compact_memory.append(thread_id, "assistant", reply)
        agent_tokens = estimate_tokens(reply)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "reply": reply,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": update.updates,
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _update_profile(self, user_id: str, message: str):
        update = extract_profile_updates(
            self._aux_call,
            message,
            self.profile_store.read_text(user_id),
            self.config.profile_min_confidence,
        )
        for key, value in update.updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        for key in update.remove:
            self.profile_store.remove_fact(user_id, key)
        return update

    def _system_prompt(self, user_id: str, thread_id: str) -> str:
        summary = str(self.compact_memory.context(thread_id)["summary"])
        parts = [SYSTEM_PROMPT, "Hồ sơ người dùng (User.md):\n" + self.profile_store.read_text(user_id)]
        if summary:
            parts.append("Tóm tắt phần hội thoại cũ:\n" + summary)
        return "\n\n".join(parts)

    def _build_prompt(self, user_id: str, thread_id: str) -> list[dict[str, str]]:
        recent = self.compact_memory.context(thread_id)["messages"]
        return [{"role": "system", "content": self._system_prompt(user_id, thread_id)}, *recent]  # type: ignore[misc]

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: User.md + summary + recent kept messages."""

        return sum(estimate_tokens(m["content"]) for m in self._build_prompt(user_id, thread_id))
