from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import LLMUnavailableError, build_chat_model, invoke_text

SYSTEM_PROMPT = "Bạn là trợ lý hữu ích. Trả lời ngắn gọn bằng ngôn ngữ của người dùng."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Every turn sends the whole thread (system prompt + all earlier messages) to the LLM.
    - No `User.md`, no compaction.
    - A new `thread_id` starts empty, so long-term facts are forgotten by design.

    The chat model comes from `llm` (any object with `.invoke(messages)`) or, when
    `config.live` is set, from `build_chat_model(config.model)`.
    """

    def __init__(self, config: LabConfig | None = None, llm: Any | None = None) -> None:
        self.config = config or load_config()
        self.sessions: dict[str, SessionState] = {}
        self.llm = llm or self._maybe_build_llm()

    def _maybe_build_llm(self):
        if not self.config.live:
            raise LLMUnavailableError(
                "No LLM configured. Set LLM_PROVIDER and its API key (see .env), or pass `llm=`."
            )
        return build_chat_model(self.config.model)

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # `user_id` is deliberately unused: the baseline has no per-user memory.
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}, *session.messages]

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in prompt)
        reply = invoke_text(self.llm, prompt)

        session.messages.append({"role": "assistant", "content": reply})
        session.prompt_tokens_processed += prompt_tokens
        agent_tokens = estimate_tokens(reply)
        session.token_usage += agent_tokens
        return {"reply": reply, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())
