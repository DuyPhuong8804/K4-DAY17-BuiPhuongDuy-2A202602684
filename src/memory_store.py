from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate: ~4 characters per token."""

    text = (text or "").strip()
    if not text:
        return 0
    return math.ceil(len(text) / 4)


# --------------------------------------------------------------------------- #
# User.md persistence
# --------------------------------------------------------------------------- #
DEFAULT_PROFILE = "# User Profile\n"
_FACT_LINE = re.compile(r"^- (?P<key>\w+): (?P<value>.*)$")


def sanitize_key(key: str) -> str:
    """Fact keys are free-form (chosen by the LLM) but stored as snake_case words."""

    return re.sub(r"\W+", "_", str(key).strip().lower()).strip("_")


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user.

    Layout: `<root_dir>/<user-slug>/User.md`, with facts as `- key: value` lines.
    The store has no fixed schema: any key is accepted.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]+", "_", (user_id or "").strip()).strip("_")
        return self.root_dir / (slug or "anonymous") / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line)
            if match:
                found[match["key"]] = match["value"].strip()
        return found

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one fact line. A new value overwrites the old one, so a
        correction never leaves the stale fact behind. Returns True when the file changed."""

        key = sanitize_key(key)
        value = re.sub(r"\s+", " ", str(value)).strip()
        if not key or not value:
            return False
        if self.facts(user_id).get(key) == value:
            return False
        content = self.read_text(user_id)
        line = f"- {key}: {value}"
        pattern = re.compile(rf"^- {re.escape(key)}: .*$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(lambda _m: line, content, count=1)
        else:
            content = content.rstrip("\n") + "\n" + line + "\n"
        self.write_text(user_id, content)
        return True

    def remove_fact(self, user_id: str, key: str) -> bool:
        key = sanitize_key(key)
        content = self.read_text(user_id)
        pattern = re.compile(rf"^- {re.escape(key)}: .*\n?", re.MULTILINE)
        if not pattern.search(content):
            return False
        self.write_text(user_id, pattern.sub("", content, count=1))
        return True


# --------------------------------------------------------------------------- #
# LLM-based fact extraction
# --------------------------------------------------------------------------- #
EXTRACTION_PROMPT = """You maintain a long-term profile of a user (a markdown file, User.md).
Given the CURRENT PROFILE and the user's LATEST MESSAGE, decide what to change.

Return ONLY a JSON object, no prose:
{"updates": {"<snake_case_key>": {"value": "<text>", "confidence": <0.0-1.0>}}, "remove": ["<key>"]}

Rules:
- Record only facts the user states about THEMSELVES that stay true across conversations
  (identity, residence, occupation, lasting preferences, interests, habits, relationships, pets).
- Do not record questions, requests to the assistant, jokes, hypotheticals, one-off plans,
  or facts about other people or places mentioned in passing.
- Reuse an existing key when the new message updates or corrects it, and give the full new value
  (for a list-like fact, merge the old and new items). A correction must replace the old value.
- Use "remove" for a fact the user says is no longer true and has no replacement.
- Every update MUST use the {"value": ..., "confidence": ...} form, and "value" is a single string
  (join list-like items with commas).
- confidence is how sure you are that this is a stable fact about the user.
- If nothing should change, return {"updates": {}, "remove": []}.
"""


@dataclass
class ProfileUpdate:
    updates: dict[str, str] = field(default_factory=dict)
    remove: list[str] = field(default_factory=list)
    skipped: dict[str, float] = field(default_factory=dict)  # key -> confidence, below threshold


def _parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _value_text(value: object) -> str:
    """Fact values are text; list-like values become a readable comma-separated string."""

    if isinstance(value, (list, tuple)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def parse_profile_update(raw: str, min_confidence: float) -> ProfileUpdate:
    """Turn the model's JSON into a ProfileUpdate, applying the confidence threshold.

    Malformed output yields an empty update: failing to learn a fact is safer than
    writing a wrong one.
    """

    data = _parse_json_object(raw)
    result = ProfileUpdate()
    updates = data.get("updates")
    for key, item in (updates.items() if isinstance(updates, dict) else []):
        key = sanitize_key(key)
        if not key:
            continue
        if isinstance(item, dict):
            value, raw_confidence = item.get("value", ""), item.get("confidence", 0)
        else:
            # The model skipped the {"value", "confidence"} wrapper. It still meant to record
            # the value; dropping it silently loses facts, so accept it as an unscored update.
            value, raw_confidence = item, 1.0
        value = _value_text(value)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if not value:
            continue
        if confidence >= min_confidence:
            result.updates[key] = value
        else:
            result.skipped[key] = confidence
    removals = data.get("remove")
    result.remove = [sanitize_key(k) for k in removals if sanitize_key(k)] if isinstance(removals, list) else []
    return result


def extract_profile_updates(
    call: Callable[[list[dict[str, str]]], str],
    message: str,
    current_profile: str,
    min_confidence: float,
) -> ProfileUpdate:
    """Ask the LLM (via `call(messages) -> text`) which stable facts the message adds,
    corrects or removes."""

    raw = call(
        [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": f"CURRENT PROFILE:\n{current_profile}\n\nLATEST MESSAGE:\n{message}",
            },
        ],
    )
    return parse_profile_update(raw, min_confidence)


# --------------------------------------------------------------------------- #
# Compact memory
# --------------------------------------------------------------------------- #
SUMMARY_PROMPT = """You compress the older part of a conversation so it can be dropped from the prompt.
Merge the PREVIOUS SUMMARY with the OLDER MESSAGES into one updated summary of at most {max_lines} short
bullet lines. Keep what is needed to continue the conversation (topics, open questions, decisions,
the user's stated opinions). Stable personal facts already live in a separate profile, so do not
spend space repeating them. Respond with the summary only, in the user's language."""

Summarizer = Callable[[str, list[dict[str, str]]], str]


def summarize_messages(
    messages: list[dict[str, str]], max_items: int = 6, max_chars: int = 140
) -> str:
    """Model-free fallback summary: one truncated line per user message.

    Used only when no summarizer is injected. It is lossy by construction.
    """

    lines = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {text}")
    return "\n".join(lines[-max_items:])


def llm_summarizer(call: Callable[[list[dict[str, str]]], str], max_lines: int) -> Summarizer:
    """Build a summarizer from an LLM call function `call(messages) -> text`."""

    def summarize(previous_summary: str, older: list[dict[str, str]]) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        return call(
            [
                {"role": "system", "content": SUMMARY_PROMPT.format(max_lines=max_lines)},
                {
                    "role": "user",
                    "content": f"PREVIOUS SUMMARY:\n{previous_summary or '(none)'}\n\nOLDER MESSAGES:\n{transcript}",
                },
            ]
        )

    return summarize


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Recent messages stay verbatim. When summary + messages exceed `threshold_tokens`,
    everything except the last `keep_messages` messages is folded into the summary
    and `compactions` is incremented. `summarizer(previous_summary, older_messages)`
    produces the new summary; without one, the model-free fallback is used.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_max_lines: int = 10
    summarizer: Summarizer | None = None

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def _tokens(self, thread: dict[str, object]) -> int:
        return estimate_tokens(str(thread["summary"])) + sum(
            estimate_tokens(m["content"]) for m in thread["messages"]  # type: ignore[index]
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        if self._tokens(thread) > self.threshold_tokens and len(messages) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        cut = len(messages) - self.keep_messages
        older, kept = messages[:cut], messages[cut:]
        previous = str(thread["summary"])
        if self.summarizer is not None:
            summary = self.summarizer(previous, older)
        else:
            new_lines = summarize_messages(older, max_items=self.summary_max_lines).splitlines()
            summary = "\n".join((previous.splitlines() + new_lines)[-self.summary_max_lines :])
        thread["messages"], thread["summary"] = kept, summary
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[call-overload]

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])  # type: ignore[call-overload]
