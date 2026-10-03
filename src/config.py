from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, has_live_credentials, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

_API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "custom": ("CUSTOM_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    `live` is True only when a provider was chosen explicitly (LLM_PROVIDER) and
    has credentials. The agents need an LLM for extraction, replies and summaries,
    so without `live` (or an injected model) they raise LLMUnavailableError.
    `profile_min_confidence` is the confidence an extracted fact needs to reach User.md.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live: bool = False
    profile_min_confidence: float = 0.7


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(root / ".env")


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _provider_config(provider_var: str, model_var: str, fallback: ProviderConfig | None) -> ProviderConfig:
    raw = os.getenv(provider_var)
    if raw is None and fallback is not None:
        return fallback
    provider = normalize_provider(raw)
    base_url = os.getenv(_BASE_URL_ENV[provider]) if provider in _BASE_URL_ENV else None
    if provider == "ollama" and not base_url:
        base_url = DEFAULT_OLLAMA_URL
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(model_var) or DEFAULT_MODELS[provider],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=_first_env(_API_KEY_ENV[provider]),
        base_url=base_url,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables (and optional `.env`) and return a LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM_PROVIDER", "LLM_MODEL", None)
    # The judge defaults to the main model unless JUDGE_PROVIDER / JUDGE_MODEL are set.
    judge_model = _provider_config("JUDGE_PROVIDER", "JUDGE_MODEL", model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        live=bool(os.getenv("LLM_PROVIDER")) and has_live_credentials(model),
        profile_min_confidence=float(os.getenv("PROFILE_MIN_CONFIDENCE", "0.7")),
    )
