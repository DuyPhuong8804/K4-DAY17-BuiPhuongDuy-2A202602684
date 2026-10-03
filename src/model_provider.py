from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "": "openai",
    "gpt": "openai",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "vllm": "custom",
    "lmstudio": "custom",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "claude": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by both agents and the judge.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None  # per-call output cap; OpenRouter reserves credit for the full default


def normalize_provider(value: str | None) -> str:
    """Map aliases and typos (e.g. `anthorpic`) onto one of SUPPORTED_PROVIDERS."""

    key = (value or "").strip().lower()
    key = _PROVIDER_ALIASES.get(key, key)
    if key not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return key


class LLMUnavailableError(RuntimeError):
    """Raised when an agent needs an LLM but none is configured."""


def invoke_text(llm, messages: list[dict[str, str]]) -> str:
    """Call a chat model with `{"role", "content"}` dicts and return the reply as plain text."""

    content = llm.invoke(messages).content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content).strip()


def has_live_credentials(config: ProviderConfig) -> bool:
    """True when the config has what a live call needs (key, or a local base URL)."""

    provider = normalize_provider(config.provider)
    if provider == "ollama":
        return bool(config.base_url)
    if provider == "custom":
        return bool(config.base_url)
    return bool(config.api_key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    SDK imports are lazy so offline runs and tests never need a provider package
    to be importable.
    """

    provider = normalize_provider(config.provider)
    common = {"temperature": config.temperature}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": config.model_name, **common}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("provider 'custom' requires a base_url")
            kwargs["base_url"] = config.base_url
            kwargs.setdefault("api_key", "not-needed")
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name, google_api_key=config.api_key, **common
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name, api_key=config.api_key, **common
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, **common}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    from langchain_openrouter import ChatOpenRouter

    if config.max_tokens:
        common["max_tokens"] = config.max_tokens
    return ChatOpenRouter(
        model_name=config.model_name, openrouter_api_key=config.api_key, **common
    )
