"""
Maps the `LLM_PROVIDER` setting to a concrete provider implementation.

Add a new provider by importing it here and adding one line to `_PROVIDERS`.
Provider classes are instantiated lazily (only the selected one is ever
constructed), so adding an entry never requires that provider's SDK/API key
to be installed/configured unless it's actually selected.
"""
from functools import lru_cache
from typing import Callable

from app.core.config import get_settings
from app.services.llm_providers.base import LLMProvider


def _make_anthropic() -> LLMProvider:
    from app.services.llm_providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _make_openai() -> LLMProvider:
    from app.services.llm_providers.openai_provider import OpenAIProvider

    return OpenAIProvider()


_PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    "anthropic": _make_anthropic,
    "openai": _make_openai,
}


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Instantiate (once) the provider selected via settings.llm_provider."""
    settings = get_settings()
    key = settings.llm_provider.lower()
    factory = _PROVIDERS.get(key)
    if factory is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown LLM provider '{settings.llm_provider}'. "
            f"Available: {available}"
        )
    return factory()
