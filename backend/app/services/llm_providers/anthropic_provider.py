"""Anthropic (Claude) implementation of the LLMProvider interface."""
from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.models.chat import Message, Role
from app.services.llm_providers.base import LLMProvider

DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    async def chat(self, messages: list[Message], **kwargs) -> str:
        # Anthropic wants system prompts separate from the message list.
        system_prompt = "\n".join(m.content for m in messages if m.role == Role.system)
        conversation = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.system
        ]

        response = await self._client.messages.create(
            model=kwargs.get("model", self._model),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            system=system_prompt or None,
            messages=conversation,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
