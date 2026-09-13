"""OpenAI implementation of the LLMProvider interface."""
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.chat import Message, Role
from app.services.llm_providers.base import LLMProvider

DEFAULT_MAX_TOKENS = 1024


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    async def chat(self, messages: list[Message], **kwargs) -> str:
        conversation = [{"role": m.role.value, "content": m.content} for m in messages]

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            messages=conversation,
        )
        return response.choices[0].message.content or ""
