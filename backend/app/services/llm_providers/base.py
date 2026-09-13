"""
Provider-agnostic interface for chat completions.

To add a new LLM provider (OpenAI, local Ollama, Gemini, ...):
  1. Create a new file in this package, e.g. `openai_provider.py`.
  2. Implement a class that subclasses `LLMProvider` and fills in `chat()`.
  3. Register it in `registry.py`.
No other code needs to change — routes and services only ever depend on
this `LLMProvider` interface, never on a concrete provider.
"""
from abc import ABC, abstractmethod

from app.models.chat import Message


class LLMProvider(ABC):
    """Common interface every chat-completion backend must implement."""

    @abstractmethod
    async def chat(self, messages: list[Message], **kwargs) -> str:
        """
        Send the full message history to the model and return its reply
        text. `messages` is ordered oldest-first and includes the latest
        user message.
        """
        raise NotImplementedError
