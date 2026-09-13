"""Shared FastAPI dependencies (DI wiring for routes)."""
from app.services.chat_service import ChatService
from app.services.llm_providers.registry import get_llm_provider
from app.services.memory.conversation_store import get_conversation_store


def get_chat_service() -> ChatService:
    return ChatService(llm=get_llm_provider(), store=get_conversation_store())
