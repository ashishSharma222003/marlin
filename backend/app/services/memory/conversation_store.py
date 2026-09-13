"""
Conversation history, backed by MongoDB.

This is the "short-term" / exact-history memory: the literal back-and-forth
of a conversation, fetched in order and replayed to the LLM as context. See
`services/memory/vector_store.py` for the separate semantic-recall layer.
"""
import uuid
from datetime import datetime, timezone

from app.db.documents import (
    CONVERSATIONS_COLLECTION,
    MESSAGES_COLLECTION,
    new_conversation_doc,
    new_message_doc,
)
from app.db.mongodb import get_database
from app.models.chat import Message, Role


class ConversationStore:
    """CRUD over conversations/messages. One instance is stateless/cheap —
    create fresh per request via the `get_conversation_store` dependency."""

    async def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        db = get_database()
        await db[CONVERSATIONS_COLLECTION].insert_one(
            new_conversation_doc(conversation_id)
        )
        return conversation_id

    async def conversation_exists(self, conversation_id: str) -> bool:
        db = get_database()
        doc = await db[CONVERSATIONS_COLLECTION].find_one({"_id": conversation_id})
        return doc is not None

    async def add_message(self, conversation_id: str, role: Role, content: str) -> None:
        db = get_database()
        await db[MESSAGES_COLLECTION].insert_one(
            new_message_doc(conversation_id, role.value, content)
        )
        await db[CONVERSATIONS_COLLECTION].update_one(
            {"_id": conversation_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}},
        )

    async def get_history(self, conversation_id: str) -> list[Message]:
        db = get_database()
        cursor = (
            db[MESSAGES_COLLECTION]
            .find({"conversation_id": conversation_id})
            .sort("created_at", 1)
        )
        return [
            Message(role=Role(doc["role"]), content=doc["content"], created_at=doc["created_at"])
            async for doc in cursor
        ]


def get_conversation_store() -> ConversationStore:
    """FastAPI dependency factory."""
    return ConversationStore()
