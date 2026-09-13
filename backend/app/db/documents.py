"""
Shape of the documents we store in MongoDB.

Kept as plain dict-building helpers rather than a heavier ODM (Beanie,
MongoEngine, ...) to keep the dependency surface small — swap in one later
if the schema grows complex enough to need migrations/validation baked in.
"""
from datetime import datetime, timezone
from typing import Any

CONVERSATIONS_COLLECTION = "conversations"
MESSAGES_COLLECTION = "messages"


def new_conversation_doc(conversation_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "_id": conversation_id,
        "title": None,
        "created_at": now,
        "updated_at": now,
    }


def new_message_doc(
    conversation_id: str, role: str, content: str
) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }
