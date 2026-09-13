"""
Pydantic schemas for chat requests/responses.

These are the shapes the API speaks — kept separate from DB documents
(see db/documents.py) since the wire format and storage format are allowed
to evolve independently.
"""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    conversation_id: str
    reply: Message


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int
