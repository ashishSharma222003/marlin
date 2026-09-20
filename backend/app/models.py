"""
Pydantic schemas for chat requests/responses.

These are the shapes the API speaks — kept separate from storage formats
(SQLite checkpoints, the JSON conversation log) since the wire format and
storage format are allowed to evolve independently.
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
    thread_id: str | None = None  # omit to start a new conversation
    model: str | None = None  # e.g. "claude-sonnet-5"; omit to use the default


class ChatResponse(BaseModel):
    thread_id: str
    reply: Message


class CreateConversationRequest(BaseModel):
    title: str = ""  # optional label for this conversation; empty by default


class ConversationResponse(BaseModel):
    thread_id: str
    title: str
