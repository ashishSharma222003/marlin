"""All API routes."""
from fastapi import APIRouter

from app.chat_service import ChatService
from app.models import ChatRequest, ChatResponse

router = APIRouter()
_chat_service = ChatService()


@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def send_message(request: ChatRequest) -> ChatResponse:
    """Send a message and get the assistant's reply. Omit `conversation_id`
    to start a new conversation; pass one back to continue it."""
    return await _chat_service.handle_message(request)
