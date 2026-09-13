"""Chat endpoints."""
from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Send a message and get the assistant's reply. Omit `conversation_id`
    to start a new conversation; pass one back to continue it."""
    return await chat_service.handle_message(request)
