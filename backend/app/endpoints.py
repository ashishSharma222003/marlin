"""All API routes."""
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage

from app import agent as agent_module
from app.config import get_settings
from app.conversation_store import (
    count_conversations,
    create_conversation,
    list_conversations,
    _append_to_log,
    _log_path,
    _read_log,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationCountResponse,
    ConversationListItem,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationResponse,
    CreateConversationRequest,
    Message,
    Role,
)

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"





@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/conversations", response_model=ConversationResponse, tags=["chat"])
async def start_conversation(
    request: CreateConversationRequest = CreateConversationRequest(),
) -> ConversationResponse:
    new_id = str(uuid.uuid4())
    await create_conversation(new_id, request.title)
    return ConversationResponse(thread_id=new_id, title=request.title)


@router.get("/conversations", response_model=ConversationListResponse, tags=["chat"])
async def get_conversations(limit: int = 20) -> ConversationListResponse:
    """List the most recent conversation threads (thread_id, title,
    created_at) — most recently created first."""
    rows = await list_conversations(limit)
    conversations = [
        ConversationListItem(
            thread_id=row["thread_id"],
            title=row["title"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return ConversationListResponse(conversations=conversations)


@router.get("/conversations/count", response_model=ConversationCountResponse, tags=["chat"])
async def get_conversation_count() -> ConversationCountResponse:
    """Total number of conversation sessions."""
    return ConversationCountResponse(count=await count_conversations())


@router.get(
    "/conversations/{thread_id}/messages",
    response_model=ConversationMessagesResponse,
    tags=["chat"],
)
async def get_conversation_messages(thread_id: str, limit: int = 20) -> ConversationMessagesResponse:
    """Last `limit` messages exchanged between user and agent in `thread_id`."""
    path = _log_path(thread_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = [Message(**m) for m in _read_log(thread_id, limit=limit)]
    return ConversationMessagesResponse(thread_id=thread_id, messages=messages)


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def send_message(request: ChatRequest) -> ChatResponse:
    """Send a message and get the assistant's reply. Omit `thread_id`
    to start a new conversation; pass one back to continue it."""
    thread_id = request.thread_id
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        await create_conversation(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    if request.model:
        config["configurable"]["model"] = request.model

    result = await agent_module.agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    print(f"Agent result: {result}")
    reply_text = result["messages"][-1].content

    _append_to_log(thread_id, request.message, reply_text)

    return ChatResponse(
        thread_id=thread_id,
        reply=Message(role=Role.assistant, content=reply_text),
    )


@router.post("/chat/stream", tags=["chat"])
async def stream_message(request: ChatRequest) -> StreamingResponse:
    """Like `/chat`, but streams the assistant's reply as it's generated
    over Server-Sent Events. Emits `progress` events from tools' `stream_writer`
    calls, `token` events with incremental reply text, then a final `done`
    event with the full reply."""
    thread_id = request.thread_id
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        await create_conversation(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    if request.model:
        config["configurable"]["model"] = request.model

    async def event_stream():
        chunks: list[str] = []
        try:
            async for stream_mode, payload in agent_module.agent.astream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode=["messages", "custom"],
            ):
                if stream_mode == "custom":
                    yield _sse("progress", {"content": payload})
                    continue

                message_chunk, _metadata = payload
                if not isinstance(message_chunk, AIMessageChunk):
                    continue
                text = message_chunk.text()
                if not text:
                    continue
                chunks.append(text)
                yield _sse("token", {"content": text})
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})
            return

        reply_text = "".join(chunks)
        _append_to_log(thread_id, request.message, reply_text)
        yield _sse("done", {"thread_id": thread_id, "reply": reply_text})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
