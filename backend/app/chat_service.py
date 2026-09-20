"""
Orchestrates a single chat turn: invoke the agent (which owns history via
its SQLite checkpointer, keyed by conversation_id as thread_id), then
append the turn to this conversation's JSON transcript. Routes stay thin
and just call this.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from langchain_core.messages import HumanMessage

from app import agent as agent_module
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, Message, Role


def _log_path(conversation_id: str) -> Path:
    return Path(get_settings().conversations_log_dir) / f"{conversation_id}.json"


def _append_to_log(conversation_id: str, user_message: str, reply: str) -> None:
    path = _log_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() else []
    now = datetime.now(timezone.utc).isoformat()
    entries.append({"role": Role.user.value, "content": user_message, "created_at": now})
    entries.append({"role": Role.assistant.value, "content": reply, "created_at": now})
    path.write_text(json.dumps(entries, indent=2))


class ChatService:
    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        config = {"configurable": {"thread_id": conversation_id}}
        if request.model:
            config["configurable"]["model"] = request.model

        result = await agent_module.agent.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        reply_text = result["messages"][-1].content

        _append_to_log(conversation_id, request.message, reply_text)

        return ChatResponse(
            conversation_id=conversation_id,
            reply=Message(role=Role.assistant, content=reply_text),
        )
