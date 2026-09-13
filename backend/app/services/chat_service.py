"""
Orchestrates a single chat turn: load history, call the LLM, persist both
sides of the exchange. This is the one place that knows about both the LLM
layer and the memory layer — routes should stay thin and just call this.
"""
from app.models.chat import ChatRequest, ChatResponse, Message, Role
from app.services.llm_providers.base import LLMProvider
from app.services.memory.conversation_store import ConversationStore

SYSTEM_PROMPT = (
    "You are Marlin, a helpful personal assistant. Be concise and direct."
)


class ChatService:
    def __init__(self, llm: LLMProvider, store: ConversationStore) -> None:
        self._llm = llm
        self._store = store

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id
        if conversation_id is None or not await self._store.conversation_exists(conversation_id):
            conversation_id = await self._store.create_conversation()

        history = await self._store.get_history(conversation_id)
        await self._store.add_message(conversation_id, Role.user, request.message)

        prompt_messages = [
            Message(role=Role.system, content=SYSTEM_PROMPT),
            *history,
            Message(role=Role.user, content=request.message),
        ]
        reply_text = await self._llm.chat(prompt_messages)

        await self._store.add_message(conversation_id, Role.assistant, reply_text)

        return ChatResponse(
            conversation_id=conversation_id,
            reply=Message(role=Role.assistant, content=reply_text),
        )
