"""
The Marlin agent: a LangGraph ReAct agent over `llm` (see app/llm.py), with
tools for structured facts (SQLite, app/fact_store.py) and semantic recall
(FAISS, app/vector_store.py).

Conversation state (message history) is owned by LangGraph's own
`AsyncSqliteSaver` checkpointer, keyed by `thread_id` — we use the
conversation's id as its thread_id, so switching sessions is just passing a
different id in `config`. This lives in the same SQLite file as
app/fact_store.py's `facts` table, but in LangGraph's own tables.

`build_agent()` must be called once at startup (see app/main.py's lifespan)
so the checkpointer's connection can be opened/closed alongside the app.
"""
from contextlib import AsyncExitStack

from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent

from app.config import get_settings
from app.fact_store import get_facts, save_fact
from app.llm import llm
from app.vector_store import index_for_recall, search_memory
from app.prompt import SYSTEM_PROMPT



_exit_stack = AsyncExitStack()
agent = None


@tool
async def remember_fact(conversation_id: str, key: str, value: str, runtime: ToolRuntime) -> str:
    """Save a structured fact (key/value) worth recalling later, e.g. a user preference."""
    runtime.stream_writer(f"Saving fact: {key}={value}")
    await save_fact(conversation_id, key, value)
    runtime.stream_writer(f"Saved fact: {key}={value}")
    return f"Saved fact: {key}={value}"


@tool
async def recall_facts(conversation_id: str, runtime: ToolRuntime) -> list[dict[str, str]]:
    """Return all structured facts saved for this conversation."""
    runtime.stream_writer("Recalling saved facts...")
    return await get_facts(conversation_id)


@tool
async def index_for_recall_tool(text: str, conversation_id: str, runtime: ToolRuntime) -> str:
    """Embed and store free-form text for later semantic search via `search_memory`."""
    runtime.stream_writer("Indexing text for recall...")
    await index_for_recall(text, conversation_id)
    return "Indexed for recall."


@tool
async def search_memory_tool(query: str, runtime: ToolRuntime) -> list[str]:
    """Semantically search previously indexed free-form memory."""
    runtime.stream_writer(f"Searching memory for: {query}")
    return await search_memory(query)


TOOLS = [remember_fact, recall_facts, index_for_recall_tool, search_memory_tool]


async def build_agent():
    """Open the SQLite checkpointer and compile the agent. Call once at startup."""
    global agent
    checkpointer = await _exit_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(get_settings().sqlite_path)
    )
    agent = create_agent(
        llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


async def close_agent() -> None:
    await _exit_stack.aclose()
