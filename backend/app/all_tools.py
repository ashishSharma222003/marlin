from langchain.tools import tool, ToolRuntime
from app.vector_store import index_for_recall, search_memory
from app.fact_store import get_facts, save_fact

@tool
async def remember_fact(key: str, value: str, runtime: ToolRuntime) -> str:
    """Save a structured fact (key/value) worth recalling later, e.g. a user preference."""
    runtime.stream_writer(f"Saving fact: {key}={value}")
    thread_id = runtime.config["configurable"]["thread_id"]
    await save_fact(thread_id, key, value)
    runtime.stream_writer(f"Saved fact: {key}={value}")
    return f"Saved fact: {key}={value}"


@tool
async def recall_facts(runtime: ToolRuntime) -> list[dict[str, str]]:
    """Return all structured facts saved for this conversation."""
    runtime.stream_writer("Recalling saved facts...")
    thread_id = runtime.config["configurable"]["thread_id"]
    facts = await get_facts(thread_id)
    if not facts:
        runtime.stream_writer("No saved facts found.")
        return [{"info": "I don't have much facts about the user yet."}]
    return facts


@tool
async def index_for_recall_tool(text: str, runtime: ToolRuntime) -> str:
    """Embed and store free-form text for later semantic search via `search_memory`."""
    runtime.stream_writer("Indexing text for recall...")
    thread_id = runtime.config["configurable"]["thread_id"]
    await index_for_recall(text, thread_id)
    return "Indexed for recall."


@tool
async def search_memory_tool(query: str, runtime: ToolRuntime) -> list[str]:
    """Semantically search previously indexed free-form memory for this conversation."""
    runtime.stream_writer(f"Searching memory for: {query}")
    thread_id = runtime.config["configurable"]["thread_id"]
    return await search_memory(query, conversation_id=thread_id)

@tool
async def runtimetool(runtime: ToolRuntime) -> str:
    """A tool that demonstrates how to use the ToolRuntime to stream output."""
    runtime.stream_writer("This is a message from the tool runtime.")
    print(runtime)
    return "Tool runtime executed."


TOOLS = [remember_fact, recall_facts, index_for_recall_tool, search_memory_tool]