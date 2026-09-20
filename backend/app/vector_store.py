"""
In-process semantic memory: FAISS (in-memory index) + FastEmbed embeddings.

Populated only when the agent calls the `index_for_recall` tool, and
queried only when it calls `search_memory` (see app/agent.py) — nothing is
auto-embedded per message. A single shared index for the process; it does
not persist across restarts.
"""
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import get_settings

_embeddings = FastEmbedEmbeddings(model_name=get_settings().vector_db_embedding_model)
_store: FAISS | None = None


def _get_store() -> FAISS:
    global _store
    if _store is None:
        _store = FAISS.from_texts(["__init__"], _embeddings, metadatas=[{"init": True}])
        _store.delete([_store.index_to_docstore_id[0]])
    return _store


async def index_for_recall(text: str, conversation_id: str) -> None:
    """Embed `text` and store it, tagged with `conversation_id`."""
    store = _get_store()
    await store.aadd_texts([text], metadatas=[{"conversation_id": conversation_id}])


async def search_memory(query: str, top_k: int = 5) -> list[str]:
    """Return the `top_k` most semantically similar stored texts."""
    store = _get_store()
    if store.index.ntotal == 0:
        return []
    docs = await store.asimilarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]
