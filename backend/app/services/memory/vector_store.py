"""
Semantic ("long-term") memory interface — NOT wired up yet.

This is a placeholder contract so the rest of the app (and future you) has
a stable shape to build against once a vector DB is chosen. To implement:

  1. Pick a backend (Chroma, Qdrant, Pinecone, pgvector, ...).
  2. Create e.g. `chroma_store.py` with a class implementing `VectorStore`.
  3. Wire an embedding model (Anthropic doesn't serve embeddings directly —
     e.g. voyageai pairs well with Claude, or use OpenAI/local embeddings).
  4. Register it similarly to how `llm_providers/registry.py` works, keyed
     off `settings.vector_db_provider`.
  5. Call `.upsert()` when storing messages/facts worth recalling later,
     and `.search()` before building the prompt to inject relevant context.

Everything here is deliberately unimplemented until that's decided.
"""
from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Embed `text` and store it, keyed by `doc_id`, with `metadata`."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the `top_k` most semantically similar stored items."""
        raise NotImplementedError
