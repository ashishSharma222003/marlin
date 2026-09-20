"""
Centralized app configuration, loaded from environment variables (.env).

Everything that might change between environments (dev/staging/prod) or
between deployments lives here, not scattered through the code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Marlin Assistant"
    debug: bool = False
    cors_origins: list[str] = ["*"]

    # --- LLM model selection ---
    # Default chat model, passed to LangChain's init_chat_model (see
    # app/llm.py). Provider is inferred from the model name; API keys
    # (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) are read straight from the
    # process environment by each provider's SDK — see app/env.py. Callers
    # can override the model per-request via ChatRequest.model.
    llm_model: str = "claude-sonnet-5"

    # --- Storage (SQLite) ---
    # One file holds both LangGraph's own checkpoint tables (conversation
    # state, keyed by conversation_id as the LangGraph thread_id) and our
    # own `facts` table (see app/fact_store.py).
    sqlite_path: str = "data/marlin.sqlite"

    # --- Conversation log ---
    # Write-only JSON transcript per conversation, for humans/export/
    # debugging. Never read back into the agent — SQLite is the source of
    # truth for actual conversation state.
    conversations_log_dir: str = "data/conversations"

    # --- Vector memory (FAISS, in-memory) ---
    vector_db_embedding_model: str = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — env is read once per process."""
    return Settings()
