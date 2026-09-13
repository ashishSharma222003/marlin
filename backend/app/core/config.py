"""
Centralized app configuration, loaded from environment variables (.env).

Everything that might change between environments (dev/staging/prod) or
between deployments lives here, not scattered through the code.
"""
from functools import lru_cache

from pydantic import Field
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

    # --- LLM provider selection ---
    # Which backend to use for chat completions. New providers just need to
    # register themselves in services/llm_providers/registry.py under a new
    # key here.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5"

    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # --- MongoDB (conversation memory) ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "marlin"

    # --- Vector DB (semantic memory, optional / future) ---
    vector_db_provider: str | None = None  # e.g. "chroma", "qdrant", "pinecone"
    vector_db_url: str | None = None
    vector_db_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — env is read once per process."""
    return Settings()
