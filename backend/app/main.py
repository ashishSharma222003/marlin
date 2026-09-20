"""FastAPI app entrypoint. Run with: uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.env import load_env

load_env()  # must run before anything imports/builds the LLM (app.llm)

from app.agent import build_agent, close_agent
from app.config import get_settings
from app.conversation_store import init_conversation_store
from app.endpoints import router
from app.fact_store import init_fact_store


def _ensure_data_dirs() -> None:
    """Create the data/ dir (and its sqlite file's parent) and the
    conversations log dir if they don't exist yet — SQLite creates the
    .sqlite file itself, but not its containing directory."""
    settings = get_settings()
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.conversations_log_dir).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_data_dirs()
    await init_fact_store()
    await init_conversation_store()
    await build_agent()
    yield
    await close_agent()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
