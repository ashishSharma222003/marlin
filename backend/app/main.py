"""FastAPI app entrypoint. Run with: uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.env import load_env

load_env()  # must run before anything imports/builds the LLM (app.llm)

from app.agent import build_agent, close_agent
from app.config import get_settings
from app.endpoints import router
from app.fact_store import init_fact_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_fact_store()
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
