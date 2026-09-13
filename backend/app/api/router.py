"""
Aggregates all route modules into one router mounted by main.py.

Adding a new feature area (e.g. "tools", "memory search", "voice"):
  1. Create app/api/routes/<feature>.py with its own `router = APIRouter(...)`.
  2. Import and include it below.
"""
from fastapi import APIRouter

from app.api.routes import chat, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
