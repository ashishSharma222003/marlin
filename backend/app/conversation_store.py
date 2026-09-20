"""
Durable record of created conversations, so a conversation exists as soon
as it's created (not only once the first message is sent). Lives in the
same SQLite file as the checkpointer and app/fact_store.py, in its own
`conversations` table.

thread_id is the LangGraph thread_id (see app/agent.py) and also the JSON
log filename (see app/endpoints.py) — there's no separate conversation_id
today.
"""
import aiosqlite

from app.config import get_settings
from pathlib import Path
import json
from datetime import datetime, timezone
from app.models import Role

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


async def init_conversation_store() -> None:
    """Create the `conversations` table if missing. Call once at app startup."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()


async def create_conversation(thread_id: str, title: str = "") -> None:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(
            "INSERT INTO conversations (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
        await db.commit()



###json

def _log_path(thread_id: str) -> Path:
    return Path(get_settings().conversations_log_dir) / f"{thread_id}.json"


def _append_to_log(thread_id: str, user_message: str, reply: str) -> None:
    path = _log_path(thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() else []
    now = datetime.now(timezone.utc).isoformat()
    entries.append({"role": Role.user.value, "content": user_message, "created_at": now})
    entries.append({"role": Role.assistant.value, "content": reply, "created_at": now})
    path.write_text(json.dumps(entries, indent=2))