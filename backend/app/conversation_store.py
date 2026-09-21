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
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


async def init_conversation_store() -> None:
    """Create the `conversations` table if missing, and add any columns a
    table created by an older version of this schema is still missing (no
    migration framework here — just enough to keep existing SQLite files
    working as the schema grows). Call once at app startup."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(_CREATE_TABLE)
        cursor = await db.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "updated_at" not in columns:
            # SQLite only allows constant defaults in ADD COLUMN, so add it
            # nullable first, then backfill existing rows from created_at
            await db.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT")
            await db.execute(
                "UPDATE conversations SET updated_at = created_at WHERE updated_at IS NULL"
            )
        await db.commit()


async def create_conversation(thread_id: str, title: str = "") -> None:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(
            "INSERT INTO conversations (thread_id, title, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (thread_id, title),
        )
        await db.commit()


async def list_conversations(limit: int = 20) -> list[dict]:
    """Most recently created conversations first, up to `limit`."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT thread_id, title, created_at, updated_at FROM conversations "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def count_conversations() -> int:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM conversations")
        (count,) = await cursor.fetchone()
        return count


async def conversation_exists(thread_id: str) -> bool:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        cursor = await db.execute(
            "SELECT 1 FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        return await cursor.fetchone() is not None


async def touch_conversation(thread_id: str) -> None:
    """Bump `updated_at` to now — call whenever a message is added to the thread."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE thread_id = ?",
            (thread_id,),
        )
        await db.commit()


async def update_conversation_title(thread_id: str, title: str) -> bool:
    """Returns False if no conversation with `thread_id` exists."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        cursor = await db.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') "
            "WHERE thread_id = ?",
            (title, thread_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_conversation(thread_id: str) -> bool:
    """Deletes the conversation row and its JSON log. Returns False if no
    conversation with `thread_id` existed."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        cursor = await db.execute(
            "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        await db.commit()
        existed = cursor.rowcount > 0

    _log_path(thread_id).unlink(missing_ok=True)
    return existed



###json

def _log_path(thread_id: str) -> Path:
    return Path(get_settings().conversations_log_dir) / f"{thread_id}.json"


def _append_to_log(
    thread_id: str, user_message: str, reply: str, metadata: dict | None = None
) -> None:
    path = _log_path(thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() else []
    now = datetime.now(timezone.utc).isoformat()
    entries.append({"role": Role.user.value, "content": user_message, "created_at": now})
    assistant_entry = {"role": Role.assistant.value, "content": reply, "created_at": now}
    if metadata is not None:
        assistant_entry["metadata"] = metadata
    entries.append(assistant_entry)
    path.write_text(json.dumps(entries, indent=2))


def _read_log(thread_id: str, limit: int | None = None) -> list[dict]:
    """Just role/content per message, for showing conversation history in the
    frontend. With `limit`, returns only the most recent `limit` messages."""
    path = _log_path(thread_id)
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    if limit is not None:
        entries = entries[-limit:]
    result = []
    for entry in entries:
        item = {"role": entry["role"], "content": entry["content"]}
        if "metadata" in entry:
            item["metadata"] = entry["metadata"]
        result.append(item)
    return result