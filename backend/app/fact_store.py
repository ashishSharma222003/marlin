"""
Structured facts the agent chooses to remember (e.g. user preferences,
extracted entities) — distinct from conversation history, which LangGraph's
own SQLite checkpointer owns (see app/agent.py). Lives in the same SQLite
file as the checkpointer, in its own `facts` table.

Written only when the agent calls the `remember_fact` tool — never
auto-populated from every message.
"""
import aiosqlite

from app.config import get_settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


async def init_fact_store() -> None:
    """Create the `facts` table if missing. Call once at app startup."""
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()


async def save_fact(conversation_id: str, key: str, value: str) -> None:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        await db.execute(
            "INSERT INTO facts (conversation_id, key, value) VALUES (?, ?, ?)",
            (conversation_id, key, value),
        )
        await db.commit()


async def get_facts(conversation_id: str) -> list[dict[str, str]]:
    async with aiosqlite.connect(get_settings().sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT key, value, created_at FROM facts WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
