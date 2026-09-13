"""
Async MongoDB connection management (via Motor).

Call `connect_to_mongo()` on app startup and `close_mongo_connection()` on
shutdown (wired up in app/main.py's lifespan). Everywhere else, just call
`get_database()` to get the current DB handle.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db_name]


async def close_mongo_connection() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_database() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected yet. Did the app startup lifespan run?"
        )
    return _db
