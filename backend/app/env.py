"""
Loads `.env` into the real process environment (os.environ), not just into
our own `Settings` object. `pydantic-settings` only populates `Settings`
fields from `.env` — it never touches `os.environ`. Provider SDKs (used
directly by LangChain's `init_chat_model`) read API keys from `os.environ`
themselves, so this must run before any chat model is constructed.
"""
from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv()
