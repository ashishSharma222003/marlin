"""
Builds the LangChain chat model used by the agent.

`init_chat_model()` called with just a default `model` (no other bound
config) returns a `ConfigurableModel` — a single instance, built once at
import time, whose actual model is picked per-call via
`config={"configurable": {"model": ...}}` on `.invoke()`/`.ainvoke()`. That
means a conversation (or a single request) can switch models without
rebuilding any client. The provider is inferred from the model name itself
(e.g. "claude-..." vs "gpt-...") — no provider field to pass around.

Each provider's SDK reads its API key straight from the process
environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) — see `app.env.load_env()`,
called at app startup, which is what actually puts `.env` into
`os.environ` for this to work.
"""
from langchain.chat_models import init_chat_model

from app.config import get_settings

llm = init_chat_model(
    model=get_settings().llm_model
)
