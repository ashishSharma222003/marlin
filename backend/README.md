# Marlin backend

FastAPI backend for the personal assistant, built around a LangGraph agent.

## Layout

```
app/
├── main.py          # FastAPI app creation, CORS, lifespan (agent + fact store startup)
├── endpoints.py      # All routes: GET /health, POST /chat
├── agent.py          # LangGraph ReAct agent, tools, SQLite checkpointer
├── llm.py            # LangChain chat model construction
├── chat_service.py   # Orchestrates one chat turn: invoke agent, write JSON log
├── fact_store.py      # SQLite `facts` table (structured facts the agent saves)
├── vector_store.py    # FAISS in-memory semantic recall
├── models.py          # Pydantic request/response schemas
├── config.py           # Settings, loaded from .env
└── env.py               # Loads .env into the process environment
```

## Memory model

- **Conversation history** — owned by LangGraph's `AsyncSqliteSaver`
  checkpointer, keyed by `conversation_id` (used as the LangGraph
  `thread_id`). Switching sessions is just passing a different
  `conversation_id`. Lives in `SQLITE_PATH`.
- **Structured facts** — a separate `facts` table in the same SQLite file
  (`app/fact_store.py`). The agent writes to it only when it calls the
  `remember_fact` tool; nothing is auto-saved.
- **Semantic memory** — an in-memory FAISS index (`app/vector_store.py`),
  populated only when the agent calls `index_for_recall`, queried via
  `search_memory`. Does not persist across restarts.
- **JSON transcript log** — `app/chat_service.py` appends every turn to
  `CONVERSATIONS_LOG_DIR/{conversation_id}.json`. This is a write-only
  mirror for humans/export/debugging — it's never read back into the agent.

## Setup

```bash
cd backend
python3 -m venv ../venv        # or reuse the repo's existing venv
source ../venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then fill in your API key(s)
```

## Run

```bash
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## Extending

**Switch or add a new LLM provider** (e.g. a local model, Gemini, ...):
`app/llm.py` builds one configurable chat model via LangChain's
`init_chat_model()` at startup; the provider is inferred from the model
name. Set `LLM_MODEL` in `.env` (default), or pass `model` in a `/chat`
request to switch per-call — no rebuild needed either way. Install that
provider's `langchain-*` integration package and make sure its API key env
var (e.g. `ANTHROPIC_API_KEY`) is set; `app/env.py` loads `.env` into the
process environment at startup so the SDK can read it directly.

**Add a new agent tool**: define it with `@tool` in `app/agent.py` and add
it to the `TOOLS` list.

**Add a new route**: add it to `app/endpoints.py`.
