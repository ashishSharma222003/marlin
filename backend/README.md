# Marlin backend

FastAPI backend for the personal assistant.

## Layout

```
app/
├── main.py                    # FastAPI app creation, CORS, lifespan (DB connect/disconnect)
├── core/
│   └── config.py              # Settings, loaded from .env
├── api/
│   ├── router.py              # Aggregates all route modules
│   ├── deps.py                # FastAPI dependency wiring (DI)
│   └── routes/
│       ├── health.py          # GET /health
│       └── chat.py            # POST /chat
├── models/
│   └── chat.py                # Pydantic request/response schemas
├── services/
│   ├── chat_service.py         # Orchestrates one chat turn: history + LLM + persist
│   ├── llm_providers/          # Pluggable LLM backends (see below)
│   └── memory/                 # Conversation + (future) semantic memory
└── db/
    ├── mongodb.py              # Motor (async Mongo) connection lifecycle
    └── documents.py             # Mongo document shapes
```

## Setup

```bash
cd backend
python3 -m venv ../venv        # or reuse the repo's existing venv
source ../venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then fill in your API key(s)
```

You'll need a MongoDB instance running (local `mongod`, Docker, or Atlas) —
point `MONGODB_URI` at it in `.env`.

## Run

```bash
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## Extending

**Add a new LLM provider** (e.g. a local model, Gemini, ...):
1. Add a class in `app/services/llm_providers/` implementing `LLMProvider.chat()`
   (see `base.py`).
2. Register it in `app/services/llm_providers/registry.py`.
3. Set `LLM_PROVIDER=<your_key>` in `.env`. No other code changes needed.

**Add semantic/vector memory**: the interface is stubbed in
`app/services/memory/vector_store.py` — implement it against your chosen
vector DB and wire it into `chat_service.py` once you're ready.

**Add a new route/feature**: create `app/api/routes/<feature>.py` with its
own `APIRouter`, then include it in `app/api/router.py`.
