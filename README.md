# CodeCopilot

AI-powered code search and Q&A over Python repositories. Point it at a
GitHub repo, ask a question in natural language, and get a grounded
answer with citations to specific files and line ranges.

> **Status:** active development. Phases 0–2 of 5 complete. The HTTP
> API streams cited answers end-to-end against any indexed repo. Multi-mode
> reasoning, web UI, evaluation pipeline, and public deployment are
> upcoming. See [Roadmap](#roadmap) for details.

## What it does today

Given any local Python repository, CodeCopilot:

1. **Walks the repo** and chunks each source file using AST-aware
   parsing (tree-sitter), producing one semantic chunk per function,
   class, and method.
2. **Embeds every chunk** with `text-embedding-3-small` (1536 dims) and
   stores them in Postgres with the `pgvector` extension.
3. **Answers questions** by running hybrid retrieval (vector + BM25
   keyword search, fused with Reciprocal Rank Fusion), passing the top
   50 candidates to an LLM reranker that picks the best 8 with
   relevance scores and reasoning, then streaming a grounded answer
   from Claude Sonnet that cites specific files and line ranges.

## Demo

After [setup](#setup) and ingesting a repo, the API streams answers
over HTTP:

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "fastapi",
    "mode": "explain",
    "question": "How does dependency injection work?"
  }'
```

Response (streamed word-by-word):

> Dependency injection in FastAPI works in two phases: a build phase
> at startup that inspects function signatures and constructs a tree
> of `Dependant` objects, and a solve phase at request time that
> recursively resolves that tree...
>
> [...]
>
> `get_dependant` in `dependencies/utils.py:284-359` is called on the
> endpoint function. It inspects every parameter via
> `get_typed_signature`...

Every claim cites a specific file and line range. When the codebase
doesn't actually answer the question (e.g. asking about JWT in a
codebase that doesn't implement JWT), the system honestly refuses
rather than fabricating an answer.

## Architecture
            ┌──────────────┐
            │  HTTP client │
            └──────┬───────┘
                   │
              POST /query
                   │
            ┌──────▼───────┐
            │   FastAPI    │
            └──────┬───────┘
                   │
   ┌───────────────┼───────────────┐
   │               │               │
┌───▼────┐     ┌────▼─────┐    ┌────▼─────┐
│ Vector │     │ Keyword  │    │ Reranker │
│ search │     │  (BM25)  │    │ (Claude) │
└───┬────┘     └────┬─────┘    └────┬─────┘
│               │               │
└───────┬───────┘               │
│                       │
(RRF fuse)                 │
│                       │
└───────────┬───────────┘
│
┌────────▼────────┐
│ Answer streamer │
│ (Claude Sonnet) │
└────────┬────────┘
│
streamed
response

### Key engineering decisions

- **AST-aware chunking** (tree-sitter): each chunk is a coherent
  semantic unit (function, class, method) rather than a fixed-size
  byte window. Class chunks include docstrings and field declarations;
  decorators and leading comments stay attached to the methods they
  describe.
- **Two-stage retrieval**: a recall-optimized first stage (hybrid +
  RRF) followed by a precision-optimized second stage (LLM reranker
  scoring 1–5 with reasoning). This is the production pattern behind
  most modern search systems.
- **HNSW over IVFFlat**: switched the vector index after diagnosing
  truncated results under selective `WHERE` filters caused by
  IVFFlat's cold-start centroid problem.
- **Calibrated uncertainty**: when the reranker produces only weak
  relevance scores, the answer prompt steers the model to refuse
  rather than fabricate.

## Tech stack

- **Backend:** Python 3.11, FastAPI, asyncpg
- **Database:** Postgres 16, `pgvector` (HNSW index), `tsvector` (GIN
  index)
- **Migrations:** Alembic
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLMs:** Anthropic Claude Haiku (reranking) and Claude Sonnet
  (answer generation)
- **Code parsing:** `tree-sitter` with `tree-sitter-python` grammar
- **Infra:** Docker Compose (Postgres + Redis)

## Setup

### Requirements

- Python 3.11+
- Docker Desktop
- An OpenAI API key (for embeddings)
- An Anthropic API key (for reranking and answer generation)

### Steps

```bash
# 1. Clone and create a virtualenv
git clone https://github.com/<your-username>/codecopilot.git
cd codecopilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and ANTHROPIC_API_KEY

# 3. Start Postgres + Redis
docker compose -f infra/docker-compose.yml up -d

# 4. Apply database migrations
cd infra && alembic upgrade head && cd ..

# 5. Ingest a repository
python -m apps.workers.ingestion /path/to/some/python/repo --name my-repo

# 6. Start the API
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000

# 7. Query it
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"repo": "my-repo", "mode": "explain",
       "question": "your question here"}'
```

Interactive API docs are auto-generated at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Project layout
codecopilot/
├── apps/
│   ├── api/              # FastAPI HTTP layer
│   │   ├── main.py
│   │   └── routers/
│   └── workers/
│       └── ingestion.py  # CLI: walk + chunk + embed + persist
├── packages/
│   ├── core/
│   │   ├── chunking/     # tree-sitter AST chunking + size cap
│   │   ├── retrieval/    # vector + keyword + RRF + reranker
│   │   ├── prompts/      # mode-specific prompt templates
│   │   ├── llm/          # OpenAI embeddings, Anthropic completions
│   │   ├── answer.py     # streaming answer generator
│   │   └── db.py         # async Postgres + pgvector client
│   └── shared/
│       └── models.py     # Pydantic request/response schemas
├── infra/
│   ├── docker-compose.yml
│   ├── alembic.ini
│   └── migrations/
└── requirements.txt

## Roadmap

- [x] **Phase 0** — Project scaffolding, Docker Compose, Alembic schema
- [x] **Phase 1** — AST-aware ingestion pipeline + first semantic search
- [x] **Phase 2** — Hybrid retrieval + LLM reranking + Explain mode +
      streaming HTTP API
- [ ] **Phase 3** — Trace, Debug, and Refactor modes; Next.js UI with
      chat interface and snippet inspector
- [ ] **Phase 4** — LLM-as-judge evaluation pipeline, golden set, and
      eval dashboard with CI integration
- [ ] **Phase 5** — Feedback loop (thumbs-down → regenerate with
      different retrieval strategy), public deployment, demo GIF

## License

MIT
