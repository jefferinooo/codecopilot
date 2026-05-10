# CodeCopilot

AI-powered code search and Q&A over Python repositories. Point it at a repo, ask a question in natural language, and get a grounded answer with citations to specific files and line ranges.

> **Live Demo:** [https://codecopilot-api-virx.onrender.com/docs](https://codecopilot-api-virx.onrender.com/docs)

> **Status:** Phases 0–5 complete. End-to-end RAG pipeline (ingestion, hybrid retrieval, LLM reranking, four answer modes, evaluation system) deployed at the URL above with a baseline of **4.33/5** on a hand-labeled 20-question golden set.

## Quality

Baseline scores from a 20-question golden set against FastAPI's source code:

| Dimension     | Score      |
|---------------|------------|
| Correctness   | 4.15 / 5   |
| Relevance     | 4.85 / 5   |
| Completeness  | 4.00 / 5   |
| **Overall**   | **4.33 / 5** |

By mode (n = number of questions):

| Mode      | n  | Correctness | Relevance | Completeness |
|-----------|----|-------------|-----------|--------------|
| explain   | 12 | 4.50        | 4.83      | 4.42         |
| trace     | 3  | 4.00        | 5.00      | 3.67         |
| debug     | 3  | 3.67        | 4.67      | 3.33         |
| refactor  | 2  | 3.00        | 5.00      | 3.00         |

Run the eval yourself: `python evals/run_evals.py --repo fastapi --tag my-run`. Results land in `evals/results/`. The runner is reproducible — same seed inputs, same scoring pipeline.

## What it does

Given any local Python repository, CodeCopilot:

1. **Walks the repo** and chunks each source file using AST-aware parsing (tree-sitter), producing one semantic chunk per function, class, and method.
2. **Embeds every chunk** with `text-embedding-3-small` (1536 dims) and stores them in Postgres with the `pgvector` extension (HNSW index).
3. **Answers questions** via hybrid retrieval (vector + BM25 keyword search, fused with Reciprocal Rank Fusion), an LLM reranker (Claude Haiku) that picks the best 8 candidates with relevance scores, then a streaming Claude Sonnet answer that cites specific files and line ranges.
4. **Judges every answer** with an LLM-as-judge pipeline scoring correctness, relevance, and completeness on a 1–5 scale, persisted alongside the query for offline analysis.

Four answer modes share the retrieval pipeline but use different prompts:

- **Explain** — natural-language explanation with inline citations
- **Trace** — numbered list of steps following a request, data flow, or control flow
- **Debug** — ranked hypotheses with evidence, no fixes proposed (diagnosis only)
- **Refactor** — minimum-viable unified diff with rationale and risks

## Demo

After [setup](#setup) and ingesting a repo, the API streams answers over HTTP:

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "fastapi",
    "mode": "explain",
    "question": "How does dependency injection work?"
  }'
```

Response (streamed token-by-token):

> Dependency injection in FastAPI works in two phases: a build phase at startup that inspects function signatures and constructs a tree of `Dependant` objects, and a solve phase at request time that recursively resolves that tree...
>
> [...]
>
> `get_dependant` in `dependencies/utils.py:284-359` is called on the endpoint function. It inspects every parameter via `get_typed_signature`...

Every claim cites a specific file and line range. When the codebase doesn't actually answer the question (e.g. asking about JWT in a codebase that doesn't implement it), the system honestly refuses rather than fabricating an answer.

The aggregated eval data is also exposed:

```bash
curl http://127.0.0.1:8000/eval/fastapi
```

Returns overall scores, per-mode breakdowns, and the 20 most recent judged queries.

## Architecture

```
                  HTTP client
                       |
                  POST /query
                       |
                    FastAPI
                       |
        +--------------+--------------+
        |              |              |
   Vector search   Keyword search   (parallel)
   (pgvector       (Postgres
    HNSW)          BM25/tsvector)
        |              |
        +------+-------+
               |
        Reciprocal Rank Fusion
               |
        LLM reranker
        (Claude Haiku, 1-5 scoring)
               |
        Answer streamer
        (Claude Sonnet, cited)
               |
        streamed response   ----> [background] LLM-as-judge
                                  scores + persists
```

### Key engineering decisions

- **AST-aware chunking** (tree-sitter): each chunk is a coherent semantic unit (function, class, method) rather than a fixed-size byte window. Class chunks include docstrings and field declarations; decorators and leading comments stay attached to the methods they describe.
- **Two-stage retrieval**: a recall-optimized first stage (hybrid + RRF) followed by a precision-optimized second stage (LLM reranker scoring 1–5 with reasoning). This is the production pattern behind most modern search systems.
- **HNSW over IVFFlat**: switched the vector index after diagnosing truncated results under selective `WHERE` filters caused by IVFFlat's cold-start centroid problem (the index was built on an empty table, so its cluster centroids were meaningless).
- **Calibrated uncertainty**: when the reranker produces only weak relevance scores, the answer prompt steers the model to refuse rather than fabricate. This is enforced both by mode-specific prompts and by the eval rubric.
- **Three-dimensional eval rubric**: a single "is it good" score is uninformative. Decomposing into correctness / relevance / completeness reveals where to direct improvements (e.g., debug mode currently has correctness 3.67 — a clear next target).

## Tech stack

- **Backend:** Python 3.11, FastAPI, asyncpg
- **Database:** Postgres 16, `pgvector` (HNSW index), `tsvector` (GIN index)
- **Migrations:** Alembic
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLMs:** Anthropic Claude Haiku (reranking and judging) and Claude Sonnet (answer generation)
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

# 8. (Optional) Run the golden-set evaluation
python evals/run_evals.py --repo my-repo --tag my-baseline
```

Interactive API docs are auto-generated at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Project layout

```
codecopilot/
  apps/
    api/                  FastAPI HTTP layer
      main.py
      routers/            query, eval
    workers/
      ingestion.py        CLI: walk, chunk, embed, persist
  packages/
    core/
      chunking/           tree-sitter AST chunking + size cap
      retrieval/          vector + keyword + RRF + reranker
      prompts/            mode-specific prompt templates
      llm/                OpenAI embeddings, Anthropic completions
      eval/               LLM-as-judge rubric and scoring
      answer.py           streaming answer generator
      db.py               async Postgres + pgvector client
    shared/
      models.py           Pydantic request/response schemas
  evals/
    golden/               hand-labeled question sets
    results/              timestamped eval runs (JSON)
    run_evals.py          golden-set runner
  infra/
    docker-compose.yml
    alembic.ini
    migrations/
  requirements.txt
```

## Roadmap

- [x] **Phase 0** — Project scaffolding, Docker Compose, Alembic schema
- [x] **Phase 1** — AST-aware ingestion pipeline + first semantic search
- [x] **Phase 2** — Hybrid retrieval + LLM reranking + Explain mode + streaming HTTP API
- [x] **Phase 3a** — Trace, Debug, and Refactor answer modes
- [x] **Phase 4** — LLM-as-judge evaluation pipeline + 20-question golden set + eval dashboard endpoint
- [ ] **Phase 5** — Public deployment, demo GIF, README polish
- [ ] **Phase 3b** — Next.js UI with chat interface and snippet inspector (optional)

## License

MIT
