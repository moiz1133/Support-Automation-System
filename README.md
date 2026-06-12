# Support Automation System

A FastAPI service that answers customer support queries by classifying intent, retrieving relevant
articles from a local knowledge base (ChromaDB), and generating a grounded answer with an OpenAI
chat model. It tracks token/cost usage per request, emits structured JSON logs, and exposes basic
monitoring and health endpoints. It does **not** do fine-tuning, does **not** maintain persistent
user sessions or conversation history (every `/query` call is stateless and independent), and does
**not** write back to any external ticketing system — escalation is currently just a canned response
telling the user a human will follow up.

## Architecture

```
                         POST /query {"query": "..."}
                                    |
                                    v
                         +-----------------------+
                         |       Classifier       |  gpt-3.5-turbo, JSON mode
                         | -> intent + confidence  |
                         +-----------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
   intent=escalate         intent=needs_more_info    intent=answerable
   (or confidence<0.6           |                   (confidence>=0.6)
    on "answerable")             |                          |
            |                     |                          v
            |                     |               +-----------------------+     +--------------+
            |                     |               |       Retrieval        |---->|   ChromaDB    |
            |                     |               | -> top 3 chunks         |     | (vector store)|
            |                     |               +-----------------------+     +--------------+
            |                     |                          |
            |                     |                          v
            |                     |               +-----------------------+     +--------------+
            |                     |               |       Generation        |---->|  OpenAI API   |
            |                     |               | -> grounded answer       |     | (chat compl.) |
            |                     |               +-----------------------+     +--------------+
            |                     |                          |
            v                     v                          v
   +-------------------------------------------------------------+
   |                         Cost Tracker                          |
   |        RequestCostTracker.summary() + "request_complete"     |
   +-------------------------------------------------------------+
                                    |
                                    v
                          Structured JSON Response
                {answer, sources, intent, confidence,
                       escalated, error, usage}
```

Component descriptions:

- **Classifier** (`app/classifier.py`) — single LLM call (gpt-3.5-turbo, `response_format=json_object`)
  that returns `{intent, confidence, reason}`; intent is one of `answerable`, `escalate`, `needs_more_info`.
- **Router** (`app/main.py`) — demotes `answerable` to `needs_more_info` if `confidence < 0.6`; routes
  `escalate` and `needs_more_info` directly to a canned response with no retrieval or generation.
- **Retrieval** (`app/retrieval.py`) — embeds the query (`text-embedding-3-large`), queries ChromaDB for
  the top 3 chunks, and returns their content, source filename, and category.
- **ChromaDB** — local persistent vector store under `data/chroma/`, populated by `scripts/ingest.py`.
- **Generation** (`app/llm.py`) — gpt-3.5-turbo call grounded in the retrieved chunks, with a 25s timeout
  and tenacity-based retry on rate limits / connection errors.
- **OpenAI API** — used for embeddings, classification, and generation chat completions.
- **Cost Tracker** (`app/cost_tracker.py`) — accumulates prompt/completion tokens and cost-per-model
  across all calls in a request, returned in the response `usage` field and logged as `request_complete`.

## Project Structure

```
support-automation/
├── app/
│   ├── __init__.py        # package marker
│   ├── main.py            # FastAPI app: POST /query, GET /monitor, GET /health
│   ├── classifier.py      # intent classification (gpt-3.5-turbo, JSON mode)
│   ├── retrieval.py        # embedding + ChromaDB query, vector DB health check
│   ├── llm.py              # grounded answer generation with timeout + retry
│   ├── ingestion.py        # document loading, chunking, embedding + storage
│   ├── cost_tracker.py      # per-request token/cost accounting
│   ├── monitor.py           # in-memory request log + summary stats
│   ├── errors.py            # custom exceptions + customer-facing fallback messages
│   ├── logger.py            # structured JSON logging setup
│   └── config.py            # pydantic-settings: env var configuration
├── scripts/
│   ├── ingest.py           # CLI: load data/docs/, chunk, embed, store in ChromaDB
│   └── eval.py              # CLI: run 10 test queries against a live server, score results
├── tests/
│   └── test_pipeline.py     # pytest suite, all OpenAI/ChromaDB calls mocked
├── data/
│   ├── docs/                # 20 sample knowledge-base articles (.txt)
│   └── chroma/               # ChromaDB persistent store (gitignored, generated by ingest.py)
├── .github/workflows/ci.yml  # GitHub Actions: install deps, run pytest
├── .env.example              # template for required environment variables
├── .env                       # local secrets (gitignored, not committed)
├── requirements.txt           # Python dependencies
└── eval_results.json          # latest output of scripts/eval.py (generated, not committed)
```

## Quickstart

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd support-automation

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
# then edit .env and set OPENAI_API_KEY

# 5. Ingest the sample knowledge base into ChromaDB
python scripts/ingest.py

# 6. Run the API server
uvicorn app.main:app --reload

# 7. Send a query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How do I reset my password?\"}"
```

## API Reference

### `POST /query`

Classifies, retrieves, and (if applicable) generates an answer for a customer query. Always returns
HTTP 200, even on internal failures — see `app/errors.py` for the fallback responses.

**Request body:**
```json
{
  "query": "How do I reset my password?"
}
```

**Response body:**
```json
{
  "answer": "string",
  "sources": ["string"],
  "intent": "answerable | escalate | needs_more_info | unknown",
  "confidence": 0.0,
  "escalated": false,
  "error": false,
  "error_type": "string | null",
  "usage": {
    "total_tokens": 0,
    "total_cost_usd": 0.0,
    "breakdown_by_model": {
      "model-name": {"tokens": 0, "cost_usd": 0.0}
    }
  }
}
```

**Example curl:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How do I reset my password?\"}"
```

**Example response:**
```json
{
  "answer": "If you've forgotten your password or want to change it for security reasons, you can reset it from the login page. Click \"Forgot Password?\" below the login form, enter the email address associated with your account, and click \"Send Reset Link.\" You will receive an email containing a secure link to reset your password.",
  "sources": ["technical_01.txt"],
  "intent": "answerable",
  "confidence": 1.0,
  "escalated": false,
  "error": false,
  "error_type": null,
  "usage": {
    "total_tokens": 446,
    "total_cost_usd": 0.00000091,
    "breakdown_by_model": {
      "gpt-3.5-turbo-0125": {"tokens": 439, "cost_usd": 0.0000000},
      "text-embedding-3-large": {"tokens": 7, "cost_usd": 0.00000091}
    }
  }
}
```

### `GET /monitor`

Returns aggregate stats and the last 100 requests, tracked in an in-memory deque (resets on restart).
No request body.

**Response body:**
```json
{
  "summary": {
    "total_requests": 0,
    "error_count": 0,
    "error_rate": 0.0,
    "avg_latency_ms": 0.0,
    "p95_latency_ms": 0,
    "avg_tokens": 0.0,
    "total_cost_usd": 0.0,
    "intent_breakdown": {"answerable": 0, "escalate": 0, "needs_more_info": 0, "unknown": 0},
    "avg_confidence": 0.0
  },
  "recent_requests": [
    {
      "request_id": "string",
      "timestamp": "ISO-8601 string",
      "query_preview": "string",
      "intent": "string",
      "confidence": 0.0,
      "escalated": false,
      "latency_ms": 0,
      "total_tokens": 0,
      "total_cost_usd": 0.0,
      "error": false,
      "error_type": null
    }
  ]
}
```

**Example curl:**
```bash
curl http://127.0.0.1:8000/monitor
```

**Example response:**
```json
{
  "summary": {
    "total_requests": 2,
    "error_count": 0,
    "error_rate": 0.0,
    "avg_latency_ms": 6411.5,
    "p95_latency_ms": 0,
    "avg_tokens": 298.0,
    "total_cost_usd": 0.000001,
    "intent_breakdown": {"answerable": 1, "escalate": 1, "needs_more_info": 0, "unknown": 0},
    "avg_confidence": 1.0
  },
  "recent_requests": [
    {
      "request_id": "5ac69b6d-d8e5-4c2c-a10d-5db00667765b",
      "timestamp": "2026-06-12T04:18:08.121160+00:00",
      "query_preview": "Someone made unauthorized charges on my account",
      "intent": "escalate",
      "confidence": 1.0,
      "escalated": true,
      "latency_ms": 2539,
      "total_tokens": 150,
      "total_cost_usd": 0.0,
      "error": false,
      "error_type": null
    },
    {
      "request_id": "75a6dcc1-d180-43ff-9238-3c79e309a5c6",
      "timestamp": "2026-06-12T04:18:05.219116+00:00",
      "query_preview": "How do I reset my password?",
      "intent": "answerable",
      "confidence": 1.0,
      "escalated": false,
      "latency_ms": 10284,
      "total_tokens": 446,
      "total_cost_usd": 0.00000091,
      "error": false,
      "error_type": null
    }
  ]
}
```

### `GET /health`

Liveness/readiness probe. Checks ChromaDB connectivity (a `get_or_create_collection` + `count()`
call, no embedding required) and whether `OPENAI_API_KEY` is set (no live OpenAI call is made).

**Response body:**
```json
{
  "status": "ok | degraded",
  "checks": {
    "vector_db": "ok | error",
    "openai_configured": true,
    "uptime_seconds": 0.0
  }
}
```

**Example curl:**
```bash
curl http://127.0.0.1:8000/health
```

**Example response:**
```json
{
  "status": "ok",
  "checks": {
    "vector_db": "ok",
    "openai_configured": true,
    "uptime_seconds": 2.44
  }
}
```

## Eval

`scripts/eval.py` sends 10 hardcoded test queries to a running server and scores each response on
three dimensions:

- **format_score** (0.0–1.0): checks the response has a non-empty `answer` between 10 and 600
  characters, a valid `intent`, and a `usage.total_tokens` field. 0.25 per check.
- **relevance_score** (0.0–1.0): fraction of expected keywords (e.g. `"password"`, `"reset"`, `"email"`)
  found (case-insensitive) in the `answer` text.
- **intent_score** (0.0 or 1.0): whether the returned `intent` matches the expected intent for that query.

`composite_score` is the mean of the three. The script prints an overall summary, a per-case table,
and a list of cases with `composite_score < 0.5`, then writes full results to `eval_results.json`.

**How to run** (server must already be running on `localhost:8000`):
```bash
uvicorn app.main:app --reload &
python scripts/eval.py
```

**Exit code:** `0` if the average composite score across all 10 cases is `>= 0.6`, otherwise `1` — so
this can be wired into CI as a regression gate on answer quality, not just on whether the code runs.

## Tradeoffs & Known Limitations

1. **In-memory monitoring.** `/monitor` is backed by a `collections.deque(maxlen=100)` inside the
   running process — chosen because it requires no extra infrastructure for a single-instance project.
   This breaks down as soon as you run more than one worker/process (each gets its own deque, so
   `/monitor` only reflects whatever fraction of traffic that process happened to handle) and all
   history is lost on every restart or deploy. A production replacement would push each request record
   to Redis (e.g. a capped list or stream) for the "recent requests" view, and write aggregates to a
   time-series-friendly store (Postgres table or Prometheus) so dashboards survive restarts and cover
   all instances.

2. **ChromaDB local persistence.** `data/chroma/` is a `PersistentClient` writing to local disk —
   chosen because it needs zero external services and is easy to inspect/reset during development.
   It doesn't scale horizontally (each instance has its own copy with no shared state, so concurrent
   writers can diverge), has no managed backups, and ties the vector index to a single machine's
   filesystem. The production alternative is **pgvector on RDS/Postgres**: a single shared, durable
   store with concurrent read/write support. Switching would mean replacing the `chromadb` calls in
   `app/retrieval.py` and `app/ingestion.py` with SQL (`INSERT`/`SELECT ... ORDER BY embedding <-> $1
   LIMIT 3`), and adding a Postgres connection to `app/config.py`.

3. **No authentication on `/monitor` and `/health`.** Acceptable for local development and demos, but
   `/monitor` exposes query previews (which may contain user-submitted PII) and per-request cost data,
   so it should not be left open on any publicly reachable deployment. In production, add a FastAPI
   dependency that checks a shared-secret header (e.g. `X-Internal-Api-Key`) or validates a JWT issued
   by internal SSO, and apply it to both routes — or simply keep them off the public-facing router
   entirely and only expose them on an internal network/reverse-proxy path.

4. **gpt-3.5-turbo for both classification and generation.** Using one model for both keeps pricing,
   retry/timeout config, and dependency management simple (one entry in `PRICING`, one set of tenacity
   settings). The cost is that classification — a simple 3-way categorization with a one-sentence
   reason — runs on the same model as grounded answer generation, where a cheaper/faster model would
   likely classify just as accurately. Once query volume grows enough that classification cost/latency
   becomes a meaningful share of the total, split `CLASSIFIER_MODEL` to a smaller model (e.g.
   `gpt-4o-mini` or an even smaller fine-tuned classifier) while keeping the larger model for
   `LLM_MODEL` in `app/llm.py`, where answer quality matters more.

5. **Mock knowledge base (20 static docs).** `data/docs/` contains 20 hand-written `.txt` files
   ingested once via `scripts/ingest.py`. This is fine for a fixed demo corpus but a real deployment
   needs: connectors/crawlers that pull from the actual sources of truth (help center CMS, internal
   wikis, resolved ticket text), parsers for real document formats (HTML, PDF, Markdown) with
   structure-aware chunking instead of the current fixed-size overlap chunking, deduplication across
   sources, and a scheduled re-ingestion job (cron or background worker) that re-embeds changed
   documents and removes deleted ones — otherwise the vector store silently drifts out of sync with
   the actual knowledge base.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | API key used for embeddings, classification, and generation calls to the OpenAI API. |
| `CHROMA_DB_PATH` | No | `./data/chroma` | Filesystem path for the ChromaDB persistent store, used by both `scripts/ingest.py` and `app/retrieval.py`. |
| `LOG_LEVEL` | No | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) for structured JSON logs. |
| `APP_ENV` | No | `development` | Environment name, included in the `startup` log line. Not currently used for branching logic. |
