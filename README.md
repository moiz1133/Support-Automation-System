# Support Automation System

A scaffold for an AI-powered customer support automation service. It exposes a FastAPI backend that will use OpenAI models and a ChromaDB vector store to answer customer questions by retrieving relevant articles from a local knowledge base (`data/docs/`), with structured JSON logging for observability.

## Running locally

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your values:
   ```
   copy .env.example .env
   ```
4. Run the development server:
   ```
   uvicorn app.main:app --reload
   ```
5. Visit `http://127.0.0.1:8000/health` to confirm the app is running.

## Architecture

```
                    POST /query {"query": "..."}
                              |
                              v
                     +-------------------+
                     |    classifier     |  gpt-3.5-turbo (JSON mode)
                     +-------------------+
                              |
                              v
                     +-------------------+
                     |      router       |  confidence < 0.6 -> needs_more_info
                     +-------------------+
                  /             |              \
                 /              |               \
       intent=escalate   intent=needs_more_info   intent=answerable
              |                   |                       |
              v                   v                       v
     +----------------+   +----------------+   +-----------------------+
     |   escalation   |   |  clarification |   |       retrieval       |
     |    response    |   |    response    |   | (ChromaDB +           |
     |   (no LLM)     |   |   (no LLM)     |   |  text-embedding-3-*)  |
     +----------------+   +----------------+   +-----------------------+
              |                   |                       |
              |                   |                       v
              |                   |             +-----------------------+
              |                   |             |       generation      |
              |                   |             |     (gpt-3.5-turbo)   |
              |                   |             +-----------------------+
               \                  |                      /
                \                 |                     /
                 \                v                    /
                  +--------------------------------------+
                  |             cost tracker              |
                  |     RequestCostTracker.summary()      |
                  +--------------------------------------+
                              |
                              v
                  +--------------------------------+
                  |   structured log: "request_     |
                  |   complete" (JSON)              |
                  +--------------------------------+
                              |
                              v
                       JSON response
            {answer, sources, intent, confidence,
                   escalated, usage}
```

## Running tests

All OpenAI calls are mocked, so the test suite makes no real API requests:

```
pytest tests/
```

## Sample requests/responses

### answerable

Request:
```json
{"query": "How do I reset my password?"}
```

Response:
```json
{
  "answer": "To reset your password, click \"Forgot Password?\" on the login page, enter your email address, and follow the link we send you. The link is valid for 60 minutes.",
  "sources": ["technical_01.txt"],
  "intent": "answerable",
  "confidence": 0.95,
  "escalated": false,
  "usage": {
    "total_tokens": 412,
    "total_cost_usd": 0.000128,
    "breakdown_by_model": {
      "gpt-3.5-turbo": {"tokens": 380, "cost_usd": 0.000114},
      "text-embedding-3-large": {"tokens": 32, "cost_usd": 0.0000042}
    }
  }
}
```

### escalate

Request:
```json
{"query": "Someone accessed my account without my permission"}
```

Response:
```json
{
  "answer": "This issue requires human support. A team member will contact you shortly.",
  "sources": [],
  "intent": "escalate",
  "confidence": 0.97,
  "escalated": true,
  "usage": {
    "total_tokens": 58,
    "total_cost_usd": 0.0000147,
    "breakdown_by_model": {
      "gpt-3.5-turbo": {"tokens": 58, "cost_usd": 0.0000147}
    }
  }
}
```

### needs_more_info

Request:
```json
{"query": "It's broken"}
```

Response:
```json
{
  "answer": "Could you provide more details? Specifically: The query is too vague to identify which feature or system is affected.",
  "sources": [],
  "intent": "needs_more_info",
  "confidence": 0.4,
  "escalated": false,
  "usage": {
    "total_tokens": 52,
    "total_cost_usd": 0.0000132,
    "breakdown_by_model": {
      "gpt-3.5-turbo": {"tokens": 52, "cost_usd": 0.0000132}
    }
  }
}
```
