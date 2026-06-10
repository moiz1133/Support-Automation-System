import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.llm import generate_answer
from app.logger import get_logger
from app.retrieval import retrieve

logger = get_logger(__name__)

app = FastAPI(title="Support Automation System")

PROMPT_TOKEN_RATE_USD = 0.00015 / 1000
COMPLETION_TOKEN_RATE_USD = 0.00060 / 1000


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    tokens_used: int
    cost_usd: float


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application startup", extra={"event": "startup", "app_env": settings.app_env})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    chunks = retrieve(request.query, settings.chroma_db_path)
    result = generate_answer(request.query, chunks)

    cost_usd = (
        result["prompt_tokens"] * PROMPT_TOKEN_RATE_USD
        + result["completion_tokens"] * COMPLETION_TOKEN_RATE_USD
    )
    sources = sorted({chunk["filename"] for chunk in chunks})
    latency_ms = round((time.perf_counter() - start) * 1000)

    logger.info(
        "query_handled",
        extra={
            "event": "query_handled",
            "query_preview": request.query[:60],
            "answer_preview": result["answer"][:80],
            "sources": sources,
            "tokens_used": result["total_tokens"],
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        },
    )

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        tokens_used=result["total_tokens"],
        cost_usd=cost_usd,
    )
