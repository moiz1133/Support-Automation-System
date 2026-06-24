import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.config import settings
from app.cost_tracker import RequestCostTracker
from app.errors import (
    ClassificationError,
    EmbeddingError,
    ESCALATE_ANSWER,
    FALLBACK_RESPONSES,
    LLMParseError,
    LLMTimeoutError,
    VectorDBError,
)
from app.llm import generate_answer
from app.logger import get_logger
from app.monitor import get_recent, get_summary, get_uptime_seconds, record_request
from app.retrieval import check_vector_db, retrieve

logger = get_logger(__name__)

app = FastAPI(title="Support Automation System")

CONFIDENCE_THRESHOLD = 0.6

HANDLED_ERRORS = (LLMTimeoutError, LLMParseError, VectorDBError, ClassificationError, EmbeddingError)


class QueryRequest(BaseModel):
    query: str


class ModelUsage(BaseModel):
    tokens: int
    cost_usd: float


class UsageSummary(BaseModel):
    total_tokens: int
    total_cost_usd: float
    breakdown_by_model: dict[str, ModelUsage]


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    intent: str
    category: str
    confidence: float
    escalated: bool
    error: bool = False
    error_type: str | None = None
    usage: UsageSummary


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application startup", extra={"event": "startup", "app_env": settings.app_env})


@app.get("/health")
async def health() -> dict:
    try:
        check_vector_db(settings.chroma_db_path)
        vector_db_status = "ok"
    except VectorDBError:
        vector_db_status = "error"

    openai_configured = bool(settings.openai_api_key)
    status = "ok" if vector_db_status == "ok" and openai_configured else "degraded"

    return {
        "status": status,
        "checks": {
            "vector_db": vector_db_status,
            "openai_configured": openai_configured,
            "uptime_seconds": get_uptime_seconds(),
        },
    }


@app.get("/monitor")
async def monitor() -> dict:
    return {
        "summary": get_summary(),
        "recent_requests": get_recent(),
    }


def _record_request_metrics(
    request_id: str,
    query: str,
    latency_ms: int,
    intent: str,
    category: str,
    confidence: float,
    escalated: bool,
    total_tokens: int,
    total_cost_usd: float,
    error: bool,
    error_type: str | None,
) -> None:
    record_request(
        {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_preview": query[:60],
            "intent": intent,
            "category": category,
            "confidence": confidence,
            "escalated": escalated,
            "latency_ms": latency_ms,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "error": error,
            "error_type": error_type,
        }
    )


def _error_response(error_type: type[Exception], fallback_answer: str) -> QueryResponse:
    return QueryResponse(
        answer=fallback_answer,
        sources=[],
        intent="unknown",
        category="unknown",
        confidence=0.0,
        escalated=True,
        error=True,
        error_type=error_type.__name__,
        usage=UsageSummary(total_tokens=0, total_cost_usd=0.0, breakdown_by_model={}),
    )


def _log_request_failed(level: int, request_id: str, error_type: str, error_message: str, latency_ms: int) -> None:
    logger.log(
        level,
        "request_failed",
        extra={
            "event": "request_failed",
            "error_type": error_type,
            "error_message": error_message,
            "request_id": request_id,
            "latency_ms": latency_ms,
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    tracker = RequestCostTracker(request_id)

    try:
        classification = classify(request.query, tracker)
        intent = classification["intent"]
        confidence = classification["confidence"]
        category = classification["category"]

        if intent == "answerable" and confidence < CONFIDENCE_THRESHOLD:
            intent = "needs_more_info"

        if intent == "escalate":
            answer = ESCALATE_ANSWER
            sources: list[str] = []
            escalated = True
        elif intent == "needs_more_info":
            answer = f"Could you provide more details? Specifically: {classification['reason']}"
            sources = []
            escalated = False
        else:
            chunks = retrieve(request.query, settings.chroma_db_path, tracker, category=category)
            result = await generate_answer(request.query, chunks, tracker)
            answer = result["answer"]
            sources = sorted({chunk["filename"] for chunk in chunks})
            escalated = False

        summary = tracker.summary()
        latency_ms = round((time.perf_counter() - start) * 1000)

        logger.info(
            "request_complete",
            extra={
                "event": "request_complete",
                "request_id": request_id,
                "intent": intent,
                "category": category,
                "escalated": escalated,
                "total_tokens": summary["total_tokens"],
                "total_cost_usd": summary["total_cost_usd"],
                "latency_ms": latency_ms,
                "calls": summary["calls"],
            },
        )

        _record_request_metrics(
            request_id=request_id,
            query=request.query,
            latency_ms=latency_ms,
            intent=intent,
            category=category,
            confidence=confidence,
            escalated=escalated,
            total_tokens=summary["total_tokens"],
            total_cost_usd=summary["total_cost_usd"],
            error=False,
            error_type=None,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            intent=intent,
            category=category,
            confidence=confidence,
            escalated=escalated,
            usage=UsageSummary(
                total_tokens=summary["total_tokens"],
                total_cost_usd=summary["total_cost_usd"],
                breakdown_by_model=summary["breakdown_by_model"],
            ),
        )

    except HANDLED_ERRORS as exc:
        latency_ms = round((time.perf_counter() - start) * 1000)
        _log_request_failed(logging.ERROR, request_id, type(exc).__name__, str(exc), latency_ms)
        _record_request_metrics(
            request_id=request_id,
            query=request.query,
            latency_ms=latency_ms,
            intent="unknown",
            category="unknown",
            confidence=0.0,
            escalated=True,
            total_tokens=0,
            total_cost_usd=0.0,
            error=True,
            error_type=type(exc).__name__,
        )
        return _error_response(type(exc), FALLBACK_RESPONSES[type(exc)])

    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000)
        _log_request_failed(logging.CRITICAL, request_id, type(exc).__name__, str(exc), latency_ms)
        _record_request_metrics(
            request_id=request_id,
            query=request.query,
            latency_ms=latency_ms,
            intent="unknown",
            category="unknown",
            confidence=0.0,
            escalated=True,
            total_tokens=0,
            total_cost_usd=0.0,
            error=True,
            error_type=type(exc).__name__,
        )
        return _error_response(type(exc), "Something went wrong on our end. Please try again or contact support.")
