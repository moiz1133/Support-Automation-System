import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.config import settings
from app.cost_tracker import RequestCostTracker
from app.llm import generate_answer
from app.logger import get_logger
from app.retrieval import retrieve

logger = get_logger(__name__)

app = FastAPI(title="Support Automation System")

CONFIDENCE_THRESHOLD = 0.6

ESCALATE_ANSWER = "This issue requires human support. A team member will contact you shortly."


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
    confidence: float
    escalated: bool
    usage: UsageSummary


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application startup", extra={"event": "startup", "app_env": settings.app_env})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    tracker = RequestCostTracker(request_id)

    classification = classify(request.query, tracker)
    intent = classification["intent"]
    confidence = classification["confidence"]

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
        chunks = retrieve(request.query, settings.chroma_db_path, tracker)
        result = generate_answer(request.query, chunks, tracker)
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
            "escalated": escalated,
            "total_tokens": summary["total_tokens"],
            "total_cost_usd": summary["total_cost_usd"],
            "latency_ms": latency_ms,
            "calls": summary["calls"],
        },
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        intent=intent,
        confidence=confidence,
        escalated=escalated,
        usage=UsageSummary(
            total_tokens=summary["total_tokens"],
            total_cost_usd=summary["total_cost_usd"],
            breakdown_by_model=summary["breakdown_by_model"],
        ),
    )
