import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.classifier import classify
from app.config import settings
from app.llm import generate_answer
from app.logger import get_logger
from app.retrieval import retrieve

logger = get_logger(__name__)

app = FastAPI(title="Support Automation System")

PROMPT_TOKEN_RATE_USD = 0.00015 / 1000
COMPLETION_TOKEN_RATE_USD = 0.00060 / 1000
CONFIDENCE_THRESHOLD = 0.6

ESCALATE_ANSWER = "This issue requires human support. A team member will contact you shortly."


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    tokens_used: int
    cost_usd: float
    intent: str
    confidence: float
    escalated: bool


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application startup", extra={"event": "startup", "app_env": settings.app_env})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _token_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * PROMPT_TOKEN_RATE_USD + completion_tokens * COMPLETION_TOKEN_RATE_USD


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    classification = classify(request.query)
    intent = classification["intent"]
    confidence = classification["confidence"]
    classifier_cost_usd = _token_cost_usd(
        classification["prompt_tokens"], classification["completion_tokens"]
    )

    if intent == "answerable" and confidence < CONFIDENCE_THRESHOLD:
        intent = "needs_more_info"

    if intent == "escalate":
        response = QueryResponse(
            answer=ESCALATE_ANSWER,
            sources=[],
            tokens_used=classification["total_tokens"],
            cost_usd=classifier_cost_usd,
            intent=intent,
            confidence=confidence,
            escalated=True,
        )
    elif intent == "needs_more_info":
        response = QueryResponse(
            answer=f"Could you provide more details? Specifically: {classification['reason']}",
            sources=[],
            tokens_used=classification["total_tokens"],
            cost_usd=classifier_cost_usd,
            intent=intent,
            confidence=confidence,
            escalated=False,
        )
    else:
        chunks = retrieve(request.query, settings.chroma_db_path)
        result = generate_answer(request.query, chunks)
        generation_cost_usd = _token_cost_usd(result["prompt_tokens"], result["completion_tokens"])

        response = QueryResponse(
            answer=result["answer"],
            sources=sorted({chunk["filename"] for chunk in chunks}),
            tokens_used=classification["total_tokens"] + result["total_tokens"],
            cost_usd=classifier_cost_usd + generation_cost_usd,
            intent=intent,
            confidence=confidence,
            escalated=False,
        )

    latency_ms = round((time.perf_counter() - start) * 1000)

    logger.info(
        "query_handled",
        extra={
            "event": "query_handled",
            "query_preview": request.query[:60],
            "answer_preview": response.answer[:80],
            "sources": response.sources,
            "tokens_used": response.tokens_used,
            "cost_usd": response.cost_usd,
            "latency_ms": latency_ms,
        },
    )

    return response
