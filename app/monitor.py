import time
from collections import deque
from dataclasses import asdict, dataclass

from app.logger import get_logger

logger = get_logger(__name__)

MAX_RECORDS = 100

_start_time = time.monotonic()

INTENTS = ("answerable", "escalate", "needs_more_info", "unknown")


@dataclass
class RequestRecord:
    request_id: str
    timestamp: str
    query_preview: str
    intent: str
    confidence: float
    escalated: bool
    latency_ms: int
    total_tokens: int
    total_cost_usd: float
    error: bool
    error_type: str | None


_records: deque[RequestRecord] = deque(maxlen=MAX_RECORDS)


def record_request(data: dict) -> None:
    try:
        record = RequestRecord(
            request_id=data["request_id"],
            timestamp=data["timestamp"],
            query_preview=data["query_preview"],
            intent=data["intent"],
            confidence=data["confidence"],
            escalated=data["escalated"],
            latency_ms=data["latency_ms"],
            total_tokens=data["total_tokens"],
            total_cost_usd=data["total_cost_usd"],
            error=data["error"],
            error_type=data["error_type"],
        )
        _records.append(record)
    except Exception as exc:
        logger.warning(
            "monitor_record_failed",
            extra={"event": "monitor_record_failed", "error_message": str(exc)},
        )


def get_recent(n: int = 100) -> list[dict]:
    records = list(_records)[-n:]
    records.reverse()
    return [asdict(r) for r in records]


def get_summary() -> dict:
    records = list(_records)
    total_requests = len(records)

    intent_breakdown = {intent: 0 for intent in INTENTS}

    if total_requests == 0:
        return {
            "total_requests": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0,
            "avg_tokens": 0.0,
            "total_cost_usd": 0.0,
            "intent_breakdown": intent_breakdown,
            "avg_confidence": 0.0,
        }

    error_count = sum(1 for r in records if r.error)
    latencies = sorted(r.latency_ms for r in records)

    if len(latencies) < 20:
        p95_latency_ms = 0
    else:
        p95_index = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p95_latency_ms = latencies[p95_index]

    for r in records:
        intent_breakdown[r.intent] = intent_breakdown.get(r.intent, 0) + 1

    return {
        "total_requests": total_requests,
        "error_count": error_count,
        "error_rate": round(error_count / total_requests, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in records) / total_requests, 2),
        "p95_latency_ms": p95_latency_ms,
        "avg_tokens": round(sum(r.total_tokens for r in records) / total_requests, 2),
        "total_cost_usd": round(sum(r.total_cost_usd for r in records), 6),
        "intent_breakdown": intent_breakdown,
        "avg_confidence": round(sum(r.confidence for r in records) / total_requests, 4),
    }


def get_uptime_seconds() -> float:
    return round(time.monotonic() - _start_time, 2)
