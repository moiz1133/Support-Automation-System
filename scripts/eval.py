import json
import sys
import time
from pathlib import Path

import requests

API_URL = "http://localhost:8000/query"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.json"
COMPOSITE_PASS_THRESHOLD = 0.6

TEST_CASES = [
    {
        "query": "How do I reset my password?",
        "expected_intent": "answerable",
        "expected_category": "technical",
        "relevance_hints": ["password", "reset", "email"],
        "should_escalate": False,
    },
    {
        "query": "How do I cancel my subscription?",
        "expected_intent": "answerable",
        "expected_category": "billing",
        "relevance_hints": ["cancel", "subscription", "billing"],
        "should_escalate": False,
    },
    {
        "query": "Someone made unauthorized charges on my account",
        "expected_intent": "escalate",
        "expected_category": "escalation",
        "relevance_hints": ["human", "support", "agent"],
        "should_escalate": True,
    },
    {
        "query": "How do I enable two-factor authentication?",
        "expected_intent": "answerable",
        "expected_category": "technical",
        "relevance_hints": ["two-factor", "2fa", "security"],
        "should_escalate": False,
    },
    {
        "query": "It's not working",
        "expected_intent": "needs_more_info",
        "expected_category": "unknown",
        "relevance_hints": ["details", "more information"],
        "should_escalate": False,
    },
    {
        "query": "How do I download my invoice?",
        "expected_intent": "answerable",
        "expected_category": "billing",
        "relevance_hints": ["invoice", "billing", "download"],
        "should_escalate": False,
    },
    {
        "query": "I want to report a security vulnerability",
        "expected_intent": "escalate",
        "expected_category": "escalation",
        "relevance_hints": ["human", "support", "team"],
        "should_escalate": True,
    },
    {
        "query": "How do I transfer account ownership?",
        "expected_intent": "answerable",
        "expected_category": "account",
        "relevance_hints": ["transfer", "ownership", "account"],
        "should_escalate": False,
    },
    {
        "query": "Help",
        "expected_intent": "needs_more_info",
        "expected_category": "unknown",
        "relevance_hints": ["details", "more information"],
        "should_escalate": False,
    },
    {
        "query": "How do I connect via API?",
        "expected_intent": "answerable",
        "expected_category": "technical",
        "relevance_hints": ["api", "curl", "endpoint"],
        "should_escalate": False,
    },
]


def format_score(response: dict) -> float:
    score = 0.0

    answer = response.get("answer")
    if isinstance(answer, str) and answer:
        score += 0.25

    if isinstance(answer, str) and 10 <= len(answer) <= 600:
        score += 0.25

    if response.get("intent") in ("answerable", "escalate", "needs_more_info", "unknown"):
        score += 0.25

    usage = response.get("usage")
    if isinstance(usage, dict) and "total_tokens" in usage:
        score += 0.25

    return score


def relevance_score(answer: str, hints: list[str]) -> float:
    answer_lower = answer.lower()
    matches = sum(1 for hint in hints if hint.lower() in answer_lower)
    return round(matches / len(hints), 2)


def intent_score(actual: str, expected: str) -> float:
    return 1.0 if actual == expected else 0.0


def category_score(actual: str | None, expected: str) -> float:
    return 1.0 if actual == expected else 0.0


def run_case(case_id: int, case: dict) -> dict:
    query = case["query"]

    try:
        start = time.perf_counter()
        response = requests.post(API_URL, json={"query": query}, timeout=60)
        latency_ms = round((time.perf_counter() - start) * 1000)
    except requests.RequestException as exc:
        return {
            "case_id": case_id,
            "query": query,
            "expected_intent": case["expected_intent"],
            "actual_intent": None,
            "expected_category": case["expected_category"],
            "actual_category": None,
            "intent_score": 0.0,
            "category_score": 0.0,
            "format_score": 0.0,
            "relevance_score": 0.0,
            "composite_score": 0.0,
            "latency_ms": None,
            "failed": True,
            "error": str(exc),
            "answer": None,
        }

    if response.status_code != 200:
        return {
            "case_id": case_id,
            "query": query,
            "expected_intent": case["expected_intent"],
            "actual_intent": None,
            "expected_category": case["expected_category"],
            "actual_category": None,
            "intent_score": 0.0,
            "category_score": 0.0,
            "format_score": 0.0,
            "relevance_score": 0.0,
            "composite_score": 0.0,
            "latency_ms": latency_ms,
            "failed": True,
            "error": f"HTTP {response.status_code}",
            "answer": None,
        }

    body = response.json()
    answer = body.get("answer", "") or ""
    actual_intent = body.get("intent")
    actual_category = body.get("category")

    f_score = format_score(body)
    r_score = relevance_score(answer, case["relevance_hints"])
    i_score = intent_score(actual_intent, case["expected_intent"])
    c_score = category_score(actual_category, case["expected_category"])
    composite = round((f_score + r_score + i_score + c_score) / 4, 2)

    return {
        "case_id": case_id,
        "query": query,
        "expected_intent": case["expected_intent"],
        "actual_intent": actual_intent,
        "expected_category": case["expected_category"],
        "actual_category": actual_category,
        "intent_score": i_score,
        "category_score": c_score,
        "format_score": f_score,
        "relevance_score": r_score,
        "composite_score": composite,
        "latency_ms": latency_ms,
        "failed": False,
        "error": None,
        "answer": answer,
    }


def print_report(results: list[dict]) -> None:
    total = len(results)
    avg_format = round(sum(r["format_score"] for r in results) / total, 2)
    avg_relevance = round(sum(r["relevance_score"] for r in results) / total, 2)
    avg_intent = round(sum(r["intent_score"] for r in results) / total, 2)
    avg_category = round(sum(r["category_score"] for r in results) / total, 2)
    avg_composite = round(sum(r["composite_score"] for r in results) / total, 2)

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0
    failed_count = sum(1 for r in results if r["failed"])

    print("OVERALL")
    print(f"  Total cases    : {total}")
    print(f"  Avg format     : {avg_format:.2f}")
    print(f"  Avg relevance  : {avg_relevance:.2f}")
    print(f"  Avg intent     : {avg_intent:.2f}")
    print(f"  Avg category   : {avg_category:.2f}")
    print(f"  Avg composite  : {avg_composite:.2f}")
    print(f"  Avg latency    : {avg_latency}ms")
    print(f"  Failed requests: {failed_count}")
    print()

    print("PER CASE")
    print(
        f"  {'id':<3} | {'query':<30} | {'intent':<15} | {'category':<10} | "
        f"{'format':<6} | {'relevance':<9} | {'composite':<9} | ms"
    )
    for r in results:
        query_trunc = r["query"][:30]
        actual_intent = r["actual_intent"] or "N/A"
        actual_category = r["actual_category"] or "N/A"
        latency = r["latency_ms"] if r["latency_ms"] is not None else "N/A"
        print(
            f"  {r['case_id']:<3} | {query_trunc:<30} | {actual_intent:<15} | {actual_category:<10} | "
            f"{r['format_score']:<6.2f} | {r['relevance_score']:<9.2f} | {r['composite_score']:<9.2f} | {latency}"
        )
    print()

    failures = [r for r in results if r["composite_score"] < 0.5]
    print(f"FAILURES (composite < 0.5): {len(failures)}")
    for r in failures:
        answer_preview = (r["answer"] or r["error"] or "")[:60]
        print(
            f"  {r['case_id']} | {r['query']} | expected={r['expected_intent']} | "
            f"actual={r['actual_intent']} | {answer_preview}"
        )

    return avg_composite


def main() -> int:
    results = [run_case(i + 1, case) for i, case in enumerate(TEST_CASES)]

    avg_composite = print_report(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results written to {RESULTS_PATH}")

    return 0 if avg_composite >= COMPOSITE_PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
