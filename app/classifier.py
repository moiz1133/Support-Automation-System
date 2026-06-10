import json

from openai import OpenAI

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

CLASSIFIER_MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = (
    "Classify the user support query into exactly one of these three categories:\n"
    "  answerable      - can be resolved with documentation\n"
    "  escalate        - sensitive: security, legal, fraud, abuse, unauthorized charges\n"
    "  needs_more_info - too vague to answer without clarification\n"
    "Respond ONLY with valid JSON:\n"
    '{"intent": "answerable" | "escalate" | "needs_more_info", '
    '"confidence": 0.0-1.0, "reason": "one sentence"}'
)


def classify(query: str) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        result = {
            "intent": parsed["intent"],
            "confidence": float(parsed["confidence"]),
            "reason": parsed["reason"],
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        result = {
            "intent": "escalate",
            "confidence": 0.0,
            "reason": "classification parse error",
        }

    usage = response.usage
    result["prompt_tokens"] = usage.prompt_tokens
    result["completion_tokens"] = usage.completion_tokens
    result["total_tokens"] = usage.total_tokens
    result["model"] = response.model

    logger.info(
        "classification_done",
        extra={
            "event": "classification_done",
            "intent": result["intent"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "query_preview": query[:60],
        },
    )
    return result
