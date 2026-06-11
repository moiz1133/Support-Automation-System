import asyncio

import openai
from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.cost_tracker import RequestCostTracker
from app.errors import LLMParseError, LLMTimeoutError
from app.logger import get_logger

logger = get_logger(__name__)

LLM_MODEL = "gpt-3.5-turbo"
LLM_TIMEOUT_SECONDS = 25

SYSTEM_PROMPT_TEMPLATE = (
    "You are a support assistant. Answer ONLY using the provided context. "
    "If the context does not contain enough information, say so explicitly.\n"
    "Context: {context}"
)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
)
def _create_chat_completion(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


async def generate_answer(query: str, chunks: list[dict], tracker: RequestCostTracker) -> dict:
    context = "\n\n".join(chunk["content"] for chunk in chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                _create_chat_completion,
                client,
                model=LLM_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        logger.warning(
            "llm_timeout",
            extra={"event": "llm_timeout", "error_message": str(exc), "query_preview": query[:60]},
        )
        raise LLMTimeoutError("LLM call exceeded time limit") from exc
    except openai.APIError as exc:
        logger.warning(
            "llm_parse_error",
            extra={"event": "llm_parse_error", "error_message": str(exc), "query_preview": query[:60]},
        )
        raise LLMParseError(str(exc)) from exc

    usage = response.usage
    result = {
        "answer": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model": response.model,
    }

    tracker.add_call(
        model=result["model"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        call_type="generation",
    )

    logger.info(
        "llm_call_done",
        extra={
            "event": "llm_call_done",
            "model": result["model"],
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        },
    )
    return result
