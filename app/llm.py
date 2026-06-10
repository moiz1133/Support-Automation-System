from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

LLM_MODEL = "gpt-3.5-turbo"

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


def generate_answer(query: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(chunk["content"] for chunk in chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    client = OpenAI(api_key=settings.openai_api_key)
    response = _create_chat_completion(
        client,
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    usage = response.usage
    result = {
        "answer": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model": response.model,
    }

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
