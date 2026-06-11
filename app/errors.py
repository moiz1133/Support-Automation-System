class LLMTimeoutError(Exception):
    pass


class LLMParseError(Exception):
    pass


class VectorDBError(Exception):
    pass


class ClassificationError(Exception):
    pass


class EmbeddingError(Exception):
    pass


ESCALATE_ANSWER = "This issue requires human support. A team member will contact you shortly."

FALLBACK_RESPONSES = {
    LLMTimeoutError: "Our system is taking longer than expected. Please try again in a moment.",
    LLMParseError: "We received an unexpected response. A support agent will follow up shortly.",
    VectorDBError: "Unable to search our knowledge base right now. Please try again or contact support directly.",
    ClassificationError: ESCALATE_ANSWER,
    EmbeddingError: "Unable to process your request right now. Please try again in a moment.",
}
