import chromadb
from openai import OpenAI

from app.config import settings
from app.cost_tracker import RequestCostTracker
from app.ingestion import EMBEDDING_MODEL
from app.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, db_path: str, tracker: RequestCostTracker, n_results: int = 3) -> list[dict]:
    openai_client = OpenAI(api_key=settings.openai_api_key)
    embedding_response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    embedding = embedding_response.data[0].embedding

    tracker.add_call(
        model=embedding_response.model,
        prompt_tokens=embedding_response.usage.prompt_tokens,
        completion_tokens=0,
        call_type="embedding",
    )

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="support_docs")
    results = collection.query(query_embeddings=[embedding], n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = [
        {
            "content": content,
            "filename": metadata["filename"],
            "category": metadata["category"],
            "distance": distance,
        }
        for content, metadata, distance in zip(documents, metadatas, distances)
    ]

    logger.info(
        "retrieval_done",
        extra={
            "event": "retrieval_done",
            "query_preview": query[:60],
            "chunks_returned": len(chunks),
            "top_distance": distances[0] if distances else None,
        },
    )
    return chunks
