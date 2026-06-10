from pathlib import Path

import chromadb
from openai import OpenAI

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

CATEGORY_PREFIXES = ("billing_", "technical_", "account_", "escalation_")
EMBEDDING_MODEL = "text-embedding-3-large"


def load_documents(docs_dir: str) -> list[dict]:
    documents = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        category = "uncategorized"
        for prefix in CATEGORY_PREFIXES:
            if path.name.startswith(prefix):
                category = prefix.rstrip("_")
                break
        documents.append({
            "filename": path.name,
            "category": category,
            "content": path.read_text(encoding="utf-8"),
        })
    return documents


def chunk_document(doc: dict, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    content = doc["content"]
    step = chunk_size - overlap

    chunks = []
    chunk_index = 0
    for start in range(0, len(content), step):
        chunk_text = content[start:start + chunk_size]
        if not chunk_text:
            break
        chunks.append({
            "chunk_id": f"{doc['filename']}_{chunk_index}",
            "filename": doc["filename"],
            "category": doc["category"],
            "content": chunk_text,
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        if start + chunk_size >= len(content):
            break
    return chunks


def embed_and_store(chunks: list[dict], db_path: str) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="support_docs")

    openai_client = OpenAI(api_key=settings.openai_api_key)
    documents = [chunk["content"] for chunk in chunks]
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=documents)
    embeddings = [item.embedding for item in response.data]

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "filename": chunk["filename"],
                "category": chunk["category"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk["filename"]] = chunk_counts.get(chunk["filename"], 0) + 1

    for filename, chunk_count in chunk_counts.items():
        logger.info(
            "document_ingested",
            extra={"event": "document_ingested", "doc_filename": filename, "chunk_count": chunk_count},
        )
