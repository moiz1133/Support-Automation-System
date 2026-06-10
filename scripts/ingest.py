import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import chunk_document, embed_and_store, load_documents
from app.logger import get_logger

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def main() -> None:
    start = time.perf_counter()

    documents = load_documents(str(DOCS_DIR))

    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc))

    embed_and_store(chunks, settings.chroma_db_path)

    elapsed_ms = round((time.perf_counter() - start) * 1000)
    logger.info(
        "ingestion_complete",
        extra={
            "event": "ingestion_complete",
            "total_docs": len(documents),
            "total_chunks": len(chunks),
            "elapsed_ms": elapsed_ms,
        },
    )


if __name__ == "__main__":
    main()
