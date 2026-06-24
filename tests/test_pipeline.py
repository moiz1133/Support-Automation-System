import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import pytest
from fastapi.testclient import TestClient

from app.classifier import classify
from app.config import settings
from app.cost_tracker import PRICING, RequestCostTracker
from app.ingestion import chunk_document, embed_and_store, load_documents
from app.main import app
from app.retrieval import retrieve

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
EMBED_DIM = 8

client = TestClient(app)


def make_embedding_response(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1] * EMBED_DIM) for _ in range(n)],
        model="text-embedding-3-large",
        usage=SimpleNamespace(prompt_tokens=10 * n, total_tokens=10 * n),
    )


def make_chat_response(content: str, prompt_tokens: int = 20, completion_tokens: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model="gpt-3.5-turbo",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.fixture(scope="module")
def chroma_db_path(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("chroma")

    with patch("app.ingestion.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.side_effect = (
            lambda model, input: make_embedding_response(len(input))
        )

        documents = load_documents(str(DOCS_DIR))
        chunks = []
        for doc in documents:
            chunks.extend(chunk_document(doc))
        embed_and_store(chunks, str(db_path))

    return str(db_path)


def test_ingestion_loads_all_20_docs(chroma_db_path):
    db_client = chromadb.PersistentClient(path=chroma_db_path)
    collection = db_client.get_collection(name="support_docs")
    assert collection.count() >= 20


def test_retrieval_returns_top_3(chroma_db_path):
    tracker = RequestCostTracker("test-retrieval")

    with patch("app.retrieval.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.embeddings.create.return_value = make_embedding_response(1)
        results = retrieve("How do I reset my password?", chroma_db_path, tracker)

    assert len(results) == 3
    for chunk in results:
        assert "content" in chunk
        assert "category" in chunk
        assert "distance" in chunk


def test_retrieval_category_filter_returns_matching_sources(chroma_db_path):
    tracker = RequestCostTracker("test-retrieval-category")

    with patch("app.retrieval.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.embeddings.create.return_value = make_embedding_response(1)
        results = retrieve("How do I reset my password?", chroma_db_path, tracker, category="technical")

    assert len(results) > 0
    for chunk in results:
        assert chunk["category"] == "technical"


def test_retrieval_category_filter_falls_back_on_no_matches(chroma_db_path):
    tracker = RequestCostTracker("test-retrieval-category-fallback")

    with patch("app.retrieval.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.embeddings.create.return_value = make_embedding_response(1)
        results = retrieve("How do I reset my password?", chroma_db_path, tracker, category="nonexistent_category")

    assert len(results) == 3


def test_classification_answerable():
    tracker = RequestCostTracker("test-classify-answerable")

    with patch("app.classifier.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.chat.completions.create.return_value = make_chat_response(
            json.dumps(
                {"intent": "answerable", "confidence": 0.9, "reason": "Documented process", "category": "billing"}
            )
        )
        result = classify("How do I cancel my subscription?", tracker)

    assert result["intent"] == "answerable"
    assert result["category"] == "billing"


def test_classification_escalate():
    with patch("app.classifier.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.chat.completions.create.return_value = make_chat_response(
            json.dumps(
                {
                    "intent": "escalate",
                    "confidence": 0.95,
                    "reason": "Possible unauthorized access",
                    "category": "escalation",
                }
            )
        )
        response = client.post("/query", json={"query": "Someone accessed my account"})

    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is True
    assert data["category"] == "escalation"
    assert "human support" in data["answer"]


def test_classification_low_confidence_demotes():
    with patch("app.classifier.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.chat.completions.create.return_value = make_chat_response(
            json.dumps(
                {
                    "intent": "answerable",
                    "confidence": 0.4,
                    "reason": "Too vague to resolve directly",
                    "category": "unknown",
                }
            )
        )
        response = client.post("/query", json={"query": "It's broken"})

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "needs_more_info"
    assert data["category"] == "unknown"


def test_cost_tracker_sums_correctly():
    tracker = RequestCostTracker("test-cost-tracker")
    tracker.add_call(model="gpt-3.5-turbo", prompt_tokens=100, completion_tokens=50, call_type="classification")
    tracker.add_call(model="text-embedding-3-large", prompt_tokens=20, completion_tokens=0, call_type="embedding")

    expected_cost = (
        100 * PRICING["gpt-3.5-turbo"]["prompt"]
        + 50 * PRICING["gpt-3.5-turbo"]["completion"]
        + 20 * PRICING["text-embedding-3-large"]["prompt"]
    )

    summary = tracker.summary()
    assert summary["total_tokens"] == 170
    assert summary["total_cost_usd"] == pytest.approx(expected_cost)


def test_full_query_endpoint(chroma_db_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_db_path", chroma_db_path)

    with patch("app.classifier.OpenAI") as mock_classifier_openai, \
            patch("app.retrieval.OpenAI") as mock_retrieval_openai, \
            patch("app.llm.OpenAI") as mock_llm_openai:

        mock_classifier_openai.return_value.chat.completions.create.return_value = make_chat_response(
            json.dumps(
                {
                    "intent": "answerable",
                    "confidence": 0.95,
                    "reason": "Documented process",
                    "category": "technical",
                }
            )
        )
        mock_retrieval_openai.return_value.embeddings.create.return_value = make_embedding_response(1)
        mock_llm_openai.return_value.chat.completions.create.return_value = make_chat_response(
            "To reset your password, click 'Forgot Password?' on the login page and follow the emailed link."
        )

        response = client.post("/query", json={"query": "How do I reset my password?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "intent" in data
    assert data["category"] == "technical"
    assert all(source.startswith("technical_") for source in data["sources"])
    assert "total_tokens" in data["usage"]
    assert "total_cost_usd" in data["usage"]
