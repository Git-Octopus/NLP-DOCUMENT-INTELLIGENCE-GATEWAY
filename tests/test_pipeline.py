# tests/test_pipeline.py

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.database import Base, engine, SessionLocal
from app.models import AnalysisModel

client = TestClient(app)


@pytest.fixture(autouse=True)
def prepare_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_analyze_pipeline_integration():
    # Send as form data (current implementation uses Form)
    response = client.post(
        "/analyze",
        data={"text": "I was charged $49.99 on my account. Please issue a refund."},
    )
    assert response.status_code == 200

    data = response.json()
    assert (
        data["original_text"]
        == "I was charged $49.99 on my account. Please issue a refund."
    )
    assert "cleaned_text" in data
    assert "topic" in data
    assert "intent" in data
    assert "sentiment" in data
    assert "keywords" in data
    assert "entities" in data
    assert "evidence_sentences" in data
    assert "duplicate_info" in data

    # Verify response structure matches current implementation
    assert data["topic"]["resolved_by"] in ["qdrant", "rules", "llm_fallback"]
    assert data["intent"]["resolved_by"] in ["qdrant", "rules", "llm_fallback"]
    assert data["sentiment"]["sentiment"] in ["positive", "neutral", "negative"]
    assert -1.0 <= data["sentiment"]["score"] <= 1.0

    # Confirm record is written to SQLite DB
    db = SessionLocal()
    try:
        record = (
            db.query(AnalysisModel)
            .filter(
                AnalysisModel.original_text
                == "I was charged $49.99 on my account. Please issue a refund."
            )
            .first()
        )
        assert record is not None
        assert record.request_id == data["request_id"]
    finally:
        db.close()


def test_analyze_with_minimal_text():
    """Test with minimal valid text (3 characters)."""
    response = client.post("/analyze", data={"text": "Hi!"})
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Hi!"


def test_analyze_validation_error():
    """Test validation error for text shorter than 3 characters."""
    response = client.post("/analyze", data={"text": "Hi"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_analyze_no_input():
    """Test error when neither text nor file is provided."""
    response = client.post("/analyze", data={})
    assert response.status_code == 400
    data = response.json()
    assert (
        "Either 'text' string or 'file' document upload must be provided"
        in data["detail"]
    )
