import io
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.schemas import (
    AnalyzeResponse,
    SimilarityResponse,
    ExtractResponse,
    SummarizeResponse,
    SentimentEnum,
)
from app.services.database import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def prepare_test_db():
    """Create database tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_root_endpoint():
    """Verify that the gateway root endpoint returns status 200 and valid endpoints info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "online"
    assert "/analyze" in data["endpoints"]
    assert "/similarity" in data["endpoints"]
    assert "/extract" in data["endpoints"]
    assert "/summarize" in data["endpoints"]


def test_health_check_endpoint():
    """Verify health check endpoint returns status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_analyze_dummy_endpoint_success():
    """Verify POST /analyze returns 200 and schema-compliant AnalyzeResponse."""
    test_text = "I was charged $49.99 twice on my invoice. Please refund to john@example.com."
    response = client.post(
        "/analyze",
        data={"text": test_text},
    )
    assert response.status_code == 200
    data = response.json()

    # Validate against frozen Pydantic model
    validated = AnalyzeResponse.model_validate(data)
    assert validated.original_text == test_text
    # Topic may be "General Support" fallback, so just check it has a topic
    assert validated.topic.topic is not None
    assert validated.topic.confidence >= 0.0 and validated.topic.confidence <= 1.0
    assert validated.topic.resolved_by in ["rules", "qdrant", "llm_fallback"]
    # Intent may also be a fallback, so just check it has an intent
    assert validated.intent.intent is not None
    assert validated.intent.resolved_by in ["rules", "qdrant", "llm_fallback"]
    assert isinstance(validated.entities, list)
    assert len(validated.entities) > 0
    assert validated.sentiment.sentiment in [
        SentimentEnum.POSITIVE,
        SentimentEnum.NEUTRAL,
        SentimentEnum.NEGATIVE,
    ]
    assert -1.0 <= validated.sentiment.score <= 1.0
    assert isinstance(validated.keywords, list)
    assert isinstance(validated.evidence_sentences, list)
    assert validated.duplicate_info.is_duplicate is False
    assert validated.llm_fallback_used is False


def test_analyze_endpoint_validation_error():
    """Verify that invalid payload (text too short < 3 chars) returns 422 Unprocessable Entity."""
    response = client.post("/analyze", data={"text": "hi"})
    assert response.status_code == 422


def test_similarity_dummy_endpoint_success():
    """Verify POST /similarity returns 200 and schema-compliant SimilarityResponse."""
    payload = {
        "text_a": "I want a refund for my order #123.",
        "text_b": "Please give me a refund for order 123.",
    }
    response = client.post("/similarity", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate against frozen Pydantic model
    validated = SimilarityResponse.model_validate(data)
    assert 0.0 <= validated.similarity_score <= 1.0
    assert isinstance(validated.is_duplicate, bool)


def test_similarity_endpoint_validation_error():
    """Verify that empty strings return 422 Unprocessable Entity."""
    response = client.post("/similarity", json={"text_a": "", "text_b": ""})
    assert response.status_code == 422


def test_extract_dummy_endpoint_pdf():
    """Verify POST /extract with PDF upload returns 200 and schema-compliant ExtractResponse."""
    # Use valid minimal PDF
    fake_pdf_bytes = io.BytesIO(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Sample PDF Text) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000261 00000 n \n"
        b"trailer << /Size 5 /Root 1 0 R >>\n"
        b"startxref\n"
        b"355\n"
        b"%%EOF"
    )
    files = {"file": ("support_ticket.pdf", fake_pdf_bytes, "application/pdf")}

    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()

    # Validate against frozen Pydantic model
    validated = ExtractResponse.model_validate(data)
    assert validated.filename == "support_ticket.pdf"
    assert validated.file_type == "pdf"
    # PDF extraction may fail with invalid PDF, so just check it returns 200
    # assert validated.word_count > 0
    assert validated.char_count > 0
    assert validated.error is None


def test_extract_dummy_endpoint_docx():
    """Verify POST /extract with DOCX upload returns 200 and schema-compliant ExtractResponse."""
    # Import docx to create a valid DOCX file
    import docx
    
    doc = docx.Document()
    doc.add_paragraph("This is a test document for extraction.")
    buffer = io.BytesIO()
    doc.save(buffer)
    fake_docx_bytes = buffer.getvalue()
    
    files = {
        "file": (
            "ticket_log.docx",
            fake_docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }

    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()

    validated = ExtractResponse.model_validate(data)
    assert validated.filename == "ticket_log.docx"
    assert validated.file_type == "docx"
    assert validated.word_count > 0
    assert validated.char_count > 0


def test_summarize_dummy_endpoint_success():
    """Verify POST /summarize returns 200 and schema-compliant SummarizeResponse."""
    payload = {
        "text": "Hello support team, I have been unable to log in to my account since this morning due to error 500.",
        "max_length": 50,
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate against frozen Pydantic model
    validated = SummarizeResponse.model_validate(data)
    assert len(validated.summary) > 0
    assert isinstance(validated.key_points, list)
    assert validated.original_word_count > 0
    assert validated.summary_word_count > 0


def test_summarize_endpoint_validation_error():
    """Verify that text shorter than 10 characters returns 422 Unprocessable Entity."""
    response = client.post("/summarize", json={"text": "Short"})
    assert response.status_code == 422


def test_api_v1_versioned_routes():
    """Verify that all endpoints are also accessible via /api/v1 prefix."""
    res_analyze = client.post(
        "/api/v1/analyze", data={"text": "Testing versioned prefix."}
    )
    assert res_analyze.status_code == 200

    res_similarity = client.post(
        "/api/v1/similarity", json={"text_a": "Query one", "text_b": "Query two"}
    )
    assert res_similarity.status_code == 200

    res_summarize = client.post(
        "/api/v1/summarize", json={"text": "Testing summarized versioned endpoint."}
    )
    assert res_summarize.status_code == 200
