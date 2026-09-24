import io
import os
import sys
import pytest
import docx
from fastapi.testclient import TestClient

# Dynamically add project root directory to Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.database import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def prepare_test_db():
    """Create database tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def create_dummy_docx_bytes(text: str) -> bytes:
    """Helper function to create an in-memory DOCX file byte stream."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_analyze_endpoint_text_input():
    """Test POST /analyze with direct text input using Form data."""
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


def test_analyze_endpoint_file_upload_docx():
    """Test POST /analyze with DOCX file upload."""
    sample_text = (
        "This is a sample document for testing the analyze endpoint with file upload."
    )
    docx_bytes = create_dummy_docx_bytes(sample_text)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "test_doc.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == sample_text
    assert data["doc_metadata"] is not None
    assert data["doc_metadata"]["filename"] == "test_doc.docx"
    assert data["doc_metadata"]["file_type"] == "docx"
    assert data["doc_metadata"]["word_count"] > 0
    assert data["doc_metadata"]["char_count"] > 0


def test_analyze_endpoint_file_upload_pdf():
    """Test POST /analyze with PDF file upload."""
    # A tiny valid binary PDF stream containing basic text data
    fake_pdf_bytes = (
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

    response = client.post(
        "/analyze",
        files={"file": ("test.pdf", fake_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "Sample PDF Text" in data["original_text"]
    assert data["doc_metadata"] is not None
    assert data["doc_metadata"]["filename"] == "test.pdf"
    assert data["doc_metadata"]["file_type"] == "pdf"


def test_analyze_endpoint_no_input():
    """Test error when neither text nor file is provided."""
    response = client.post("/analyze", data={})
    assert response.status_code == 400
    data = response.json()
    assert (
        "Either 'text' string or 'file' document upload must be provided"
        in data["detail"]
    )


def test_analyze_endpoint_minimal_text():
    """Test with minimal valid text (3 characters)."""
    response = client.post("/analyze", data={"text": "Hi!"})
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Hi!"


def test_analyze_endpoint_text_too_short():
    """Test validation error for text shorter than 3 characters."""
    response = client.post("/analyze", data={"text": "Hi"})
    assert response.status_code == 422
    data = response.json()
    assert (
        "Text content must contain at least 3 non-whitespace characters"
        in data["detail"]
    )


def test_analyze_endpoint_oversized_file():
    """Test file size validation with oversized file."""
    # Create a file larger than 10MB (in practice, use a smaller size for testing)
    large_file = b"x" * (11 * 1024 * 1024)  # 11MB

    response = client.post(
        "/analyze",
        files={"file": ("large.txt", large_file, "text/plain")},
    )

    assert response.status_code == 413
    data = response.json()
    assert "File size exceeds maximum allowed limit" in data["detail"]


def test_analyze_response_structure():
    """Test that response structure matches AnalyzeResponse schema."""
    response = client.post(
        "/analyze", data={"text": "Hello, I need help with my account."}
    )
    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    required_fields = [
        "request_id",
        "original_text",
        "cleaned_text",
        "topic",
        "intent",
        "entities",
        "keywords",
        "sentiment",
        "evidence_sentences",
        "duplicate_info",
        "llm_fallback_used",
    ]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Verify nested object structure
    assert "topic" in data["topic"]
    assert "confidence" in data["topic"]
    assert "resolved_by" in data["topic"]

    assert "intent" in data["intent"]
    assert "confidence" in data["intent"]
    assert "resolved_by" in data["intent"]

    assert "sentiment" in data["sentiment"]
    assert "score" in data["sentiment"]

    assert "is_duplicate" in data["duplicate_info"]

    # Verify data types
    assert isinstance(data["entities"], list)
    assert isinstance(data["keywords"], list)
    assert isinstance(data["evidence_sentences"], list)
    assert isinstance(data["llm_fallback_used"], bool)


def test_analyze_endpoint_with_api_prefix():
    """Test that /api/v1/analyze also works."""
    response = client.post(
        "/api/v1/analyze", data={"text": "Testing API prefix endpoint."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Testing API prefix endpoint."
