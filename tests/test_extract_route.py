import io
import os
import sys
import docx
from fastapi.testclient import TestClient

# Dynamically add project root directory to Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


def create_dummy_docx_bytes(text: str) -> bytes:
    """Helper function to create an in-memory DOCX file byte stream."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_extract_endpoint_docx():
    """Test POST /api/v1/extract with a DOCX file."""
    sample_text = "This is a sample document for testing the extraction endpoint."
    docx_bytes = create_dummy_docx_bytes(sample_text)

    response = client.post(
        "/api/v1/extract",
        files={"file": ("test_doc.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.docx"
    assert data["file_type"] == "docx"
    assert sample_text in data["extracted_text"]
    assert data["word_count"] > 0
    assert data["char_count"] > 0
    assert data["error"] is None


def test_extract_endpoint_unsupported_file():
    """Test POST /api/v1/extract with an unsupported file extension."""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("test_file.txt", b"Plain text content", "text/plain")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]


