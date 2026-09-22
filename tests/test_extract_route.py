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
    assert data["filename"] == "test_doc.docx"
    assert data["file_type"] == "docx"
    assert sample_text in data["extracted_text"]
    assert data["word_count"] > 0
    assert data["char_count"] > 0
    assert data["error"] is None


def test_extract_endpoint_pdf():
    """Test POST /api/v1/extract with a PDF file."""
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
        "/api/v1/extract",
        files={"file": ("test.pdf", fake_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["file_type"] == "pdf"
    assert "Sample PDF Text" in data["extracted_text"]
    assert data["word_count"] > 0
    assert data["char_count"] > 0


def test_extract_endpoint_empty_file():
    """Test POST /api/v1/extract with an empty file."""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "Uploaded file is empty" in data["detail"]


def test_extract_endpoint_unsupported_file():
    """Test POST /api/v1/extract with an unsupported file extension."""
    response = client.post(
        "/api/v1/extract",
        files={"file": ("test_file.txt", b"Plain text content", "text/plain")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]
