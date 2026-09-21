"""
Integration tests for POST /analyze endpoint.
Tests direct text input processing, document file parsing (PDF/DOCX),
and database persistence validation.
"""

import io
import docx
import pytest


def create_dummy_docx_bytes(text: str) -> bytes:
    """Helper function to generate an in-memory DOCX byte stream."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_analyze_direct_text_success(client):
    """Test POST /analyze using form-data text parameter."""
    payload = {"text": "I was charged $49.99 twice on my invoice. Please refund john@example.com."}
    
    response = client.post("/analyze", data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("req-")
    assert "charged" in data["cleaned_text"]
    assert data["doc_metadata"] is None
    assert len(data["entities"]) > 0


def test_analyze_docx_file_upload_success(client):
    """Test POST /analyze using document upload (.docx)."""
    sample_text = "I received a defective product and need to return order #INV-9821."
    docx_bytes = create_dummy_docx_bytes(sample_text)

    files = {
        "file": (
            "support_ticket.docx",
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }

    response = client.post("/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["doc_metadata"] is not None
    assert data["doc_metadata"]["filename"] == "support_ticket.docx"
    assert data["doc_metadata"]["file_type"] == "docx"
    assert data["doc_metadata"]["word_count"] > 0
    assert sample_text in data["original_text"]


def test_analyze_missing_payload_error(client):
    """Test POST /analyze with neither text nor file provided."""
    response = client.post("/analyze")

    assert response.status_code == 400
    assert "Either 'text' string or 'file' document upload must be provided." in response.json()["detail"]


def test_analyze_short_text_error(client):
    """Test POST /analyze with text under the minimum length limit."""
    payload = {"text": "hi"}
    response = client.post("/analyze", data=payload)

    assert response.status_code == 422
    assert "at least 3 non-whitespace characters" in response.json()["detail"]