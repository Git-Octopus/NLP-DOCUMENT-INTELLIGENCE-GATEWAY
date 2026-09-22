import io
import os
import sys
import docx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


def create_dummy_docx_bytes(text: str) -> bytes:
    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_day14_extract_and_validation_checkpoint():
    """Day 14 Checkpoint: Verify file extraction, size limits, and error handling."""
    sample_text = "This is a Day 14 test ticket document."
    docx_bytes = create_dummy_docx_bytes(sample_text)

    # 1. Valid extraction
    response = client.post(
        "/extract",
        files={
            "file": (
                "day14_test.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "day14_test.docx"
    assert sample_text in data["extracted_text"]
    assert data["error"] is None

    # 2. Empty file handling
    empty_res = client.post(
        "/extract", files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert empty_res.status_code == 400
