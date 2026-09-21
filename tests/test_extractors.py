import io
import os
import sys
import pytest
import docx

# Dynamically add project root directory to Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.extractors import extract_text_from_pdf, extract_text_from_docx


def create_dummy_docx_bytes(text: str) -> bytes:
    """Helper function to create an in-memory DOCX file byte stream."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_extract_text_from_docx():
    """Verify that extract_text_from_docx correctly extracts paragraph text."""
    sample_text = "Hello, this is a test support ticket in DOCX format."
    docx_bytes = create_dummy_docx_bytes(sample_text)
    
    extracted = extract_text_from_docx(docx_bytes)
    assert sample_text in extracted


def test_extract_text_from_pdf():
    """Verify that extract_text_from_pdf handles binary PDF stream input."""
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
    
    extracted = extract_text_from_pdf(fake_pdf_bytes)
    assert "Sample PDF Text" in extracted


