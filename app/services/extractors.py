import io
import os
import sys

import fitz
import docx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class ExtractionError(Exception):
    """Custom exception raised when file text extraction fails."""

    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text content from a PDF file byte stream with error handling."""
    if not file_bytes:
        raise ExtractionError("Provided PDF file is empty.")

    text_content = []

    try:
        # Open the byte stream using PyMuPDF (fitz)
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            if pdf_doc.is_encrypted:
                raise ExtractionError(
                    "Encrypted or password-protected PDF files are not supported."
                )

            for page in pdf_doc:
                text = page.get_text()
                if text:
                    text_content.append(text)
    except Exception as e:
        if isinstance(e, ExtractionError):
            raise e
        raise ExtractionError(f"Failed to parse PDF document: {str(e)}")

    extracted_text = "\n".join(text_content).strip()

    if not extracted_text:
        raise ExtractionError(
            "PDF file processed successfully but contains no readable text content."
        )

    return extracted_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts raw text content from a DOCX file byte stream with error handling."""
    if not file_bytes:
        raise ExtractionError("Provided DOCX file is empty.")

    try:
        # Convert raw bytes to an in-memory binary stream
        docx_stream = io.BytesIO(file_bytes)

        # Open the document using python-docx
        doc = docx.Document(docx_stream)

        # Extract text from each paragraph
        text_content = [
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text and paragraph.text.strip()
        ]
    except Exception as e:
        raise ExtractionError(f"Failed to parse DOCX document: {str(e)}")

    extracted_text = "\n".join(text_content).strip()

    if not extracted_text:
        raise ExtractionError(
            "DOCX file processed successfully but contains no readable text content."
        )

    return extracted_text


def parse_document_stream(file_bytes: bytes, filename: str) -> dict:
    """
    Routes incoming byte streams to the appropriate extractor based on extension
    and calculates metadata properties (filename, extension type, word count, character count).
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext == "pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        extracted_text = extract_text_from_docx(file_bytes)
    else:
        raise ExtractionError(
            f"Unsupported file format: .{ext}. Only .pdf and .docx are supported."
        )

    words = extracted_text.split()
    return {
        "filename": filename,
        "file_type": ext,
        "extracted_text": extracted_text,
        "word_count": len(words),
        "char_count": len(extracted_text),
    }
