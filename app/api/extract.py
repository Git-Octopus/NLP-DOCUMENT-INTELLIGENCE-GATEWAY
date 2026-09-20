import os
import sys
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.schemas import ExtractResponse
from app.services.extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    ExtractionError,
)
from app.services.database import get_db

router = APIRouter(tags=["Document Extraction"])

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Text from Document (PDF/DOCX)",
)
async def extract_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ExtractResponse:
    """Reads uploaded PDF or DOCX binary streams and extracts raw text content with error handling."""
    filename = file.filename or "uploaded_document"
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Check file size limit
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # Determine file type from extension
    lower_name = filename.lower()
    extracted_text = ""
    file_type = "unknown"

    try:
        if lower_name.endswith(".pdf"):
            file_type = "pdf"
            extracted_text = extract_text_from_pdf(file_bytes)
        elif lower_name.endswith(".docx"):
            file_type = "docx"
            extracted_text = extract_text_from_docx(file_bytes)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload a .pdf or .docx file.",
            )

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        if isinstance(exc, ExtractionError):
            return ExtractResponse(
                filename=filename,
                file_type=file_type,
                extracted_text="",
                word_count=0,
                char_count=0,
                error=str(exc),
            )
        return ExtractResponse(
            filename=filename,
            file_type=file_type,
            extracted_text="",
            word_count=0,
            char_count=0,
            error=f"Failed to extract document content: {str(exc)}",
        )

    # Calculate word and character counts
    words = extracted_text.split()
    word_count = len(words)
    char_count = len(extracted_text)

    return ExtractResponse(
        filename=filename,
        file_type=file_type,
        extracted_text=extracted_text,
        word_count=word_count,
        char_count=char_count,
        error=None,
    )


# --- DIRECT TEST RUNNER ---
if __name__ == "__main__":
    print("\n--- Testing Extract API Route Logic ---")
    print("Router object imported successfully!")
    print(f"Registered route tags: {router.tags}\n")