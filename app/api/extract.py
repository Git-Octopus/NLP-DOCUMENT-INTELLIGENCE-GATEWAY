from fastapi import APIRouter, File, UploadFile, status, HTTPException
from app.schemas.extract import ExtractResponse

router = APIRouter(tags=["Document Extraction"])


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Text from Document (Dummy Skeleton for Day 2)",
    description="Extracts raw text and metadata from uploaded PDF or DOCX support documents.",
)
async def extract_document(
    file: UploadFile = File(..., description="Document file to extract text from (.pdf or .docx)")
) -> ExtractResponse:
    """
    Day 2 skeleton endpoint. Returns schema-compliant dummy extracted text.
    Will be wired to PyMuPDF (PDF) and python-docx (DOCX) in Day 5-6 / Week 2.
    """
    filename = file.filename or "sample_document.pdf"
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    if extension not in ["pdf", "docx"]:
        file_type = "pdf"
    else:
        file_type = extension

    dummy_text = (
        f"Customer Support Inquiry Document\n"
        f"Filename: {filename}\n"
        f"Issue Description: Customer reported a duplicate billing charge of $49.99 on invoice #INV-9821.\n"
        f"Contact: support-user@company.org\n"
        f"Status: Under Investigation."
    )

    return ExtractResponse(
        filename=filename,
        file_type=file_type,
        extracted_text=dummy_text,
        word_count=len(dummy_text.split()),
        char_count=len(dummy_text),
        error=None,
    )
