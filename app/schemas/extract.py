from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ExtractResponse(BaseModel):
    filename: str = Field(..., description="Original filename of the uploaded document")
    file_type: str = Field(..., description="Detected file type: 'pdf' or 'docx'")
    extracted_text: str = Field(..., description="Raw text extracted from the document")
    word_count: int = Field(..., ge=0, description="Total word count of extracted text")
    char_count: int = Field(
        ..., ge=0, description="Total character count of extracted text"
    )
    error: Optional[str] = Field(
        None, description="Error message if text extraction failed"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "support_ticket_invoice.pdf",
                "file_type": "pdf",
                "extracted_text": (
                    "Invoice #INV-9821\nCustomer: John Doe\nAmount: $49.99\nDescription: Duplicate charge inquiry."
                ),
                "word_count": 12,
                "char_count": 82,
                "error": None,
            }
        }
    )
