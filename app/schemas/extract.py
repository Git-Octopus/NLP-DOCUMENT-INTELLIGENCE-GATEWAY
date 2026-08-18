from pydantic import BaseModel, Field
from typing import Optional

class ExtractResponse(BaseModel):
    filename: str
    file_type: str = Field(..., description="'pdf' or 'docx'")
    extracted_text: str
    word_count: int
    char_count: int
    error: Optional[str] = None