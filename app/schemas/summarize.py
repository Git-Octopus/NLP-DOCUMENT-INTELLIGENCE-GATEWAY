from pydantic import BaseModel, Field
from typing import List, Optional

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text or document content to summarize")
    max_length: Optional[int] = Field(150, description="Target word count for summary")

class SummarizeResponse(BaseModel):
    summary: str
    key_points: List[str]
    original_word_count: int
    summary_word_count: int