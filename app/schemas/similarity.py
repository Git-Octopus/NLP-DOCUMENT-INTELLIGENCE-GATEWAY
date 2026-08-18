from pydantic import BaseModel, Field

class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1, description="First text snippet")
    text_b: str = Field(..., min_length=1, description="Second text snippet")

class SimilarityResponse(BaseModel):
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity between text_a and text_b embeddings")
    is_duplicate: bool