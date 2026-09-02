from pydantic import BaseModel, Field, ConfigDict


class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1, description="First text snippet to compare")
    text_b: str = Field(..., min_length=1, description="Second text snippet to compare")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_a": "I want a refund for my order #12345.",
                "text_b": "Can I please get my money back for order 12345?",
            }
        }
    )


class SimilarityResponse(BaseModel):
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between text_a and text_b embeddings (0.0 to 1.0)",
    )
    is_duplicate: bool = Field(
        ..., description="Whether similarity meets or exceeds the duplicate threshold"
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"similarity_score": 0.91, "is_duplicate": True}}
    )
