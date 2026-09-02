from fastapi import APIRouter, status
from app.schemas.similarity import SimilarityRequest, SimilarityResponse
from app.config import settings

router = APIRouter(tags=["Similarity"])


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Semantic Similarity (Dummy Skeleton for Day 2)",
    description="Compares two text snippets and computes cosine similarity between dense embeddings.",
)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Day 2 skeleton endpoint. Returns schema-compliant dummy similarity score.
    Will be wired to SentenceTransformers and Qdrant in Week 2.
    """
    # Dummy similarity score: higher if exact match or long overlap, default mock 0.88
    if request.text_a.strip().lower() == request.text_b.strip().lower():
        score = 1.0
    else:
        score = 0.88

    is_dup = score >= settings.DUPLICATE_SIMILARITY_THRESHOLD

    return SimilarityResponse(
        similarity_score=score,
        is_duplicate=is_dup,
    )
