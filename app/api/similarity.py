"""
Semantic Similarity API endpoint for NLP Document Intelligence Gateway.
"""

from fastapi import APIRouter, status
from app.schemas.similarity import SimilarityRequest, SimilarityResponse
from app.nlp.embeddings import embed_text, compute_cosine_similarity
from app.config import settings

router = APIRouter(tags=["Similarity"])


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Semantic Similarity",
    description="Compares two text snippets and computes cosine similarity between dense embeddings.",
)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Computes dense vector representations of text_a and text_b using SentenceTransformers
    and calculates their cosine similarity. Determines if the pair represents a duplicate message.
    """
    # Fast path for identical strings
    if request.text_a.strip().lower() == request.text_b.strip().lower():
        score = 1.0
    else:
        vec_a = embed_text(request.text_a)
        vec_b = embed_text(request.text_b)
        score = round(compute_cosine_similarity(vec_a, vec_b), 4)

    threshold = getattr(settings, "DUPLICATE_SIMILARITY_THRESHOLD", 0.85)
    is_dup = score >= threshold

    return SimilarityResponse(
        similarity_score=score,
        is_duplicate=is_dup,
    )
