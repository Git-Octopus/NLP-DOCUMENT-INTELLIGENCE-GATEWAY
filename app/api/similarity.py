import difflib
from fastapi import APIRouter, status, HTTPException 
from app.schemas.similarity import SimilarityRequest, SimilarityResponse
from app.config import settings
from app.nlp.embeddings import get_embedding, compute_cosine_similarity

router = APIRouter(tags=["Similarity"])

def _tfidf_fallback_similarity(text_a: str, text_b: str) -> float:
    """Fallback similarity calculator using scikit-learn TF-IDF / sequence matching."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer().fit_transform([text_a, text_b])
        vectors = vectorizer.toarray()
        score = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
        return round(max(0.0, min(1.0, score)), 4)
    except Exception:
        # Ultimate fallback using SequenceMatcher ratio
        ratio = difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
        return round(ratio, 4)

@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Semantic Similarity (Dummy Skeleton for Day 2)",
    description="Compares two text snippets and computes cosine similarity between dense embeddings.",
)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Computes semantic similarity between text_a and text_b using dense vector 
    embeddings and cosine similarity, evaluating against duplicate threshold.
    """
    # 1. Input Validation Guard
    text_a_clean = request.text_a.strip()
    text_b_clean = request.text_b.strip()
    
    if not text_a_clean or not text_b_clean:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both 'text_a' and 'text_b' must contain non-whitespace text.",
        )

    # 2. Exact match check
    if text_a_clean.lower() == text_b_clean.lower():
        return SimilarityResponse(
            similarity_score=1.0,
            is_duplicate=True
        )

    # 3. Compute vector embeddings
    vec_a = get_embedding(text_a_clean)
    vec_b = get_embedding(text_b_clean)

    # Check if dense vectors were generated properly
    if any(vec_a) and any(vec_b):
        score = compute_cosine_similarity(vec_a, vec_b)
        score = round(score, 4)
    else:
        # Fallback to local TF-IDF similarity if dense model is not loaded
        score = _tfidf_fallback_similarity(text_a_clean, text_b_clean)

    # 4. Evaluate duplicate threshold
    is_dup = bool(score >= settings.DUPLICATE_SIMILARITY_THRESHOLD)

    return SimilarityResponse(
        similarity_score=score,
        is_duplicate=is_dup,
    )