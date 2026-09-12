"""
Text Summarization API endpoint for NLP Document Intelligence Gateway.
"""

from fastapi import APIRouter, status
from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.providers import get_llm_provider

router = APIRouter(tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize Support Text",
    description="Generates an executive summary and key highlight points from customer conversations or documents.",
)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Generates a concise summary and key takeaway bullet points using the configured LLM provider
    or deterministic heuristic summarizer.
    """
    words = request.text.split()
    original_len = len(words)

    provider = get_llm_provider()
    result = provider.summarize_text(request.text, max_length=request.max_length or 150)

    summary = result.get("summary", "")
    key_points = result.get("key_points", [])

    return SummarizeResponse(
        summary=summary,
        key_points=key_points,
        original_word_count=original_len,
        summary_word_count=len(summary.split()),
    )
