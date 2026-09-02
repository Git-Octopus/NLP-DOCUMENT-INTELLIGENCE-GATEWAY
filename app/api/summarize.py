from fastapi import APIRouter, status
from app.schemas.summarize import SummarizeRequest, SummarizeResponse

router = APIRouter(tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize Support Text (Dummy Skeleton for Day 2)",
    description="Generates an executive summary and key highlight points from customer conversations or documents.",
)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Day 2 skeleton endpoint. Returns schema-compliant dummy summarization data.
    Will be wired to Gemini/OpenAI provider fallback in Week 2-3.
    """
    words = request.text.split()
    original_len = len(words)

    dummy_summary = (
        "Customer reported repeated difficulty accessing corporate account features due to "
        "authentication errors, requesting urgent support."
    )
    dummy_key_points = [
        "Persistent service access errors reported",
        "Initial customer troubleshooting attempts unsuccessful",
        "Action required by tier-2 support team",
    ]

    return SummarizeResponse(
        summary=dummy_summary,
        key_points=dummy_key_points,
        original_word_count=original_len,
        summary_word_count=len(dummy_summary.split()),
    )
