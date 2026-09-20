from fastapi import APIRouter, status , HTTPException
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
     Returns schema-compliant dummy summarization data.
    Will be wired to Gemini/OpenAI provider fallback in Week 2-3.
    """
    
    # Input Validation Guard 
    if len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text input for summarization must be at least 10 characters long.",
        )
        
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
