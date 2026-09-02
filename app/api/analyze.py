import uuid
from fastapi import APIRouter, status
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    TopicMatch,
    IntentMatch,
    ExtractedEntity,
    SentimentAnalysis,
    SentimentEnum,
    DuplicateCheck,
)

router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Customer Support Text (Dummy Skeleton for Day 2)",
    description="Takes unstructured customer support text and returns structured NLP analysis (Topic, Intent, Entities, Keywords, Sentiment, Evidence).",
)
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Day 2 skeleton endpoint. Returns schema-compliant dummy structured data.
    Will be wired to local NLP pipeline (Dev 2) and Qdrant/LLM fallback (Dev 3) in Week 2-3.
    """
    # Simple dummy text cleaning for response illustration
    cleaned = request.text.lower().strip()

    return AnalyzeResponse(
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        original_text=request.text,
        cleaned_text=cleaned,
        topic=TopicMatch(
            topic="Billing & Refunds",
            confidence=0.92,
            resolved_by="qdrant",
        ),
        intent=IntentMatch(
            intent="request_refund",
            confidence=0.88,
            resolved_by="rules",
        ),
        entities=[
            ExtractedEntity(
                text="$49.99",
                label="MONEY",
                source="regex",
                start_char=21,
                end_char=27,
            ),
            ExtractedEntity(
                text="john@example.com",
                label="EMAIL",
                source="regex",
                start_char=request.text.find("@") - 4 if "@" in request.text else None,
                end_char=request.text.find("@") + 12 if "@" in request.text else None,
            ),
        ],
        keywords=["refund", "invoice", "payment", "charge"],
        sentiment=SentimentAnalysis(
            sentiment=SentimentEnum.NEGATIVE,
            score=-0.65,
        ),
        evidence_sentences=[
            request.text.split(".")[0] if "." in request.text else request.text
        ],
        duplicate_info=DuplicateCheck(
            is_duplicate=False,
            matched_message_id=None,
            similarity_score=0.15,
        ),
        llm_fallback_used=False,
    )
