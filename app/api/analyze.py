"""
Orchestrator Analysis API endpoint for NLP Document Intelligence Gateway.
Integrates local NLP pipeline (Dev 2) with Qdrant vector retrieval and LLM fallback (Dev 3).
"""

import uuid
import logging
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
from app.nlp.pipeline import run_local_nlp_pipeline
from app.nlp.embeddings import embed_text
from app.nlp.vector_store import search_duplicate_message, record_analyzed_message
from app.nlp.topics import match_topic, match_intent
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Customer Support Text",
    description=(
        "Executes end-to-end NLP document intelligence: text normalization, linguistic analysis, "
        "topic classification via Qdrant, intent detection, NER, sentiment analysis, duplicate check, "
        "and evidence sentence extraction with LLM fallback."
    ),
)
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Primary gateway orchestrator endpoint combining local NLP and dense vector matching.
    """
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    # Step 1: Run Dev 2 local NLP pipeline (cleaning, tokenization, POS, TF-IDF, NER, sentiment)
    nlp_results = run_local_nlp_pipeline(request.text)
    cleaned_text = nlp_results.get("cleaned_text", request.text.lower().strip())

    # Step 2: Dense vector generation for duplicate detection & topic matching
    text_vector = embed_text(request.text)

    # Step 3: Duplicate detection against previously analyzed messages in Qdrant
    dup_res = search_duplicate_message(
        text_vector, threshold=getattr(settings, "DUPLICATE_SIMILARITY_THRESHOLD", 0.85)
    )
    duplicate_info = DuplicateCheck(
        is_duplicate=dup_res.get("is_duplicate", False),
        matched_message_id=dup_res.get("matched_message_id"),
        similarity_score=dup_res.get("similarity_score", 0.0),
    )

    # Record message embedding for future duplicate detection
    record_analyzed_message(req_id, request.text, text_vector)

    # Step 4: Topic & Intent classification (Qdrant semantic search + rules + LLM fallback)
    topic_res = match_topic(request.text)
    intent_res = match_intent(request.text)

    topic_match = TopicMatch(
        topic=topic_res["topic"],
        confidence=topic_res["confidence"],
        resolved_by=topic_res["resolved_by"],
    )

    intent_match = IntentMatch(
        intent=intent_res["intent"],
        confidence=intent_res["confidence"],
        resolved_by=intent_res["resolved_by"],
    )

    llm_fallback_used = (
        topic_res.get("resolved_by") == "llm_fallback"
        or intent_res.get("resolved_by") == "llm_fallback"
    )

    # Step 5: Format Extracted Entities
    formatted_entities = []
    raw_entities = nlp_results.get("entities", [])
    for ent in raw_entities:
        ent_text = ent.get("text") or ent.get("entity") or ""
        label = ent.get("label") or "MISC"
        source = ent.get("source") or "regex"
        start_char = ent.get("start_char") if ent.get("start_char") is not None else ent.get("start")
        end_char = ent.get("end_char") if ent.get("end_char") is not None else ent.get("end")

        formatted_entities.append(
            ExtractedEntity(
                text=ent_text,
                label=label,
                source=source,
                start_char=start_char,
                end_char=end_char,
            )
        )

    # Step 6: Extract Keywords
    raw_keywords = nlp_results.get("keywords", [])
    keyword_list = []
    for kw in raw_keywords:
        if isinstance(kw, dict):
            keyword_list.append(kw.get("keyword", ""))
        elif isinstance(kw, str):
            keyword_list.append(kw)
    keyword_list = [k for k in keyword_list if k]

    # Step 7: Local Sentiment Classification
    raw_sentiment = nlp_results.get("sentiment", {"sentiment": "neutral", "score": 0.0})
    sentiment_label = raw_sentiment.get("sentiment", "neutral").lower()
    if sentiment_label not in [e.value for e in SentimentEnum]:
        sentiment_label = "neutral"

    sentiment_analysis = SentimentAnalysis(
        sentiment=SentimentEnum(sentiment_label),
        score=float(raw_sentiment.get("score", 0.0)),
    )

    # Step 8: Evidence Sentences Selection
    # Prefer sentences containing entities or strong intent keywords
    all_sentences = nlp_results.get("sentences", [])
    if not all_sentences and "." in request.text:
        all_sentences = [s.strip() for s in request.text.split(".") if s.strip()]

    evidence = []
    lower_intent = intent_res["intent"].replace("_", " ")
    for sent in all_sentences:
        lower_sent = sent.lower()
        if any(w in lower_sent for w in lower_intent.split()):
            evidence.append(sent)
        elif any(ent.text.lower() in lower_sent for ent in formatted_entities):
            evidence.append(sent)

    if not evidence and all_sentences:
        evidence = all_sentences[:2]
    elif not evidence:
        evidence = [request.text]

    return AnalyzeResponse(
        request_id=req_id,
        original_text=request.text,
        cleaned_text=cleaned_text,
        topic=topic_match,
        intent=intent_match,
        entities=formatted_entities,
        keywords=keyword_list,
        sentiment=sentiment_analysis,
        evidence_sentences=evidence,
        duplicate_info=duplicate_info,
        llm_fallback_used=llm_fallback_used,
    )
