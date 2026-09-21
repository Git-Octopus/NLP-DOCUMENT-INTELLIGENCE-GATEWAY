"""
API endpoint handling direct text and document-stream analysis requests.
Extracts linguistic features, matches topic/intent, parses document metadata,
and persists analysis outputs to relational storage.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, status, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.config import settings
from app.schemas.analysis import (
    AnalyzeResponse,
    DocumentMetadata,
    TopicMatch,
    IntentMatch,
    ExtractedEntity,
    SentimentAnalysis,
    SentimentEnum,
    DuplicateCheck,
)
from app.services.database import get_db
from app.services.crud import create_analysis
from app.services.extractors import parse_document_stream

from app.nlp.preprocess import clean_text
from app.nlp.linguistic import analyze_linguistics
from app.nlp.ner import extract_all_entities
from app.nlp.tfidf import extract_keywords
from app.nlp.rules import extract_evidence_sentences
from app.nlp.topics import match_topic, match_intent
from app.nlp.vector_store import search_duplicate_message

router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Customer Support Text or Document Stream",
    description="Accepts direct text or uploaded document files (PDF/DOCX) to extract structured NLP analytics.",
)
async def analyze_text_or_document(
    text: Optional[str] = Form(None, description="Direct text input to analyze"),
    file: Optional[UploadFile] = File(None, description="Optional document file (.pdf, .docx)"),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """
    Parses incoming text or extracts document streams, runs full NLP feature processing,
    formats analytical responses, and persists outputs to the relational database.
    """
    doc_meta: Optional[DocumentMetadata] = None
    input_text: str = ""

    # Multi-Modal Stream Handling: File Extraction vs Direct Text
    if file:
        file_bytes = await file.read()

        if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File size exceeds maximum allowed limit of "
                    f"{settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )

        try:
            extracted_data = parse_document_stream(file_bytes, file.filename)
            input_text = extracted_data["extracted_text"]
            doc_meta = DocumentMetadata(
                filename=extracted_data["filename"],
                file_type=extracted_data["file_type"],
                word_count=extracted_data["word_count"],
                char_count=extracted_data["char_count"],
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract document stream: {str(err)}",
            )
    elif text:
        input_text = text.strip()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' string or 'file' document upload must be provided.",
        )

    # Input Validation Guard
    if len(input_text.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text content must contain at least 3 non-whitespace characters.",
        )

    # Preprocessing and Entity Extraction
    raw_text = input_text.strip()
    cleaned = clean_text(raw_text)

    raw_entities = extract_all_entities(raw_text)
    formatted_entities = [
        ExtractedEntity(
            text=e.get("text") or e.get("entity", ""),
            label=e.get("label", "MISC"),
            source=e.get("source", "spacy"),
            start_char=e.get("start_char", e.get("start", 0)),
            end_char=e.get("end_char", e.get("end", 0)),
        )
        for e in raw_entities
    ]

    # Keyword & Evidence Extraction
    kw_dicts = extract_keywords(cleaned, top_n=5)
    keyword_strings = [k["keyword"] for k in kw_dicts] if kw_dicts else []

    evidence = extract_evidence_sentences(raw_text, keywords=keyword_strings, max_sentences=3)
    if not evidence:
        evidence = [raw_text]

    # Vector Store & Intent Resolution
    try:
        topic_data = match_topic(raw_text) or {}
    except Exception:
        topic_data = {}

    try:
        intent_data = match_intent(raw_text) or {}
    except Exception:
        intent_data = {}

    try:
        dup_data = search_duplicate_message(raw_text) or {}
    except Exception:
        dup_data = {}

    # Response Assembly
    response = AnalyzeResponse(
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        original_text=raw_text,
        cleaned_text=cleaned,
        doc_metadata=doc_meta,
        topic=TopicMatch(
            topic=topic_data.get("topic", "General Support"),
            confidence=topic_data.get("confidence", 0.50),
            resolved_by=topic_data.get("resolved_by", "rules"),
        ),
        intent=IntentMatch(
            intent=intent_data.get("intent", "general_query"),
            confidence=intent_data.get("confidence", 0.50),
            resolved_by=intent_data.get("resolved_by", "rules"),
        ),
        entities=formatted_entities,
        keywords=keyword_strings,
        sentiment=SentimentAnalysis(
            sentiment=SentimentEnum.NEUTRAL,
            score=0.0,
        ),
        evidence_sentences=evidence,
        duplicate_info=DuplicateCheck(
            is_duplicate=dup_data.get("is_duplicate", False),
            matched_message_id=dup_data.get("matched_message_id"),
            similarity_score=dup_data.get("similarity_score", 0.0),
        ),
        llm_fallback_used=topic_data.get("resolved_by") == "llm_fallback"
        or intent_data.get("resolved_by") == "llm_fallback",
    )

    # Relational Database Persistence
    create_analysis(db, response)

    return response