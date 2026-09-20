from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum


class SentimentEnum(str, Enum):
    """Enumeration of possible text sentiment classifications."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ExtractedEntity(BaseModel):
    text: str = Field(
        ..., description="The entity text, e.g. '$50.00' or 'john@example.com'"
    )
    label: str = Field(
        ...,
        description="Entity category: MONEY, EMAIL, PHONE, DATE, PERSON, ORG, ORDER_ID, etc.",
    )
    source: str = Field(
        ..., description="Source of extraction: 'regex', 'spacy', or 'llm_fallback'"
    )
    start_char: Optional[int] = Field(None, description="Start character index in text")
    end_char: Optional[int] = Field(None, description="End character index in text")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "$49.99",
                "label": "MONEY",
                "source": "regex",
                "start_char": 28,
                "end_char": 34,
            }
        }
    )


class TopicMatch(BaseModel):
    """Schema representing the classification topic match and confidence score."""
    topic: str = Field(..., description="Identified topic from topics.yaml, e.g. 'Billing & Refunds'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Similarity score from Qdrant vector search")
    resolved_by: str = Field(..., description="Resolution method: 'qdrant' or 'llm_fallback'")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "Billing & Refunds",
                "confidence": 0.92,
                "resolved_by": "qdrant",
            }
        }
    )


class IntentMatch(BaseModel):
    """Schema representing the detected user intent and resolution method."""
    intent: str = Field(..., description="Detected user intent, e.g. 'request_refund'")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    resolved_by: str = Field(
        ..., description="Resolution method: 'rules', 'qdrant', or 'llm_fallback'"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "request_refund",
                "confidence": 0.88,
                "resolved_by": "rules",
            }
        }
    )


class SentimentAnalysis(BaseModel):
    """Schema representing the overall text sentiment and numerical polarity score."""
    sentiment: SentimentEnum = Field(..., description="Overall sentiment classification")
    score: float = Field(..., ge=-1.0, le=1.0, description="Polarity score ranging from -1.0 (negative) to +1.0 (positive)")

    model_config = ConfigDict(
        json_schema_extra={"example": {"sentiment": "negative", "score": -0.65}}
    )


class DuplicateCheck(BaseModel):
    """Schema containing result metrics for message deduplication checks."""
    is_duplicate: bool = Field(False, description="Whether this message duplicates an earlier indexed message")
    matched_message_id: Optional[str] = Field(None, description="ID of the matching message in Qdrant if duplicate")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score to matched message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_duplicate": False,
                "matched_message_id": None,
                "similarity_score": 0.12,
            }
        }
    )


class DocumentMetadata(BaseModel):
    """Schema containing structural and file metadata extracted from uploaded documents."""
    filename: Optional[str] = Field(None, description="Uploaded document filename")
    file_type: Optional[str] = Field(None, description="File type extension, e.g. 'pdf' or 'docx'")
    word_count: Optional[int] = Field(None, description="Total word count in document")
    char_count: Optional[int] = Field(None, description="Total character count in document")
    

# --- REQUEST / RESPONSE FOR /analyze ---


class AnalyzeRequest(BaseModel):
    """Incoming request payload schema for raw support text analysis."""
    text: str = Field(..., min_length=3, description="Unstructured customer support text to analyze")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata (user_id, ticket_id, channel)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hello, I was charged $49.99 twice on my invoice #INV-9821. Please issue a refund to john@example.com immediately.",
                "metadata": {"user_id": "usr_12345", "channel": "email"},
            }
        }
    )


class AnalyzeResponse(BaseModel):
    """Outbound response payload schema containing complete analytical output."""
    request_id: Optional[str] = Field(None, description="Unique ID for this analysis request")
    original_text: str = Field(..., description="Original customer input text")
    cleaned_text: str = Field(..., description="Preprocessed and normalized text")
    doc_metadata: Optional[DocumentMetadata] = Field(None, description="Optional document metadata if extracted from file")
    topic: TopicMatch = Field(..., description="Topic categorization result")
    intent: IntentMatch = Field(..., description="Intent detection result")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    keywords: List[str] = Field(default_factory=list, description="Top keywords / key phrases extracted via TF-IDF")
    sentiment: SentimentAnalysis = Field(..., description="Sentiment analysis output")
    evidence_sentences: List[str] = Field(default_factory=list, description="Key sentences justifying intent/topic")
    duplicate_info: DuplicateCheck = Field(..., description="Duplicate message detection result")
    llm_fallback_used: bool = Field(False, description="True if LLM was invoked for fallback")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req-dummy-123456",
                "original_text": "Hello, I was charged $49.99 twice on my invoice #INV-9821. Please issue a refund to john@example.com immediately.",
                "cleaned_text": "hello i was charged $49.99 twice on my invoice #inv-9821 please issue a refund to john@example.com immediately",
                "doc_metadata": {
                    "filename": "ticket_doc.pdf",
                    "file_type": "pdf",
                    "word_count": 18,
                    "char_count": 115
                },
                "topic": {
                    "topic": "Billing & Refunds",
                    "confidence": 0.94,
                    "resolved_by": "qdrant",
                },
                "intent": {
                    "intent": "request_refund",
                    "confidence": 0.89,
                    "resolved_by": "rules",
                },
                "entities": [
                    {
                        "text": "$49.99",
                        "label": "MONEY",
                        "source": "regex",
                        "start_char": 21,
                        "end_char": 27,
                    },
                    {
                        "text": "#INV-9821",
                        "label": "ORDER_ID",
                        "source": "regex",
                        "start_char": 48,
                        "end_char": 57,
                    },
                    {
                        "text": "john@example.com",
                        "label": "EMAIL",
                        "source": "regex",
                        "start_char": 84,
                        "end_char": 100,
                    },
                ],
                "keywords": ["charged", "invoice", "refund", "billing"],
                "sentiment": {"sentiment": "negative", "score": -0.65},
                "evidence_sentences": [
                    "I was charged $49.99 twice on my invoice #INV-9821.",
                    "Please issue a refund to john@example.com immediately.",
                ],
                "duplicate_info": {
                    "is_duplicate": False,
                    "matched_message_id": None,
                    "similarity_score": 0.15,
                },
                "llm_fallback_used": False,
            }
        }
    )
