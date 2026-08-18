from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class ExtractedEntity(BaseModel):
    text: str = Field(..., description="The entity text, e.g. '$50.00' or 'john@example.com'")
    label: str = Field(..., description="Entity category: MONEY, EMAIL, PHONE, DATE, PERSON, ORG, etc.")
    source: str = Field(..., description="'regex', 'spacy', or 'llm_fallback'")
    start_char: Optional[int] = Field(None, description="Start character index in text")
    end_char: Optional[int] = Field(None, description="End character index in text")

class TopicMatch(BaseModel):
    topic: str = Field(..., description="Identified topic from topics.yaml, e.g. 'Billing & Refunds'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Similarity score from Qdrant")
    resolved_by: str = Field(..., description="'qdrant' or 'llm_fallback'")

class IntentMatch(BaseModel):
    intent: str = Field(..., description="Detected user intent, e.g. 'request_refund'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    resolved_by: str = Field(..., description="'rules', 'qdrant', or 'llm_fallback'")

class SentimentAnalysis(BaseModel):
    sentiment: SentimentEnum
    score: float = Field(..., ge=-1.0, le=1.0, description="Polarity score from -1.0 to +1.0")

class DuplicateCheck(BaseModel):
    is_duplicate: bool = False
    matched_message_id: Optional[str] = None
    similarity_score: Optional[float] = None

# --- REQUEST / RESPONSE FOR /analyze ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Unstructured customer support text to analyze")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata (user_id, ticket_id, channel)")

class AnalyzeResponse(BaseModel):
    request_id: Optional[str] = None
    original_text: str
    cleaned_text: str
    topic: TopicMatch
    intent: IntentMatch
    entities: List[ExtractedEntity]
    keywords: List[str]
    sentiment: SentimentAnalysis
    evidence_sentences: List[str] = Field(..., description="Key sentences justifying the intent/topic")
    duplicate_info: DuplicateCheck
    llm_fallback_used: bool = Field(..., description="True if LLM was invoked for topic/entity ambiguity")