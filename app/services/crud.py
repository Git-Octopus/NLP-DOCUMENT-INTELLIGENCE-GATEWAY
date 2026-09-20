import os
import sys
from typing import List, Optional
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.models import AnalysisModel, EntityModel, EvidenceModel
from app.schemas import AnalyzeResponse

def create_analysis(db: Session, analysis_data: AnalyzeResponse) -> AnalysisModel:
    """Saves a complete AnalyzeResponse object into SQLite tables."""
    
    # Extract string value for sentiment cleanly
    sentiment_val = None
    if analysis_data.sentiment and analysis_data.sentiment.sentiment:
        sentiment_val = (
            analysis_data.sentiment.sentiment.value 
            if hasattr(analysis_data.sentiment.sentiment, 'value') 
            else str(analysis_data.sentiment.sentiment)
        )
        
    # Build main Analysis record
    db_analysis = AnalysisModel(
        request_id=analysis_data.request_id,
        original_text=analysis_data.original_text,
        cleaned_text=analysis_data.cleaned_text,
        topic=analysis_data.topic.topic if analysis_data.topic else None,
        topic_confidence=analysis_data.topic.confidence if analysis_data.topic else None,
        topic_resolved_by=analysis_data.topic.resolved_by if analysis_data.topic else None,
        intent=analysis_data.intent.intent if analysis_data.intent else None,  # Fixed .label to .intent
        intent_confidence=analysis_data.intent.confidence if analysis_data.intent else None,
        intent_resolved_by=analysis_data.intent.resolved_by if analysis_data.intent else None,
        sentiment=sentiment_val,
        sentiment_score=analysis_data.sentiment.score if analysis_data.sentiment else None,
        is_duplicate=analysis_data.duplicate_info.is_duplicate if analysis_data.duplicate_info else False,
        llm_fallback_used=analysis_data.llm_fallback_used,
    )
    
    db.add(db_analysis)
    db.flush()  # Generates db_analysis.id for foreign keys

    # Add associated entities
    for ent in analysis_data.entities:
        db_entity = EntityModel(
            analysis_id=db_analysis.id,
            text=ent.text,
            label=ent.label,
            source=ent.source,
            start_char=ent.start_char,
            end_char=ent.end_char,
        )
        db.add(db_entity)

    # Add associated evidence sentences
    for ev in analysis_data.evidence_sentences:
        db_evidence = EvidenceModel(
            analysis_id=db_analysis.id,
            sentence=ev,
        )
        db.add(db_evidence)

    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_analysis_by_request_id(db: Session, request_id: str) -> Optional[AnalysisModel]:
    """Retrieves an analysis record by its unique request ID."""
    return db.query(AnalysisModel).filter(AnalysisModel.request_id == request_id).first()


def list_analyses(db: Session, skip: int = 0, limit: int = 100) -> List[AnalysisModel]:
    """Fetches a paginated list of analysis records."""
    return db.query(AnalysisModel).offset(skip).limit(limit).all()


