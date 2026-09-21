"""
SQLAlchemy ORM models defining the database storage schemas for text analysis,
document metadata, extracted entities, and evidence keyphrases.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.database import Base, engine

class AnalysisModel(Base):
    """
    Main entity table storing top-level document analysis results, 
    classification metrics, and document source metadata.
    """
    __tablename__ = "analysis"

    # Primary key and tracking identifiers
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    original_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    
    # Optional document metadata fields for uploaded file processing
    filename = Column(String, nullable=True)
    file_type = Column(String, nullable=True)
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    
    # Topic & Intent
    topic = Column(String, nullable=True)
    topic_confidence = Column(Float, nullable=True)
    topic_resolved_by = Column(String, nullable=True)
    
    intent = Column(String, nullable=True)
    intent_confidence = Column(Float, nullable=True)
    intent_resolved_by = Column(String, nullable=True)
    
    # Sentiment
    sentiment = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    
    # Flags & Timestamps
    is_duplicate = Column(Boolean, default=False)
    llm_fallback_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships to child tables
    entities = relationship("EntityModel", back_populates="analysis", cascade="all, delete-orphan")
    evidence = relationship("EvidenceModel", back_populates="analysis", cascade="all, delete-orphan")
    
class EntityModel(Base):
    """"Child table storing individual named entities linked to a primary analysis record."""
    
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis.id"), nullable=False)
    
    text = Column(String, nullable=False)
    label = Column(String, nullable=False)
    source = Column(String, nullable=True)  # e.g., 'regex', 'spacy', 'llm'
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)

    # Many-to-one relationship back to parent analysis table
    analysis = relationship("AnalysisModel", back_populates="entities")
    
class EvidenceModel(Base):
    """Stores key evidence sentences matching extracted intent/topics."""
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis.id"), nullable=False)
    
    sentence = Column(Text, nullable=False)

    analysis = relationship("AnalysisModel", back_populates="evidence")
    
