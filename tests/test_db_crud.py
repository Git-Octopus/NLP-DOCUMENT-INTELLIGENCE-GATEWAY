import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.database import Base
from app.services.crud import create_analysis, get_analysis_by_request_id, list_analyses
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

@pytest.fixture(scope="function")
def db_session():
    """
    Creates an in-memory SQLite database session for testing.
    Ensures isolated state per test function.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create all tables in memory
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        
def test_create_and_retrieve_analysis(db_session):
    """Verify that a full AnalyzeResponse object is correctly saved and retrieved from SQLite."""
    dummy_response = AnalyzeResponse(
        request_id="test-req-001",
        original_text="I was charged $49.99 twice. Please refund john@example.com.",
        cleaned_text="i was charged $49.99 twice please refund john@example.com",
        topic=TopicMatch(
            topic="Billing & Refunds",
            confidence=0.95,
            resolved_by="qdrant",
        ),
        intent=IntentMatch(
            intent="request_refund",
            confidence=0.90,
            resolved_by="rules",
        ),
        entities=[
            ExtractedEntity(
                text="$49.99",
                label="MONEY",
                source="regex",
                start_char=14,
                end_char=20,
            ),
            ExtractedEntity(
                text="john@example.com",
                label="EMAIL",
                source="regex",
                start_char=42,
                end_char=58,
            ),
        ],
        keywords=["charged", "refund", "billing"],
        sentiment=SentimentAnalysis(
            sentiment=SentimentEnum.NEGATIVE,
            score=-0.7,
        ),
        evidence_sentences=["I was charged $49.99 twice."],
        duplicate_info=DuplicateCheck(
            is_duplicate=False,
            similarity_score=0.10,
        ),
        llm_fallback_used=False,
    )
    
    # 1. Insert record
    saved_record = create_analysis(db_session, dummy_response)
    assert saved_record.id is not None
    assert saved_record.request_id == "test-req-001"
    assert saved_record.topic == "Billing & Refunds"
    assert saved_record.intent == "request_refund"
    assert saved_record.sentiment == "negative"

    # 2. Verify child relationships (entities and evidence)
    assert len(saved_record.entities) == 2
    assert saved_record.entities[0].text == "$49.99"
    assert saved_record.entities[0].label == "MONEY"
    assert saved_record.entities[1].text == "john@example.com"
    assert saved_record.entities[1].label == "EMAIL"                        
    assert len(saved_record.evidence) == 1
    assert saved_record.evidence[0].sentence == "I was charged $49.99 twice."

    # 3. Retrieve by request ID
    retrieved = get_analysis_by_request_id(db_session, "test-req-001")
    assert retrieved is not None
    assert retrieved.request_id == "test-req-001"
    assert retrieved.original_text == dummy_response.original_text

    # 4. Paginated list check
    all_records = list_analyses(db_session)
    assert len(all_records) == 1
    assert all_records[0].request_id == "test-req-001"

def test_create_analysis_with_doc_metadata(db_session):
    """Verify that file metadata (filename, file_type, word_count, char_count) is persisted."""
    doc_response = AnalyzeResponse(
        request_id="test-doc-002",
        original_text="Product issue reported in document.",
        cleaned_text="product issue reported in document",
        doc_metadata=DocumentMetadata(
            filename="issue_report.docx",
            file_type="docx",
            word_count=5,
            char_count=35,
        ),
        topic=TopicMatch(
            topic="Technical Issue",
            confidence=0.88,
            resolved_by="rules",
        ),
        intent=IntentMatch(
            intent="report_bug",
            confidence=0.85,
            resolved_by="rules",
        ),
        entities=[],
        keywords=["issue", "product"],
        sentiment=SentimentAnalysis(
            sentiment=SentimentEnum.NEUTRAL,
            score=0.0,
        ),
        evidence_sentences=["Product issue reported in document."],
        duplicate_info=DuplicateCheck(
            is_duplicate=False,
            similarity_score=0.0,
        ),
        llm_fallback_used=False,
    )
    saved = create_analysis(db_session, doc_response)
    assert saved.id is not None
    assert saved.filename == "issue_report.docx"
    assert saved.file_type == "docx"
    assert saved.word_count == 5
    assert saved.char_count == 35
    retrieved = get_analysis_by_request_id(db_session, "test-doc-002")
    assert retrieved is not None
    assert retrieved.filename == "issue_report.docx"
    assert retrieved.file_type == "docx"
    assert retrieved.word_count == 5
    assert retrieved.char_count == 35