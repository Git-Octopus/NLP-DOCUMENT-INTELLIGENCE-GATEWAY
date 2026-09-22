import pytest
from pydantic import ValidationError
from app.schemas import (
    SentimentEnum,
    ExtractedEntity,
    TopicMatch,
    IntentMatch,
    SentimentAnalysis,
    DuplicateCheck,
    DocumentMetadata,
    AnalyzeRequest,
    AnalyzeResponse,
    SimilarityRequest,
    SimilarityResponse,
    ExtractResponse,
    SummarizeRequest,
    SummarizeResponse,
)


def test_schema_extracted_entity():
    entity = ExtractedEntity(
        text="$99.00",
        label="MONEY",
        source="regex",
        start_char=0,
        end_char=6,
    )
    assert entity.text == "$99.00"
    assert entity.label == "MONEY"
    assert entity.source == "regex"
    assert entity.start_char == 0
    assert entity.end_char == 6


def test_schema_topic_match():
    topic = TopicMatch(topic="Technical Support", confidence=0.85, resolved_by="qdrant")
    assert topic.topic == "Technical Support"
    assert topic.confidence == 0.85
    assert topic.resolved_by == "qdrant"

    # Out of range confidence should fail
    with pytest.raises(ValidationError):
        TopicMatch(topic="Billing", confidence=1.5, resolved_by="qdrant")


def test_schema_intent_match():
    intent = IntentMatch(
        intent="cancel_subscription", confidence=0.75, resolved_by="rules"
    )
    assert intent.intent == "cancel_subscription"
    assert intent.confidence == 0.75

    with pytest.raises(ValidationError):
        IntentMatch(intent="cancel_subscription", confidence=-0.1, resolved_by="rules")


def test_schema_sentiment_analysis():
    sentiment = SentimentAnalysis(sentiment=SentimentEnum.POSITIVE, score=0.8)
    assert sentiment.sentiment == SentimentEnum.POSITIVE
    assert sentiment.score == 0.8

    with pytest.raises(ValidationError):
        SentimentAnalysis(sentiment=SentimentEnum.POSITIVE, score=2.0)


def test_schema_duplicate_check():
    dup = DuplicateCheck(
        is_duplicate=True, matched_message_id="msg_999", similarity_score=0.95
    )
    assert dup.is_duplicate is True
    assert dup.matched_message_id == "msg_999"
    assert dup.similarity_score == 0.95


def test_schema_analyze_request_and_response():
    req = AnalyzeRequest(
        text="Account locked after failed login attempts", metadata={"channel": "chat"}
    )
    assert len(req.text) >= 3
    assert req.metadata["channel"] == "chat"

    doc_meta = DocumentMetadata(
        filename="account_issue.pdf", file_type="pdf", word_count=7, char_count=42
    )

    res = AnalyzeResponse(
        request_id="req-12345",
        original_text=req.text,
        cleaned_text="account locked after failed login attempts",
        doc_metadata=doc_meta,
        topic=TopicMatch(topic="Account Access", confidence=0.91, resolved_by="qdrant"),
        intent=IntentMatch(
            intent="unlock_account", confidence=0.87, resolved_by="rules"
        ),
        entities=[ExtractedEntity(text="locked", label="STATUS", source="regex")],
        keywords=["account", "locked", "login"],
        sentiment=SentimentAnalysis(sentiment=SentimentEnum.NEGATIVE, score=-0.5),
        evidence_sentences=["Account locked after failed login attempts"],
        duplicate_info=DuplicateCheck(is_duplicate=False),
        llm_fallback_used=False,
    )
    assert res.request_id == "req-12345"
    assert res.topic.topic == "Account Access"
    assert res.sentiment.sentiment == SentimentEnum.NEGATIVE
    assert res.doc_metadata.filename == "account_issue.pdf"
    assert res.doc_metadata.word_count == 7


def test_schema_similarity():
    sim_req = SimilarityRequest(text_a="Hello world", text_b="Hi world")
    assert sim_req.text_a == "Hello world"
    assert sim_req.text_b == "Hi world"

    sim_res = SimilarityResponse(similarity_score=0.88, is_duplicate=True)
    assert sim_res.similarity_score == 0.88
    assert sim_res.is_duplicate is True


def test_schema_extract_response():
    extract_res = ExtractResponse(
        filename="invoice.pdf",
        file_type="pdf",
        extracted_text="Order total: $100",
        word_count=3,
        char_count=17,
    )
    assert extract_res.filename == "invoice.pdf"
    assert extract_res.file_type == "pdf"
    assert extract_res.word_count == 3


def test_schema_summarize():
    sum_req = SummarizeRequest(
        text="This is a test message that has more than ten characters.", max_length=100
    )
    assert sum_req.max_length == 100

    sum_res = SummarizeResponse(
        summary="Test message summary.",
        key_points=["Point 1", "Point 2"],
        original_word_count=10,
        summary_word_count=3,
    )
    assert len(sum_res.key_points) == 2
