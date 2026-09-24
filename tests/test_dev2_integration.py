"""
Comprehensive Integration Tests for Dev 2: Core NLP Pipeline & Traditional ML.

Verifies:
1. Complete customer support analysis with realistic multi-entity queries.
2. Rule-based entity extraction and spaCy NER merging with priority deduplication.
3. TF-IDF and noun chunk candidate keyword extraction.
4. Linguistic feature parsing (lemmas, POS tags, dependencies).
5. Dynamic local sentiment analysis.
6. Multi-modal Gateway API interoperability (Form POST, JSON POST).
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.nlp.pipeline import (
    CoreNLPService,
    get_core_nlp_service,
    run_local_nlp_pipeline,
)
from app.services.database import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def prepare_test_db():
    """Create database tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_dev2_realistic_customer_support_scenario():
    """
    Section 45 Integration Test:
    Realistic customer-support message containing person, money, order ID,
    date, email, phone, and domain intent keywords.
    """
    text = (
        "Hello, my name is Rahul Sharma. I paid ₹1,500 for order ORD-2026-1042 "
        "on 12 September 2026, but the payment was deducted and my order is "
        "still showing unpaid. Please contact me at rahul@example.com or "
        "+91 9876543210. I also requested a refund but have not received it."
    )

    nlp_service = get_core_nlp_service()
    result = nlp_service.analyze(text)

    # 1. Linguistic & Preprocessing validation
    assert len(result["tokens"]) > 10
    assert len(result["sentences"]) >= 3
    assert "refund" in result["lemmas"] or "payment" in result["lemmas"]

    # 2. Entity extraction validation across rules + spaCy
    entities = result["entities"]
    entity_labels = {e["label"] for e in entities}
    entity_texts = {e["text"] for e in entities}

    # Verify high-confidence deterministic entities
    assert "EMAIL" in entity_labels
    assert "rahul@example.com" in entity_texts

    assert "MONEY" in entity_labels
    assert any("1,500" in t or "1500" in t for t in entity_texts)

    assert "ORDER_ID" in entity_labels
    assert any("ORD-2026-1042" in t for t in entity_texts)

    assert "PHONE" in entity_labels
    assert any("9876543210" in t for t in entity_texts)

    assert "DATE" in entity_labels
    assert any("September" in t or "2026" in t for t in entity_texts)

    # Verify spaCy NER or person recognition
    assert "PERSON" in entity_labels or any("Rahul" in t for t in entity_texts)

    # 3. Keyword extraction validation
    kw_strings = [k["keyword"].lower() for k in result["keywords"]]
    assert any(
        target in kw_strings
        for target in ["refund", "payment", "order", "unpaid", "rahul sharma"]
    )

    # 4. Sentiment validation (customer query regarding deducted money & unpaid order is negative)
    assert result["sentiment"]["sentiment"] in ["negative", "neutral"]
    assert -1.0 <= result["sentiment"]["score"] <= 1.0

    # 5. Evidence sentence validation
    assert len(result["evidence_sentences"]) > 0
    assert any(
        "payment" in s.lower() or "refund" in s.lower()
        for s in result["evidence_sentences"]
    )


def test_core_nlp_service_standalone_methods():
    """Validates individual methods exposed by CoreNLPService."""
    service = CoreNLPService()
    text = "Please cancel my annual subscription and issue a full refund."

    # Preprocessing
    prep = service.preprocess(text)
    assert "cleaned_text" in prep
    assert "original_text" in prep
    assert "tokens" in prep
    assert "stems" in prep

    # Linguistic
    ling = service.linguistic_analysis(text)
    assert len(ling["lemmas"]) > 0
    assert len(ling["pos_tags"]) > 0
    assert len(ling["noun_chunks"]) > 0
    assert "tokens_info" in ling

    # N-grams
    tokens = ["credit", "card", "declined", "error"]
    ng = service.all_ngrams(tokens, max_n=3)
    assert "unigrams" in ng
    assert "bigrams" in ng
    assert "trigrams" in ng
    assert "credit card" in ng["bigrams"]
    assert "credit card declined" in ng["trigrams"]

    # BoW
    bow_res = service.bow([text, "Another ticket about billing refund"])
    assert "vocabulary" in bow_res
    assert "counts" in bow_res
    assert len(bow_res["counts"]) == len(bow_res["vocabulary"])

    # TF-IDF
    tfidf_res = service.tfidf([text, "Different issue regarding login"])
    assert "ranked_terms" in tfidf_res
    assert len(tfidf_res["ranked_terms"]) > 0
    # First term should have non-negative score
    assert tfidf_res["ranked_terms"][0]["score"] >= 0.0


def test_analyze_endpoint_json_and_form_parity():
    """Verifies POST /analyze accepts both JSON payloads and Form payloads."""
    text_content = (
        "My payment of ₹2,000 for invoice INV-2026-9012 failed on 2026-05-10. "
        "Contact me at support@client.in."
    )

    # 1. Test via Form data (Content-Type: multipart/form-data or application/x-www-form-urlencoded)
    res_form = client.post("/analyze", data={"text": text_content})
    assert res_form.status_code == 200
    data_form = res_form.json()
    assert data_form["original_text"] == text_content
    assert len(data_form["entities"]) > 0
    assert data_form["sentiment"]["sentiment"] in ["positive", "neutral", "negative"]

    # 2. Test via JSON payload (Content-Type: application/json)
    res_json = client.post(
        "/analyze",
        json={"text": text_content, "metadata": {"client": "test"}},
    )
    assert res_json.status_code == 200
    data_json = res_json.json()
    assert data_json["original_text"] == text_content
    assert len(data_json["entities"]) == len(data_form["entities"])
    assert data_json["topic"]["topic"] is not None


def test_long_and_unicode_input_handling():
    """Validates robust handling of very long inputs and Unicode normalization."""
    # Unicode text
    unicode_text = "Héllo, café réfund for €50 please contact user@café.fr."
    res = run_local_nlp_pipeline(unicode_text)
    assert res["cleaned_text"] != ""
    assert len(res["entities"]) > 0

    # Pathological long text (exceeding default limit should not crash)
    long_text = "Billing issue with order ORD-1234. " * 2000
    res_long = run_local_nlp_pipeline(long_text)
    assert "cleaned_text" in res_long
    assert len(res_long["entities"]) > 0
