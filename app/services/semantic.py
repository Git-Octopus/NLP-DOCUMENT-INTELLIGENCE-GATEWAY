"""
Semantic Fallback Adapter for NLP Document Intelligence Gateway.

Provides wrapper functions for topic, intent, and duplicate matching that attempt
Dev 3's Qdrant/LLM implementation first, then fall back to rule-based approaches
when those services are unavailable. This ensures the API continues to return valid
JSON even when external services are down.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("gateway")


def load_topics_config() -> Dict[str, Any]:
    """Load topics configuration from config/topics.yaml."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "topics.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load topics.yaml: {e}. Using empty config.")
        return {}


def fallback_topic_match(text: str) -> Dict[str, Any]:
    """
    Rule-based topic matching as fallback when Qdrant is unavailable.
    Uses keyword matching aligned with taxonomy from config/topics.yaml.
    """
    text_lower = text.lower()
    topics_config = load_topics_config()

    topic_keywords = {}
    if topics_config and "topics" in topics_config:
        for topic_key, topic_data in topics_config["topics"].items():
            name = topic_data.get("name", topic_key)
            keywords = topic_data.get("keywords", [])
            if not keywords:
                keywords = [name.lower(), topic_key.lower()]
            topic_keywords[name] = [k.lower() for k in keywords]

    # fixed (Step 4): Aligned fallback topic labels with config/topics.yaml taxonomy names
    if not topic_keywords:
        topic_keywords = {
            "Payment": [
                "payment",
                "card",
                "charge",
                "checkout",
                "transaction",
                "unauthorized",
            ],
            "Billing & Refunds": [
                "refund",
                "money back",
                "reimburse",
                "overcharge",
                "double charge",
            ],
            "Invoice & Billing": [
                "invoice",
                "tax",
                "gst",
                "receipt",
                "surcharge",
                "bill",
            ],
            "Account Access": [
                "account",
                "login",
                "locked",
                "2fa",
                "suspended",
                "reactivate",
            ],
            "Password Reset": [
                "password",
                "reset",
                "forgot password",
                "pin",
                "credentials",
            ],
            "Delivery & Shipping": [
                "shipping",
                "delivery",
                "tracking",
                "package",
                "parcel",
                "shipment",
            ],
            "Cancellation": [
                "cancel",
                "cancellation",
                "stop order",
                "abort",
                "cancel booking",
            ],
            "Technical Issue": [
                "error",
                "bug",
                "crash",
                "broken",
                "not working",
                "glitch",
                "500",
            ],
            "Subscription & Plans": [
                "subscription",
                "plan",
                "upgrade",
                "downgrade",
                "renewal",
                "membership",
            ],
            "General Support": [
                "question",
                "help",
                "information",
                "support",
                "assist",
                "agent",
                "hours",
            ],
        }

    best_topic = "General Support"
    best_score = 0.0

    for topic_name, keywords in topic_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > best_score:
            best_score = matches
            best_topic = topic_name

    confidence = min(0.60, (best_score / 3.0)) if best_score > 0 else 0.50

    return {
        "topic": best_topic,
        "confidence": round(confidence, 2),
        "resolved_by": "rules",
    }


def fallback_intent_match(text: str) -> Dict[str, Any]:
    """
    Rule-based intent matching as fallback when Qdrant/LLM is unavailable.
    Uses simple keyword rules for common intents.
    """
    text_lower = text.lower()

    intent_rules = {
        "request_refund": ["refund", "money back", "charge back", "reimburse"],
        "request_password_reset": [
            "password",
            "reset",
            "change password",
            "forgot password",
        ],
        "track_order": [
            "shipping",
            "delivery",
            "tracking",
            "where is my order",
            "status",
        ],
        "cancel_order": ["cancel order", "cancel", "stop order", "cancellation"],
        "report_issue": [
            "issue",
            "problem",
            "not working",
            "broken",
            "error",
            "bug",
            "crash",
        ],
        "upgrade_plan": ["upgrade", "plan", "subscription", "tier"],
        "account_access": ["login", "locked", "access", "unblock"],
        "general_query": ["question", "help", "information", "how to", "what is"],
    }

    best_intent = "general_query"
    best_score = 0.0

    for intent, keywords in intent_rules.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > best_score:
            best_score = matches
            best_intent = intent

    confidence = min(0.60, (best_score / 2.0)) if best_score > 0 else 0.50

    return {
        "intent": best_intent,
        "confidence": round(confidence, 2),
        "resolved_by": "rules",
    }


def fallback_duplicate_check(text: str) -> Dict[str, Any]:
    """
    Fallback duplicate check when Qdrant is unavailable.
    Returns no duplicate found with zero similarity score.
    """
    return {"is_duplicate": False, "matched_message_id": None, "similarity_score": 0.0}


def safe_topic_match(text: str) -> Dict[str, Any]:
    """
    Wrapper for topic matching that tries Dev 3's implementation first,
    then falls back to rule-based keyword matching.
    """
    try:
        from app.nlp.topics import match_topic

        result = match_topic(text)
        if result and result.get("confidence", 0) >= 0.5:
            logger.debug(f"Topic matched via Qdrant/LLM: {result.get('topic')}")
            return result
    except Exception as e:
        logger.warning(
            f"Qdrant/LLM topic matching failed: {e}. Using rule-based fallback."
        )

    return fallback_topic_match(text)


def safe_intent_match(text: str) -> Dict[str, Any]:
    """
    Wrapper for intent matching that tries Dev 3's implementation first,
    then falls back to rule-based keyword matching.
    """
    try:
        from app.nlp.topics import match_intent

        result = match_intent(text)
        if result and result.get("confidence", 0) >= 0.5:
            logger.debug(f"Intent matched via Qdrant/LLM: {result.get('intent')}")
            return result
    except Exception as e:
        logger.warning(
            f"Qdrant/LLM intent matching failed: {e}. Using rule-based fallback."
        )

    return fallback_intent_match(text)


def safe_duplicate_check(text: str) -> Dict[str, Any]:
    """
    Wrapper for duplicate checking that embeds text first, calls Dev 3's Qdrant implementation,
    then falls back safely if Qdrant is unavailable.
    """
    # fixed (Step 4): Call get_embedding first, passing dense vector into search_duplicate_message
    try:
        from app.nlp.embeddings import get_embedding
        from app.nlp.vector_store import search_duplicate_message

        vector = get_embedding(text)
        if any(vector):
            result = search_duplicate_message(vector)
            if result:
                logger.debug(
                    f"Duplicate check via Qdrant: is_duplicate={result.get('is_duplicate')}"
                )
                return result
    except Exception as e:
        # fixed (Step 4): Honest logging of Qdrant unavailability
        logger.warning(f"semantic fallback: qdrant_unavailable ({e})")

    return fallback_duplicate_check(text)


def safe_sentiment_match(text: str) -> Dict[str, Any]:
    """
    Computes sentiment polarity using local NLP (VADER + domain lexicon) first.
    If local scoring is ambiguous/borderline (e.g. score == 0.0 or abs(score) < 0.15)
    or lacks decisive sentiment signals, falls back to active LLM provider (Ollama / Gemini).
    """
    from app.nlp.sentiment import analyze_sentiment_local

    local_res = analyze_sentiment_local(text)
    local_score = float(local_res.get("score", 0.0))
    local_sent = local_res.get("sentiment", "neutral")

    # If local sentiment has clear confidence, resolve immediately via local NLP
    if abs(local_score) >= 0.15 and local_sent != "neutral":
        return {
            "sentiment": local_sent,
            "score": local_score,
            "resolved_by": "local_nlp",
        }

    # Ambiguous, borderline, or neutral: trigger LLM fallback
    try:
        from app.providers import get_llm_provider

        provider = get_llm_provider()
        if provider and provider.is_available():
            llm_res = provider.analyze_sentiment(text)
            if llm_res and "sentiment" in llm_res and "score" in llm_res:
                return {
                    "sentiment": str(llm_res["sentiment"]).lower(),
                    "score": round(float(llm_res["score"]), 2),
                    "resolved_by": "llm_fallback",
                }
    except Exception as e:
        logger.warning(f"LLM sentiment fallback failed: {e}. Keeping local sentiment.")

    return {
        "sentiment": local_sent,
        "score": local_score,
        "resolved_by": "local_nlp",
    }
