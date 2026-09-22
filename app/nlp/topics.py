"""
Topic & Intent Classification module for NLP Document Intelligence Gateway.

Coordinates Qdrant vector similarity search against the controlled topic taxonomy (config/topics.yaml)
with rule-based intent matching and external LLM fallback for ambiguous cases.
"""

import os
import uuid
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import settings
from app.nlp.embeddings import embed_text, embed_texts
from app.nlp.vector_store import (
    search_similar_topics,
    upsert_topic_vectors,
)
from app.providers import get_llm_provider

logger = logging.getLogger(__name__)

_TOPICS_CACHE: Optional[Dict[str, Any]] = None

# Rule-based intent keyword mapping for high-precision fast resolution
INTENT_RULE_PATTERNS = [
    (r"\b(refund|money back|charge back|reimburse|reversal)\b", "request_refund", 0.92),
    (
        r"\b(reset|forgot|change)\s+(my\s+)?(password|passcode|pin)\b",
        "reset_password",
        0.94,
    ),
    (
        r"\b(cancel|terminate|stop)\s+(my\s+)?(subscription|membership|plan|account)\b",
        "cancel_subscription",
        0.91,
    ),
    (
        r"\b(where\s+is|track|status\s+of)\s+(my\s+)?(order|shipment|delivery|package)\b",
        "track_order",
        0.93,
    ),
    (
        r"\b(return|exchange)\s+(the\s+)?(item|product|goods|order)\b",
        "return_item",
        0.90,
    ),
    (
        r"\b(invoice|receipt|tax invoice|billing statement|bill copy)\b",
        "request_invoice",
        0.88,
    ),
    (
        r"\b(update|change)\s+(my\s+)?(email|phone|address|profile|name)\b",
        "update_account_details",
        0.89,
    ),
    (
        r"\b(damaged|broken|defective|faulty|cracked|ruined)\b",
        "report_damaged_item",
        0.91,
    ),
    (
        r"\b(charged twice|duplicate charge|double billed|overcharged)\b",
        "dispute_duplicate_charge",
        0.95,
    ),
    (
        r"\b(error\s+\d+|crash|bug|not loading|blank screen|freeze)\b",
        "report_technical_bug",
        0.89,
    ),
]


def load_topics_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads and caches the topic taxonomy from config/topics.yaml.
    """
    global _TOPICS_CACHE
    if _TOPICS_CACHE is not None and config_path is None:
        return _TOPICS_CACHE

    target_path = config_path or settings.TOPICS_CONFIG_PATH
    if not os.path.isabs(target_path):
        candidate = Path(__file__).resolve().parent.parent.parent / target_path
        if candidate.exists():
            target_path = str(candidate)
        else:
            candidate = Path(target_path).resolve()
            target_path = str(candidate)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if config_path is None:
                _TOPICS_CACHE = data
            return data
    except Exception as e:
        logger.error(f"Failed to load topics config from {target_path}: {e}")
        return {"topics": {}}


def get_available_topics() -> List[Dict[str, Any]]:
    """
    Returns a flattened list of all available topics from topics.yaml.
    """
    cfg = load_topics_config()
    topics_dict = cfg.get("topics", {})
    results = []
    for topic_id, details in topics_dict.items():
        results.append(
            {
                "id": details.get("id", topic_id),
                "name": details.get("name", topic_id.title()),
                "description": details.get("description", ""),
                "intents": details.get("intents", []),
                "examples": details.get("examples", []),
            }
        )
    return results


def seed_topics_to_vector_store() -> int:
    """
    Indexes topic descriptions and anchor example sentences into Qdrant.
    Returns the total number of points indexed.
    """
    taxonomy = get_available_topics()
    if not taxonomy:
        logger.warning("No topics found in topics taxonomy to seed.")
        return 0

    texts_to_embed = []
    points_meta = []

    for item in taxonomy:
        topic_id = item["id"]
        topic_name = item["name"]

        # Anchor 1: Topic Description
        if item.get("description"):
            texts_to_embed.append(item["description"])
            points_meta.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "type": "description",
                    "text": item["description"],
                }
            )

        # Anchors 2+: Example Utterances
        for example in item.get("examples", []):
            if example.strip():
                texts_to_embed.append(example.strip())
                points_meta.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "type": "example",
                        "text": example.strip(),
                    }
                )

    if not texts_to_embed:
        return 0

    embeddings = embed_texts(texts_to_embed)
    points_data = []

    for idx, (meta, vector) in enumerate(zip(points_meta, embeddings)):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS, f"{meta['topic_id']}_{idx}_{meta['text'][:30]}"
            )
        )
        points_data.append({"id": point_id, "vector": vector, "payload": meta})

    upsert_topic_vectors(points_data)
    logger.info(f"Successfully seeded {len(points_data)} topic vectors into Qdrant.")
    return len(points_data)


_topics_seeded: bool = False


def ensure_topics_seeded() -> int:
    """
    Ensures topic taxonomy vectors are indexed in Qdrant before search queries.
    """
    global _topics_seeded
    if not _topics_seeded:
        count = seed_topics_to_vector_store()
        _topics_seeded = True
        return count
    return 0


def match_topic(
    text: str, confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Determines document topic matching TopicMatch schema:
    {'topic': str, 'confidence': float, 'resolved_by': 'qdrant' | 'llm_fallback'}
    """
    ensure_topics_seeded()
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else getattr(settings, "SIMILARITY_THRESHOLD", 0.58)
    )
    query_vector = embed_text(text)

    similar = search_similar_topics(query_vector, top_k=3)

    if similar and similar[0]["score"] >= threshold:
        top_match = similar[0]
        return {
            "topic": top_match["topic"],
            "confidence": round(float(top_match["score"]), 4),
            "resolved_by": "qdrant",
        }

    # Low confidence or ambiguous: Trigger LLM provider fallback
    provider = get_llm_provider()
    resolved = provider.resolve_ambiguous(
        text=text, candidate_topics=similar, available_taxonomy=get_available_topics()
    )

    conf = round(float(resolved.get("confidence", 0.75)), 4)
    return {
        "topic": resolved.get(
            "topic", similar[0]["topic"] if similar else "General Support"
        ),
        "confidence": conf,
        "resolved_by": "llm_fallback",
    }


def match_intent(
    text: str, confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Determines customer intent matching IntentMatch schema:
    {'intent': str, 'confidence': float, 'resolved_by': 'rules' | 'qdrant' | 'llm_fallback'}
    """
    import re

    ensure_topics_seeded()
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else getattr(settings, "SIMILARITY_THRESHOLD", 0.58)
    )

    # Step 1: High precision rule-based patterns
    lower_text = text.lower()
    for pattern, intent_name, conf in INTENT_RULE_PATTERNS:
        if re.search(pattern, lower_text):
            return {
                "intent": intent_name,
                "confidence": round(float(conf), 4),
                "resolved_by": "rules",
            }

    # Step 2: Vector search against topic taxonomy
    query_vector = embed_text(text)
    similar = search_similar_topics(query_vector, top_k=3)

    if similar and similar[0]["score"] >= threshold:
        top_topic = similar[0]
        topic_id = top_topic.get("topic_id", "")
        # Find default intent for this topic
        cfg = load_topics_config()
        topic_info = cfg.get("topics", {}).get(topic_id, {})
        intents = topic_info.get("intents", [])
        primary_intent = intents[0] if intents else f"inquire_{topic_id}"

        return {
            "intent": primary_intent,
            "confidence": round(float(top_topic["score"]), 4),
            "resolved_by": "qdrant",
        }

    # Step 3: LLM fallback
    provider = get_llm_provider()
    resolved = provider.resolve_ambiguous(
        text=text, candidate_topics=similar, available_taxonomy=get_available_topics()
    )
    conf = round(float(resolved.get("confidence", 0.75)), 4)
    return {
        "intent": resolved.get("intent", "general_inquiry"),
        "confidence": conf,
        "resolved_by": "llm_fallback",
    }
