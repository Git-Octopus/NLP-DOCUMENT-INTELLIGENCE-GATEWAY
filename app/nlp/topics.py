"""
Topic & Intent Classification module for NLP Document Intelligence Gateway.

High-level coordinator that calls vector store similarity and rule logic to match
topics and intents.
"""

from typing import Dict, Any


def match_topic(text: str, confidence_threshold: float = 0.60) -> Dict[str, Any]:
    """
    Determines document topic matching TopicMatch schema:
    {'topic': str, 'confidence': float, 'resolved_by': 'qdrant' | 'llm_fallback'}
    """
    pass


def match_intent(text: str, confidence_threshold: float = 0.60) -> Dict[str, Any]:
    """
    Determines customer intent matching IntentMatch schema:
    {'intent': str, 'confidence': float, 'resolved_by': 'rules' | 'qdrant' | 'llm_fallback'}
    """
    pass
