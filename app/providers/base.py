"""
Model Provider Interface for NLP Document Intelligence Gateway.

Abstract Base Class defining the contract for external LLM integrations (Google Gemini,
OpenAI, and Mock/Local fallback).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM operations used strictly as a fallback brain
    when local NLP pipelines require disambiguation or domain expansion.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def resolve_ambiguous(
        self,
        text: str,
        candidate_topics: List[Dict[str, Any]],
        available_taxonomy: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_domain_entities(
        self, text: str, existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def summarize_text(
        self, text: str, max_length: Optional[int] = 150
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        pass
