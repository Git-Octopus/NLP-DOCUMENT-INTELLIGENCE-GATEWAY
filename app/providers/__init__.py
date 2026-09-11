"""
LLM Provider package for NLP Document Intelligence Gateway.
Provides dynamic provider selection (Gemini or OpenAI) with automatic fallback.
"""

import logging
from typing import Optional

from app.config import settings
from app.providers.base import BaseLLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

_global_provider: Optional[BaseLLMProvider] = None


def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function returning the active LLM provider.
    Defaults to settings.LLM_PROVIDER ('gemini' or 'openai').
    """
    global _global_provider
    selected = (provider_name or settings.LLM_PROVIDER or "gemini").lower()

    if _global_provider is not None and _global_provider.provider_name == selected:
        return _global_provider

    if selected == "openai":
        _global_provider = OpenAIProvider()
    else:
        _global_provider = GeminiProvider()

    return _global_provider


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "get_llm_provider",
]
