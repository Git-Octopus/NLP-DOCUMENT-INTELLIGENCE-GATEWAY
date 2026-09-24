"""
LLM Provider package for NLP Document Intelligence Gateway.
Provides dynamic provider selection (Ollama, Gemini, OpenAI) with automatic fallback.
"""

import logging
from typing import Optional

from app.config import settings
from app.providers.base import BaseLLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

_global_provider: Optional[BaseLLMProvider] = None
_global_model_name: Optional[str] = None


def reset_llm_provider():
    """Resets cached global LLM provider instance on model/provider configuration change."""
    global _global_provider, _global_model_name
    _global_provider = None
    _global_model_name = None
    logger.info("Active LLM provider cache invalidated.")


def get_llm_provider(
    provider_name: Optional[str] = None, model_name: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function returning the active LLM provider.
    Resolves dynamically from model_manager runtime settings if not explicitly passed.
    """
    global _global_provider, _global_model_name

    # Import locally to avoid circular dependencies
    try:
        from app.services.model_manager import model_manager

        active_provider = provider_name or model_manager.reasoning_provider
        active_model = model_name or model_manager.reasoning_model
    except Exception:
        active_provider = provider_name or getattr(
            settings, "REASONING_PROVIDER", "gemini"
        )
        active_model = model_name or getattr(
            settings, "REASONING_MODEL", "gemini-1.5-flash"
        )

    selected = (active_provider or "gemini").lower()

    if (
        _global_provider is not None
        and _global_provider.provider_name == selected
        and _global_model_name == active_model
    ):
        return _global_provider

    if selected == "ollama":
        _global_provider = OllamaProvider(model_name=active_model)
    elif selected == "openai":
        _global_provider = OpenAIProvider(model_name=active_model)
    else:
        _global_provider = GeminiProvider(model_name=active_model)

    _global_model_name = active_model
    return _global_provider


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "get_llm_provider",
    "reset_llm_provider",
]
