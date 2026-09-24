"""
Model Manager Service for NLP Document Intelligence Gateway.

Maintains active runtime model selections for embeddings and reasoning, dynamically queries
the Ollama server for installed models, performs validation, and triggers provider resets.
"""

import logging
import requests
from typing import List, Optional
from fastapi import HTTPException, status

from app.config import settings
from app.schemas.models import (
    ModelConfigRequest,
    ModelConfigResponse,
    OllamaStatus,
    AvailableModelsResponse,
)

logger = logging.getLogger("gateway.models")


class ModelManager:
    """Singleton service managing runtime embedding and reasoning configurations."""

    def __init__(self):
        self._active_ollama_url: Optional[str] = None
        self._embedding_provider: str = (
            getattr(settings, "EMBEDDING_PROVIDER", "sentence_transformer")
            or "sentence_transformer"
        ).lower()
        self._embedding_model: str = (
            getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            or "all-MiniLM-L6-v2"
        )
        self._reasoning_provider: str = (
            getattr(settings, "REASONING_PROVIDER", "gemini") or "gemini"
        ).lower()
        self._reasoning_model: str = (
            getattr(settings, "REASONING_MODEL", "gemini-1.5-flash")
            or "gemini-1.5-flash"
        )

    def get_ollama_url(self) -> str:
        """Returns the verified working Ollama base URL."""
        if self._active_ollama_url:
            return self._active_ollama_url
        return getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
            "/"
        )

    @property
    def embedding_provider(self) -> str:
        return self._embedding_provider

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def reasoning_provider(self) -> str:
        return self._reasoning_provider

    @property
    def reasoning_model(self) -> str:
        return self._reasoning_model

    def get_active_config(self) -> ModelConfigResponse:
        return ModelConfigResponse(
            embedding_provider=self._embedding_provider,
            embedding_model=self._embedding_model,
            reasoning_provider=self._reasoning_provider,
            reasoning_model=self._reasoning_model,
            status="active",
            message="Currently active model configuration",
        )

    def fetch_ollama_status(self, base_url: Optional[str] = None) -> OllamaStatus:
        """
        Dynamically queries the Ollama server (/api/tags) to detect installed models.
        Automatically probes host endpoints (127.0.0.1, host.docker.internal, localhost)
        to discover models installed on the user's host machine.
        """
        primary_url = (
            base_url or self._active_ollama_url or settings.OLLAMA_BASE_URL
        ).rstrip("/")

        candidate_urls = [
            primary_url,
            "http://127.0.0.1:11434",
            "http://host.docker.internal:11434",
            "http://localhost:11434",
        ]
        # Preserve order, deduplicate
        seen = set()
        deduped = []
        for u in candidate_urls:
            cleaned = u.rstrip("/")
            if cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)

        first_reachable_stat = None
        last_error = None

        for target_url in deduped:
            try:
                resp = requests.get(f"{target_url}/api/tags", timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_models = data.get("models", [])
                    model_names = [
                        m.get("name", "") for m in raw_models if m.get("name")
                    ]
                    status_obj = OllamaStatus(
                        reachable=True,
                        base_url=target_url,
                        installed_models=model_names,
                        error=None,
                    )
                    # If this endpoint has models, lock onto it immediately
                    if model_names:
                        self._active_ollama_url = target_url
                        return status_obj
                    if first_reachable_stat is None:
                        first_reachable_stat = status_obj
            except Exception as e:
                last_error = str(e)

        if first_reachable_stat:
            self._active_ollama_url = first_reachable_stat.base_url
            return first_reachable_stat

        return OllamaStatus(
            reachable=False,
            base_url=primary_url,
            installed_models=[],
            error=f"Cannot reach Ollama at {primary_url}: {last_error}",
        )

    def _model_matches(self, target: str, available_list: List[str]) -> bool:
        """Checks if target model exists, matching either 'model' or 'model:tag'."""
        target_clean = target.strip().lower()
        target_base = target_clean.split(":")[0]
        for candidate in available_list:
            cand_clean = candidate.strip().lower()
            cand_base = cand_clean.split(":")[0]
            if (
                target_clean == cand_clean
                or target_clean == cand_base
                or target_base == cand_clean
            ):
                return True
        return False

    def validate_and_update(self, req: ModelConfigRequest) -> ModelConfigResponse:
        """
        Validates the incoming model selection request and updates active runtime configuration.
        Rejects invalid or unreachable models with HTTP 400 without silent fallback.
        """
        emb_prov = req.embedding_provider.strip().lower()
        emb_model = req.embedding_model.strip()
        reas_prov = req.reasoning_provider.strip().lower()
        reas_model = req.reasoning_model.strip()

        # Check provider names
        if emb_prov not in ["sentence_transformer", "ollama"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid embedding provider: '{emb_prov}'. Must be 'sentence_transformer' or 'ollama'.",
            )
        if reas_prov not in ["ollama", "gemini"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid reasoning provider: '{reas_prov}'. Must be 'ollama' or 'gemini'.",
            )

        # Check Ollama requirements if any selected provider is Ollama
        if emb_prov == "ollama" or reas_prov == "ollama":
            ollama_status = self.fetch_ollama_status()
            if not ollama_status.reachable:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Ollama server is unreachable at {ollama_status.base_url}. "
                        f"Please ensure Ollama is running and accessible. Error: {ollama_status.error}"
                    ),
                )

            installed = ollama_status.installed_models

            if emb_prov == "ollama":
                if not self._model_matches(emb_model, installed):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Selected Ollama embedding model '{emb_model}' is not installed. "
                            f"Installed models: {installed}. Please run 'ollama pull {emb_model}' first."
                        ),
                    )

            if reas_prov == "ollama":
                if not self._model_matches(reas_model, installed):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Selected Ollama reasoning model '{reas_model}' is not installed. "
                            f"Installed models: {installed}. Please run 'ollama pull {reas_model}' first."
                        ),
                    )

        # Non-Ollama validations
        if emb_prov == "sentence_transformer" and not emb_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Embedding model name cannot be empty.",
            )
        if reas_prov == "gemini" and not reas_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gemini reasoning model name cannot be empty.",
            )

        # Check what changed
        embedding_changed = (emb_prov != self._embedding_provider) or (
            emb_model != self._embedding_model
        )
        reasoning_changed = (reas_prov != self._reasoning_provider) or (
            reas_model != self._reasoning_model
        )

        # Update runtime state
        self._embedding_provider = emb_prov
        self._embedding_model = emb_model
        self._reasoning_provider = reas_prov
        self._reasoning_model = reas_model

        logger.info(
            f"Active models updated: Embedding=[{emb_prov} | {emb_model}], "
            f"Reasoning=[{reas_prov} | {reas_model}]"
        )

        # If embedding changed, reset embedder cache and re-seed Qdrant topic vectors
        if embedding_changed:
            try:
                from app.nlp.embeddings import reset_embedding_model

                reset_embedding_model()
            except Exception as e:
                logger.warning(f"Error resetting embedding model: {e}")

            try:
                from app.nlp.topics import reseed_topics_for_active_model

                reseed_topics_for_active_model()
            except Exception as e:
                logger.warning(f"Error reseeding topic vectors: {e}")

        # If reasoning changed, reset active LLM provider singleton
        if reasoning_changed:
            try:
                from app.providers import reset_llm_provider

                reset_llm_provider()
            except Exception as e:
                logger.warning(f"Error resetting LLM provider: {e}")

        return ModelConfigResponse(
            embedding_provider=self._embedding_provider,
            embedding_model=self._embedding_model,
            reasoning_provider=self._reasoning_provider,
            reasoning_model=self._reasoning_model,
            status="success",
            message="AI Model configuration updated and synchronized successfully.",
        )

    def get_catalog(self) -> AvailableModelsResponse:
        """Gathers available models across all providers and status metadata."""
        ollama_status = self.fetch_ollama_status()
        installed_ollama = ollama_status.installed_models

        # Check Qdrant status
        qdrant_stat = None
        try:
            from app.nlp.vector_store import check_qdrant_status

            qdrant_stat = check_qdrant_status()
        except Exception:
            qdrant_stat = {
                "reachable": False,
                "mode": "memory",
                "error": "Diagnostic unavailable",
            }

        # Separate Ollama models into embedding vs reasoning
        ollama_embed_models = [
            m
            for m in installed_ollama
            if any(term in m.lower() for term in ["embed", "bert", "bge", "minilm"])
        ]
        if not ollama_embed_models:
            ollama_embed_models = installed_ollama

        ollama_reason_models = [
            m
            for m in installed_ollama
            if not any(term in m.lower() for term in ["embed", "bge"])
        ]
        if not ollama_reason_models:
            ollama_reason_models = installed_ollama

        embedding_catalog = {
            "sentence_transformer": getattr(
                settings,
                "SENTENCE_TRANSFORMER_MODELS",
                ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2"],
            ),
            "ollama": ollama_embed_models,
        }

        reasoning_catalog = {
            "ollama": ollama_reason_models,
            "gemini": getattr(
                settings,
                "GEMINI_MODELS",
                ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            ),
        }

        return AvailableModelsResponse(
            embedding_providers=embedding_catalog,
            reasoning_providers=reasoning_catalog,
            current_config=self.get_active_config(),
            ollama_status=ollama_status,
            qdrant_status=qdrant_stat,
        )


model_manager = ModelManager()
