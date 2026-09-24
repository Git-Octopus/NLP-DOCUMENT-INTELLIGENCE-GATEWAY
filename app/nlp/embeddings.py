"""
Embeddings module for NLP Document Intelligence Gateway.

Generates dense vector representations using SentenceTransformers (e.g. all-MiniLM-L6-v2)
for topic classification and duplicate detection.
"""

import hashlib
import logging
import math
import re
from typing import List, Optional
import numpy as np
import requests

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_loaded_model_name: Optional[str] = None


def reset_embedding_model():
    """Invalidates the cached SentenceTransformer model instance."""
    global _model, _loaded_model_name
    _model = None
    _loaded_model_name = None
    logger.info("Active embedding model cache invalidated.")


def _get_active_provider_and_model(
    model_name: Optional[str] = None, provider: Optional[str] = None
) -> tuple[str, str]:
    """Helper to dynamically resolve embedding provider and model name."""
    try:
        from app.services.model_manager import model_manager

        active_provider = provider or model_manager.embedding_provider
        active_model = model_name or model_manager.embedding_model
    except Exception:
        active_provider = provider or getattr(
            settings, "EMBEDDING_PROVIDER", "sentence_transformer"
        )
        active_model = model_name or getattr(
            settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

    return (
        active_provider or "sentence_transformer"
    ).lower(), active_model or "all-MiniLM-L6-v2"


def _get_sentence_transformer_model(target_model_name: str):
    """Lazy loader for SentenceTransformer model supporting dynamic model names."""
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != target_model_name:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(target_model_name)
            _loaded_model_name = target_model_name
            logger.info(
                f"Loaded SentenceTransformer embedding model: {target_model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load SentenceTransformer model '{target_model_name}' ({e}). "
                f"Falling back to deterministic hash vectors."
            )
            _model = False
            _loaded_model_name = target_model_name
    return _model


def _fallback_embedding(text: str, dimension: int = 384) -> List[float]:
    vector = [0.0] * dimension
    tokens = re.findall(r"\w+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0

    return [value / norm for value in vector]


def _ollama_embedding(text: str, model_name: str) -> Optional[List[float]]:
    """Generates dense vector using local Ollama instance."""
    base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
        "/"
    )
    try:
        # Try modern Ollama /api/embed endpoint
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": model_name, "input": text},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return [float(x) for x in embeddings[0]]

        # Fallback to classic /api/embeddings endpoint
        resp = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            embedding = data.get("embedding", [])
            if embedding:
                return [float(x) for x in embedding]
    except Exception as e:
        logger.warning(f"Ollama embedding request failed for model '{model_name}': {e}")
    return None


def get_embedding(
    text: str,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[float]:
    """
    Generates a dense vector embedding for input text string using either
    the configured SentenceTransformer or Ollama model.
    """
    active_provider, active_model = _get_active_provider_and_model(model_name, provider)

    if active_provider == "ollama":
        vector = _ollama_embedding(text, active_model)
        if vector:
            return vector
        logger.warning(
            f"Ollama embedding unavailable for '{active_model}'. Using deterministic fallback."
        )
        return _fallback_embedding(text, getattr(settings, "EMBEDDING_DIMENSION", 384))

    # Default to SentenceTransformer
    st_model = _get_sentence_transformer_model(active_model)
    if st_model:
        embedding = st_model.encode(text, convert_to_numpy=True)
        return embedding.astype(float).tolist()

    return _fallback_embedding(text, getattr(settings, "EMBEDDING_DIMENSION", 384))


def get_embeddings_batch(
    texts: List[str],
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of input texts.
    """
    active_provider, active_model = _get_active_provider_and_model(model_name, provider)

    if active_provider == "ollama":
        base_url = getattr(
            settings, "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/")
        try:
            resp = requests.post(
                f"{base_url}/api/embed",
                json={"model": active_model, "input": texts},
                timeout=20.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts):
                    return [[float(x) for x in emb] for emb in embeddings]
        except Exception as e:
            logger.warning(f"Ollama batch embed failed for '{active_model}': {e}")

        # Fallback element by element
        return [
            get_embedding(t, model_name=active_model, provider=active_provider)
            for t in texts
        ]

    # SentenceTransformer batch
    st_model = _get_sentence_transformer_model(active_model)
    if st_model:
        embeddings = st_model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(float).tolist()

    return [
        _fallback_embedding(text, getattr(settings, "EMBEDDING_DIMENSION", 384))
        for text in texts
    ]


def get_active_embedding_dimension() -> int:
    """Returns the vector dimension produced by the currently active embedding model."""
    sample_vec = get_embedding("dimension_probe")
    return len(sample_vec)


def embed_text(text: str) -> List[float]:
    return get_embedding(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embeddings_batch(texts)


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity score between two dense vectors."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    score = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, score))
