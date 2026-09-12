"""
Embeddings module for NLP Document Intelligence Gateway.

Generates dense vector representations using SentenceTransformers (e.g. all-MiniLM-L6-v2)
for semantic topic classification and duplicate detection.
"""

from typing import List, Union, Optional
import numpy as np
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_model_instance = None


def get_sentence_transformer_model(model_name: Optional[str] = None):
    """
    Lazy singleton loader for SentenceTransformer model.
    """
    global _model_instance
    if _model_instance is None:
        target_model = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Loading SentenceTransformer embedding model: {target_model}")
        try:
            from sentence_transformers import SentenceTransformer
            _model_instance = SentenceTransformer(target_model)
        except Exception as e:
            logger.warning(f"SentenceTransformer could not load '{target_model}' ({e}). Falling back to lightweight embeddings.")
            _model_instance = "fallback"
    return _model_instance


def get_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generates a normalized dense vector embedding (e.g. 384 dimensions) for input text string.
    """
    if not text or not text.strip():
        return [0.0] * settings.EMBEDDING_DIMENSION

    model = get_sentence_transformer_model(model_name)
    if model != "fallback" and hasattr(model, "encode"):
        try:
            emb = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return emb.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding via SentenceTransformer: {e}")

    return _fallback_text_embedding(text, settings.EMBEDDING_DIMENSION)


def get_embeddings_batch(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of input text strings in batches.
    """
    if not texts:
        return []

    model = get_sentence_transformer_model(model_name)
    if model != "fallback" and hasattr(model, "encode"):
        try:
            embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [e.tolist() for e in embs]
        except Exception as e:
            logger.error(f"Error in batch embeddings: {e}")

    return [_fallback_text_embedding(t, settings.EMBEDDING_DIMENSION) for t in texts]


def compute_cosine_similarity(vec_a: Union[List[float], np.ndarray], vec_b: Union[List[float], np.ndarray]) -> float:
    """
    Computes cosine similarity between two vector representations.
    Returns float score between 0.0 and 1.0.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot = np.dot(a, b)
    similarity = float(dot / (norm_a * norm_b))
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0 if similarity < 0 else similarity))


def _fallback_text_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Lightweight deterministic hashing vector for offline/testing scenarios.
    """
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    if not words:
        return vec.tolist()
    for word in words:
        h = hash(word) % dim
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# Convenience aliases for cross-module compatibility
embed_text = get_embedding
embed_texts = get_embeddings_batch
cosine_similarity = compute_cosine_similarity
