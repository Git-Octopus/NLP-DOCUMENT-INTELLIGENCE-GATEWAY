"""
Embeddings module for NLP Document Intelligence Gateway.

Generates dense vector representations using SentenceTransformers (e.g. all-MiniLM-L6-v2)
for topic classification and duplicate detection.
"""

import hashlib
import math
import re
from typing import List
from functools import lru_cache
import numpy as np

_model = None


def _get_model():
    """Lazy loader for SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from app.config import settings

            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception:
            _model = False
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


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Generates a dense vector embedding (384 dimensions) for input text string.
    """
    model = _get_model()

    if model:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(float).tolist()

    from app.config import settings

    return _fallback_embedding(text, settings.EMBEDDING_DIMENSION)


def get_embeddings_batch(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of input text strings in batches.
    """
    model = _get_model()

    if model:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(float).tolist()

    from app.config import settings

    return [_fallback_embedding(text, settings.EMBEDDING_DIMENSION) for text in texts]


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
