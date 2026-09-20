"""
Embeddings module for NLP Document Intelligence Gateway.

Generates dense vector representations using SentenceTransformers (e.g. all-MiniLM-L6-v2)
for topic classification and duplicate detection.
"""

import numpy as np
from typing import List
from functools import lru_cache

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

def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Generates a dense vector embedding (384 dimensions) for input text string.
    """
    model = _get_model()
    if model:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    # Fallback pseudo-embedding if sentence-transformers is missing
    return [0.0] * 384

def get_embeddings_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of input text strings in batches.
    """
    model = _get_model()
    if model:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    return [[0.0] * 384 for _ in texts]

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
