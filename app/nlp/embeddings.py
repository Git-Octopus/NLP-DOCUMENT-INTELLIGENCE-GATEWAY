"""
Embeddings module for NLP Document Intelligence Gateway.

Generates dense vector representations using SentenceTransformers (e.g. all-MiniLM-L6-v2)
for topic classification and duplicate detection.
"""

from typing import List


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Generates a dense vector embedding (e.g. 384 dimensions) for input text string.
    """
    pass


def get_embeddings_batch(
    texts: List[str], model_name: str = "all-MiniLM-L6-v2"
) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of input text strings in batches.
    """
    pass
