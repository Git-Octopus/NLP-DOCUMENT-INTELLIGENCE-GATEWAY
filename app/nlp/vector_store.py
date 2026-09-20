"""
Vector Store module for NLP Document Intelligence Gateway.

Wrapper around Qdrant client for collection initialization, topic embeddings indexing,
and nearest-neighbor duplicate detection vector search.
Provides automatic fallback to in-memory Qdrant storage if no live server is configured.
"""

import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from app.config import settings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None


def connect_qdrant(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    """
    Initializes or returns Qdrant client connection.
    Uses live URL if provided in config; otherwise runs in-memory.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    target_url = url or settings.QDRANT_URL
    target_key = api_key or settings.QDRANT_API_KEY

    if target_url and target_url.strip():
        try:
            logger.info(f"Connecting to Qdrant Vector DB at {target_url}...")
            _qdrant_client = QdrantClient(url=target_url, api_key=target_key)
            _qdrant_client.get_collections()
            logger.info("Successfully connected to live Qdrant instance.")
            return _qdrant_client
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant at {target_url} ({e}). Falling back to in-memory Qdrant instance.")

    logger.info("Initializing in-memory Qdrant vector database.")
    _qdrant_client = QdrantClient(":memory:")
    return _qdrant_client


def get_qdrant_client() -> QdrantClient:
    """
    Returns existing Qdrant client or initializes one.
    """
    return connect_qdrant()


def init_collections(vector_size: int = 384):
    """
    Ensures 'topics' and 'messages' collections exist in Qdrant with specified vector size.
    """
    client = get_qdrant_client()
    collections_response = client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    topics_col = settings.QDRANT_COLLECTION_TOPICS
    messages_col = settings.QDRANT_COLLECTION_MESSAGES

    if topics_col not in existing_collections:
        logger.info(f"Creating Qdrant collection: '{topics_col}' (dim: {vector_size})")
        client.create_collection(
            collection_name=topics_col,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    if messages_col not in existing_collections:
        logger.info(f"Creating Qdrant collection: '{messages_col}' (dim: {vector_size})")
        client.create_collection(
            collection_name=messages_col,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _search_points(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int) -> List[Any]:
    """
    Cross-version query helper supporting both qdrant-client >=1.10 (query_points)
    and older versions (search).
    """
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )
        return res.points if hasattr(res, "points") else (res if isinstance(res, list) else [])
    elif hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
    return []


def upsert_topic_vectors(points_data: List[Dict[str, Any]]):
    """
    Upserts topic points into the Qdrant 'topics' collection.
    """
    if not points_data:
        return

    client = get_qdrant_client()
    init_collections(vector_size=len(points_data[0]["vector"]))

    points = [
        PointStruct(
            id=item["id"],
            vector=item["vector"],
            payload=item["payload"]
        )
        for item in points_data
    ]

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_TOPICS,
        points=points
    )
    logger.info(f"Upserted {len(points)} topic anchor vectors into Qdrant collection '{settings.QDRANT_COLLECTION_TOPICS}'.")


def search_similar_topics(text_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches Qdrant 'topics' collection for top matching topic labels and confidence scores.
    """
    client = get_qdrant_client()
    init_collections(vector_size=len(text_vector))

    try:
        results = _search_points(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_TOPICS,
            query_vector=text_vector,
            limit=top_k * 4
        )
    except Exception as e:
        logger.error(f"Error querying Qdrant 'topics' collection: {e}")
        return []

    topic_best: Dict[str, Dict[str, Any]] = {}
    for hit in results:
        payload = hit.payload or {}
        topic_id = payload.get("topic_id", "unknown")
        topic_name = payload.get("topic_name", topic_id)
        score = float(hit.score)
        
        normalized_score = max(0.0, min(1.0, score))

        if topic_id not in topic_best or normalized_score > topic_best[topic_id]["score"]:
            topic_best[topic_id] = {
                "topic_id": topic_id,
                "topic": topic_name,
                "score": round(normalized_score, 4),
                "matched_text": payload.get("text", "")
            }

    sorted_topics = sorted(topic_best.values(), key=lambda x: x["score"], reverse=True)
    return sorted_topics[:top_k]


def search_duplicate_message(text_vector: List[float], threshold: float = 0.85) -> Dict[str, Any]:
    """
    Searches Qdrant 'messages' collection for existing messages with cosine similarity >= threshold.
    """
    client = get_qdrant_client()
    init_collections(vector_size=len(text_vector))

    try:
        results = _search_points(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_MESSAGES,
            query_vector=text_vector,
            limit=1
        )
        if results and results[0].score >= threshold:
            best_hit = results[0]
            matched_id = str(best_hit.id)
            if best_hit.payload and "message_id" in best_hit.payload:
                matched_id = str(best_hit.payload["message_id"])
            return {
                "is_duplicate": True,
                "matched_message_id": matched_id,
                "similarity_score": round(float(best_hit.score), 4)
            }
        elif results:
            return {
                "is_duplicate": False,
                "matched_message_id": None,
                "similarity_score": round(float(results[0].score), 4)
            }
    except Exception as e:
        logger.error(f"Error searching duplicate messages in Qdrant: {e}")

    return {
        "is_duplicate": False,
        "matched_message_id": None,
        "similarity_score": 0.0
    }


def record_analyzed_message(message_id: str, text: str, text_vector: List[float]):
    """
    Stores an analyzed message embedding into the Qdrant 'messages' collection
    for future duplicate detection.
    """
    client = get_qdrant_client()
    init_collections(vector_size=len(text_vector))

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, message_id)) if not _is_valid_uuid(message_id) else message_id

    try:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_MESSAGES,
            points=[
                PointStruct(
                    id=point_id,
                    vector=text_vector,
                    payload={"message_id": message_id, "text": text[:200]}
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error recording message to Qdrant: {e}")


def _is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False
