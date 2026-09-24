"""
Vector Store module for NLP Document Intelligence Gateway.

Wrapper around Qdrant client for collection initialization, topic embeddings indexing,
and nearest-neighbor duplicate detection vector search.
Provides seamless fallback to in-memory Qdrant client when external server is offline.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from app.config import settings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_client_mode: str = "disconnected"


def connect_qdrant(
    url: Optional[str] = None, api_key: Optional[str] = None
) -> QdrantClient:
    """
    Initializes Qdrant client connection using settings from config / environment variables.
    Falls back to in-memory Qdrant instance if external connection is unreachable.
    """
    global _qdrant_client, _client_mode
    if _qdrant_client is not None:
        return _qdrant_client

    target_url = url or settings.QDRANT_URL
    target_api_key = api_key or settings.QDRANT_API_KEY

    # Try connecting to remote Qdrant first if configured
    if target_url and not target_url.startswith(":memory:"):
        try:
            client = QdrantClient(url=target_url, api_key=target_api_key, timeout=2.0)
            client.get_collections()
            _qdrant_client = client
            _client_mode = "remote"
            logger.info(f"Connected to remote Qdrant at {target_url}")
            return _qdrant_client
        except Exception as e:
            logger.warning(
                f"Could not connect to remote Qdrant at {target_url} ({e}). "
                f"Falling back to local in-memory Qdrant client."
            )

    # In-memory fallback: allows 100% genuine Qdrant vector search without external Docker
    _qdrant_client = QdrantClient(":memory:")
    _client_mode = "memory"
    logger.info("Initialized local in-memory Qdrant client.")
    return _qdrant_client


def check_qdrant_status() -> Dict[str, Any]:
    """Returns connectivity and diagnostic status for Qdrant vector database."""
    client = connect_qdrant()
    try:
        colls = [c.name for c in client.get_collections().collections]
        return {
            "reachable": True,
            "mode": _client_mode,
            "url": settings.QDRANT_URL if _client_mode == "remote" else "in-memory",
            "collections": colls,
            "error": None,
        }
    except Exception as e:
        return {
            "reachable": False,
            "mode": _client_mode,
            "url": settings.QDRANT_URL,
            "collections": [],
            "error": str(e),
        }


def init_collections(vector_size: Optional[int] = None):
    """
    Ensures 'topics' and 'messages' collections exist in Qdrant with specified vector size.
    """
    client = connect_qdrant()
    dim = vector_size or settings.EMBEDDING_DIMENSION

    for collection_name in [settings.TOPICS_COLLECTION, settings.MESSAGES_COLLECTION]:
        try:
            collections = [c.name for c in client.get_collections().collections]
            if collection_name not in collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dim, distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name} (dim={dim})")
        except Exception as e:
            logger.warning(f"Error ensuring Qdrant collection {collection_name}: {e}")


def recreate_collections_for_dimension(vector_size: int):
    """
    Recreates Qdrant collections when active embedding model dimension changes.
    """
    client = connect_qdrant()
    for collection_name in [settings.TOPICS_COLLECTION, settings.MESSAGES_COLLECTION]:
        try:
            collections = [c.name for c in client.get_collections().collections]
            if collection_name in collections:
                client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            logger.info(
                f"Recreated Qdrant collection: {collection_name} with new dim={vector_size}"
            )
        except Exception as e:
            logger.warning(f"Error recreating collection {collection_name}: {e}")


def upsert_topic_vectors(topics_data: List[Dict[str, Any]]):
    """
    Upserts embedded topic descriptions / anchor phrases into Qdrant 'topics' collection.
    """
    if not topics_data:
        return

    client = connect_qdrant()
    init_collections()

    points = []
    for item in topics_data:
        point_id = item.get("id")
        vector = item.get("vector")
        payload = item.get("payload", {})
        if point_id and vector:
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

    if points:
        try:
            client.upsert(collection_name=settings.TOPICS_COLLECTION, points=points)
            logger.info(
                f"Upserted {len(points)} topic vectors to {settings.TOPICS_COLLECTION}."
            )
        except Exception as e:
            logger.error(f"Failed to upsert topic vectors to Qdrant: {e}")


def search_similar_topics(
    text_vector: List[float], top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches Qdrant 'topics' collection for top matching topic labels and confidence scores.
    """
    client = connect_qdrant()
    init_collections()

    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=settings.TOPICS_COLLECTION,
                query=text_vector,
                limit=top_k,
                with_payload=True,
            )
            raw_points = response.points
        elif hasattr(client, "search"):
            raw_points = client.search(
                collection_name=settings.TOPICS_COLLECTION,
                query_vector=text_vector,
                limit=top_k,
                with_payload=True,
            )
        else:
            raw_points = []

        results = []
        for point in raw_points:
            payload = point.payload or {}
            topic_name = payload.get("topic_name") or payload.get(
                "topic", "General Support"
            )
            topic_id = payload.get("topic_id", topic_name.lower().replace(" ", "_"))
            results.append(
                {
                    "topic": topic_name,
                    "topic_id": topic_id,
                    "score": round(float(point.score), 4),
                    "payload": payload,
                }
            )
        return results
    except Exception as e:
        logger.warning(f"Error querying Qdrant topics collection: {e}")
        return []


def search_duplicate_message(
    text_vector: List[float], threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Searches Qdrant 'messages' collection for existing messages with cosine similarity >= threshold.
    Returns DuplicateCheck information.
    """
    client = connect_qdrant()
    init_collections()
    thresh = (
        threshold if threshold is not None else settings.DUPLICATE_SIMILARITY_THRESHOLD
    )

    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=settings.MESSAGES_COLLECTION,
                query=text_vector,
                limit=1,
                with_payload=True,
            )
            raw_points = response.points
        elif hasattr(client, "search"):
            raw_points = client.search(
                collection_name=settings.MESSAGES_COLLECTION,
                query_vector=text_vector,
                limit=1,
                with_payload=True,
            )
        else:
            raw_points = []

        if raw_points:
            best_match = raw_points[0]
            score = round(float(best_match.score), 4)
            is_dup = score >= thresh
            return {
                "is_duplicate": is_dup,
                "matched_message_id": str(best_match.id),
                "similarity_score": score,
            }
    except Exception as e:
        logger.warning(f"Error querying Qdrant messages collection: {e}")

    return {
        "is_duplicate": False,
        "matched_message_id": None,
        "similarity_score": 0.0,
    }


def index_message(
    message_id: str,
    text_vector: List[float],
    payload: Optional[Dict[str, Any]] = None,
):
    """
    Indexes a processed customer message vector into Qdrant for future duplicate detection.
    """
    client = connect_qdrant()
    init_collections()
    try:
        client.upsert(
            collection_name=settings.MESSAGES_COLLECTION,
            points=[
                models.PointStruct(
                    id=message_id,
                    vector=text_vector,
                    payload=payload or {},
                )
            ],
        )
    except Exception as e:
        logger.warning(f"Error indexing message {message_id} in Qdrant: {e}")
