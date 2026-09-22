import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Gateway Application Settings
    APP_NAME: str = "NLP Document Intelligence Gateway"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Vector Store / Qdrant Settings
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    TOPICS_COLLECTION: str = "topics"
    MESSAGES_COLLECTION: str = "messages"
    QDRANT_COLLECTION_TOPICS: str = "topics"
    QDRANT_COLLECTION_MESSAGES: str = "messages"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    TOPICS_CONFIG_PATH: str = str(BASE_DIR / "config" / "topics.yaml")

    # Decision Thresholds
    TOPIC_CONFIDENCE_THRESHOLD: float = 0.58  # Below this threshold, fallback to LLM
    SIMILARITY_THRESHOLD: float = 0.58
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85
    DUPLICATE_THRESHOLD: float = 0.85

    # LLM Fallback Providers (Only used when local NLP confidence is below threshold)
    PRIMARY_LLM_PROVIDER: str = "gemini"  # "gemini" or "openai"
    LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Database
    DATABASE_URL: str = "sqlite:///./nlp_gateway.db"

    MAX_FILE_SIZE_BYTES: int = 10485760

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
