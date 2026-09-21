import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Gateway Application Settings
    APP_NAME: str = "NLP Document Intelligence Gateway"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Vector Store / Qdrant Settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    TOPICS_COLLECTION: str = "support_topics"
    MESSAGES_COLLECTION: str = "support_messages"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Decision Thresholds
    TOPIC_CONFIDENCE_THRESHOLD: float = 0.70  # Below this threshold, fallback to LLM
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85

    # LLM Fallback Providers (Only used when local NLP confidence is below threshold)
    PRIMARY_LLM_PROVIDER: str = "gemini"  # "gemini" or "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # SQLite Database
    DATABASE_URL: str = "sqlite:///./nlp_gateway.db"

    MAX_FILE_SIZE_BYTES: int = 10485760

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
