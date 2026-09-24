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
    TOPICS_CONFIG_PATH: str = "config/topics.yaml"
    EMBEDDING_PROVIDER: str = (
        "sentence_transformer"  # "sentence_transformer" or "ollama"
    )
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Ollama Service Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Decision Thresholds
    TOPIC_CONFIDENCE_THRESHOLD: float = 0.70  # Below this threshold, fallback to LLM
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85

    # LLM / Reasoning Provider Settings
    PRIMARY_LLM_PROVIDER: str = "gemini"  # "gemini", "openai", or "ollama"
    LLM_PROVIDER: str = "gemini"
    REASONING_PROVIDER: str = "gemini"  # "ollama" or "gemini"
    REASONING_MODEL: str = "gemini-1.5-flash"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Configurable Provider Model Lists (Non-Ollama)
    SENTENCE_TRANSFORMER_MODELS: list[str] = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    GEMINI_MODELS: list[str] = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-1.0-pro",
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./nlp_gateway.db"

    MAX_FILE_SIZE_BYTES: int = 10485760

    # Core NLP Pipeline Settings (Dev 2)
    MAX_KEYWORDS: int = 10
    MAX_NGRAMS: int = 50
    TFIDF_MAX_FEATURES: int = 1000
    SPACY_MODEL: str = "en_core_web_sm"
    MAX_TEXT_LENGTH: int = 50000
    ENABLE_NER: bool = True
    ENABLE_RULE_EXTRACTION: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
