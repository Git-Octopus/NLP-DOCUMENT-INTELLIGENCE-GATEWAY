"""
Pydantic Schemas for AI Model Selection and Provider Configuration.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ModelConfigRequest(BaseModel):
    """Payload to update active embedding and reasoning model configuration."""

    embedding_provider: str = Field(
        ...,
        description="Embedding provider name: 'sentence_transformer' or 'ollama'",
        examples=["ollama", "sentence_transformer"],
    )
    embedding_model: str = Field(
        ...,
        description="Selected embedding model name",
        examples=["nomic-embed-text", "all-MiniLM-L6-v2"],
    )
    reasoning_provider: str = Field(
        ...,
        description="Reasoning/LLM provider name: 'ollama' or 'gemini'",
        examples=["ollama", "gemini"],
    )
    reasoning_model: str = Field(
        ...,
        description="Selected reasoning model name",
        examples=["qwen2.5:7b", "gemini-1.5-flash"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "reasoning_provider": "ollama",
                "reasoning_model": "qwen2.5:7b",
            }
        }
    )


class ModelConfigResponse(BaseModel):
    """Current active model configuration state."""

    embedding_provider: str = Field(..., description="Active embedding provider")
    embedding_model: str = Field(..., description="Active embedding model")
    reasoning_provider: str = Field(..., description="Active reasoning provider")
    reasoning_model: str = Field(..., description="Active reasoning model")
    status: str = Field("success", description="Status indicator")
    message: Optional[str] = Field(
        None, description="Optional status or warning message"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "reasoning_provider": "ollama",
                "reasoning_model": "qwen2.5:7b",
                "status": "success",
                "message": "Model configuration synchronized successfully.",
            }
        }
    )


class OllamaStatus(BaseModel):
    """Connectivity and diagnostic info for Ollama server."""

    reachable: bool = Field(..., description="Whether Ollama HTTP API responded")
    base_url: str = Field(..., description="Configured Ollama base URL")
    installed_models: List[str] = Field(
        default_factory=list, description="Models currently installed on Ollama server"
    )
    error: Optional[str] = Field(None, description="Error detail if unreachable")


class AvailableModelsResponse(BaseModel):
    """Complete catalog of available models per provider, active config, and status."""

    embedding_providers: Dict[str, List[str]] = Field(
        ..., description="Available embedding models grouped by provider"
    )
    reasoning_providers: Dict[str, List[str]] = Field(
        ..., description="Available reasoning models grouped by provider"
    )
    current_config: ModelConfigResponse = Field(
        ..., description="Currently active model configuration"
    )
    ollama_status: OllamaStatus = Field(
        ..., description="Ollama server connectivity state"
    )
    qdrant_status: Optional[Dict[str, Any]] = Field(
        None, description="Qdrant vector database connectivity status"
    )
