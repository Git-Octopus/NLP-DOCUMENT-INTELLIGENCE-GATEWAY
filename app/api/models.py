"""
AI Model Selection & Management API Routes.

Provides endpoints to inspect available embedding/reasoning models across providers,
check Ollama server connectivity, and dynamically update runtime model selections.
"""

from fastapi import APIRouter, status
from app.schemas.models import (
    ModelConfigRequest,
    ModelConfigResponse,
    AvailableModelsResponse,
)
from app.services.model_manager import model_manager

router = APIRouter(tags=["AI Models"])


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Available Models and Provider Status",
    description="Returns available embedding and reasoning models, dynamically querying Ollama if available.",
)
async def get_available_models() -> AvailableModelsResponse:
    """Returns available models catalog, active configuration, and provider connectivity."""
    return model_manager.get_catalog()


@router.get(
    "/models/config",
    response_model=ModelConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Active Model Configuration",
    description="Returns currently active embedding and reasoning providers and models.",
)
async def get_active_model_config() -> ModelConfigResponse:
    """Returns the current active model configuration."""
    return model_manager.get_active_config()


@router.post(
    "/models/config",
    response_model=ModelConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Active Model Configuration",
    description="Validates selected models and updates runtime embedding and reasoning engine configurations.",
)
async def update_model_config(request: ModelConfigRequest) -> ModelConfigResponse:
    """
    Validates model selection against Ollama/provider status.
    Rejects invalid models with HTTP 400.
    """
    return model_manager.validate_and_update(request)
