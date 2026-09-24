"""
Test Suite for Dynamic AI Model Selection, Ollama Discovery, and Validation.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_manager import model_manager
from app.providers.ollama_provider import OllamaProvider
from app.nlp.embeddings import reset_embedding_model

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_manager_baseline():
    """Ensure baseline models are restored after each test."""
    yield
    model_manager._embedding_provider = "sentence_transformer"
    model_manager._embedding_model = "all-MiniLM-L6-v2"
    model_manager._reasoning_provider = "gemini"
    model_manager._reasoning_model = "gemini-1.5-flash"
    reset_embedding_model()


def test_get_models_catalog_endpoint():
    """Verify GET /models returns schema-compliant catalog and status."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()

    assert "embedding_providers" in data
    assert "sentence_transformer" in data["embedding_providers"]
    assert "ollama" in data["embedding_providers"]

    assert "reasoning_providers" in data
    assert "ollama" in data["reasoning_providers"]
    assert "gemini" in data["reasoning_providers"]

    assert "current_config" in data
    assert "ollama_status" in data
    assert "reachable" in data["ollama_status"]


def test_get_model_config_endpoint():
    """Verify GET /models/config returns active model selections."""
    response = client.get("/models/config")
    assert response.status_code == 200
    data = response.json()

    assert "embedding_provider" in data
    assert "embedding_model" in data
    assert "reasoning_provider" in data
    assert "reasoning_model" in data


def test_update_valid_sentence_transformer_and_gemini_config():
    """Verify updating to valid SentenceTransformer and Gemini model succeeds."""
    payload = {
        "embedding_provider": "sentence_transformer",
        "embedding_model": "all-mpnet-base-v2",
        "reasoning_provider": "gemini",
        "reasoning_model": "gemini-1.5-pro",
    }
    with patch("app.nlp.embeddings.get_active_embedding_dimension", return_value=768):
        response = client.post("/models/config", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["embedding_provider"] == "sentence_transformer"
    assert data["embedding_model"] == "all-mpnet-base-v2"
    assert data["reasoning_provider"] == "gemini"
    assert data["reasoning_model"] == "gemini-1.5-pro"
    assert data["status"] == "success"

    # Confirm manager state is synchronized
    assert model_manager.embedding_model == "all-mpnet-base-v2"
    assert model_manager.reasoning_model == "gemini-1.5-pro"


def test_update_rejects_invalid_provider():
    """Verify invalid provider name is rejected with 400."""
    payload = {
        "embedding_provider": "unsupported_provider",
        "embedding_model": "some-model",
        "reasoning_provider": "gemini",
        "reasoning_model": "gemini-1.5-flash",
    }
    response = client.post("/models/config", json=payload)
    assert response.status_code == 400
    assert "Invalid embedding provider" in response.json()["detail"]


def test_update_rejects_nonexistent_ollama_model():
    """Verify selecting an Ollama model not installed on the server is rejected with 400."""
    # Mock Ollama server as reachable with installed models ['qwen2.5:7b', 'nomic-embed-text']
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text"}]
        }
        mock_get.return_value = mock_resp

        payload = {
            "embedding_provider": "ollama",
            "embedding_model": "nonexistent-model-xyz",
            "reasoning_provider": "ollama",
            "reasoning_model": "qwen2.5:7b",
        }
        response = client.post("/models/config", json=payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "not installed" in detail


def test_update_rejects_when_ollama_unreachable():
    """Verify selecting Ollama when the server is down is rejected with 400."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        payload = {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "reasoning_provider": "gemini",
            "reasoning_model": "gemini-1.5-flash",
        }
        response = client.post("/models/config", json=payload)
        assert response.status_code == 400
        assert "unreachable" in response.json()["detail"].lower()


def test_dynamic_ollama_model_detection_and_selection():
    """Verify that when a new model is pulled into Ollama, it is accepted dynamically."""
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
        mock_get.return_value = mock_resp

        # 1. Check GET /models discovers the models dynamically
        get_res = client.get("/models")
        assert get_res.status_code == 200
        catalog = get_res.json()
        assert "llama3.2:latest" in catalog["reasoning_providers"]["ollama"]
        assert "nomic-embed-text:latest" in catalog["embedding_providers"]["ollama"]

        # 2. Select the dynamically discovered model
        payload = {
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "reasoning_provider": "ollama",
            "reasoning_model": "llama3.2",
        }
        post_res = client.post("/models/config", json=payload)
        assert post_res.status_code == 200
        assert post_res.json()["status"] == "success"
        assert model_manager.reasoning_model == "llama3.2"


def test_embedding_and_reasoning_models_are_separate():
    """Verify changing embedding model does not accidentally alter reasoning model."""
    with patch("app.nlp.embeddings.get_active_embedding_dimension", return_value=384):
        # Reset to baseline
        client.post(
            "/models/config",
            json={
                "embedding_provider": "sentence_transformer",
                "embedding_model": "all-MiniLM-L6-v2",
                "reasoning_provider": "gemini",
                "reasoning_model": "gemini-1.5-flash",
            },
        )

        # Change only embedding model
        client.post(
            "/models/config",
            json={
                "embedding_provider": "sentence_transformer",
                "embedding_model": "all-MiniLM-L12-v2",
                "reasoning_provider": "gemini",
                "reasoning_model": "gemini-1.5-flash",
            },
        )
        assert model_manager.embedding_model == "all-MiniLM-L12-v2"
        assert model_manager.reasoning_model == "gemini-1.5-flash"

        # Change only reasoning model
        client.post(
            "/models/config",
            json={
                "embedding_provider": "sentence_transformer",
                "embedding_model": "all-MiniLM-L12-v2",
                "reasoning_provider": "gemini",
                "reasoning_model": "gemini-2.0-flash",
            },
        )
        assert model_manager.embedding_model == "all-MiniLM-L12-v2"
        assert model_manager.reasoning_model == "gemini-2.0-flash"


def test_ollama_provider_heuristic_fallback():
    """Verify OllamaProvider gracefully falls back to heuristics if server fails."""
    provider = OllamaProvider(
        model_name="test-model", base_url="http://invalid-host:9999"
    )
    assert provider.is_available() is False

    # Topic resolution fallback
    res = provider.resolve_ambiguous(
        text="Need refund for double charge",
        candidate_topics=[{"topic": "Billing & Refunds", "score": 0.88}],
    )
    assert "topic" in res
    assert res["topic"] == "Billing & Refunds"

    # Summarization fallback
    summary_res = provider.summarize_text("Customer reported repeated login failure.")
    assert "summary" in summary_res
    assert "key_points" in summary_res


def test_versioned_models_route_prefix():
    """Verify /api/v1/models is accessible under versioned prefix."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert "embedding_providers" in response.json()
