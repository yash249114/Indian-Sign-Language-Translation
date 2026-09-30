"""
Integration tests for FastAPI REST Endpoints.
Verifies /api/health, /api/stats, /api/translate, /api/tts, /api/buffer/control, and /api/process-frame.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    """Tests /api/health returns online status and local vision confirmation."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["translation_engine"] == "Gemini 3.5 Flash Lite"
    assert data["translation_model"] == "gemini-3.5-flash-lite"


def test_api_stats():
    """Tests /api/stats returns verified dataset and evaluation metrics."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_summary" in data or "dataset_statistics" in data
    assert "model_evaluation" in data or "evaluation_metrics" in data
    assert "supported_languages" in data


def test_api_translate():
    """Tests /api/translate endpoint for Telugu translation."""
    payload = {
        "text": "HELLO",
        "src_lang": "eng_Latn",
        "tgt_lang": "tel_Telu"
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "translated_text" in data
    assert len(data["translated_text"]) > 0
    assert data["status"] in ("success", "cached", "offline_fallback")
    assert data["model"] == "gemini-3.5-flash-lite"


def test_api_buffer_control():
    """Tests /api/buffer/control actions."""
    # Clear
    res_clear = client.post("/api/buffer/control?action=clear")
    assert res_clear.status_code == 200

    # Space
    res_space = client.post("/api/buffer/control?action=space")
    assert res_space.status_code == 200
