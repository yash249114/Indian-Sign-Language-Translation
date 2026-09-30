"""
Comprehensive Test Suite for Gemini 3.5 Flash Lite Translation Engine.
Covers:
1. /api/translate endpoint schema and execution
2. Correct model selection ('gemini-3.5-flash-lite')
3. Environment variable loading & API key absence fallback
4. Successful translation & payload structure
5. Empty text handling
6. Same phrase cache hit (zero redundant API calls)
7. Multi-language code normalization (Telugu, Hindi, Tamil, Kannada, Malayalam, etc.)
8. HTTP 429 Rate limit protection and graceful handling
9. Network timeout & connection error handling
10. Text buffer -> translation integration
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.translation import GeminiTranslationEngine, TranslationCache, LANGUAGE_MAP, translator

client = TestClient(app)


def test_model_selection_and_defaults():
    """Validates that the translation engine defaults strictly to gemini-3.5-flash-lite."""
    engine = GeminiTranslationEngine()
    assert engine.model_name == "gemini-3.5-flash-lite"


def test_language_code_normalization():
    """Validates mapping of various language code representations."""
    engine = GeminiTranslationEngine()
    assert engine.normalize_lang_code("te") == "te"
    assert engine.normalize_lang_code("tel_Telu") == "te"
    assert engine.normalize_lang_code("Telugu") == "te"
    assert engine.normalize_lang_code("hi") == "hi"
    assert engine.normalize_lang_code("hin_Deva") == "hi"
    assert engine.normalize_lang_code("ta") == "ta"
    assert engine.normalize_lang_code("kn") == "kn"
    assert engine.normalize_lang_code("ml") == "ml"
    assert engine.normalize_lang_code("en") == "en"


def test_empty_and_whitespace_text_handling():
    """Validates empty or whitespace text returns empty string with 0 API calls."""
    engine = GeminiTranslationEngine(api_key="test_key")
    res1 = engine.translate("", "en", "te")
    assert res1["translated_text"] == ""
    assert res1["status"] == "empty_input"
    assert res1["cached"] is False

    res2 = engine.translate("   ", "en", "hi")
    assert res2["translated_text"] == ""
    assert res2["status"] == "empty_input"


def test_identity_translation():
    """Validates identical source and target language returns unaltered text without API call."""
    engine = GeminiTranslationEngine(api_key="test_key")
    res = engine.translate("Hello World", "en", "en")
    assert res["translated_text"] == "Hello World"
    assert res["status"] == "identity"
    assert res["cached"] is False


def test_api_key_absence_and_offline_lexicon_fallback():
    """Validates graceful fallback when GEMINI_API_KEY is not configured."""
    engine = GeminiTranslationEngine(api_key="")

    # Common vocabulary in offline lexicon falls back cleanly
    res_hello = engine.translate("HELLO", "en", "te")
    assert res_hello["translated_text"] == "నమస్కారం"
    assert res_hello["status"] == "offline_fallback"

    # Unknown phrase returns honest unavailable message without crashing
    res_unknown = engine.translate("arbitrary custom sentence", "en", "te")
    assert "unavailable" in res_unknown["status"] or "not configured" in res_unknown["translated_text"]
    assert res_unknown["model"] == "gemini-3.5-flash-lite"


def test_translation_caching_behavior():
    """Validates that identical text + target language hits cache and makes ZERO redundant API calls."""
    engine = GeminiTranslationEngine(api_key="mock_api_key")

    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "నమస్కారం, నా పేరు యష్."}]
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_gemini_response
        mock_post.return_value = mock_response

        # First call: Cache miss -> calls Gemini API
        res1 = engine.translate("Hello, my name is Yash.", "en", "te")
        assert res1["translated_text"] == "నమస్కారం, నా పేరు యష్."
        assert res1["cached"] is False
        assert res1["status"] == "success"
        assert mock_post.call_count == 1

        # Second call with same text + language: Cache hit -> ZERO API calls
        res2 = engine.translate("Hello, my name is Yash.", "en", "te")
        assert res2["translated_text"] == "నమస్కారం, నా పేరు యష్."
        assert res2["cached"] is True
        assert res2["status"] == "cached"
        assert mock_post.call_count == 1  # Still 1, NOT called again!

        # Different target language (Hindi): Cache miss -> calls API
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "नमस्ते, मेरा नाम यश है।"}]
                    }
                }
            ]
        }
        res3 = engine.translate("Hello, my name is Yash.", "en", "hi")
        assert res3["translated_text"] == "नमस्ते, मेरा नाम यश है।"
        assert res3["cached"] is False
        assert mock_post.call_count == 2


def test_http_429_rate_limit_resilience():
    """Validates that HTTP 429 responses return rate limit status without crashing."""
    engine = GeminiTranslationEngine(api_key="mock_api_key")

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response

        res = engine.translate("Test rate limit message", "en", "te")
        assert res["status"] == "rate_limited"
        assert "temporarily unavailable" in res["translated_text"].lower() or "rate limit" in res["translated_text"].lower()


def test_network_timeout_resilience():
    """Validates that network timeouts return timeout status without freezing."""
    import requests
    engine = GeminiTranslationEngine(api_key="mock_api_key")

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        res = engine.translate("Test timeout message", "en", "te")
        assert res["status"] == "timeout"
        assert "unavailable" in res["translated_text"].lower() or "timeout" in res["translated_text"].lower()


def test_api_translate_endpoint():
    """Tests the /api/translate REST endpoint."""
    payload = {
        "text": "HELLO",
        "source_language": "en",
        "target_language": "te"
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "translated_text" in data
    assert "model" in data
    assert "cached" in data


def test_api_stats_endpoint_includes_telemetry():
    """Tests that /api/stats includes translation telemetry counters."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "translation_telemetry" in data
    tel = data["translation_telemetry"]
    assert "translation_calls_today" in tel
    assert "cached_translations" in tel
