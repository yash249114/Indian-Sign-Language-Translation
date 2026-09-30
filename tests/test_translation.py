"""
Unit tests for Gemini Translation Engine.
Verifies translation across Telugu, Hindi, Tamil, Kannada, Malayalam,
and graceful offline fallbacks.
"""

import pytest
from backend.app.translation import GeminiTranslationEngine, LANGUAGE_MAP


def test_supported_languages_presence():
    """Validates required Indian language codes in LANGUAGE_MAP."""
    required = ["te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "en"]
    for lang in required:
        assert lang in LANGUAGE_MAP


def test_translation_all_five_indian_languages_fallback():
    """Validates offline fallback translation across all 5 requested Indian languages."""
    engine = GeminiTranslationEngine(api_key="")

    # Telugu
    assert engine.translate("HELLO", "en", "te")["translated_text"] == "నమస్కారం"
    # Hindi
    assert engine.translate("HELLO", "en", "hi")["translated_text"] == "नमस्ते"
    # Tamil
    assert engine.translate("HELLO", "en", "ta")["translated_text"] == "வணக்கம்"
    # Kannada
    assert engine.translate("HELLO", "en", "kn")["translated_text"] == "ನಮಸ್ಕಾರ"
    # Malayalam
    assert engine.translate("HELLO", "en", "ml")["translated_text"] == "നമസ്കാരം"


def test_translation_vocabulary_terms_fallback():
    """Validates common ISL vocabulary offline fallback translation."""
    engine = GeminiTranslationEngine(api_key="")

    assert engine.translate("WATER", "en", "te")["translated_text"] == "నీరు"
    assert engine.translate("WATER", "en", "hi")["translated_text"] == "पानी"
    assert engine.translate("THANK YOU", "en", "te")["translated_text"] == "ధన్యవాదాలు"
    assert engine.translate("GOOD", "en", "ta")["translated_text"] == "நல்லது"
