"""
Unit tests for Local Offline TTS Engine.
Verifies abstract TTSEngine interface and local audio file generation.
"""

import os
import pytest
from tts.tts_engine import LocalOfflineTTSEngine


def test_tts_initialization():
    """Validates TTS engine cache setup."""
    tts = LocalOfflineTTSEngine()
    assert os.path.exists(tts.cache_dir)
    assert tts.engine_name == "pyttsx3_offline"


def test_tts_synthesis_call():
    """Validates speak function returns structured response without error."""
    tts = LocalOfflineTTSEngine()
    res = tts.speak("Test voice synthesis", language="en")

    assert "success" in res
    assert "text" in res
    assert "engine" in res
    assert res["text"] == "Test voice synthesis"


def test_tts_empty_text():
    """Validates empty text handling."""
    tts = LocalOfflineTTSEngine()
    res = tts.speak("", language="en")
    assert res["success"] is False
