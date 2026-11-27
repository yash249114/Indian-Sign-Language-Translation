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


def test_tts_reports_unrenderable_text_instead_of_silent_file(monkeypatch):
    """A header-only WAV (no voice for the script) must be reported as a failure, not returned."""
    import sys, types
    tts = LocalOfflineTTSEngine()

    class FakeEngine:
        def setProperty(self, *a): pass
        def save_to_file(self, text, path):
            with open(path, "wb") as f:
                f.write(b"RIFF" + b"\0" * 42)  # 46-byte header-only WAV
        def runAndWait(self): pass

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=lambda: FakeEngine()))
    res = tts.speak("వీడ్కోలు", language="te")
    assert res["success"] is False
    assert res["audio_file"] is None
    assert "voice" in res["error"]
