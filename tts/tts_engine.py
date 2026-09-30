"""
Local Text-to-Speech (TTS) Engine.
Implements an abstract interface `TTSEngine` using 100% offline local synthesis (pyttsx3 / Piper TTS)
with zero paid APIs, subscription dependencies, or cloud inference.
"""

import os
import sys
import uuid
from abc import ABC, abstractmethod

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR = os.path.join(PROJECT_ROOT, "tts", "cache")


class TTSEngine(ABC):
    """Abstract interface for offline Text-to-Speech synthesis."""

    @abstractmethod
    def speak(self, text: str, language: str = "en") -> dict:
        """
        Synthesizes text into spoken audio.

        Args:
            text: Sentence or word to synthesize.
            language: ISO language code ('en', 'te', 'hi', 'ta', etc.)

        Returns:
            dict containing:
                - 'success': bool
                - 'text': str
                - 'audio_file': str or None (relative path to cached audio file)
                - 'engine': str
        """
        pass


class LocalOfflineTTSEngine(TTSEngine):
    """Local offline TTS engine powered by pyttsx3 / system speech synthesizer."""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.engine_name = "pyttsx3_offline"

    def speak(self, text: str, language: str = "en") -> dict:
        if not text or not text.strip():
            return {"success": False, "text": "", "audio_file": None, "engine": self.engine_name, "error": "Empty text"}

        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_filepath = os.path.join(self.cache_dir, filename)

        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            
            # Save audio to file for web streaming
            engine.save_to_file(text, output_filepath)
            engine.runAndWait()

            return {
                "success": True,
                "text": text,
                "language": language,
                "audio_file": f"/static/tts_cache/{filename}",
                "engine": self.engine_name
            }
        except Exception as e:
            print(f"[TTS Error] Local synthesis error: {e}")
            return {
                "success": False,
                "text": text,
                "language": language,
                "audio_file": None,
                "engine": self.engine_name,
                "error": str(e)
            }


# Global instance
tts_engine = LocalOfflineTTSEngine()


if __name__ == "__main__":
    res = tts_engine.speak("Hello, Indian Sign Language real-time MVP is online!")
    print("TTS test output:", res)
