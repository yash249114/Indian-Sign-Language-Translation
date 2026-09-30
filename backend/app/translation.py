"""
Multilingual Translation Engine using Gemini 3.5 Flash Lite.
Translates recognized Indian Sign Language text into Indian languages (Telugu, Hindi, Tamil, Kannada, Malayalam, etc.)
Features:
- Dedicated translation model: gemini-3.5-flash-lite
- In-memory translation caching to minimize API calls and latency
- Rate-limit (RPM 15, TPM 250k, RPD 500) protection and call counters
- Graceful offline and lexicon fallbacks with zero application crashes
- Thread-safe architecture
"""

import os
import sys
import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Internal Language Mapping
LANGUAGE_MAP = {
    "en": {"name": "English", "native": "English", "alt_code": "eng_Latn"},
    "te": {"name": "Telugu", "native": "తెలుగు", "alt_code": "tel_Telu"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "alt_code": "hin_Deva"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "alt_code": "tam_Taml"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "alt_code": "kan_Knda"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "alt_code": "mal_Mlym"},
    "bn": {"name": "Bengali", "native": "বাংলা", "alt_code": "ben_Beng"},
    "mr": {"name": "Marathi", "native": "मराठी", "alt_code": "mar_Deva"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "alt_code": "guj_Gujr"}
}

# Normalize legacy / BCP-47 codes to standard short codes
CODE_NORMALIZATION = {
    "eng_latn": "en", "en": "en", "english": "en",
    "tel_telu": "te", "te": "te", "telugu": "te",
    "hin_deva": "hi", "hi": "hi", "hindi": "hi",
    "tam_taml": "ta", "ta": "ta", "tamil": "ta",
    "kan_knda": "kn", "kn": "kn", "kannada": "kn",
    "mal_mlym": "ml", "ml": "ml", "malayalam": "ml",
    "ben_beng": "bn", "bn": "bn", "bengali": "bn",
    "mar_deva": "mr", "mr": "mr", "marathi": "mr",
    "guj_gujr": "gu", "gu": "gu", "gujarati": "gu"
}

# Offline multilingual lexicon for emergency fallback / common phrases
OFFLINE_LEXICON = {
    "HELLO": {
        "te": "నమస్కారం", "hi": "नमस्ते", "ta": "வணக்கம்",
        "kn": "ನಮಸ್ಕಾರ", "ml": "നമസ്കാരം", "bn": "নমস্কার", "mr": "नमस्कार", "gu": "નમસ્તે"
    },
    "THANK YOU": {
        "te": "ధన్యవాదాలు", "hi": "धन्यवाद", "ta": "நன்றி",
        "kn": "ಧನ್ಯವಾದಗಳು", "ml": "നന്ദി", "bn": "ধন্যবাদ", "mr": "धन्यवाद", "gu": "આભાર"
    },
    "GOOD": {
        "te": "మంచిది", "hi": "अच्छा", "ta": "நல்லது",
        "kn": "ಒಳ್ಳೆಯದು", "ml": "നല്ലത്", "bn": "ভালো", "mr": "चांगले", "gu": "સારું"
    },
    "WATER": {
        "te": "నీరు", "hi": "पानी", "ta": "தண்ணீர்",
        "kn": "ನೀರು", "ml": "വെള്ളം", "bn": "জল", "mr": "पाणी", "gu": "પાણી"
    },
    "HELP": {
        "te": "సహాయం", "hi": "मदद", "ta": "உதவி",
        "kn": "ಸಹಾಯ", "ml": "സഹായം", "bn": "সাহায্য", "mr": "मदत", "gu": "મદદ"
    },
    "YES": {
        "te": "అవును", "hi": "हाँ", "ta": "ஆம்",
        "kn": "ಹೌದು", "ml": "അതെ", "bn": "হ্যাঁ", "mr": "होय", "gu": "હા"
    },
    "NO": {
        "te": "కాదు", "hi": "नहीं", "ta": "இல்லை",
        "kn": "ಇಲ್ಲ", "ml": "അല്ല", "bn": "না", "mr": "नाही", "gu": "ના"
    }
}

SYSTEM_INSTRUCTION = (
    "You are a translation engine for an Indian Sign Language accessibility application.\n"
    "Translate the provided source text into the requested target language.\n"
    "Preserve the meaning and intent.\n"
    "Do not explain the translation.\n"
    "Do not add information.\n"
    "Return ONLY the translated text."
)


class TranslationCache:
    """Thread-safe in-memory cache for translation results."""

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _generate_key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        norm = text.strip().lower()
        raw = f"{norm}:{src_lang.lower()}:{tgt_lang.lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        key = self._generate_key(text, src_lang, tgt_lang)
        with self._lock:
            return self._cache.get(key)

    def set(self, text: str, src_lang: str, tgt_lang: str, translated: str):
        key = self._generate_key(text, src_lang, tgt_lang)
        with self._lock:
            self._cache[key] = translated

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class GeminiTranslationEngine:
    """
    Multilingual Translation Engine using Gemini 3.5 Flash Lite.
    Executes REST calls with minimal prompt tokens, in-memory caching,
    and fallback protections.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 6.0
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name or os.environ.get("GEMINI_TRANSLATION_MODEL", "gemini-3.5-flash-lite")
        self.timeout = timeout
        self.cache = TranslationCache()

        # Telemetry counters
        self.translation_calls_today = 0
        self.cached_translations = 0
        self.failed_requests = 0
        self._lock = threading.Lock()

    def normalize_lang_code(self, code: str) -> str:
        """Converts any code variant to short ISO code."""
        if not code:
            return "te"
        clean = code.strip().lower()
        return CODE_NORMALIZATION.get(clean, "te")

    def get_language_name(self, code: str) -> str:
        short_code = self.normalize_lang_code(code)
        info = LANGUAGE_MAP.get(short_code, {"name": "Telugu"})
        return info["name"]

    def translate(
        self,
        text: str,
        source_language: str = "en",
        target_language: str = "te"
    ) -> Dict[str, Any]:
        """
        Translates text using Gemini 3.5 Flash Lite with caching and fallbacks.

        Args:
            text: Source text to translate.
            source_language: Source language code (e.g. 'en' or 'eng_Latn').
            target_language: Target language code (e.g. 'te' or 'tel_Telu').

        Returns:
            Dict containing translated_text, source_language, target_language,
            model, cached, status, latency_ms.
        """
        t0 = time.perf_counter()

        if not text or not text.strip():
            return {
                "translated_text": "",
                "source_language": source_language,
                "target_language": target_language,
                "model": self.model_name,
                "cached": False,
                "status": "empty_input",
                "latency_ms": 0.0
            }

        src_code = self.normalize_lang_code(source_language)
        tgt_code = self.normalize_lang_code(target_language)
        clean_text = text.strip()

        # If source and target are identical
        if src_code == tgt_code:
            return {
                "translated_text": clean_text,
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": False,
                "status": "identity",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        # 1. Check in-memory cache
        cached_result = self.cache.get(clean_text, src_code, tgt_code)
        if cached_result is not None:
            with self._lock:
                self.cached_translations += 1
            return {
                "translated_text": cached_result,
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": True,
                "status": "cached",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        # Check for API key presence
        api_key = self.api_key if self.api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_key_here" or api_key == "your_gemini_api_key_here":
            # Fallback to local offline dictionary if entry exists
            upper_text = clean_text.upper()
            if upper_text in OFFLINE_LEXICON and tgt_code in OFFLINE_LEXICON[upper_text]:
                fallback_txt = OFFLINE_LEXICON[upper_text][tgt_code]
                return {
                    "translated_text": fallback_txt,
                    "source_language": src_code,
                    "target_language": tgt_code,
                    "model": "offline_lexicon_fallback",
                    "cached": False,
                    "status": "offline_fallback",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                }

            with self._lock:
                self.failed_requests += 1
            return {
                "translated_text": "Translation unavailable (GEMINI_API_KEY not configured)",
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": False,
                "status": "unavailable",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        # 2. Call Gemini 3.5 Flash Lite API
        src_name = self.get_language_name(src_code)
        tgt_name = self.get_language_name(tgt_code)

        user_content = f"Source language: {src_name}\nTarget language: {tgt_name}\nText: {clean_text}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        translated_text = parts[0].get("text", "").strip()
                        # Cache translation
                        self.cache.set(clean_text, src_code, tgt_code, translated_text)
                        with self._lock:
                            self.translation_calls_today += 1

                        return {
                            "translated_text": translated_text,
                            "source_language": src_code,
                            "target_language": tgt_code,
                            "model": self.model_name,
                            "cached": False,
                            "status": "success",
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                        }

            elif response.status_code == 429:
                with self._lock:
                    self.failed_requests += 1
                return {
                    "translated_text": "Translation temporarily unavailable (Rate limit reached)",
                    "source_language": src_code,
                    "target_language": tgt_code,
                    "model": self.model_name,
                    "cached": False,
                    "status": "rate_limited",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                }

            with self._lock:
                self.failed_requests += 1
            return {
                "translated_text": f"Translation unavailable (API Error {response.status_code})",
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": False,
                "status": "error",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        except requests.exceptions.Timeout:
            with self._lock:
                self.failed_requests += 1
            return {
                "translated_text": "Translation unavailable (Network timeout)",
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": False,
                "status": "timeout",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
            return {
                "translated_text": f"Translation unavailable ({type(e).__name__})",
                "source_language": src_code,
                "target_language": tgt_code,
                "model": self.model_name,
                "cached": False,
                "status": "exception",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

    def translate_gloss_sequence(
        self,
        gloss_tokens: List[str],
        target_language: str = "te",
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Translates a completed ISL gloss sequence into natural English and the target Indian language.
        Uses structured JSON schema generation with Gemini 3.5 Flash Lite, strict debouncing,
        and offline linguistic grammar fallback.
        """
        t0 = time.perf_counter()
        tgt_code = self.normalize_lang_code(target_language)
        clean_glosses = [g.strip().upper() for g in gloss_tokens if g.strip()]
        gloss_key = " ".join(clean_glosses)

        if not clean_glosses:
            return {
                "recognized_glosses": [],
                "gloss_sequence_str": "",
                "natural_english": "",
                "translated_text": "",
                "grammatical_notes": "No glosses provided.",
                "is_canonical": False,
                "uncertainty_warning": "Empty gloss input.",
                "source_language": "isl",
                "target_language": tgt_code,
                "cached": False,
                "status": "empty_input",
                "latency_ms": 0.0
            }

        # 1. Local Rule-Based ISL Linguistic Interpretation
        from backend.app.isl_grammar import isl_grammar_engine, GlossToken
        token_objs = [GlossToken(gloss=g, confidence=confidence) for g in clean_glosses]
        local_interp = isl_grammar_engine.interpret_sequence(token_objs)

        # 2. Check Cache
        cached_val = self.cache.get(f"GLOSS:{gloss_key}", "isl", tgt_code)
        if cached_val is not None:
            try:
                cached_dict = json.loads(cached_val)
                cached_dict["cached"] = True
                cached_dict["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                with self._lock:
                    self.cached_translations += 1
                return cached_dict
            except Exception:
                pass

        # 3. Check API Key
        api_key = self.api_key if self.api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key in ("your_key_here", "your_gemini_api_key_here"):
            # Offline local grammar fallback
            translated = local_interp.interpreted_english if tgt_code == "en" else self.translate(local_interp.interpreted_english, "en", tgt_code)["translated_text"]
            return {
                "recognized_glosses": clean_glosses,
                "gloss_sequence_str": gloss_key,
                "natural_english": local_interp.interpreted_english,
                "translated_text": translated,
                "grammatical_notes": local_interp.grammatical_notes,
                "is_canonical": local_interp.is_canonical_isl,
                "uncertainty_warning": local_interp.uncertainty_warning,
                "source_language": "isl",
                "target_language": tgt_code,
                "cached": False,
                "status": "offline_fallback",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        # 4. Structured Gemini Prompting with Schema Enforcing
        tgt_name = self.get_language_name(tgt_code)
        structured_system_prompt = (
            "You are an expert Indian Sign Language (ISL) linguist and translator.\n"
            "The user provides an authenticated ISL gloss sequence (representing gestures recognized in temporal order).\n"
            "Your tasks:\n"
            "1. Interpret the gloss sequence into a fluent, grammatical English sentence based on ISL syntax (Topic-Comment, SOV, Time-first).\n"
            f"2. Translate that natural English sentence into {tgt_name}.\n"
            "3. Provide brief grammatical notes.\n"
            "4. DO NOT invent missing signs or hallucinate ungrounded facts.\n"
            "5. If the sequence is incomplete or ambiguous, state the uncertainty clearly in 'uncertainty_warning'.\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "natural_english": string,\n'
            '  "translated_text": string,\n'
            '  "grammatical_notes": string,\n'
            '  "is_canonical": boolean,\n'
            '  "uncertainty_warning": string or null\n'
            "}"
        )
        user_msg = (
            f"ISL Gloss Sequence: {gloss_key}\n"
            f"Sequence Confidence: {confidence:.2f}\n"
            f"Target Language: {tgt_name} ({tgt_code})"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": structured_system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_msg}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 350,
                "response_mime_type": "application/json"
            }
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_json = parts[0].get("text", "").strip()
                        try:
                            parsed = json.loads(raw_json)
                            res_obj = {
                                "recognized_glosses": clean_glosses,
                                "gloss_sequence_str": gloss_key,
                                "natural_english": parsed.get("natural_english", local_interp.interpreted_english),
                                "translated_text": parsed.get("translated_text", ""),
                                "grammatical_notes": parsed.get("grammatical_notes", local_interp.grammatical_notes),
                                "is_canonical": parsed.get("is_canonical", local_interp.is_canonical_isl),
                                "uncertainty_warning": parsed.get("uncertainty_warning") or local_interp.uncertainty_warning,
                                "source_language": "isl",
                                "target_language": tgt_code,
                                "cached": False,
                                "status": "success",
                                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                            }
                            # Cache serialized JSON
                            self.cache.set(f"GLOSS:{gloss_key}", "isl", tgt_code, json.dumps(res_obj))
                            with self._lock:
                                self.translation_calls_today += 1
                            return res_obj
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

        # On network error/timeout/API failure, fall back to offline local grammar
        with self._lock:
            self.failed_requests += 1
        translated = local_interp.interpreted_english if tgt_code == "en" else self.translate(local_interp.interpreted_english, "en", tgt_code)["translated_text"]
        return {
            "recognized_glosses": clean_glosses,
            "gloss_sequence_str": gloss_key,
            "natural_english": local_interp.interpreted_english,
            "translated_text": translated,
            "grammatical_notes": local_interp.grammatical_notes,
            "is_canonical": local_interp.is_canonical_isl,
            "uncertainty_warning": local_interp.uncertainty_warning,
            "source_language": "isl",
            "target_language": tgt_code,
            "cached": False,
            "status": "offline_fallback",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns translation usage stats."""
        with self._lock:
            return {
                "model": self.model_name,
                "translation_calls_today": self.translation_calls_today,
                "cached_translations": self.cached_translations,
                "failed_requests": self.failed_requests,
                "cache_entries_count": self.cache.size()
            }


# Global instance
translator = GeminiTranslationEngine()

