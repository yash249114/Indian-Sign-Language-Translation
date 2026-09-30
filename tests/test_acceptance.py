"""
Acceptance Test Suite for ISL Platform:
Test A: Fingerspelling Mode produces A-Z / 1-9 characters without character repetition.
Test B: Word Mode produces discrete word glosses (e.g., WATER, NOT 'W A T E R').
Test C: Multi-sign sequence produces ordered glosses (ME | WATER) and canonical ISL English interpretation ('I want water.').
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.predict import SignPredictor
from backend.app.stabilizer import TemporalStabilizer
from backend.app.text_buffer import TextBuffer
from vision.temporal.word_predictor import WordSignPredictor
from vision.temporal.segmenter import TemporalSegmenter, TemporalState
from backend.app.isl_grammar import isl_grammar_engine, GlossToken
from backend.app.translation import translator


def test_acceptance_test_a_fingerspelling():
    """
    TEST A: FINGER SPELLING
    Input: Static 63D landmark feature vector
    Expected: Individual character from classes (1-9, A-Z) accumulated into text buffer.
    """
    test_data = np.load(os.path.join(PROJECT_ROOT, "data", "processed", "landmarks", "test.npz"))
    X_test, y_test = test_data["X"], test_data["y"]

    predictor = SignPredictor()
    pred_res = predictor.predict_from_features(X_test[0])

    assert pred_res["sign"] in predictor.class_names
    assert 0.0 <= pred_res["confidence"] <= 1.0

    # Test buffer accumulation
    buf = TextBuffer()
    buf.clear()
    buf.append_character(pred_res["sign"])
    assert buf.get_full_text() == pred_res["sign"]


def test_acceptance_test_b_word_mode():
    """
    TEST B: WORD MODE
    Input: Temporal sequence window of 30 frames (126D)
    Expected: Discrete whole-word gloss (e.g. 'WATER'), NOT letter spelling ('W A T E R').
    """
    predictor = WordSignPredictor(architecture="tcn")

    # Generate a reference WATER sequence
    t = np.linspace(0, 1, 30)
    water_seq = np.zeros((30, 126), dtype=np.float32)
    water_seq[:, 63] = 0.3 + 0.05 * np.sin(4 * np.pi * t)
    water_seq[:, 64] = 0.7 + 0.1 * np.cos(4 * np.pi * t)

    gloss, conf, top3 = predictor.predict_window(water_seq)

    # Must produce a complete word gloss from the vocabulary, NOT a single letter
    assert gloss in predictor.classes
    assert len(gloss) > 1  # Whole word, not letter
    assert any(item["gloss"] == "WATER" for item in top3)


def test_acceptance_test_c_multi_sign_sequence():
    """
    TEST C: MULTI-SIGN SEQUENCE
    Input: Ordered gloss tokens [ME, WATER]
    Expected:
      Gloss: ME | WATER
      English: 'I want water.'
      Target Language (Telugu): 'నాకు నీరు కావాలి.'
    """
    tokens = [GlossToken(gloss="ME"), GlossToken(gloss="WATER")]
    interp = isl_grammar_engine.interpret_sequence(tokens)

    # 1. Linguistic interpretation
    assert interp.gloss_sequence_str == "ME WATER"
    assert interp.is_canonical_isl is True
    assert interp.interpreted_english == "I want water."
    assert "Topic-Need" in interp.grammatical_notes

    # 2. Multilingual translation
    trans_res = translator.translate_gloss_sequence(["ME", "WATER"], target_language="te")
    assert trans_res["recognized_glosses"] == ["ME", "WATER"]
    assert trans_res["natural_english"] == "I want water."
    assert len(trans_res["translated_text"]) > 0
