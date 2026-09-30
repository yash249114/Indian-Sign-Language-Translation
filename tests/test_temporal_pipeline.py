"""
Unit Tests for Temporal Word Recognition, ISL Grammar, and Dataset Collection.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.hand_tracking.multi_hand_tracker import MultiHandTracker
from vision.temporal.segmenter import TemporalSegmenter, TemporalState
from vision.temporal.word_predictor import WordSignPredictor
from backend.app.isl_grammar import isl_grammar_engine, GlossToken
from backend.app.collector import DatasetCollector
from backend.app.translation import translator


def test_multi_hand_tracker_output_shape():
    """Verifies that MultiHandTracker produces a 126D feature vector."""
    tracker = MultiHandTracker()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(blank_frame)

    assert "detected" in res
    assert "feature_vector" in res
    assert res["feature_vector"].shape == (126,)
    assert res["detected"] is False
    tracker.close()


def test_temporal_segmenter_lifecycle():
    """Tests the complete state machine transitions of TemporalSegmenter."""
    segmenter = TemporalSegmenter(window_size=5, motion_threshold=0.01, confidence_threshold=0.7)
    assert segmenter.state == TemporalState.IDLE

    # Dummy predictor returning HELLO with 0.95 confidence
    def mock_predictor(win):
        return "HELLO", 0.95, [{"gloss": "HELLO", "confidence": 0.95}]

    # Step 1: Initial resting frame then frame with high motion -> SIGN_START -> COLLECTING
    feat0 = np.zeros(126, dtype=np.float32)
    segmenter.update(feat0, detected=True, predictor_fn=mock_predictor)
    feat1 = np.ones(126, dtype=np.float32) * 0.1
    s1 = segmenter.update(feat1, detected=True, predictor_fn=mock_predictor)
    assert s1["state"] in (TemporalState.SIGN_START.value, TemporalState.COLLECTING.value)

    # Step 2: Feed consecutive frames until buffer fills (5 frames)
    for _ in range(6):
        feat = np.ones(126, dtype=np.float32) * 0.15
        s = segmenter.update(feat, detected=True, predictor_fn=mock_predictor)

    # Should commit HELLO
    assert segmenter.last_committed_gloss == "HELLO"
    assert s["state"] in (TemporalState.COMMIT.value, TemporalState.COOLDOWN.value)

    # Reset
    segmenter.reset()
    assert segmenter.state == TemporalState.IDLE


def test_word_predictor_inference():
    """Verifies WordSignPredictor takes a (30, 126) window and outputs top-3 classes."""
    predictor = WordSignPredictor(architecture="tcn")
    dummy_window = np.random.randn(30, 126).astype(np.float32)
    gloss, conf, top3 = predictor.predict_window(dummy_window)

    assert isinstance(gloss, str)
    assert 0.0 <= conf <= 1.0
    assert len(top3) == 3
    assert all("gloss" in item and "confidence" in item for item in top3)


def test_isl_grammar_canonical_rules():
    """Tests rule-based translation of canonical ISL sequences."""
    # Test 1: GREETING
    res1 = isl_grammar_engine.interpret_sequence([GlossToken(gloss="HELLO")])
    assert res1.is_canonical_isl is True
    assert "Hello" in res1.interpreted_english

    # Test 2: TOPIC + OBJECT NEED (ME WATER)
    res2 = isl_grammar_engine.interpret_sequence([GlossToken(gloss="ME"), GlossToken(gloss="WATER")])
    assert res2.is_canonical_isl is True
    assert res2.interpreted_english == "I want water."
    assert res2.uncertainty_warning is None

    # Test 3: TOPIC + IDENTITY (YOU NAME)
    res3 = isl_grammar_engine.interpret_sequence([GlossToken(gloss="YOU"), GlossToken(gloss="NAME")])
    assert res3.is_canonical_isl is True
    assert res3.interpreted_english == "What is your name?"


def test_isl_grammar_uncertainty_flagging():
    """Verifies that uncatalogued or low-confidence sequences trigger uncertainty warnings."""
    # Low confidence token
    res = isl_grammar_engine.interpret_sequence([
        GlossToken(gloss="ME", confidence=0.4),
        GlossToken(gloss="GOOD", confidence=0.5)
    ])
    assert res.uncertainty_warning is not None


def test_dataset_collector_signer_splits(tmp_path):
    """Verifies signer-independent dataset partitioning."""
    manifest_file = os.path.join(str(tmp_path), "test_manifest.json")
    col = DatasetCollector(manifest_file=manifest_file)

    # Signer 01 and 02 must map to train
    assert col.assign_split("signer_01") == "train"
    assert col.assign_split("signer_02") == "train"
    # Signer 03 must map to val
    assert col.assign_split("signer_03") == "val"
    # Signer 04 must map to test
    assert col.assign_split("signer_04") == "test"


def test_gloss_sequence_translation_offline_fallback():
    """Tests offline linguistic translation of gloss sequences."""
    res = translator.translate_gloss_sequence(["ME", "WATER"], target_language="te")
    assert res["status"] in ("success", "offline_fallback", "cached")
    assert "recognized_glosses" in res
    assert res["recognized_glosses"] == ["ME", "WATER"]
    assert len(res["natural_english"]) > 0
