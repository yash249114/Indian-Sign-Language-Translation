"""
Unit tests for Sign Classifier Inference Engine.
Verifies model prediction, top-3 candidates, and confidence bounds.
"""

import numpy as np
import pytest
from training.predict import SignPredictor


def test_predictor_initialization():
    """Validates predictor loads model and 35 classes."""
    predictor = SignPredictor()
    assert len(predictor.class_names) == 35
    assert "A" in predictor.class_names
    assert "1" in predictor.class_names


def test_prediction_output_structure():
    """Validates predictor returns structured predictions."""
    predictor = SignPredictor()
    dummy_feat = np.random.randn(63).astype(np.float32)
    res = predictor.predict_from_features(dummy_feat)

    assert "sign" in res
    assert "confidence" in res
    assert "class_index" in res
    assert "top3" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top3"]) == 3
    assert res["sign"] in predictor.class_names
