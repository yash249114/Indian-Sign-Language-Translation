"""
Unit tests for Landmark Normalizer.
Verifies wrist-relative translation invariance, scale invariance, and 63D feature shape.
"""

import numpy as np
import pytest
from vision.preprocessing.normalizer import LandmarkNormalizer


def test_normalizer_output_shape():
    """Validates that normalizer produces exactly 63D output for 21 3D landmarks."""
    normalizer = LandmarkNormalizer()
    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)
    normalized = normalizer.normalize_landmarks(dummy_landmarks)

    assert normalized.shape == (63,)
    assert normalized.dtype == np.float32


def test_translation_invariance():
    """Validates that translating all coordinates preserves normalized feature vector."""
    normalizer = LandmarkNormalizer()
    base_lms = np.random.rand(21, 3).astype(np.float32) * 100.0
    shift_vector = np.array([25.0, -40.0, 15.0], dtype=np.float32)
    shifted_lms = base_lms + shift_vector

    norm_base = normalizer.normalize_landmarks(base_lms)
    norm_shifted = normalizer.normalize_landmarks(shifted_lms)

    np.testing.assert_allclose(norm_base, norm_shifted, atol=1e-5)


def test_scale_invariance():
    """Validates that scaling the hand preserves normalized feature vector."""
    normalizer = LandmarkNormalizer()
    base_lms = np.random.rand(21, 3).astype(np.float32) * 50.0
    scaled_lms = base_lms * 3.5

    norm_base = normalizer.normalize_landmarks(base_lms)
    norm_scaled = normalizer.normalize_landmarks(scaled_lms)

    np.testing.assert_allclose(norm_base, norm_scaled, atol=1e-5)


def test_zero_division_safety():
    """Validates that identical/collapsed points do not throw ZeroDivisionError."""
    normalizer = LandmarkNormalizer()
    collapsed_lms = np.zeros((21, 3), dtype=np.float32)
    normalized = normalizer.normalize_landmarks(collapsed_lms)

    assert normalized.shape == (63,)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
