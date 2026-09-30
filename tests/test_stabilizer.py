"""
Unit tests for Temporal Stabilizer and State Machine.
Verifies consensus voting, single commit on continuous hold, and neutral resets.
"""

import pytest
from backend.app.stabilizer import TemporalStabilizer, RecognitionState


def test_stabilizer_majority_vote():
    """Validates that a sign is stabilized only when it achieves consensus."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60)

    # Send 2 'A's -> not consensus yet (2/6 = 0.33 < 0.60)
    res1 = stabilizer.update("A", 0.85)
    res2 = stabilizer.update("A", 0.85)
    assert res1["stable_sign"] is None
    assert res2["stable_sign"] is None

    # Send 3rd and 4th 'A' -> 4 out of 6 is 66.7% >= 60%
    res3 = stabilizer.update("A", 0.90)
    res4 = stabilizer.update("A", 0.90)
    assert res4["stable_sign"] == "A"
    assert res4["is_new_emission"] is True
    assert res4["state"] in ("COMMIT_ONCE", "LOCKED_HOLD")


def test_confidence_threshold_filtering():
    """Validates that low-confidence frames are filtered out as noise."""
    stabilizer = TemporalStabilizer(window_size=4, confidence_threshold=0.75)

    # 10 frames with low confidence
    for _ in range(10):
        res = stabilizer.update("B", 0.60)
        assert res["stable_sign"] is None
        assert res["is_new_emission"] is False


def test_debouncing_and_single_emission_on_hold():
    """Validates that holding a sign continuously for many frames emits EXACTLY ONE character."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60)

    emissions = []
    # Simulate holding 'C' for 60 frames (~2 seconds at 30 FPS)
    for _ in range(60):
        res = stabilizer.update("C", 0.92)
        if res["is_new_emission"]:
            emissions.append(res["stable_sign"])

    # Must produce exactly 1 emission
    assert len(emissions) == 1
    assert emissions[0] == "C"
    assert stabilizer.state == RecognitionState.LOCKED_HOLD


def test_neutral_reset_behavior():
    """Validates that returning to neutral (no hand) unlocks the gesture for subsequent emissions."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60, neutral_reset_frames=3)

    # 1. Hold 'A' for 10 frames -> emits 'A' once
    emissions = []
    for _ in range(10):
        res = stabilizer.update("A", 0.90)
        if res["is_new_emission"]:
            emissions.append(res["stable_sign"])
    assert emissions == ["A"]

    # 2. Return to neutral (no hand) for 3 frames
    for _ in range(3):
        res = stabilizer.update(None, 0.0)
        assert res["is_new_emission"] is False
    assert stabilizer.state == RecognitionState.NO_SIGN

    # 3. Hold 'A' again for 10 frames -> must emit second 'A'
    for _ in range(10):
        res = stabilizer.update("A", 0.90)
        if res["is_new_emission"]:
            emissions.append(res["stable_sign"])
    assert emissions == ["A", "A"]
