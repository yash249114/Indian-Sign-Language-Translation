"""
Comprehensive Regression Test Suite for Recognition State Machine & Text Buffer.
Tests strict requirements:
1. 'A' held for 2 seconds -> exactly one 'A'
2. 'B' held for 2 seconds -> exactly one 'B'
3. 'A' -> neutral -> 'A' -> 'AA'
4. 'A' -> 'B' -> neutral -> 'AB'
5. No hand -> no new character
6. Low confidence -> no character
"""

import pytest
from backend.app.stabilizer import TemporalStabilizer, RecognitionState
from backend.app.text_buffer import TextBuffer


def test_regression_case_1_a_held_2_seconds():
    """Requirement: 'A' held for 2 seconds (60 frames at 30 FPS) -> exactly ONE 'A'."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60)
    text_buffer = TextBuffer()

    for _ in range(60):
        res = stabilizer.update("A", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == "A"
    assert text_buffer.get_state()["character_count"] == 1


def test_regression_case_2_b_held_2_seconds():
    """Requirement: 'B' held for 2 seconds -> exactly ONE 'B'."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60)
    text_buffer = TextBuffer()

    for _ in range(60):
        res = stabilizer.update("B", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == "B"
    assert text_buffer.get_state()["character_count"] == 1


def test_regression_case_3_a_neutral_a_produces_aa():
    """Requirement: 'A' -> neutral -> 'A' -> produces 'AA'."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60, neutral_reset_frames=3)
    text_buffer = TextBuffer()

    # Step 1: Hold 'A' for 30 frames
    for _ in range(30):
        res = stabilizer.update("A", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])
    assert text_buffer.get_full_text() == "A"

    # Step 2: Return to neutral (no hand) for 5 frames
    for _ in range(5):
        res = stabilizer.update(None, 0.0)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])
    assert text_buffer.get_full_text() == "A"

    # Step 3: Hold 'A' again for 30 frames
    for _ in range(30):
        res = stabilizer.update("A", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == "AA"
    assert text_buffer.get_state()["character_count"] == 2


def test_regression_case_4_a_to_b_to_neutral_produces_ab():
    """Requirement: 'A' -> 'B' -> neutral -> produces 'AB'."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75, min_consensus_ratio=0.60, neutral_reset_frames=3)
    text_buffer = TextBuffer()

    # Step 1: Hold 'A'
    for _ in range(20):
        res = stabilizer.update("A", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])
    assert text_buffer.get_full_text() == "A"

    # Step 2: Transition to 'B'
    for _ in range(20):
        res = stabilizer.update("B", 0.95)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])
    assert text_buffer.get_full_text() == "AB"

    # Step 3: Return to neutral
    for _ in range(5):
        res = stabilizer.update(None, 0.0)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == "AB"
    assert text_buffer.get_state()["character_count"] == 2


def test_regression_case_5_no_hand_no_character():
    """Requirement: No hand (None, 0.0) -> no character appended."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75)
    text_buffer = TextBuffer()

    for _ in range(50):
        res = stabilizer.update(None, 0.0)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == ""
    assert text_buffer.get_state()["character_count"] == 0


def test_regression_case_6_low_confidence_no_character():
    """Requirement: Low confidence (e.g. 0.60 < 0.75) -> no character appended."""
    stabilizer = TemporalStabilizer(window_size=6, confidence_threshold=0.75)
    text_buffer = TextBuffer()

    for _ in range(50):
        res = stabilizer.update("A", 0.60)
        if res["is_new_emission"] and res["stable_sign"]:
            text_buffer.append_character(res["stable_sign"])

    assert text_buffer.get_full_text() == ""
    assert text_buffer.get_state()["character_count"] == 0
