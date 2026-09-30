"""
Unit tests for Text Buffer and Word Builder.
Verifies character appending, word boundary commits, backspacing, and clear operations.
"""

import pytest
from backend.app.text_buffer import TextBuffer


def test_buffer_character_accumulation():
    """Validates sequential character appending."""
    tb = TextBuffer()
    assert tb.get_full_text() == ""

    tb.append_character("H")
    tb.append_character("E")
    tb.append_character("L")
    tb.append_character("L")
    tb.append_character("O")

    assert tb.get_full_text() == "HELLO"
    assert tb.current_word == "HELLO"


def test_buffer_space_word_commit():
    """Validates space creates word boundaries."""
    tb = TextBuffer()
    tb.append_character("H")
    tb.append_character("I")
    tb.add_space()

    assert tb.words == ["HI"]
    assert tb.current_word == ""

    tb.append_character("A")
    tb.append_character("L")
    tb.append_character("L")

    assert tb.get_full_text() == "HI ALL"


def test_buffer_backspace_and_clear():
    """Validates backspace across characters/words and clearing."""
    tb = TextBuffer()
    tb.append_character("A")
    tb.append_character("B")
    tb.backspace()
    assert tb.get_full_text() == "A"

    tb.add_space()
    tb.append_character("C")
    assert tb.get_full_text() == "A C"

    tb.backspace()
    assert tb.get_full_text() == "A"

    tb.clear()
    assert tb.get_full_text() == ""
    assert len(tb.words) == 0
