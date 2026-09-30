"""
Live Text Buffer & Word Builder Module.
Accumulates recognized signs into words and sentences with full editing controls.
"""


class TextBuffer:
    """Manages the real-time recognized character stream, words, and sentence buffer."""

    def __init__(self):
        self.current_word = ""
        self.words = []

    def append_character(self, char):
        """Appends a recognized sign character."""
        if not char:
            return self.get_full_text()
        self.current_word += str(char)
        return self.get_full_text()

    def add_space(self):
        """Commits current word and adds a space boundary."""
        if self.current_word:
            self.words.append(self.current_word)
            self.current_word = ""
        return self.get_full_text()

    def backspace(self):
        """Removes the last character or trailing word."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.words:
            last_word = self.words.pop()
            self.current_word = last_word[:-1]
        return self.get_full_text()

    def clear(self):
        """Clears all words and current word."""
        self.current_word = ""
        self.words = []
        return ""

    def get_full_text(self):
        """Returns full assembled sentence string."""
        if not self.words and not self.current_word:
            return ""
        all_words = list(self.words)
        if self.current_word:
            all_words.append(self.current_word)
        return " ".join(all_words)

    def get_state(self):
        """Returns structured buffer state."""
        return {
            "current_word": self.current_word,
            "words": self.words,
            "full_text": self.get_full_text(),
            "character_count": len(self.get_full_text())
        }
