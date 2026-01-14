"""
Tests for the shared alignment types module.

These tests verify that:
1. WordTimestamp correctly calculates duration
2. AlignmentResult groups words into phrases correctly
3. Serialization/deserialization works correctly
4. SubtitleStyle generates valid ASS format
"""

import pytest

from src.pipeline.alignment_types import (
    AlignmentResult,
    SubtitleStyle,
    WordTimestamp,
)


class TestWordTimestamp:
    """Tests for WordTimestamp dataclass."""

    def test_duration_calculation(self):
        """Test that duration is calculated correctly."""
        word = WordTimestamp(word="hello", start=1.0, end=2.5)
        assert word.duration == 1.5

    def test_duration_zero(self):
        """Test duration when start equals end."""
        word = WordTimestamp(word="a", start=1.0, end=1.0)
        assert word.duration == 0.0

    def test_default_score(self):
        """Test that default score is 1.0."""
        word = WordTimestamp(word="test", start=0.0, end=1.0)
        assert word.score == 1.0

    def test_custom_score(self):
        """Test custom score assignment."""
        word = WordTimestamp(word="test", start=0.0, end=1.0, score=0.85)
        assert word.score == 0.85

    def test_repr(self):
        """Test string representation."""
        word = WordTimestamp(word="hello", start=1.23, end=2.45)
        repr_str = repr(word)
        assert "hello" in repr_str
        assert "1.23" in repr_str
        assert "2.45" in repr_str

    def test_to_dict(self):
        """Test conversion to dictionary."""
        word = WordTimestamp(word="test", start=1.0, end=2.0, score=0.9)
        d = word.to_dict()

        assert d["word"] == "test"
        assert d["start"] == 1.0
        assert d["end"] == 2.0
        assert d["score"] == 0.9

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"word": "hello", "start": 0.5, "end": 1.5, "score": 0.95}
        word = WordTimestamp.from_dict(d)

        assert word.word == "hello"
        assert word.start == 0.5
        assert word.end == 1.5
        assert word.score == 0.95

    def test_from_dict_default_score(self):
        """Test that from_dict uses default score if not provided."""
        d = {"word": "test", "start": 0.0, "end": 1.0}
        word = WordTimestamp.from_dict(d)

        assert word.score == 1.0


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    @pytest.fixture
    def sample_words(self):
        """Create sample word timestamps."""
        return [
            WordTimestamp("Hello", 0.0, 0.5),
            WordTimestamp("world,", 0.6, 1.0),
            WordTimestamp("this", 1.1, 1.3),
            WordTimestamp("is", 1.4, 1.5),
            WordTimestamp("a", 1.6, 1.7),
            WordTimestamp("test.", 1.8, 2.2),
        ]

    @pytest.fixture
    def sample_result(self, sample_words):
        """Create sample alignment result."""
        return AlignmentResult(
            words=sample_words,
            full_text="Hello world, this is a test.",
            duration=2.5,
            language="en",
        )

    def test_get_words_in_range(self, sample_result):
        """Test filtering words by time range."""
        words = sample_result.get_words_in_range(0.0, 1.0)
        assert len(words) == 2
        assert words[0].word == "Hello"
        assert words[1].word == "world,"

    def test_get_words_in_range_empty(self, sample_result):
        """Test empty result when no words in range."""
        words = sample_result.get_words_in_range(10.0, 15.0)
        assert len(words) == 0

    def test_group_into_phrases_default(self, sample_result):
        """Test grouping with default words per group."""
        groups = sample_result.group_into_phrases(words_per_group=3)

        # First group should end at comma (punctuation break)
        assert len(groups) >= 1
        # Verify groups contain WordTimestamp objects
        assert all(isinstance(w, WordTimestamp) for g in groups for w in g)

    def test_group_into_phrases_sentence_break(self):
        """Test that sentence-ending punctuation creates breaks."""
        words = [
            WordTimestamp("First.", 0.0, 0.5),
            WordTimestamp("Second", 0.6, 1.0),
            WordTimestamp("word", 1.1, 1.5),
        ]
        result = AlignmentResult(words=words, full_text="", duration=2.0, language="en")
        groups = result.group_into_phrases(words_per_group=5)

        # Should break after "First." due to period
        assert len(groups) == 2
        assert groups[0][0].word == "First."
        assert groups[1][0].word == "Second"

    def test_len(self, sample_result):
        """Test __len__ returns word count."""
        assert len(sample_result) == 6

    def test_iter(self, sample_result):
        """Test iteration over words."""
        words = list(sample_result)
        assert len(words) == 6
        assert words[0].word == "Hello"

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        d = sample_result.to_dict()

        assert d["full_text"] == "Hello world, this is a test."
        assert d["duration"] == 2.5
        assert d["language"] == "en"
        assert len(d["words"]) == 6
        assert d["words"][0]["word"] == "Hello"

    def test_from_dict(self, sample_result):
        """Test round-trip serialization."""
        d = sample_result.to_dict()
        restored = AlignmentResult.from_dict(d)

        assert restored.full_text == sample_result.full_text
        assert restored.duration == sample_result.duration
        assert restored.language == sample_result.language
        assert len(restored.words) == len(sample_result.words)
        assert restored.words[0].word == sample_result.words[0].word


class TestSubtitleStyle:
    """Tests for SubtitleStyle dataclass."""

    def test_default_values(self):
        """Test default style values."""
        style = SubtitleStyle()

        assert style.font_name == "Arial Black"
        assert style.font_size == 48
        assert style.alignment == 2  # Bottom center
        assert style.margin_v == 50

    def test_to_ass_style(self):
        """Test ASS style line generation."""
        style = SubtitleStyle(font_name="Impact", font_size=64)
        ass_line = style.to_ass_style("MyStyle")

        assert ass_line.startswith("Style: MyStyle,Impact,64,")
        assert "Impact" in ass_line
        assert "64" in ass_line

    def test_to_highlight_style(self):
        """Test ASS highlight style line generation."""
        style = SubtitleStyle(
            font_name="Arial",
            font_size=48,
            highlight_color="&H00FFFF00",  # Cyan
        )
        ass_line = style.to_highlight_style("Highlight")

        assert ass_line.startswith("Style: Highlight,Arial,48,")
        assert "&H00FFFF00" in ass_line

    def test_custom_style(self):
        """Test fully customized style."""
        style = SubtitleStyle(
            font_name="Comic Sans MS",
            font_size=72,
            primary_color="&H00FF0000",
            highlight_color="&H0000FF00",
            outline_color="&H00FFFFFF",
            outline_width=5,
            shadow_depth=3,
            alignment=8,  # Top center
            margin_v=100,
        )

        assert style.font_name == "Comic Sans MS"
        assert style.font_size == 72
        assert style.outline_width == 5
        assert style.alignment == 8
        assert style.margin_v == 100


class TestAlignmentResultEdgeCases:
    """Edge case tests for AlignmentResult."""

    def test_empty_words(self):
        """Test with no words."""
        result = AlignmentResult(words=[], full_text="", duration=0.0, language="en")

        assert len(result) == 0
        assert result.group_into_phrases() == []
        assert result.get_words_in_range(0, 10) == []

    def test_single_word(self):
        """Test with single word."""
        result = AlignmentResult(
            words=[WordTimestamp("Hello", 0.0, 1.0)],
            full_text="Hello",
            duration=1.0,
            language="en",
        )

        groups = result.group_into_phrases(words_per_group=3)
        assert len(groups) == 1
        assert groups[0][0].word == "Hello"

    def test_group_with_question_mark(self):
        """Test grouping breaks on question marks."""
        words = [
            WordTimestamp("What?", 0.0, 0.5),
            WordTimestamp("Yes", 0.6, 0.8),
        ]
        result = AlignmentResult(
            words=words, full_text="What? Yes", duration=1.0, language="en"
        )
        groups = result.group_into_phrases(words_per_group=5)

        assert len(groups) == 2

    def test_group_with_exclamation(self):
        """Test grouping breaks on exclamation marks."""
        words = [
            WordTimestamp("Wow!", 0.0, 0.5),
            WordTimestamp("Amazing", 0.6, 1.0),
        ]
        result = AlignmentResult(
            words=words, full_text="Wow! Amazing", duration=1.0, language="en"
        )
        groups = result.group_into_phrases(words_per_group=5)

        assert len(groups) == 2
