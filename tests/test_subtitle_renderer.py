"""
Tests for the subtitle renderer module.

These tests verify that:
1. SubtitleStyle correctly stores appearance settings
2. SubtitleConfig correctly stores behavior settings
3. SubtitleRenderer creates subtitles based on configuration
4. Karaoke mode vs standard mode work correctly
5. Style adjustments for video format work correctly
6. Text processing utilities work correctly
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.subtitle_renderer import (
    SubtitleConfig,
    SubtitleRenderer,
    SubtitleStyle,
)


class TestSubtitleStyle:
    """Tests for SubtitleStyle dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        style = SubtitleStyle()

        assert style.font_size == 70
        assert style.color == "white"
        assert style.highlight_color == "#FFD700"
        assert style.stroke_color == "black"
        assert style.stroke_width == 4
        assert style.bg_color is None
        assert style.position == ("center", 0.78)
        assert style.max_chars_per_line == 20
        assert style.words_per_group == 4
        assert style.min_word_duration == 0.15

    def test_custom_values(self):
        """Test creating style with custom values."""
        style = SubtitleStyle(
            font_size=80,
            color="yellow",
            highlight_color="#00FF00",
            stroke_color="white",
            stroke_width=6,
            bg_color="black",
            position=("center", 0.85),
            max_chars_per_line=30,
            words_per_group=5,
            min_word_duration=0.2,
        )

        assert style.font_size == 80
        assert style.color == "yellow"
        assert style.highlight_color == "#00FF00"
        assert style.stroke_color == "white"
        assert style.stroke_width == 6
        assert style.bg_color == "black"
        assert style.position == ("center", 0.85)
        assert style.max_chars_per_line == 30
        assert style.words_per_group == 5
        assert style.min_word_duration == 0.2

    def test_emphasis_colors_default(self):
        """Test default emphasis colors."""
        style = SubtitleStyle()

        assert "strong" in style.emphasis_colors
        assert "action" in style.emphasis_colors
        assert "stats" in style.emphasis_colors
        assert style.emphasis_colors["strong"] == "#FFD700"
        assert style.emphasis_colors["action"] == "#00D4FF"
        assert style.emphasis_colors["stats"] == "#00FF88"

    def test_custom_emphasis_colors(self):
        """Test custom emphasis colors."""
        custom_colors = {
            "important": "#FF0000",
            "warning": "#FFAA00",
        }
        style = SubtitleStyle(emphasis_colors=custom_colors)

        assert style.emphasis_colors == custom_colors


class TestSubtitleConfig:
    """Tests for SubtitleConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SubtitleConfig()

        assert config.enabled is True
        assert config.karaoke_mode is True
        assert config.emoji_enabled is False
        assert config.color_emphasis is True
        assert config.aligner == "wav2vec2"
        assert config.save_ass_files is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = SubtitleConfig(
            enabled=False,
            karaoke_mode=False,
            emoji_enabled=True,
            color_emphasis=False,
            aligner="whisper",
            save_ass_files=False,
        )

        assert config.enabled is False
        assert config.karaoke_mode is False
        assert config.emoji_enabled is True
        assert config.color_emphasis is False
        assert config.aligner == "whisper"
        assert config.save_ass_files is False

    def test_partial_config(self):
        """Test creating config with partial overrides."""
        config = SubtitleConfig(
            karaoke_mode=False,
            aligner="whisper",
        )

        assert config.enabled is True  # Default
        assert config.karaoke_mode is False  # Overridden
        assert config.emoji_enabled is False  # Default
        assert config.aligner == "whisper"  # Overridden


class TestSubtitleRenderer:
    """Tests for SubtitleRenderer class."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance with default settings."""
        return SubtitleRenderer(output_dir=tmp_path)

    @pytest.fixture
    def disabled_renderer(self, tmp_path):
        """Create a SubtitleRenderer with subtitles disabled."""
        config = SubtitleConfig(enabled=False)
        return SubtitleRenderer(config=config, output_dir=tmp_path)

    def test_init_default_config(self, tmp_path):
        """Test initialization with default config."""
        renderer = SubtitleRenderer(output_dir=tmp_path)

        assert renderer.config is not None
        assert renderer.style is not None
        assert renderer.config.enabled is True
        assert renderer.config.karaoke_mode is True

    def test_init_custom_config(self, tmp_path):
        """Test initialization with custom config."""
        style = SubtitleStyle(font_size=80)
        config = SubtitleConfig(karaoke_mode=False)

        renderer = SubtitleRenderer(
            style=style,
            config=config,
            output_dir=tmp_path,
        )

        assert renderer.style.font_size == 80
        assert renderer.config.karaoke_mode is False

    def test_init_custom_font_path(self, tmp_path):
        """Test initialization with custom font path."""
        font_path = tmp_path / "custom_font.ttf"

        renderer = SubtitleRenderer(
            font_path=font_path,
            output_dir=tmp_path,
        )

        assert renderer.font_path == font_path


class TestCreateSubtitles:
    """Tests for create_subtitles method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_disabled_returns_empty(self, tmp_path):
        """Test that disabled renderer returns empty list."""
        config = SubtitleConfig(enabled=False)
        renderer = SubtitleRenderer(config=config, output_dir=tmp_path)

        result = renderer.create_subtitles(
            narration="Test narration",
            total_duration=10.0,
            video_width=1920,
            video_height=1080,
            audio_duration=10.0,
        )

        assert result == []

    def test_empty_narration_returns_empty(self, renderer):
        """Test that empty narration returns empty list."""
        result = renderer.create_subtitles(
            narration="",
            total_duration=10.0,
            video_width=1920,
            video_height=1080,
            audio_duration=10.0,
        )

        assert result == []

    def test_none_narration_returns_empty(self, renderer):
        """Test that None narration returns empty list."""
        result = renderer.create_subtitles(
            narration=None,
            total_duration=10.0,
            video_width=1920,
            video_height=1080,
            audio_duration=10.0,
        )

        assert result == []


class TestAdjustStyleForFormat:
    """Tests for adjust_style_for_format method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_short_viral_mode(self, renderer):
        """Test style adjustments for short-form viral video."""
        renderer.adjust_style_for_format(is_short=True, viral_mode=True)

        assert renderer.style.font_size == 75
        assert renderer.style.stroke_width == 5
        assert renderer.style.max_chars_per_line == 18
        assert renderer.style.position == ("center", 0.75)
        assert renderer.style.words_per_group == 4

    def test_short_non_viral_mode(self, renderer):
        """Test style adjustments for short-form non-viral video."""
        renderer.adjust_style_for_format(is_short=True, viral_mode=False)

        assert renderer.style.font_size == 55
        assert renderer.style.stroke_width == 3
        assert renderer.style.max_chars_per_line == 25
        assert renderer.style.position == ("center", 0.80)

    def test_long_viral_mode(self, renderer):
        """Test style adjustments for long-form viral video."""
        renderer.adjust_style_for_format(is_short=False, viral_mode=True)

        assert renderer.style.font_size == 55
        assert renderer.style.stroke_width == 4
        assert renderer.style.max_chars_per_line == 35
        assert renderer.style.position == ("center", 0.82)
        assert renderer.style.words_per_group == 4

    def test_long_non_viral_mode(self, renderer):
        """Test style adjustments for long-form non-viral video."""
        renderer.adjust_style_for_format(is_short=False, viral_mode=False)

        assert renderer.style.font_size == 45
        assert renderer.style.stroke_width == 2
        assert renderer.style.max_chars_per_line == 50
        assert renderer.style.position == ("center", 0.85)


class TestTokenizeWords:
    """Tests for _tokenize_words method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_simple_text(self, renderer):
        """Test tokenizing simple text."""
        words = renderer._tokenize_words("Hello world test")

        assert words == ["Hello", "world", "test"]

    def test_text_with_punctuation(self, renderer):
        """Test tokenizing text with punctuation."""
        words = renderer._tokenize_words("Hello, world! How are you?")

        assert words == ["Hello,", "world!", "How", "are", "you?"]

    def test_empty_string(self, renderer):
        """Test tokenizing empty string."""
        words = renderer._tokenize_words("")

        assert words == []

    def test_whitespace_only(self, renderer):
        """Test tokenizing whitespace-only string."""
        words = renderer._tokenize_words("   \n\t  ")

        assert words == []

    def test_multiple_spaces(self, renderer):
        """Test tokenizing text with multiple spaces."""
        words = renderer._tokenize_words("Hello    world")

        assert words == ["Hello", "world"]


class TestGroupWords:
    """Tests for _group_words method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_group_by_size(self, renderer):
        """Test grouping words by size."""
        words = ["one", "two", "three", "four", "five", "six"]
        groups = renderer._group_words(words, group_size=3)

        assert len(groups) == 2
        assert groups[0] == ["one", "two", "three"]
        assert groups[1] == ["four", "five", "six"]

    def test_group_breaks_at_sentence_end(self, renderer):
        """Test that grouping breaks at sentence-ending punctuation."""
        words = ["Hello.", "World", "test"]
        groups = renderer._group_words(words, group_size=5)

        assert len(groups) == 2
        assert groups[0] == ["Hello."]
        assert groups[1] == ["World", "test"]

    def test_group_breaks_at_question_mark(self, renderer):
        """Test that grouping breaks at question marks."""
        words = ["What?", "Yes", "indeed"]
        groups = renderer._group_words(words, group_size=5)

        assert len(groups) == 2
        assert groups[0] == ["What?"]

    def test_group_breaks_at_exclamation(self, renderer):
        """Test that grouping breaks at exclamation marks."""
        words = ["Wow!", "Amazing", "stuff"]
        groups = renderer._group_words(words, group_size=5)

        assert len(groups) == 2
        assert groups[0] == ["Wow!"]

    def test_group_clause_break_near_limit(self, renderer):
        """Test clause break when near group size limit."""
        words = ["one", "two", "three,", "four"]
        groups = renderer._group_words(words, group_size=4)

        # Should break at comma since we're at group_size - 1
        assert len(groups) == 2
        assert groups[0] == ["one", "two", "three,"]
        assert groups[1] == ["four"]

    def test_empty_words(self, renderer):
        """Test grouping empty word list."""
        groups = renderer._group_words([], group_size=3)

        assert groups == []

    def test_remaining_words(self, renderer):
        """Test that remaining words are included."""
        words = ["one", "two", "three", "four", "five"]
        groups = renderer._group_words(words, group_size=3)

        assert len(groups) == 2
        assert groups[0] == ["one", "two", "three"]
        assert groups[1] == ["four", "five"]


class TestSplitIntoPhrases:
    """Tests for _split_into_phrases method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_simple_sentence(self, renderer):
        """Test splitting a simple sentence."""
        phrases = renderer._split_into_phrases("Hello world.")

        assert len(phrases) == 1
        assert phrases[0] == "Hello world."

    def test_multiple_sentences(self, renderer):
        """Test splitting multiple sentences."""
        phrases = renderer._split_into_phrases("First sentence. Second sentence.")

        assert len(phrases) == 2
        assert phrases[0] == "First sentence."
        assert phrases[1] == "Second sentence."

    def test_long_sentence_split(self, renderer):
        """Test that long sentences are split into chunks."""
        long_text = "This is a very long sentence with many words that should be split into smaller chunks for readability."
        phrases = renderer._split_into_phrases(long_text)

        assert len(phrases) > 1
        # Each phrase should be reasonably short
        for phrase in phrases:
            assert len(phrase.split()) <= 8

    def test_empty_text(self, renderer):
        """Test splitting empty text."""
        phrases = renderer._split_into_phrases("")

        assert phrases == []

    def test_comma_splits(self, renderer):
        """Test that commas can create splits in long sentences."""
        text = "First part of sentence, second part of sentence, third part"
        phrases = renderer._split_into_phrases(text)

        # Should split at commas for long text
        assert len(phrases) >= 1


class TestWrapText:
    """Tests for _wrap_text method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_short_text_no_wrap(self, renderer):
        """Test that short text is not wrapped."""
        renderer.style.max_chars_per_line = 50
        result = renderer._wrap_text("Short text")

        assert result == "Short text"
        assert "\n" not in result

    def test_long_text_wrapped(self, renderer):
        """Test that long text is wrapped."""
        renderer.style.max_chars_per_line = 20
        result = renderer._wrap_text("This is a longer text that should be wrapped")

        assert "\n" in result
        for line in result.split("\n"):
            assert len(line) <= 20


class TestGetWordColor:
    """Tests for _get_word_color method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_color_emphasis_disabled(self, tmp_path):
        """Test color when emphasis is disabled."""
        config = SubtitleConfig(color_emphasis=False)
        renderer = SubtitleRenderer(config=config, output_dir=tmp_path)

        color = renderer._get_word_color("amazing", is_highlighted=True)

        assert color == renderer.style.highlight_color

    def test_color_emphasis_disabled_not_highlighted(self, tmp_path):
        """Test color when emphasis disabled and not highlighted."""
        config = SubtitleConfig(color_emphasis=False)
        renderer = SubtitleRenderer(config=config, output_dir=tmp_path)

        color = renderer._get_word_color("amazing", is_highlighted=False)

        assert color == renderer.style.color

    def test_number_gets_stats_color(self, renderer):
        """Test that numbers get stats color."""
        color = renderer._get_word_color("100", is_highlighted=True)

        assert color == renderer.style.emphasis_colors.get("stats", "#00FF88")

    def test_highlighted_word_default_color(self, renderer):
        """Test default highlight color for non-special words."""
        color = renderer._get_word_color("regular", is_highlighted=True)

        assert color == renderer.style.highlight_color

    def test_non_highlighted_word_default_color(self, renderer):
        """Test default color for non-highlighted words."""
        color = renderer._get_word_color("regular", is_highlighted=False)

        assert color == renderer.style.color


class TestCreateKaraokeFromTimestamps:
    """Tests for create_karaoke_from_timestamps method."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create a SubtitleRenderer instance."""
        return SubtitleRenderer(output_dir=tmp_path)

    def test_empty_timestamps(self, renderer):
        """Test with empty timestamps list."""
        result = renderer.create_karaoke_from_timestamps(
            word_timestamps=[],
            video_width=1920,
            video_height=1080,
            audio_duration=10.0,
        )

        assert result == []


class TestSubtitleRendererIntegration:
    """Integration tests for SubtitleRenderer."""

    def test_full_configuration(self, tmp_path):
        """Test creating renderer with full custom configuration."""
        style = SubtitleStyle(
            font_size=80,
            color="yellow",
            highlight_color="#00FF00",
            stroke_width=5,
            max_chars_per_line=25,
            words_per_group=3,
        )
        config = SubtitleConfig(
            enabled=True,
            karaoke_mode=True,
            emoji_enabled=False,
            color_emphasis=True,
            aligner="whisper",
            save_ass_files=False,
        )
        font_path = Path("assets/fonts/arial.ttf")

        renderer = SubtitleRenderer(
            style=style,
            config=config,
            font_path=font_path,
            output_dir=tmp_path,
        )

        assert renderer.style.font_size == 80
        assert renderer.config.aligner == "whisper"
        assert renderer.font_path == font_path
        assert renderer.output_dir == tmp_path

    def test_style_modification_persists(self, tmp_path):
        """Test that style modifications persist."""
        renderer = SubtitleRenderer(output_dir=tmp_path)

        # Modify style
        renderer.style.font_size = 100
        renderer.style.color = "red"

        assert renderer.style.font_size == 100
        assert renderer.style.color == "red"

    def test_config_modification_persists(self, tmp_path):
        """Test that config modifications persist."""
        renderer = SubtitleRenderer(output_dir=tmp_path)

        # Modify config
        renderer.config.karaoke_mode = False
        renderer.config.emoji_enabled = True

        assert renderer.config.karaoke_mode is False
        assert renderer.config.emoji_enabled is True
