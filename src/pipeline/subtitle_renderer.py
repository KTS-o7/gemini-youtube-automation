"""
Subtitle Renderer Module - Handles all subtitle generation and rendering.

This module extracts subtitle-related functionality from VideoComposer,
including karaoke-style word-by-word highlighting, standard subtitles,
emoji insertion, and word emphasis coloring.

FEATURES:
- 🔥 Karaoke-style captions with word-by-word highlighting
- 🎤 Whisper/Wav2Vec2-powered audio alignment for precise word sync
- 🎨 Color emphasis for key words
- 😀 Emoji support in captions
"""

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from moviepy.editor import TextClip

from .config import load_emoji_mappings, load_emphasis_keywords


@dataclass
class SubtitleStyle:
    """Configuration for subtitle appearance."""

    font_size: int = 70
    color: str = "white"
    highlight_color: str = "#FFD700"  # Gold for current word
    emphasis_colors: dict = field(
        default_factory=lambda: {
            "strong": "#FFD700",  # Gold/yellow
            "action": "#00D4FF",  # Cyan
            "stats": "#00FF88",  # Green
        }
    )
    stroke_color: str = "black"
    stroke_width: int = 4
    bg_color: Optional[str] = None
    position: tuple = ("center", 0.78)
    max_chars_per_line: int = 20
    words_per_group: int = 4
    min_word_duration: float = 0.15


@dataclass
class SubtitleConfig:
    """Configuration for subtitle behavior."""

    enabled: bool = True
    karaoke_mode: bool = True
    emoji_enabled: bool = False
    color_emphasis: bool = True
    aligner: str = "wav2vec2"  # "wav2vec2" or "whisper"
    save_ass_files: bool = True


class SubtitleRenderer:
    """
    Renders subtitles for video clips.

    Supports both karaoke-style (word-by-word highlighting) and
    standard phrase-based subtitles.
    """

    def __init__(
        self,
        style: Optional[SubtitleStyle] = None,
        config: Optional[SubtitleConfig] = None,
        font_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the subtitle renderer.

        Args:
            style: Subtitle appearance settings
            config: Subtitle behavior settings
            font_path: Path to font file for subtitles
            output_dir: Directory for saving .ass files
        """
        self.style = style or SubtitleStyle()
        self.config = config or SubtitleConfig()
        self.font_path = font_path or Path("assets/fonts/arial.ttf")
        self.output_dir = output_dir or Path("output")

        # Store last alignment result for .ass file generation
        self._last_alignment_result = None

    def create_subtitles(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
        audio_path: Optional[str | Path] = None,
    ) -> list:
        """
        Create subtitle clips for a video scene.

        Args:
            narration: The narration text
            total_duration: Total duration of the scene
            video_width: Width of the video
            video_height: Height of the video
            audio_duration: Duration of the audio
            audio_path: Path to audio file for alignment

        Returns:
            List of TextClip objects
        """
        if not self.config.enabled or not narration:
            return []

        if self.config.karaoke_mode:
            return self._create_karaoke_subtitles(
                narration,
                total_duration,
                video_width,
                video_height,
                audio_duration,
                audio_path,
            )
        else:
            return self._create_standard_subtitles(
                narration,
                total_duration,
                video_width,
                video_height,
                audio_duration,
            )

    def create_karaoke_from_timestamps(
        self,
        word_timestamps: list,
        video_width: int,
        video_height: int,
        audio_duration: float,
    ) -> list:
        """
        Create karaoke clips using pre-computed timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'
            video_width: Width of video
            video_height: Height of video
            audio_duration: Total audio duration

        Returns:
            List of subtitle clips
        """
        y_pos = int(video_height * self.style.position[1])
        return self._create_karaoke_from_timestamps(
            word_timestamps, video_width, y_pos, audio_duration
        )

    def adjust_style_for_format(self, is_short: bool, viral_mode: bool = True) -> None:
        """
        Adjust subtitle style based on video format.

        Args:
            is_short: Whether this is a short-form video (vertical)
            viral_mode: Whether viral optimizations are enabled
        """
        if is_short:
            if viral_mode:
                self.style.font_size = 75
                self.style.stroke_width = 5
                self.style.max_chars_per_line = 18
                self.style.position = ("center", 0.75)
                self.style.words_per_group = 4
            else:
                self.style.font_size = 55
                self.style.stroke_width = 3
                self.style.max_chars_per_line = 25
                self.style.position = ("center", 0.80)
        else:
            if viral_mode:
                self.style.font_size = 55
                self.style.stroke_width = 4
                self.style.max_chars_per_line = 35
                self.style.position = ("center", 0.82)
                self.style.words_per_group = 4
            else:
                self.style.font_size = 45
                self.style.stroke_width = 2
                self.style.max_chars_per_line = 50
                self.style.position = ("center", 0.85)

    # =========================================================================
    # KARAOKE-STYLE SUBTITLES
    # =========================================================================

    def _create_karaoke_subtitles(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
        audio_path: Optional[str | Path] = None,
    ) -> list:
        """
        Create karaoke-style subtitles with word-by-word highlighting.

        TRUE KARAOKE: Single line where current word is highlighted in color,
        other words stay in default color. No duplicate text.
        """
        subtitle_clips = []

        # Preprocess: add emojis if enabled
        if self.config.emoji_enabled:
            narration = self._add_emojis_to_text(narration)

        y_pos = int(video_height * self.style.position[1])

        # Get word-level timestamps using configured aligner
        word_timestamps = None
        if audio_path:
            word_timestamps = self._get_word_timestamps(audio_path, narration)

        if word_timestamps:
            # Use precise word timestamps
            print(f"      🎤 Aligned {len(word_timestamps)} words")

            # Save .ass file if enabled
            if self.config.save_ass_files and audio_path:
                self._save_ass_file(word_timestamps, audio_path)

            return self._create_karaoke_from_timestamps(
                word_timestamps, video_width, y_pos, audio_duration
            )

        # Fallback to estimated timing
        print(f"      ⏱️ Using estimated timing (no alignment)")
        return self._create_estimated_karaoke(
            narration, audio_duration, video_width, y_pos
        )

    def _get_word_timestamps(
        self, audio_path: str | Path, transcript: Optional[str] = None
    ) -> Optional[list]:
        """
        Get word-level timestamps using configured aligner.

        Args:
            audio_path: Path to the audio file
            transcript: The transcript text (required for wav2vec2)

        Returns:
            List of dicts with 'word', 'start', 'end' keys, or None if failed
        """
        audio_path_str = str(audio_path) if isinstance(audio_path, Path) else audio_path

        if self.config.aligner == "wav2vec2":
            if not transcript:
                print(
                    "      ⚠️ wav2vec2 requires transcript, falling back to Whisper..."
                )
                return self._get_whisper_timestamps(audio_path_str)
            print("      🆓 Using FREE wav2vec2 alignment...")
            return self._get_wav2vec2_timestamps(audio_path_str, transcript)
        else:
            print("      💰 Using Whisper alignment...")
            return self._get_whisper_timestamps(audio_path_str)

    def _get_wav2vec2_timestamps(
        self, audio_path: str, transcript: str
    ) -> Optional[list]:
        """
        Get word-level timestamps using FREE wav2vec2 forced alignment.

        Args:
            audio_path: Path to the audio file
            transcript: The transcript text to align

        Returns:
            List of dicts with 'word', 'start', 'end' keys, or None if failed
        """
        try:
            from .wav2vec2_aligner import Wav2Vec2Aligner

            aligner = Wav2Vec2Aligner()
            result = aligner.align(audio_path, transcript)

            self._last_alignment_result = result

            return [
                {"word": w.word, "start": w.start, "end": w.end} for w in result.words
            ]
        except Exception as e:
            print(f"      ⚠️ Wav2Vec2 alignment failed: {e}")
            return []

    def _get_whisper_timestamps(self, audio_path: str) -> Optional[list]:
        """
        Get word-level timestamps from audio using OpenAI Whisper.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of dicts with 'word', 'start', 'end' keys, or None if failed
        """
        try:
            from .audio_aligner import AudioAligner

            aligner = AudioAligner()
            result = aligner.get_word_timestamps(audio_path)

            self._last_alignment_result = result

            return [
                {"word": w.word, "start": w.start, "end": w.end} for w in result.words
            ]
        except Exception as e:
            print(f"      ⚠️ Whisper alignment failed: {e}")
            return []

    def _save_ass_file(self, word_timestamps: list, audio_path: str | Path) -> None:
        """
        Save .ass subtitle file from word timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'
            audio_path: Path to the audio file (used for naming)
        """
        try:
            from .audio_aligner import AlignmentResult, AudioAligner, WordTimestamp

            words = [
                WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
                for w in word_timestamps
            ]

            result = AlignmentResult(
                words=words,
                full_text=" ".join(w["word"] for w in word_timestamps),
                duration=word_timestamps[-1]["end"] if word_timestamps else 0,
                language="en",
            )

            audio_path_obj = (
                Path(audio_path) if isinstance(audio_path, str) else audio_path
            )
            ass_path = self.output_dir / "subtitles" / f"{audio_path_obj.stem}.ass"
            ass_path.parent.mkdir(parents=True, exist_ok=True)

            aligner = AudioAligner()
            aligner.generate_ass_file(
                result,
                ass_path,
                words_per_line=self.style.words_per_group,
                font_size=self.style.font_size,
            )
            print(f"      💾 Saved: {ass_path}")

        except Exception as e:
            print(f"      ⚠️ Failed to save .ass file: {e}")

    def _create_karaoke_from_timestamps(
        self,
        word_timestamps: list,
        video_width: int,
        y_pos: int,
        audio_duration: float,
    ) -> list:
        """
        Create karaoke clips using precise timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'
            video_width: Width of video
            y_pos: Y position for subtitles
            audio_duration: Total audio duration

        Returns:
            List of subtitle clips
        """
        clips = []

        # Group words into phrases
        groups = []
        current_group = []

        for wt in word_timestamps:
            current_group.append(wt)
            word = wt["word"]

            ends_sentence = any(word.rstrip().endswith(p) for p in [".", "!", "?"])
            ends_clause = any(word.rstrip().endswith(p) for p in [",", ";", ":"])

            if len(current_group) >= self.style.words_per_group or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= self.style.words_per_group - 1:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        # Create clips for each group with precise timing
        for group in groups:
            words = [wt["word"] for wt in group]

            for i, wt in enumerate(group):
                word_start = wt["start"]
                word_end = wt["end"]
                word_duration = word_end - word_start

                if word_duration <= 0:
                    continue

                word_lower = wt["word"].lower().strip(".,!?;:")
                highlight_color = self._get_word_color(word_lower, is_highlighted=True)

                try:
                    word_clips = self._create_word_group_clips_precise(
                        words,
                        i,
                        word_start,
                        word_duration,
                        video_width,
                        y_pos,
                        highlight_color,
                    )
                    clips.extend(word_clips)
                except Exception as e:
                    print(f"      ⚠️ Failed to create clip for '{wt['word']}': {e}")

        return clips

    def _create_word_group_clips_precise(
        self,
        words: list,
        highlight_idx: int,
        start_time: float,
        duration: float,
        video_width: int,
        y_pos: int,
        highlight_color: str,
    ) -> list:
        """
        Create text clips for a word group with precise timing.
        """
        clips = []

        try:
            # Calculate total width for centering
            temp_clips = []
            for word in words:
                tc = TextClip(
                    word + " ",
                    fontsize=self.style.font_size,
                    color=self.style.color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.style.stroke_color,
                    stroke_width=self.style.stroke_width,
                    method="label",
                )
                temp_clips.append(tc)

            total_width = sum(tc.w for tc in temp_clips)

            for tc in temp_clips:
                tc.close()

            # Create the actual clips
            x_offset = (video_width - total_width) // 2

            for i, word in enumerate(words):
                is_current = i == highlight_idx
                word_color = highlight_color if is_current else self.style.color
                font_size = int(self.style.font_size * (1.1 if is_current else 1.0))

                word_clip = TextClip(
                    word + (" " if i < len(words) - 1 else ""),
                    fontsize=font_size,
                    color=word_color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.style.stroke_color,
                    stroke_width=self.style.stroke_width,
                    method="label",
                )

                word_y = y_pos - (int(self.style.font_size * 0.05) if is_current else 0)

                word_clip = (
                    word_clip.set_start(start_time)
                    .set_duration(duration)
                    .set_position((x_offset, word_y))
                )

                clips.append(word_clip)
                x_offset += word_clip.w

        except Exception as e:
            print(f"      ⚠️ Precise clip creation failed: {e}")

        return clips

    def _create_estimated_karaoke(
        self,
        narration: str,
        audio_duration: float,
        video_width: int,
        y_pos: int,
    ) -> list:
        """
        Create karaoke subtitles using estimated timing (no alignment).
        """
        subtitle_clips = []

        words = self._tokenize_words(narration)
        if not words:
            return subtitle_clips

        available_duration = audio_duration * 0.95
        time_per_word = available_duration / len(words)
        time_per_word = max(time_per_word, self.style.min_word_duration)

        word_groups = self._group_words(words, self.style.words_per_group)
        words_processed = 0

        for group_idx, word_group in enumerate(word_groups):
            group_start = words_processed * time_per_word
            group_duration = len(word_group) * time_per_word

            if group_start + group_duration > audio_duration:
                group_duration = max(0.1, audio_duration - group_start - 0.05)

            if group_duration <= 0:
                break

            group_clips = self._create_word_group_clips(
                word_group,
                group_start,
                group_duration,
                video_width,
                y_pos,
            )
            subtitle_clips.extend(group_clips)

            words_processed += len(word_group)

        return subtitle_clips

    def _create_word_group_clips(
        self,
        words: list[str],
        start_time: float,
        duration: float,
        video_width: int,
        y_pos: int,
    ) -> list:
        """
        Create text clips for a group of words with karaoke highlighting.

        TRUE KARAOKE: Single line where current word is highlighted in color,
        other words stay in default color. No duplicate text.
        """
        clips = []
        time_per_word = duration / len(words)

        for highlight_idx in range(len(words)):
            word_start = start_time + (highlight_idx * time_per_word)
            word_duration = time_per_word

            current_word = words[highlight_idx]
            word_lower = current_word.lower().strip(".,!?;:")
            highlight_color = self._get_word_color(word_lower, is_highlighted=True)

            try:
                temp_clips = []
                for word in words:
                    tc = TextClip(
                        word + " ",
                        fontsize=self.style.font_size,
                        color=self.style.color,
                        font=str(self.font_path)
                        if self.font_path.exists()
                        else "Arial-Bold",
                        stroke_color=self.style.stroke_color,
                        stroke_width=self.style.stroke_width,
                        method="label",
                    )
                    temp_clips.append(tc)

                total_width = sum(tc.w for tc in temp_clips)

                for tc in temp_clips:
                    tc.close()

                x_offset = (video_width - total_width) // 2

                for i, word in enumerate(words):
                    is_current = i == highlight_idx
                    word_color = highlight_color if is_current else self.style.color
                    font_size = int(self.style.font_size * (1.1 if is_current else 1.0))

                    word_clip = TextClip(
                        word + (" " if i < len(words) - 1 else ""),
                        fontsize=font_size,
                        color=word_color,
                        font=str(self.font_path)
                        if self.font_path.exists()
                        else "Arial-Bold",
                        stroke_color=self.style.stroke_color,
                        stroke_width=self.style.stroke_width,
                        method="label",
                    )

                    word_y = y_pos - (
                        int(self.style.font_size * 0.05) if is_current else 0
                    )

                    word_clip = (
                        word_clip.set_start(word_start)
                        .set_duration(word_duration)
                        .set_position((x_offset, word_y))
                    )

                    clips.append(word_clip)
                    x_offset += word_clip.w

            except Exception as e:
                print(f"  ⚠️ Failed to create karaoke clip: {e}")
                try:
                    full_phrase = " ".join(words)
                    txt_clip = TextClip(
                        full_phrase,
                        fontsize=self.style.font_size,
                        color=self.style.highlight_color,
                        stroke_color=self.style.stroke_color,
                        stroke_width=self.style.stroke_width,
                        method="label",
                    )
                    txt_clip = (
                        txt_clip.set_start(word_start)
                        .set_duration(word_duration)
                        .set_position(("center", y_pos))
                    )
                    clips.append(txt_clip)
                except Exception:
                    pass

        return clips

    # =========================================================================
    # STANDARD SUBTITLES (NON-KARAOKE)
    # =========================================================================

    def _create_standard_subtitles(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
    ) -> list[TextClip]:
        """
        Create standard subtitle clips (phrase-based, non-karaoke).
        """
        subtitle_clips = []

        if self.config.emoji_enabled:
            narration = self._add_emojis_to_text(narration)

        phrases = self._split_into_phrases(narration)

        if not phrases:
            return subtitle_clips

        available_duration = audio_duration * 0.95
        time_per_phrase = available_duration / len(phrases)

        y_pos = int(video_height * self.style.position[1])

        current_time = 0.0
        for phrase in phrases:
            wrapped_text = self._wrap_text(phrase)

            try:
                txt_clip = TextClip(
                    wrapped_text,
                    fontsize=self.style.font_size,
                    color=self.style.color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.style.stroke_color,
                    stroke_width=self.style.stroke_width,
                    method="caption",
                    size=(video_width - 80, None),
                    align="center",
                )

                phrase_duration = min(
                    time_per_phrase, audio_duration - current_time - 0.1
                )
                if phrase_duration <= 0:
                    break

                txt_clip = (
                    txt_clip.set_start(current_time)
                    .set_duration(phrase_duration)
                    .set_position(("center", y_pos))
                )

                fade_duration = min(0.08, phrase_duration / 4)
                if fade_duration > 0.03:
                    txt_clip = txt_clip.crossfadein(fade_duration).crossfadeout(
                        fade_duration
                    )

                subtitle_clips.append(txt_clip)

            except Exception as e:
                print(f"  ⚠️ Failed to create subtitle clip: {e}")
                try:
                    phrase_duration = min(
                        time_per_phrase, audio_duration - current_time - 0.1
                    )
                    if phrase_duration <= 0:
                        break
                    txt_clip = TextClip(
                        wrapped_text,
                        fontsize=self.style.font_size,
                        color=self.style.color,
                        stroke_color=self.style.stroke_color,
                        stroke_width=self.style.stroke_width,
                        method="label",
                    )
                    txt_clip = (
                        txt_clip.set_start(current_time)
                        .set_duration(phrase_duration)
                        .set_position(("center", y_pos))
                    )
                    subtitle_clips.append(txt_clip)
                except Exception as e2:
                    print(f"  ⚠️ Subtitle fallback also failed: {e2}")

            current_time += time_per_phrase

        return subtitle_clips

    # =========================================================================
    # TEXT PROCESSING UTILITIES
    # =========================================================================

    def _tokenize_words(self, text: str) -> list[str]:
        """Split text into words while preserving emojis and punctuation."""
        text = text.strip()
        words = text.split()
        words = [w for w in words if w.strip()]
        return words

    def _group_words(self, words: list[str], group_size: int) -> list[list[str]]:
        """Group words for display, respecting punctuation boundaries."""
        groups = []
        current_group = []

        for word in words:
            current_group.append(word)

            ends_sentence = any(word.endswith(p) for p in [".", "!", "?", ":"])
            ends_clause = any(word.endswith(p) for p in [",", ";", "-"])

            if len(current_group) >= group_size or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= group_size - 1:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        return groups

    def _split_into_phrases(self, text: str) -> list[str]:
        """
        Split text into readable phrases for subtitles.
        Shorter phrases for more impact.
        """
        text = text.strip()

        # Handle empty text early
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)

        phrases = []
        chunk_size = 5 if self.config.karaoke_mode else 6

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= chunk_size + 2:
                phrases.append(sentence.strip())
            else:
                if "," in sentence:
                    parts = sentence.split(",")
                    for part in parts:
                        part = part.strip()
                        if part:
                            part_words = part.split()
                            if len(part_words) <= chunk_size + 3:
                                phrases.append(part)
                            else:
                                for i in range(0, len(part_words), chunk_size):
                                    chunk = " ".join(part_words[i : i + chunk_size])
                                    if chunk:
                                        phrases.append(chunk)
                else:
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i : i + chunk_size])
                        if chunk:
                            phrases.append(chunk)

        return phrases

    def _wrap_text(self, text: str) -> str:
        """Wrap text to fit within max characters per line."""
        return "\n".join(textwrap.wrap(text, width=self.style.max_chars_per_line))

    def _get_word_color(self, word: str, is_highlighted: bool = False) -> str:
        """Determine the color for a word based on emphasis rules."""
        word_lower = word.lower().strip(".,!?;:\"'")

        if not self.config.color_emphasis:
            return self.style.highlight_color if is_highlighted else self.style.color

        for category, keywords in load_emphasis_keywords().items():
            if word_lower in keywords:
                return self.style.emphasis_colors.get(
                    category, self.style.highlight_color
                )

        if any(c.isdigit() for c in word):
            return self.style.emphasis_colors.get("stats", "#00FF88")

        return self.style.highlight_color if is_highlighted else self.style.color

    def _add_emojis_to_text(self, text: str) -> str:
        """Add relevant emojis to text based on keyword mappings."""
        words = text.split()
        result_words = []

        added_emojis = set()

        for i, word in enumerate(words):
            result_words.append(word)

            word_lower = word.lower().strip(".,!?;:")

            emoji_mappings = load_emoji_mappings()
            if word_lower in emoji_mappings and word_lower not in added_emojis:
                emoji = emoji_mappings[word_lower]
                if len(added_emojis) < 3:
                    result_words.append(emoji)
                    added_emojis.add(word_lower)

            if i > 0:
                prev_word = words[i - 1].lower().strip(".,!?;:")
                phrase = f"{prev_word} {word_lower}"
                if phrase in emoji_mappings and phrase not in added_emojis:
                    emoji = emoji_mappings[phrase]
                    if len(added_emojis) < 3:
                        result_words.append(emoji)
                        added_emojis.add(phrase)

        return " ".join(result_words)
