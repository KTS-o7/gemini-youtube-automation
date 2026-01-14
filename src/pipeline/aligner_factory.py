"""
Aligner Factory - Unified interface for audio-text alignment.

This module provides a factory function to create the appropriate aligner
based on configuration, making it easy to switch between:

1. Whisper (OpenAI API) - Auto-transcribes, costs ~$0.006/minute
2. Wav2Vec2 (local) - FREE, requires transcript text

Usage:
    from src.pipeline.aligner_factory import create_aligner, AlignerType

    # Use Whisper (default, auto-transcribes)
    aligner = create_aligner(AlignerType.WHISPER)
    result = aligner.get_word_timestamps("audio.wav")

    # Use Wav2Vec2 (free, needs transcript)
    aligner = create_aligner(AlignerType.WAV2VEC2)
    result = aligner.align("audio.wav", "Hello world this is a test")

    # Auto-select based on environment variable SUBTITLE_ALIGNER
    aligner = create_aligner()  # Reads from env or defaults to wav2vec2
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, Union


class AlignerType(Enum):
    """Available alignment methods."""

    WHISPER = "whisper"  # OpenAI API - auto-transcribes, costs money
    WAV2VEC2 = "wav2vec2"  # Local model - FREE, needs transcript


@dataclass
class WordTimestamp:
    """A single word with its timing information."""

    word: str
    start: float
    end: float
    score: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AlignmentResult:
    """Result of audio alignment."""

    words: list[WordTimestamp]
    full_text: str
    duration: float
    language: str = "en"
    aligner: str = "unknown"

    def group_into_phrases(self, words_per_group: int = 3) -> list[list[WordTimestamp]]:
        """Group words into phrases for subtitle display."""
        groups = []
        current_group = []

        for word in self.words:
            current_group.append(word)

            ends_sentence = any(word.word.rstrip().endswith(p) for p in [".", "!", "?"])
            ends_clause = any(word.word.rstrip().endswith(p) for p in [",", ";", ":"])

            if len(current_group) >= words_per_group or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= words_per_group - 1:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        return groups


class AlignerProtocol(Protocol):
    """Protocol defining the aligner interface."""

    def get_word_timestamps(
        self,
        audio_path: Union[str, Path],
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """Get word-level timestamps from audio."""
        ...

    def generate_ass_file(
        self,
        result: AlignmentResult,
        output_path: Union[str, Path],
        words_per_line: int = 3,
        font_name: str = "Arial",
        font_size: int = 24,
    ) -> Path:
        """Generate .ass subtitle file from alignment result."""
        ...

    def save_timestamps_json(
        self,
        result: AlignmentResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Save timestamps to JSON file."""
        ...


class UnifiedAligner:
    """
    Unified aligner that wraps both Whisper and Wav2Vec2.

    Provides a consistent interface regardless of which backend is used.
    """

    def __init__(
        self,
        aligner_type: AlignerType = AlignerType.WAV2VEC2,
        device: Optional[str] = None,
    ):
        """
        Initialize the unified aligner.

        Args:
            aligner_type: Which alignment method to use
            device: Device for wav2vec2 ('cuda', 'cpu', or None for auto)
        """
        self.aligner_type = aligner_type
        self._aligner = None
        self._device = device

    def _get_aligner(self):
        """Lazy-load the appropriate aligner."""
        if self._aligner is None:
            if self.aligner_type == AlignerType.WHISPER:
                from .audio_aligner import AudioAligner

                self._aligner = AudioAligner()
            else:
                from .wav2vec2_aligner import Wav2Vec2Aligner

                self._aligner = Wav2Vec2Aligner(device=self._device)
        return self._aligner

    def align(
        self,
        audio_path: Union[str, Path],
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Align audio with optional transcript.

        Args:
            audio_path: Path to audio file
            transcript: Text transcript (required for wav2vec2, optional for whisper)
            language: Language code (used by whisper)

        Returns:
            AlignmentResult with word timestamps

        Raises:
            ValueError: If using wav2vec2 without a transcript
        """
        aligner = self._get_aligner()

        if self.aligner_type == AlignerType.WAV2VEC2:
            if transcript is None:
                raise ValueError(
                    "Transcript is required for wav2vec2 alignment. "
                    "Either provide a transcript or use AlignerType.WHISPER."
                )
            result = aligner.align(audio_path, transcript)
        else:
            result = aligner.get_word_timestamps(audio_path, language=language)

        # Convert to unified result format
        words = [
            WordTimestamp(
                word=w.word,
                start=w.start,
                end=w.end,
                score=getattr(w, "score", 1.0),
            )
            for w in result.words
        ]

        return AlignmentResult(
            words=words,
            full_text=result.full_text,
            duration=result.duration,
            language=result.language,
            aligner=self.aligner_type.value,
        )

    def get_word_timestamps(
        self,
        audio_path: Union[str, Path],
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """Alias for align() for API compatibility."""
        return self.align(audio_path, transcript, language)

    def generate_ass_file(
        self,
        result: AlignmentResult,
        output_path: Union[str, Path],
        words_per_line: int = 3,
        font_name: str = "Arial",
        font_size: int = 24,
        primary_color: str = "&H00FFFFFF",
        highlight_color: str = "&H0000D7FF",
    ) -> Path:
        """
        Generate .ass subtitle file from alignment result.

        Args:
            result: AlignmentResult from align()
            output_path: Path to save .ass file
            words_per_line: Words per subtitle line
            font_name: Font for subtitles
            font_size: Font size
            primary_color: Main text color (ASS format)
            highlight_color: Karaoke highlight color

        Returns:
            Path to generated .ass file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ass_content = f"""[Script Info]
; Generated by Unified Aligner ({result.aligner})
Title: Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,50,1
Style: Karaoke,{font_name},{font_size},{highlight_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        groups = result.group_into_phrases(words_per_line)

        for group in groups:
            if not group:
                continue

            start_time = self._format_ass_time(group[0].start)
            end_time = self._format_ass_time(group[-1].end)

            karaoke_text = ""
            for word in group:
                duration_cs = int((word.end - word.start) * 100)
                karaoke_text += f"{{\\k{duration_cs}}}{word.word} "

            karaoke_text = karaoke_text.strip()
            ass_content += (
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}\n"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        print(f"  💾 Saved .ass file: {output_path}")
        return output_path

    def _format_ass_time(self, seconds: float) -> str:
        """Format time in ASS format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"

    def save_timestamps_json(
        self, result: AlignmentResult, output_path: Union[str, Path]
    ) -> Path:
        """Save timestamps to JSON file."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "full_text": result.full_text,
            "duration": result.duration,
            "language": result.language,
            "aligner": result.aligner,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "score": w.score}
                for w in result.words
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  💾 Saved timestamps JSON: {output_path}")
        return output_path


def create_aligner(
    aligner_type: Optional[AlignerType] = None,
    device: Optional[str] = None,
) -> UnifiedAligner:
    """
    Factory function to create an aligner.

    Args:
        aligner_type: Which aligner to use (defaults to env var or wav2vec2)
        device: Device for wav2vec2 ('cuda', 'cpu', or None for auto)

    Returns:
        UnifiedAligner instance

    Environment Variables:
        SUBTITLE_ALIGNER: Set to 'whisper' or 'wav2vec2' to control default

    Example:
        # Use default (checks SUBTITLE_ALIGNER env var, defaults to wav2vec2)
        aligner = create_aligner()

        # Explicitly use Whisper
        aligner = create_aligner(AlignerType.WHISPER)

        # Explicitly use Wav2Vec2
        aligner = create_aligner(AlignerType.WAV2VEC2)
    """
    if aligner_type is None:
        env_aligner = os.environ.get("SUBTITLE_ALIGNER", "").lower()
        if env_aligner == "whisper":
            aligner_type = AlignerType.WHISPER
        else:
            # Default to wav2vec2 (free!)
            aligner_type = AlignerType.WAV2VEC2

    return UnifiedAligner(aligner_type=aligner_type, device=device)


def get_recommended_aligner(has_transcript: bool, has_api_key: bool) -> AlignerType:
    """
    Get recommended aligner based on available resources.

    Args:
        has_transcript: Whether you have the transcript text
        has_api_key: Whether you have an OpenAI API key

    Returns:
        Recommended AlignerType
    """
    if has_transcript:
        # Always prefer wav2vec2 when we have transcript - it's free!
        return AlignerType.WAV2VEC2
    elif has_api_key:
        # Need Whisper to transcribe
        return AlignerType.WHISPER
    else:
        raise ValueError(
            "Either provide a transcript (for free wav2vec2 alignment) "
            "or set OPENAI_API_KEY (for Whisper transcription)"
        )
