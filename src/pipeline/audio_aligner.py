"""
Audio Aligner Module - Uses OpenAI Whisper for precise word-level timestamps.

This module transcribes audio files and returns exact timing for each word,
enabling true karaoke-style subtitle synchronization.

Usage:
    aligner = AudioAligner()
    word_timestamps = aligner.get_word_timestamps("audio.wav")
    # Returns: [{"word": "Hello", "start": 0.0, "end": 0.32}, ...]

    # Generate .ass subtitle file
    aligner.generate_ass_file(word_timestamps, "output/subtitles.ass")
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


@dataclass
class WordTimestamp:
    """A single word with its timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"WordTimestamp('{self.word}', {self.start:.2f}s - {self.end:.2f}s)"


@dataclass
class AlignmentResult:
    """Result of audio alignment containing all word timestamps."""

    words: list[WordTimestamp]
    full_text: str
    duration: float  # Total audio duration
    language: str

    def get_words_in_range(self, start: float, end: float) -> List[WordTimestamp]:
        """Get all words within a time range."""
        return [w for w in self.words if w.start >= start and w.end <= end]

    def group_into_phrases(self, words_per_group: int = 3) -> List[List[WordTimestamp]]:
        """Group words into phrases for display."""
        groups = []
        current_group = []

        for word in self.words:
            current_group.append(word)

            # Check for sentence-ending punctuation
            ends_sentence = any(word.word.rstrip().endswith(p) for p in [".", "!", "?"])
            ends_clause = any(word.word.rstrip().endswith(p) for p in [",", ";", ":"])

            if len(current_group) >= words_per_group or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= words_per_group - 1:
                groups.append(current_group)
                current_group = []

        # Add remaining words
        if current_group:
            groups.append(current_group)

        return groups


class AudioAligner:
    """
    Aligns audio with text using OpenAI Whisper for word-level timestamps.

    This provides precise timing for each word spoken in the audio,
    enabling accurate karaoke-style subtitle synchronization.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize the audio aligner.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model to use (whisper-1 is the only option currently)
        """
        # Try multiple ways to get the API key
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key:
            # Try loading from .env file directly
            from pathlib import Path

            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            self.api_key = (
                                line.split("=", 1)[1].strip().strip('"').strip("'")
                            )
                            break

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required for audio alignment. Set OPENAI_API_KEY env var."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def get_word_timestamps(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Get word-level timestamps from an audio file.

        Args:
            audio_path: Path to the audio file (mp3, wav, etc.)
            language: Optional language code (e.g., "en") for better accuracy

        Returns:
            AlignmentResult containing word timestamps and metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  🎤 Aligning audio: {audio_path.name}...")

        # Open and transcribe the audio file
        with open(audio_path, "rb") as audio_file:
            # Use verbose_json with word timestamps
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language=language,
            )

        # Extract word timestamps
        words = []
        if hasattr(response, "words") and response.words:
            for word_data in response.words:
                words.append(
                    WordTimestamp(
                        word=word_data.word,
                        start=word_data.start,
                        end=word_data.end,
                    )
                )

        # Get full text and duration
        full_text = response.text if hasattr(response, "text") else ""
        duration = response.duration if hasattr(response, "duration") else 0.0
        language_detected = response.language if hasattr(response, "language") else "en"

        print(f"  ✅ Aligned {len(words)} words in {duration:.1f}s")

        return AlignmentResult(
            words=words,
            full_text=full_text,
            duration=duration,
            language=language_detected,
        )

    def align_with_narration(
        self,
        audio_path: Union[str, Path],
        expected_text: str,
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Align audio with expected narration text.

        This method transcribes the audio and can optionally verify
        against expected text (useful for debugging).

        Args:
            audio_path: Path to the audio file
            expected_text: The expected narration text
            language: Optional language code

        Returns:
            AlignmentResult with word timestamps
        """
        result = self.get_word_timestamps(audio_path, language)

        # Log any significant differences
        expected_words = len(expected_text.split())
        actual_words = len(result.words)

        if abs(expected_words - actual_words) > expected_words * 0.1:
            print(
                f"  ⚠️ Word count mismatch: expected ~{expected_words}, got {actual_words}"
            )

        return result

    def generate_ass_file(
        self,
        result: AlignmentResult,
        output_path: Union[str, Path],
        words_per_line: int = 3,
        font_name: str = "Arial",
        font_size: int = 24,
        primary_color: str = "&H00FFFFFF",  # White
        highlight_color: str = "&H0000D7FF",  # Gold (BGR format)
    ) -> Path:
        """
        Generate an .ass subtitle file from alignment result.

        Args:
            result: AlignmentResult from get_word_timestamps()
            output_path: Path to save the .ass file
            words_per_line: Number of words per subtitle line
            font_name: Font name for subtitles
            font_size: Font size
            primary_color: Primary text color (ASS format &HAABBGGRR)
            highlight_color: Highlight color for karaoke effect

        Returns:
            Path to the generated .ass file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ASS file header
        ass_content = f"""[Script Info]
; Generated by Audio Aligner - Whisper word-level timestamps
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

        # Group words into lines
        groups = result.group_into_phrases(words_per_line)

        for group in groups:
            if not group:
                continue

            start_time = self._format_ass_time(group[0].start)
            end_time = self._format_ass_time(group[-1].end)

            # Build karaoke text with timing
            karaoke_text = ""
            for i, word in enumerate(group):
                # Duration in centiseconds
                duration_cs = int((word.end - word.start) * 100)
                # Use \k for karaoke timing (fill effect)
                karaoke_text += f"{{\\k{duration_cs}}}{word.word} "

            karaoke_text = karaoke_text.strip()

            # Add dialogue line
            ass_content += (
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}\n"
            )

        # Write file
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
        """
        Save word timestamps to a JSON file for debugging/reuse.

        Args:
            result: AlignmentResult from get_word_timestamps()
            output_path: Path to save the JSON file

        Returns:
            Path to the generated JSON file
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "full_text": result.full_text,
            "duration": result.duration,
            "language": result.language,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end} for w in result.words
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  💾 Saved timestamps JSON: {output_path}")
        return output_path


def align_audio_file(
    audio_path: Union[str, Path],
    language: Optional[str] = None,
    save_ass: Optional[Union[str, Path]] = None,
    save_json: Optional[Union[str, Path]] = None,
) -> List[dict]:
    """
    Convenience function to get word timestamps as a list of dicts.

    Args:
        audio_path: Path to the audio file
        language: Optional language code
        save_ass: Optional path to save .ass subtitle file
        save_json: Optional path to save timestamps JSON

    Returns:
        List of dicts with 'word', 'start', 'end' keys
    """
    aligner = AudioAligner()
    result = aligner.get_word_timestamps(audio_path, language)

    # Save files if requested
    if save_ass:
        aligner.generate_ass_file(result, save_ass)
    if save_json:
        aligner.save_timestamps_json(result, save_json)

    return [{"word": w.word, "start": w.start, "end": w.end} for w in result.words]


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_aligner.py <audio_file>")
        print("Example: python audio_aligner.py output/audio/voice_01.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"\n🎤 Aligning: {audio_file}\n")

    aligner = AudioAligner()
    result = aligner.get_word_timestamps(audio_file)

    print(f"\n📝 Full text: {result.full_text}\n")
    print(f"⏱️  Duration: {result.duration:.2f}s")
    print(f"🌍 Language: {result.language}")
    print(f"📊 Words: {len(result.words)}\n")

    print("Word timestamps:")
    print("-" * 50)
    for word in result.words[:20]:  # Show first 20 words
        print(f"  {word.start:6.2f}s - {word.end:6.2f}s : '{word.word}'")

    if len(result.words) > 20:
        print(f"  ... and {len(result.words) - 20} more words")

    # Save .ass file
    ass_path = Path(audio_file).with_suffix(".ass")
    aligner.generate_ass_file(result, ass_path)

    # Save JSON
    json_path = Path(audio_file).with_suffix(".timestamps.json")
    aligner.save_timestamps_json(result, json_path)

    print("\n✅ Alignment complete!")
