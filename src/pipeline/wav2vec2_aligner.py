"""
Wav2Vec2 Force Alignment Module - FREE local audio-text alignment using wav2vec2.

This module provides word-level timestamp alignment without requiring any API calls,
based on the PyTorch forced alignment tutorial by Motu Hira.

The wav2vec2 approach works by:
1. Loading a pre-trained wav2vec2 ASR model
2. Getting frame-wise character probabilities from audio
3. Building a trellis matrix for alignment
4. Backtracking to find the optimal path
5. Merging character segments into word timestamps

Usage:
    from wav2vec2_aligner import Wav2Vec2Aligner

    aligner = Wav2Vec2Aligner()
    result = aligner.align(audio_path="audio.wav", transcript="Hello world")
    # Returns AlignmentResult with word timestamps

    # Generate .ass subtitle file
    aligner.generate_ass_file(result, "output/subtitles.ass")

Requirements:
    - torch
    - torchaudio
    - Audio must be converted to 16kHz mono for best results

Based on: https://github.com/harvestingmoon/OBrainRot
Reference: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
"""

import json
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio

# Import shared types - eliminates duplication with audio_aligner
from .alignment_types import AlignmentResult, SubtitleStyle, WordTimestamp

# Re-export for backwards compatibility
__all__ = ["Wav2Vec2Aligner", "AlignmentResult", "WordTimestamp", "align_with_wav2vec2"]


@dataclass
class _Point:
    """Internal class for backtracking path."""

    token_index: int
    time_index: int
    score: float


@dataclass
class _Segment:
    """Internal class for character segments."""

    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


class Wav2Vec2Aligner:
    """
    Free audio-text alignment using wav2vec2 forced alignment.

    This provides word-level timestamps without any API costs by running
    the wav2vec2 model locally. Requires PyTorch and torchaudio.

    The alignment works best when:
    - Audio is clear speech (TTS output works great)
    - Audio is 16kHz mono WAV format
    - Transcript matches the spoken content exactly
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the wav2vec2 aligner.

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"  🔧 Wav2Vec2 Aligner using device: {self.device}")

        # Load model lazily on first use
        self._bundle = None
        self._model = None
        self._labels = None
        self._sample_rate = None

    def _load_model(self):
        """Lazy-load the wav2vec2 model."""
        if self._model is None:
            print("  📥 Loading wav2vec2 model...")
            self._bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self._model = self._bundle.get_model().to(self.device)
            self._labels = self._bundle.get_labels()
            self._sample_rate = self._bundle.sample_rate
            print(f"  ✅ Model loaded (sample rate: {self._sample_rate}Hz)")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for alignment.

        - Converts to uppercase (wav2vec2 uses uppercase labels)
        - Removes punctuation except apostrophes
        - Converts numbers to words
        - Normalizes whitespace
        """
        # Convert to uppercase
        text = text.upper()

        # Remove punctuation but keep apostrophes and spaces
        # The model expects only A-Z, apostrophe, and space
        cleaned = ""
        for char in text:
            if char.isalpha() or char == "'" or char == " ":
                cleaned += char
            elif char.isdigit():
                # Simple number to word conversion
                cleaned += self._number_to_word(char)
            else:
                cleaned += " "

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    def _number_to_word(self, digit: str) -> str:
        """Convert a single digit to word."""
        mapping = {
            "0": "ZERO",
            "1": "ONE",
            "2": "TWO",
            "3": "THREE",
            "4": "FOUR",
            "5": "FIVE",
            "6": "SIX",
            "7": "SEVEN",
            "8": "EIGHT",
            "9": "NINE",
        }
        return mapping.get(digit, "")

    def _format_transcript(self, text: str) -> str:
        """Format text with pipe separators for alignment."""
        words = text.split()
        return "|" + "|".join(words) + "|"

    def _convert_audio_to_16khz(self, audio_path: Path) -> Tuple[torch.Tensor, Path]:
        """
        Convert audio to 16kHz mono format required by wav2vec2.

        Returns the waveform tensor and path to converted file.
        """
        audio_path = Path(audio_path)

        # Try multiple methods to load audio
        waveform = None
        sample_rate = None

        # Method 1: Try Python's built-in wave module (most reliable for WAV)
        if str(audio_path).lower().endswith(".wav"):
            try:
                import numpy as np

                with wave.open(str(audio_path), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    n_frames = wav_file.getnframes()
                    sample_width = wav_file.getsampwidth()

                    # Read raw audio data
                    raw_data = wav_file.readframes(n_frames)

                    # Convert to numpy array based on sample width
                    if sample_width == 1:
                        dtype = np.uint8
                    elif sample_width == 2:
                        dtype = np.int16
                    elif sample_width == 4:
                        dtype = np.int32
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")

                    audio_data = np.frombuffer(raw_data, dtype=dtype)

                    # Reshape for stereo
                    if n_channels > 1:
                        audio_data = audio_data.reshape(-1, n_channels)
                        # Take first channel or average
                        audio_data = audio_data[:, 0]

                    # Normalize to float32 [-1, 1]
                    if dtype == np.uint8:
                        audio_data = (audio_data.astype(np.float32) - 128) / 128
                    else:
                        audio_data = audio_data.astype(np.float32) / np.iinfo(dtype).max

                    # Convert to torch tensor [1, samples]
                    waveform = torch.tensor(audio_data).unsqueeze(0)
                    print(f"  ✅ Loaded via wave module")
            except Exception as e:
                print(f"  ⚠️ wave module failed: {e}")

        # Method 2: Try torchaudio directly
        if waveform is None:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                print(f"  ✅ Loaded via torchaudio")
            except (RuntimeError, Exception) as e:
                print(f"  ⚠️ torchaudio.load failed: {e}")

        # Method 3: Try soundfile if available
        if waveform is None:
            try:
                import soundfile as sf

                data, sample_rate = sf.read(str(audio_path))
                # soundfile returns (samples, channels) or (samples,)
                if len(data.shape) == 1:
                    waveform = torch.tensor(data).unsqueeze(0).float()
                else:
                    waveform = torch.tensor(data.T).float()
                print(f"  ✅ Loaded via soundfile")
            except ImportError:
                pass
            except Exception as e:
                print(f"  ⚠️ soundfile failed: {e}")

        # Method 4: Convert with ffmpeg to raw PCM and read directly
        if waveform is None:
            try:
                import numpy as np

                print(f"  🔄 Converting audio with ffmpeg...")

                # Use ffmpeg to output raw PCM to stdout
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",  # mono
                    "-ar",
                    "16000",  # 16kHz
                    "-f",
                    "s16le",  # raw 16-bit little-endian PCM
                    "-acodec",
                    "pcm_s16le",
                    "pipe:1",  # output to stdout
                ]
                result = subprocess.run(cmd, capture_output=True)

                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

                # Convert raw bytes to numpy array
                audio_data = np.frombuffer(result.stdout, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0

                waveform = torch.tensor(audio_data).unsqueeze(0)
                sample_rate = 16000  # We requested 16kHz from ffmpeg

                print(f"  ✅ Converted via ffmpeg")

                # Already at target sample rate, return early
                return waveform, audio_path

            except FileNotFoundError:
                print(f"  ⚠️ ffmpeg not found")
            except Exception as e:
                print(f"  ⚠️ ffmpeg conversion failed: {e}")

        if waveform is None:
            raise RuntimeError(
                f"Could not load audio file: {audio_path}\n"
                "The file may be corrupted or in an unsupported format.\n"
                "Try installing soundfile: pip install soundfile"
            )

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self._sample_rate:
            print(f"  🔄 Resampling from {sample_rate}Hz to {self._sample_rate}Hz...")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self._sample_rate
            )
            waveform = resampler(waveform)

        return waveform, audio_path

    def _get_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """Get frame-wise label probabilities from audio."""
        with torch.inference_mode():
            emissions, _ = self._model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)
        return emissions[0].cpu().detach()

    def _build_trellis(
        self, emission: torch.Tensor, tokens: List[int], blank_id: int = 0
    ) -> torch.Tensor:
        """
        Build trellis matrix for alignment.

        The trellis represents the probability of transcript labels
        occurring at each time frame.
        """
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frames, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frames - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens[1:]],
            )

        return trellis

    def _backtrack(
        self,
        trellis: torch.Tensor,
        emission: torch.Tensor,
        tokens: List[int],
        blank_id: int = 0,
    ) -> List[_Point]:
        """Find the most likely path through the trellis using backtracking."""
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [_Point(j, t, emission[t, blank_id].exp().item())]

        while j > 0:
            if t <= 0:
                break

            # Determine if we stayed or changed
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            t -= 1
            if changed > stayed:
                j -= 1

            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(_Point(j, t, prob))

        # Fill remaining path
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(_Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    def _merge_repeats(self, path: List[_Point], transcript: str) -> List[_Segment]:
        """Merge repeated characters into segments."""
        i1, i2 = 0, 0
        segments = []

        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1

            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                _Segment(
                    label=transcript[path[i1].token_index],
                    start=path[i1].time_index,
                    end=path[i2 - 1].time_index + 1,
                    score=score,
                )
            )
            i1 = i2

        return segments

    def _merge_words(
        self, segments: List[_Segment], separator: str = "|"
    ) -> List[_Segment]:
        """Merge character segments into word segments."""
        words = []
        i1, i2 = 0, 0

        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    total_length = sum(seg.length for seg in segs)
                    if total_length > 0:
                        score = (
                            sum(seg.score * seg.length for seg in segs) / total_length
                        )
                    else:
                        score = sum(seg.score for seg in segs) / len(segs)
                    words.append(
                        _Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                    )
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1

        return words

    def align(
        self,
        audio_path: Union[str, Path],
        transcript: str,
    ) -> AlignmentResult:
        """
        Align audio with transcript to get word-level timestamps.

        Args:
            audio_path: Path to the audio file
            transcript: The text transcript to align

        Returns:
            AlignmentResult with word timestamps

        Note:
            The transcript should match what's spoken in the audio.
            Minor differences are tolerated but large differences
            will result in poor alignment.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  🎤 Aligning audio with wav2vec2: {audio_path.name}...")

        # Load model if needed
        self._load_model()

        # Preprocess transcript
        processed_text = self._preprocess_text(transcript)
        formatted_transcript = self._format_transcript(processed_text)

        print(f"  📝 Transcript: {processed_text[:50]}...")

        # Load and convert audio
        waveform, _ = self._convert_audio_to_16khz(audio_path)

        # Get duration
        duration = waveform.size(1) / self._sample_rate

        # Get emissions (frame-wise probabilities)
        emission = self._get_emissions(waveform)

        # Build dictionary mapping characters to indices
        dictionary = {c: i for i, c in enumerate(self._labels)}

        # Convert transcript to token indices
        tokens = []
        for c in formatted_transcript:
            if c in dictionary:
                tokens.append(dictionary[c])
            else:
                # Unknown character, use blank
                tokens.append(0)

        if not tokens:
            raise ValueError("No valid tokens found in transcript")

        # Build trellis and find optimal path
        trellis = self._build_trellis(emission, tokens)
        path = self._backtrack(trellis, emission, tokens)

        # Merge into segments and words
        segments = self._merge_repeats(path, formatted_transcript)
        word_segments = self._merge_words(segments)

        # Convert frame indices to timestamps
        ratio = waveform.size(1) / trellis.size(0)

        words = []
        original_words = transcript.split()

        for i, seg in enumerate(word_segments):
            x0 = int(ratio * seg.start)
            x1 = int(ratio * seg.end)
            start_time = x0 / self._sample_rate
            end_time = x1 / self._sample_rate

            # Try to use original word (with punctuation) if available
            if i < len(original_words):
                word_text = original_words[i]
            else:
                word_text = seg.label

            words.append(
                WordTimestamp(
                    word=word_text, start=start_time, end=end_time, score=seg.score
                )
            )

        print(f"  ✅ Aligned {len(words)} words in {duration:.1f}s (FREE!)")

        return AlignmentResult(
            words=words, full_text=transcript, duration=duration, language="en"
        )

    def get_word_timestamps(
        self,
        audio_path: Union[str, Path],
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Get word-level timestamps from an audio file.

        This method signature matches AudioAligner for easy swapping.

        Args:
            audio_path: Path to the audio file
            transcript: The text transcript (required for wav2vec2)
            language: Ignored (wav2vec2 only supports English)

        Returns:
            AlignmentResult containing word timestamps
        """
        if transcript is None:
            raise ValueError(
                "Transcript is required for wav2vec2 alignment. "
                "Use AudioAligner (Whisper) if you don't have a transcript."
            )

        return self.align(audio_path, transcript)

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
            result: AlignmentResult from align()
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
; Generated by Wav2Vec2 Aligner - FREE forced alignment
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
        """
        Save word timestamps to a JSON file.

        Args:
            result: AlignmentResult from align()
            output_path: Path to save the JSON file

        Returns:
            Path to the generated JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use shared to_dict method, add aligner info
        data = result.to_dict()
        data["aligner"] = "wav2vec2"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  💾 Saved timestamps JSON: {output_path}")
        return output_path


def align_with_wav2vec2(
    audio_path: Union[str, Path],
    transcript: str,
    save_ass: Optional[Union[str, Path]] = None,
    save_json: Optional[Union[str, Path]] = None,
) -> List[dict]:
    """
    Convenience function to get word timestamps as a list of dicts.

    This is the simplest way to use wav2vec2 alignment.

    Args:
        audio_path: Path to the audio file
        transcript: The text transcript to align
        save_ass: Optional path to save .ass subtitle file
        save_json: Optional path to save timestamps JSON

    Returns:
        List of dicts with 'word', 'start', 'end', 'score' keys

    Example:
        >>> timestamps = align_with_wav2vec2(
        ...     "audio.wav",
        ...     "Hello world, this is a test.",
        ...     save_ass="output.ass"
        ... )
        >>> print(timestamps[0])
        {'word': 'Hello', 'start': 0.12, 'end': 0.45, 'score': 0.89}
    """
    aligner = Wav2Vec2Aligner()
    result = aligner.align(audio_path, transcript)

    if save_ass:
        aligner.generate_ass_file(result, save_ass)
    if save_json:
        aligner.save_timestamps_json(result, save_json)

    # Use shared to_dict method
    return [w.to_dict() for w in result.words]


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python wav2vec2_aligner.py <audio_file> <transcript>")
        print('Example: python wav2vec2_aligner.py audio.wav "Hello world"')
        print("\nNote: Transcript must match what is spoken in the audio!")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcript = sys.argv[2]

    print(f"\n🎤 Wav2Vec2 Forced Alignment (FREE!)")
    print(f"=" * 50)
    print(f"📁 Audio: {audio_file}")
    print(f"📝 Transcript: {transcript[:50]}...")
    print(f"=" * 50)

    aligner = Wav2Vec2Aligner()
    result = aligner.align(audio_file, transcript)

    print(f"\n📊 Results:")
    print(f"  ⏱️  Duration: {result.duration:.2f}s")
    print(f"  📝 Words: {len(result.words)}")
    print(f"\n  Word timestamps:")
    print(f"  {'-' * 45}")

    for word in result.words[:20]:
        print(
            f"    {word.start:6.2f}s - {word.end:6.2f}s : '{word.word}' (score: {word.score:.2f})"
        )

    if len(result.words) > 20:
        print(f"    ... and {len(result.words) - 20} more words")

    # Save .ass file
    ass_path = Path(audio_file).with_suffix(".ass")
    aligner.generate_ass_file(result, ass_path)

    # Save JSON
    json_path = Path(audio_file).with_suffix(".wav2vec2.json")
    aligner.save_timestamps_json(result, json_path)

    print(f"\n✅ Alignment complete!")
    print(f"   💰 Cost: $0.00 (wav2vec2 runs locally)")
