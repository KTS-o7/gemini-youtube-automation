"""
Shared Alignment Types - Common data structures for audio alignment.

This module defines the shared types used by both the Whisper-based
AudioAligner and the free Wav2Vec2Aligner, eliminating code duplication.

Usage:
    from .alignment_types import WordTimestamp, AlignmentResult
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class WordTimestamp:
    """A single word with its timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    score: float = 1.0  # Confidence score (used by wav2vec2)

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"WordTimestamp('{self.word}', {self.start:.2f}s - {self.end:.2f}s)"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTimestamp":
        """Create from dictionary."""
        return cls(
            word=data["word"],
            start=data["start"],
            end=data["end"],
            score=data.get("score", 1.0),
        )


@dataclass
class AlignmentResult:
    """Result of audio alignment containing all word timestamps."""

    words: List[WordTimestamp]
    full_text: str
    duration: float  # Total audio duration
    language: str = "en"

    def get_words_in_range(self, start: float, end: float) -> List[WordTimestamp]:
        """Get all words within a time range."""
        return [w for w in self.words if w.start >= start and w.end <= end]

    def group_into_phrases(self, words_per_group: int = 3) -> List[List[WordTimestamp]]:
        """
        Group words into phrases for display.

        Groups are split on:
        - Reaching words_per_group limit
        - Sentence-ending punctuation (. ! ?)
        - Clause-ending punctuation (, ; :) if near limit

        Args:
            words_per_group: Target number of words per group

        Returns:
            List of word groups
        """
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

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "words": [w.to_dict() for w in self.words],
            "full_text": self.full_text,
            "duration": self.duration,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentResult":
        """Create from dictionary."""
        return cls(
            words=[WordTimestamp.from_dict(w) for w in data["words"]],
            full_text=data["full_text"],
            duration=data["duration"],
            language=data.get("language", "en"),
        )

    def __len__(self) -> int:
        """Return number of words."""
        return len(self.words)

    def __iter__(self):
        """Iterate over words."""
        return iter(self.words)


@dataclass
class SubtitleStyle:
    """Configuration for subtitle appearance."""

    font_name: str = "Arial Black"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"  # White (ASS format: AABBGGRR)
    highlight_color: str = "&H0000D4FF"  # Gold/Yellow
    outline_color: str = "&H00000000"  # Black
    outline_width: int = 3
    shadow_depth: int = 2
    alignment: int = 2  # Bottom center
    margin_v: int = 50  # Vertical margin

    def to_ass_style(self, name: str = "Default") -> str:
        """Generate ASS style line."""
        return (
            f"Style: {name},{self.font_name},{self.font_size},"
            f"{self.primary_color},{self.primary_color},{self.outline_color},{self.outline_color},"
            f"0,0,0,0,100,100,0,0,1,{self.outline_width},{self.shadow_depth},"
            f"{self.alignment},10,10,{self.margin_v},1"
        )

    def to_highlight_style(self, name: str = "Highlight") -> str:
        """Generate ASS highlight style line."""
        return (
            f"Style: {name},{self.font_name},{self.font_size},"
            f"{self.highlight_color},{self.highlight_color},{self.outline_color},{self.outline_color},"
            f"0,0,0,0,100,100,0,0,1,{self.outline_width},{self.shadow_depth},"
            f"{self.alignment},10,10,{self.margin_v},1"
        )
