"""
Data models for the video generation pipeline.

These dataclasses define the structure of data flowing through each stage
of the pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class VideoFormat(str, Enum):
    """Video format options."""

    SHORT = "short"  # < 60s, 9:16 (vertical)
    LONG = "long"  # 3-10 min, 16:9 (horizontal)


class TTSProvider(str, Enum):
    """Text-to-speech provider options."""

    GTTS = "gtts"
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"


# ============================================================================
# Stage 1: Input Processing
# ============================================================================


@dataclass
class VideoRequest:
    """Input request for video generation."""

    topic: str
    target_audience: str
    format: VideoFormat
    style: str = "educational"
    language: str = "en"

    def validate(self) -> bool:
        """Validate input parameters."""
        if not self.topic or not self.topic.strip():
            raise ValueError("Topic cannot be empty")
        if not self.target_audience or not self.target_audience.strip():
            raise ValueError("Target audience cannot be empty")
        if self.format not in VideoFormat:
            raise ValueError(f"Invalid format: {self.format}")
        return True

    def get_video_dimensions(self) -> tuple[int, int]:
        """Return (width, height) based on format."""
        if self.format == VideoFormat.LONG:
            return (1920, 1080)
        return (1080, 1920)

    def get_target_duration(self) -> tuple[int, int]:
        """Return (min_seconds, max_seconds) based on format."""
        if self.format == VideoFormat.LONG:
            return (180, 600)  # 3-10 minutes
        return (30, 60)  # 30-60 seconds

    def get_image_size(self) -> str:
        """Return image size string for AI image generation."""
        if self.format == VideoFormat.LONG:
            return "1536x1024"
        return "1024x1536"


# ============================================================================
# Stage 2: Research
# ============================================================================


@dataclass
class ResearchResult:
    """Output from the web research stage."""

    topic: str
    key_points: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)  # {"fact": str, "source": str}
    examples: list[str] = field(default_factory=list)
    analogies: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    raw_content: str = ""

    def summary(self) -> str:
        """Get a summary of the research for script generation."""
        parts = [f"Topic: {self.topic}"]

        if self.key_points:
            parts.append("\nKey Points:")
            parts.extend(f"- {point}" for point in self.key_points)

        if self.facts:
            parts.append("\nFacts:")
            parts.extend(f"- {f['fact']}" for f in self.facts)

        if self.examples:
            parts.append("\nExamples:")
            parts.extend(f"- {ex}" for ex in self.examples)

        if self.analogies:
            parts.append("\nAnalogies:")
            parts.extend(f"- {a}" for a in self.analogies)

        return "\n".join(parts)


# ============================================================================
# Stage 3: Script Generation
# ============================================================================


@dataclass
class Scene:
    """A single scene in the video script."""

    scene_number: int
    narration: str  # Text for TTS
    visual_description: str  # Description for image generation
    duration_seconds: float = 0.0  # Estimated/actual duration
    mood: str = "informative"  # Emotional tone
    key_visual_elements: list[str] = field(default_factory=list)

    def word_count(self) -> int:
        """Get word count of narration."""
        return len(self.narration.split())

    def estimate_duration(self, words_per_minute: int = 150) -> float:
        """Estimate duration based on word count."""
        return (self.word_count() / words_per_minute) * 60


@dataclass
class Script:
    """Complete script for the video."""

    title: str
    hook: str
    scenes: list[Scene]
    total_duration_seconds: float = 0.0
    hashtags: list[str] = field(default_factory=list)
    thumbnail_prompt: str = ""
    description: str = ""

    def scene_count(self) -> int:
        """Get total number of scenes."""
        return len(self.scenes)

    def calculate_total_duration(self) -> float:
        """Calculate total duration from all scenes."""
        self.total_duration_seconds = sum(s.duration_seconds for s in self.scenes)
        return self.total_duration_seconds


# ============================================================================
# Stage 4: Scene Planning
# ============================================================================


@dataclass
class PlannedScene:
    """A scene with all planned assets."""

    scene_number: int
    narration: str
    visual_description: str
    image_prompt: str
    duration_seconds: float = 0.0
    mood: str = "informative"
    transition: str = "crossfade"  # crossfade, cut, fade_to_black

    # Paths set during asset generation
    audio_path: Optional[Path] = None
    image_path: Optional[Path] = None


# ============================================================================
# Stage 5: Asset Generation (stored in PlannedScene)
# ============================================================================


@dataclass
class GeneratedAssets:
    """Container for all generated assets for a scene."""

    scene_number: int
    image_path: Path
    audio_path: Path
    duration_seconds: float


# ============================================================================
# Stage 6 & 7: Output
# ============================================================================


@dataclass
class VideoMetadata:
    """Metadata for the generated video."""

    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    format: str = "long"
    sources: list[str] = field(default_factory=list)

    def youtube_description(self) -> str:
        """Generate YouTube-ready description."""
        parts = [self.description]

        if self.hashtags:
            parts.append("\n" + " ".join(f"#{tag}" for tag in self.hashtags))

        if self.sources:
            parts.append("\n\nSources:")
            parts.extend(f"- {src}" for src in self.sources)

        return "\n".join(parts)

    def youtube_tags(self) -> str:
        """Generate comma-separated tags for YouTube."""
        return ", ".join(self.tags)


@dataclass
class VideoOutput:
    """Final output from the video generation pipeline."""

    video_path: Path
    thumbnail_path: Optional[Path] = None
    metadata: Optional[VideoMetadata] = None
    success: bool = True
    error_message: str = ""

    def __str__(self) -> str:
        if self.success:
            return (
                f"VideoOutput(video={self.video_path}, thumbnail={self.thumbnail_path})"
            )
        return f"VideoOutput(error={self.error_message})"


# ============================================================================
# Pipeline State
# ============================================================================


@dataclass
class PipelineState:
    """Tracks the state of the pipeline as it progresses."""

    request: VideoRequest
    research: Optional[ResearchResult] = None
    script: Optional[Script] = None
    planned_scenes: list[PlannedScene] = field(default_factory=list)
    output: Optional[VideoOutput] = None
    current_stage: str = "initialized"
    errors: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error to the pipeline state."""
        self.errors.append(error)

    def is_successful(self) -> bool:
        """Check if pipeline has completed without errors."""
        return len(self.errors) == 0 and self.output is not None and self.output.success
