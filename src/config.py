"""
Centralized Configuration Module

This is the ONLY place where environment variables should be read.
All other modules should receive configuration via dependency injection.

Usage:
    from src.config import AppConfig

    # At application entry point (once):
    config = AppConfig.from_environment()

    # Pass to components that need it:
    pipeline = VideoPipeline(config=config)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class AIProviderConfig:
    """Configuration for AI providers (OpenAI/Gemini)."""

    provider: str = "openai"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_image_model: str = "gpt-image-1"
    openai_image_quality: str = "low"

    # Gemini settings
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"

    def validate(self) -> None:
        """Validate that required API keys are present."""
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        if self.provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when using Gemini provider")


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for Text-to-Speech."""

    provider: str = "openai"  # "gtts", "openai", "elevenlabs"

    # OpenAI TTS settings
    openai_voice: str = "marin"
    openai_speed: float = 1.0
    openai_model: str = "gpt-4o-mini-tts"
    openai_instructions: Optional[str] = None

    # ElevenLabs settings
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    def validate(self) -> None:
        """Validate TTS configuration."""
        valid_providers = ("gtts", "openai", "elevenlabs")
        if self.provider not in valid_providers:
            raise ValueError(f"TTS provider must be one of: {valid_providers}")
        if self.provider == "elevenlabs" and not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is required when using ElevenLabs")


@dataclass(frozen=True)
class SubtitleConfig:
    """Configuration for subtitle generation and alignment."""

    aligner: str = "wav2vec2"  # "whisper" or "wav2vec2"
    font: str = "Arial-Bold"
    font_size: int = 80
    highlight_color: str = "yellow"
    text_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 4

    def validate(self) -> None:
        """Validate subtitle configuration."""
        valid_aligners = ("whisper", "wav2vec2")
        if self.aligner not in valid_aligners:
            raise ValueError(f"Subtitle aligner must be one of: {valid_aligners}")


@dataclass(frozen=True)
class VideoConfig:
    """Configuration for video composition."""

    background_music_volume: float = 0.1
    transition_duration: float = 0.5
    enable_motion_effects: bool = True
    enable_sound_effects: bool = True


@dataclass(frozen=True)
class PathConfig:
    """Configuration for file paths."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    assets_dir: Path = field(default_factory=lambda: Path("assets"))
    temp_dir: Path = field(default_factory=lambda: Path("output/temp"))

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    """
    Main application configuration.

    This is an immutable configuration object that should be created once
    at application startup and passed to all components that need it.
    """

    ai: AIProviderConfig
    tts: TTSConfig
    subtitle: SubtitleConfig
    video: VideoConfig
    paths: PathConfig

    # YouTube upload settings
    youtube_client_secrets_path: Optional[Path] = None

    # Feature flags
    viral_mode: bool = True
    auto_cleanup: bool = False
    fallback_tts_enabled: bool = True  # Whether to fall back to gTTS on OpenAI failure

    @classmethod
    def from_environment(cls, dotenv_path: Optional[Path] = None) -> "AppConfig":
        """
        Create configuration from environment variables.

        This method should be called ONCE at application startup.
        The resulting config object should be passed to all components
        via dependency injection.

        Args:
            dotenv_path: Optional path to .env file. If not provided,
                        python-dotenv will search for it automatically.

        Returns:
            Immutable AppConfig instance
        """
        # Load .env file once
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        # Build sub-configs
        ai_config = AIProviderConfig(
            provider=os.environ.get("AI_PROVIDER", "openai").lower(),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            openai_image_model=os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            openai_image_quality=os.environ.get("OPENAI_IMAGE_QUALITY", "low"),
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        )

        tts_config = TTSConfig(
            provider=os.environ.get("TTS_PROVIDER", "openai").lower(),
            openai_voice=os.environ.get("TTS_VOICE", "marin"),
            openai_speed=float(os.environ.get("TTS_SPEED", "1.0")),
            openai_model=os.environ.get("TTS_MODEL", "gpt-4o-mini-tts"),
            openai_instructions=os.environ.get("TTS_INSTRUCTIONS"),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.environ.get(
                "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
            ),
        )

        subtitle_config = SubtitleConfig(
            aligner=os.environ.get("SUBTITLE_ALIGNER", "wav2vec2").lower(),
            font=os.environ.get("SUBTITLE_FONT", "Arial-Bold"),
            font_size=int(os.environ.get("SUBTITLE_FONT_SIZE", "80")),
            highlight_color=os.environ.get("SUBTITLE_HIGHLIGHT_COLOR", "yellow"),
            text_color=os.environ.get("SUBTITLE_TEXT_COLOR", "white"),
            stroke_color=os.environ.get("SUBTITLE_STROKE_COLOR", "black"),
            stroke_width=int(os.environ.get("SUBTITLE_STROKE_WIDTH", "4")),
        )

        video_config = VideoConfig(
            background_music_volume=float(
                os.environ.get("BACKGROUND_MUSIC_VOLUME", "0.1")
            ),
            transition_duration=float(os.environ.get("TRANSITION_DURATION", "0.5")),
            enable_motion_effects=os.environ.get(
                "ENABLE_MOTION_EFFECTS", "true"
            ).lower()
            == "true",
            enable_sound_effects=os.environ.get("ENABLE_SOUND_EFFECTS", "true").lower()
            == "true",
        )

        paths_config = PathConfig(
            output_dir=Path(os.environ.get("OUTPUT_DIR", "output")),
            assets_dir=Path(os.environ.get("ASSETS_DIR", "assets")),
            temp_dir=Path(os.environ.get("TEMP_DIR", "output/temp")),
        )

        # YouTube settings
        youtube_secrets = os.environ.get("YOUTUBE_CLIENT_SECRETS")
        youtube_path = Path(youtube_secrets) if youtube_secrets else None

        return cls(
            ai=ai_config,
            tts=tts_config,
            subtitle=subtitle_config,
            video=video_config,
            paths=paths_config,
            youtube_client_secrets_path=youtube_path,
            viral_mode=os.environ.get("VIRAL_MODE", "true").lower() == "true",
            auto_cleanup=os.environ.get("AUTO_CLEANUP", "false").lower() == "true",
            fallback_tts_enabled=os.environ.get("FALLBACK_TTS_ENABLED", "true").lower()
            == "true",
        )

    def validate(self) -> None:
        """
        Validate all configuration settings.

        Raises:
            ValueError: If any configuration is invalid
        """
        self.ai.validate()
        self.tts.validate()
        self.subtitle.validate()

    def ensure_ready(self) -> None:
        """
        Prepare the configuration for use.

        This validates settings and creates necessary directories.
        Call this once after creating the config.
        """
        self.validate()
        self.paths.ensure_directories()


# Lazy-loaded default config for backwards compatibility
# Prefer explicit config passing via dependency injection
_default_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get or create the default application configuration.

    DEPRECATED: Prefer creating an AppConfig explicitly and passing it
    via dependency injection. This function exists only for backwards
    compatibility during the refactoring period.

    Returns:
        AppConfig instance (created on first call, cached thereafter)
    """
    global _default_config
    if _default_config is None:
        _default_config = AppConfig.from_environment()
    return _default_config


def reset_config() -> None:
    """
    Reset the cached default configuration.

    This is primarily useful for testing.
    """
    global _default_config
    _default_config = None
