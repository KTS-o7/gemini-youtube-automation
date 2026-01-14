"""
Voice Generator Module - Converts text to speech for video narration.

This module handles text-to-speech generation with support for multiple
providers: gTTS (free), OpenAI TTS, and ElevenLabs.

OpenAI TTS (gpt-4o-mini-tts):
- Voices: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar
- Best quality: marin, cedar
- Cost: $0.015/minute
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from ..utils.ai_client import AIClient, create_ai_client
from .models import PlannedScene, TTSProvider

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Raised when TTS generation fails."""

    pass


class TTSFallbackError(TTSError):
    """Raised when TTS fails and fallback is disabled."""

    pass


@dataclass
class VoiceConfig:
    """Configuration for voice generation.

    This dataclass holds all TTS-related settings. Create this once
    at application startup and pass it to VoiceGenerator.
    """

    provider: TTSProvider = TTSProvider.GTTS

    # OpenAI TTS settings
    openai_voice: str = "marin"
    openai_speed: float = 1.0
    openai_instructions: Optional[str] = None

    # ElevenLabs settings
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # Fallback behavior
    fallback_enabled: bool = True  # If True, falls back to gTTS on failure
    fallback_silent: bool = False  # If True, don't log fallback warnings

    def validate(self) -> None:
        """Validate the configuration."""
        if self.provider == TTSProvider.ELEVENLABS and not self.elevenlabs_api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is required when using ElevenLabs provider"
            )


class VoiceGenerator:
    """Generates voice audio from text using various TTS providers."""

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        ai_client: Optional[AIClient] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the voice generator.

        Args:
            config: Voice generation configuration. If not provided,
                   defaults will be used.
            ai_client: AI client for OpenAI TTS. If not provided,
                      a new client will be created when needed.
            output_dir: Directory to save audio files.
        """
        self.config = config or VoiceConfig()
        self._ai_client = ai_client
        self.output_dir = output_dir or Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ai_client(self) -> AIClient:
        """Lazy-load AI client when needed."""
        if self._ai_client is None:
            self._ai_client = create_ai_client()
        return self._ai_client

    @property
    def provider(self) -> TTSProvider:
        """Get the current TTS provider."""
        return self.config.provider

    def generate(
        self,
        text: str,
        output_path: Optional[Path] = None,
        scene_number: int = 0,
    ) -> tuple[Path, float]:
        """
        Generate voice audio from text.

        Args:
            text: Text to convert to speech
            output_path: Optional output path for the audio file
            scene_number: Scene number for default filename

        Returns:
            Tuple of (audio_path, duration_seconds)

        Raises:
            TTSError: If generation fails and fallback is disabled
        """
        if output_path is None:
            output_path = self.output_dir / f"voice_{scene_number:02d}.wav"

        print(f"  🎤 Generating voice ({self.provider.value})...")

        # Route to appropriate provider
        if self.provider == TTSProvider.OPENAI:
            return self._generate_with_fallback(
                primary_fn=lambda: self._generate_openai(text, output_path),
                fallback_fn=lambda: self._generate_gtts(text, output_path),
                provider_name="OpenAI TTS",
            )
        elif self.provider == TTSProvider.ELEVENLABS:
            return self._generate_with_fallback(
                primary_fn=lambda: self._generate_elevenlabs(text, output_path),
                fallback_fn=lambda: self._generate_gtts(text, output_path),
                provider_name="ElevenLabs",
            )
        else:
            return self._generate_gtts(text, output_path)

    def _generate_with_fallback(
        self,
        primary_fn,
        fallback_fn,
        provider_name: str,
    ) -> tuple[Path, float]:
        """
        Execute primary TTS function with optional fallback.

        Args:
            primary_fn: Primary TTS generation function
            fallback_fn: Fallback TTS generation function (gTTS)
            provider_name: Name of primary provider for error messages

        Returns:
            Tuple of (audio_path, duration_seconds)

        Raises:
            TTSFallbackError: If primary fails and fallback is disabled
            TTSError: If both primary and fallback fail
        """
        try:
            return primary_fn()
        except Exception as e:
            error_msg = f"{provider_name} failed: {e}"
            logger.error(error_msg)
            print(f"  ❌ {error_msg}")

            if not self.config.fallback_enabled:
                raise TTSFallbackError(
                    f"{error_msg}. Fallback is disabled. "
                    f"Set fallback_enabled=True in VoiceConfig to use gTTS as fallback."
                ) from e

            # Log fallback explicitly
            if not self.config.fallback_silent:
                warning_msg = (
                    f"Falling back to gTTS. Original error: {e}. "
                    f"To disable fallback and fail fast, set fallback_enabled=False."
                )
                logger.warning(warning_msg)
                print(f"  ⚠️ {warning_msg}")

            try:
                return fallback_fn()
            except Exception as fallback_error:
                raise TTSError(
                    f"Both {provider_name} and fallback gTTS failed. "
                    f"Primary error: {e}. Fallback error: {fallback_error}"
                ) from fallback_error

    def _generate_gtts(self, text: str, output_path: Path) -> tuple[Path, float]:
        """
        Generate voice using Google Text-to-Speech (free).

        Args:
            text: Text to convert
            output_path: Output path for audio file

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        from gtts import gTTS

        # Generate MP3 first
        temp_mp3 = output_path.with_suffix(".temp.mp3")

        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(str(temp_mp3))

        # Convert to WAV for better compatibility
        wav_path = output_path.with_suffix(".wav")
        audio = AudioSegment.from_mp3(str(temp_mp3))
        audio.export(str(wav_path), format="wav", codec="pcm_s16le")

        # Get duration
        duration = len(audio) / 1000.0  # Convert ms to seconds

        # Cleanup temp file
        temp_mp3.unlink(missing_ok=True)

        print(f"  ✅ Voice generated: {wav_path.name} ({duration:.1f}s)")
        return wav_path, duration

    def _generate_openai(self, text: str, output_path: Path) -> tuple[Path, float]:
        """
        Generate voice using OpenAI's TTS API (gpt-4o-mini-tts).

        Args:
            text: Text to convert
            output_path: Output path for audio file

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        audio_bytes = self.ai_client.generate_speech(
            text=text,
            voice=self.config.openai_voice,
            speed=self.config.openai_speed,
            instructions=self.config.openai_instructions,
        )

        # Save MP3
        mp3_path = output_path.with_suffix(".mp3")
        with open(mp3_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to WAV for better compatibility with moviepy
        wav_path = output_path.with_suffix(".wav")
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(wav_path), format="wav", codec="pcm_s16le")

        # Get duration
        duration = len(audio) / 1000.0

        # Cleanup MP3
        mp3_path.unlink(missing_ok=True)

        print(f"  ✅ Voice generated: {wav_path.name} ({duration:.1f}s)")
        return wav_path, duration

    def _generate_elevenlabs(self, text: str, output_path: Path) -> tuple[Path, float]:
        """
        Generate voice using ElevenLabs API (premium quality).

        Args:
            text: Text to convert
            output_path: Output path for audio file

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        import requests

        if not self.config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")

        voice_id = self.config.elevenlabs_voice_id

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.config.elevenlabs_api_key,
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }

        response = requests.post(url, json=data, headers=headers, timeout=120)
        response.raise_for_status()

        # Save MP3
        mp3_path = output_path.with_suffix(".mp3")
        with open(mp3_path, "wb") as f:
            f.write(response.content)

        # Convert to WAV
        wav_path = output_path.with_suffix(".wav")
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(wav_path), format="wav", codec="pcm_s16le")

        # Get duration
        duration = len(audio) / 1000.0

        # Cleanup MP3
        mp3_path.unlink(missing_ok=True)

        print(f"  ✅ Voice generated: {wav_path.name} ({duration:.1f}s)")
        return wav_path, duration

    def batch_generate(
        self,
        scenes: list[PlannedScene],
    ) -> dict[int, tuple[Path, float]]:
        """
        Generate voice for multiple scenes.

        Args:
            scenes: List of planned scenes with narration

        Returns:
            Dictionary mapping scene numbers to (audio_path, duration) tuples
        """
        print(f"🎤 Generating voice for {len(scenes)} scenes...")

        results = {}
        for scene in scenes:
            audio_path, duration = self.generate(
                text=scene.narration,
                scene_number=scene.scene_number,
            )
            results[scene.scene_number] = (audio_path, duration)

            # Update scene with actual values
            scene.audio_path = audio_path
            scene.duration_seconds = duration

        print(f"✅ Generated voice for {len(results)} scenes")
        return results

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0

    def cleanup(self) -> None:
        """Remove all generated audio files."""
        if self.output_dir.exists():
            for pattern in ["*.wav", "*.mp3"]:
                for file in self.output_dir.glob(pattern):
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file}: {e}")


# =============================================================================
# Factory function for backwards compatibility
# =============================================================================


def create_voice_generator(
    provider: TTSProvider = TTSProvider.GTTS,
    ai_client: Optional[AIClient] = None,
    output_dir: Optional[Path] = None,
    openai_voice: str = "marin",
    openai_speed: float = 1.0,
    openai_instructions: Optional[str] = None,
    elevenlabs_api_key: Optional[str] = None,
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    fallback_enabled: bool = True,
) -> VoiceGenerator:
    """
    Create a VoiceGenerator with the given settings.

    This is a convenience factory that creates both the config
    and generator in one call.

    Args:
        provider: TTS provider to use
        ai_client: Optional AI client for OpenAI TTS
        output_dir: Directory to save audio files
        openai_voice: Voice for OpenAI TTS
        openai_speed: Speed for OpenAI TTS
        openai_instructions: Instructions for OpenAI TTS
        elevenlabs_api_key: API key for ElevenLabs
        elevenlabs_voice_id: Voice ID for ElevenLabs
        fallback_enabled: Whether to fall back to gTTS on failure

    Returns:
        Configured VoiceGenerator instance
    """
    config = VoiceConfig(
        provider=provider,
        openai_voice=openai_voice,
        openai_speed=openai_speed,
        openai_instructions=openai_instructions,
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        fallback_enabled=fallback_enabled,
    )

    return VoiceGenerator(
        config=config,
        ai_client=ai_client,
        output_dir=output_dir,
    )
