"""
Voice Generator Module - Converts text to speech for video narration.

This module handles text-to-speech generation with support for multiple
providers: gTTS (free), OpenAI TTS, and ElevenLabs.
"""

import os
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from ..utils.ai_client import AIClient, get_ai_client
from .models import PlannedScene, TTSProvider


class VoiceGenerator:
    """Generates voice audio from text using various TTS providers."""

    def __init__(
        self,
        provider: TTSProvider = TTSProvider.GTTS,
        ai_client: Optional[AIClient] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the voice generator.

        Args:
            provider: TTS provider to use (gtts, openai, elevenlabs)
            ai_client: AI client for OpenAI TTS
            output_dir: Directory to save audio files
        """
        # Check environment variable for provider override
        env_provider = os.environ.get("TTS_PROVIDER", "").lower()
        if env_provider and env_provider in [p.value for p in TTSProvider]:
            provider = TTSProvider(env_provider)

        self.provider = provider
        self.ai_client = ai_client or get_ai_client()
        self.output_dir = output_dir or Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # OpenAI TTS settings
        self.openai_voice = os.environ.get("TTS_VOICE", "alloy")
        self.openai_speed = float(os.environ.get("TTS_SPEED", "1.0"))

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
        """
        if output_path is None:
            output_path = self.output_dir / f"voice_{scene_number:02d}.wav"

        print(f"  🎤 Generating voice ({self.provider.value})...")

        # Route to appropriate provider
        if self.provider == TTSProvider.OPENAI:
            return self._generate_openai(text, output_path)
        elif self.provider == TTSProvider.ELEVENLABS:
            return self._generate_elevenlabs(text, output_path)
        else:
            return self._generate_gtts(text, output_path)

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

        try:
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

        except Exception as e:
            print(f"  ❌ gTTS generation failed: {e}")
            raise

    def _generate_openai(self, text: str, output_path: Path) -> tuple[Path, float]:
        """
        Generate voice using OpenAI's TTS API.

        Args:
            text: Text to convert
            output_path: Output path for audio file

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        try:
            # Generate audio using AI client
            audio_bytes = self.ai_client.generate_speech(
                text=text,
                voice=self.openai_voice,
                speed=self.openai_speed,
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

        except Exception as e:
            print(f"  ❌ OpenAI TTS failed: {e}")
            # Fall back to gTTS
            print("  ⚠️ Falling back to gTTS...")
            return self._generate_gtts(text, output_path)

    def _generate_elevenlabs(self, text: str, output_path: Path) -> tuple[Path, float]:
        """
        Generate voice using ElevenLabs API (premium quality).

        Args:
            text: Text to convert
            output_path: Output path for audio file

        Returns:
            Tuple of (audio_path, duration_seconds)
        """
        try:
            import requests

            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY not set")

            # Default voice ID (Rachel) - can be customized
            voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key,
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

        except Exception as e:
            print(f"  ❌ ElevenLabs TTS failed: {e}")
            # Fall back to gTTS
            print("  ⚠️ Falling back to gTTS...")
            return self._generate_gtts(text, output_path)

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
            try:
                audio_path, duration = self.generate(
                    text=scene.narration,
                    scene_number=scene.scene_number,
                )
                results[scene.scene_number] = (audio_path, duration)

                # Update scene with actual values
                scene.audio_path = audio_path
                scene.duration_seconds = duration

            except Exception as e:
                print(
                    f"  ❌ Failed to generate voice for scene {scene.scene_number}: {e}"
                )
                raise

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
