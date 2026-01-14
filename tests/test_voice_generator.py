"""
Tests for the voice generator module.

These tests verify that:
1. VoiceConfig is properly structured
2. VoiceGenerator handles different providers correctly
3. Fallback behavior works as expected
4. Error handling is proper with TTSError and TTSFallbackError
"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.pipeline.models import TTSProvider
from src.pipeline.voice_generator import (
    TTSError,
    TTSFallbackError,
    VoiceConfig,
    VoiceGenerator,
    create_voice_generator,
)


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = VoiceConfig()

        assert config.provider == TTSProvider.GTTS
        assert config.openai_voice == "marin"
        assert config.openai_speed == 1.0
        assert config.openai_instructions is None
        assert config.elevenlabs_api_key is None
        assert config.elevenlabs_voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert config.fallback_enabled is True
        assert config.fallback_silent is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VoiceConfig(
            provider=TTSProvider.OPENAI,
            openai_voice="nova",
            openai_speed=1.5,
            openai_instructions="Speak cheerfully",
            fallback_enabled=False,
        )

        assert config.provider == TTSProvider.OPENAI
        assert config.openai_voice == "nova"
        assert config.openai_speed == 1.5
        assert config.openai_instructions == "Speak cheerfully"
        assert config.fallback_enabled is False

    def test_validate_elevenlabs_without_key(self):
        """Test validation fails for ElevenLabs without API key."""
        config = VoiceConfig(
            provider=TTSProvider.ELEVENLABS,
            elevenlabs_api_key=None,
        )

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY is required"):
            config.validate()

    def test_validate_elevenlabs_with_key(self):
        """Test validation passes for ElevenLabs with API key."""
        config = VoiceConfig(
            provider=TTSProvider.ELEVENLABS,
            elevenlabs_api_key="test-key",
        )
        config.validate()  # Should not raise

    def test_validate_gtts_no_key_needed(self):
        """Test gTTS doesn't require any keys."""
        config = VoiceConfig(provider=TTSProvider.GTTS)
        config.validate()  # Should not raise

    def test_validate_openai_no_key_in_config(self):
        """Test OpenAI config validation (key is in AIClient)."""
        config = VoiceConfig(provider=TTSProvider.OPENAI)
        config.validate()  # Should not raise - key is handled elsewhere


class TestVoiceGenerator:
    """Tests for VoiceGenerator class."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = MagicMock()
        client.generate_speech.return_value = b"fake audio data"
        return client

    @pytest.fixture
    def gtts_generator(self, tmp_path):
        """Create a VoiceGenerator with gTTS provider."""
        config = VoiceConfig(provider=TTSProvider.GTTS)
        return VoiceGenerator(config=config, output_dir=tmp_path)

    @pytest.fixture
    def openai_generator(self, tmp_path, mock_ai_client):
        """Create a VoiceGenerator with OpenAI provider."""
        config = VoiceConfig(provider=TTSProvider.OPENAI)
        return VoiceGenerator(
            config=config,
            ai_client=mock_ai_client,
            output_dir=tmp_path,
        )

    def test_init_default_config(self, tmp_path):
        """Test initialization with default config."""
        generator = VoiceGenerator(output_dir=tmp_path)

        assert generator.config.provider == TTSProvider.GTTS
        assert generator.output_dir == tmp_path

    def test_init_custom_config(self, tmp_path):
        """Test initialization with custom config."""
        config = VoiceConfig(
            provider=TTSProvider.OPENAI,
            openai_voice="nova",
        )
        generator = VoiceGenerator(config=config, output_dir=tmp_path)

        assert generator.config.provider == TTSProvider.OPENAI
        assert generator.config.openai_voice == "nova"

    def test_provider_property(self, gtts_generator):
        """Test provider property returns correct value."""
        assert gtts_generator.provider == TTSProvider.GTTS

    def test_output_dir_created(self, tmp_path):
        """Test that output directory is created on init."""
        output_dir = tmp_path / "audio" / "nested"
        generator = VoiceGenerator(output_dir=output_dir)

        assert output_dir.exists()

    @patch("gtts.gTTS")
    @patch("src.pipeline.voice_generator.AudioSegment")
    def test_generate_gtts_success(
        self, mock_audio_segment, mock_gtts, gtts_generator, tmp_path
    ):
        """Test successful gTTS generation."""
        # Setup mocks
        mock_tts_instance = MagicMock()
        mock_gtts.return_value = mock_tts_instance

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=5000)  # 5 seconds in ms
        mock_audio_segment.from_mp3.return_value = mock_audio

        # Generate
        path, duration = gtts_generator.generate(
            text="Hello world",
            scene_number=1,
        )

        # Verify
        assert path.suffix == ".wav"
        assert duration == 5.0
        mock_gtts.assert_called_once_with(text="Hello world", lang="en", slow=False)

    @patch("src.pipeline.voice_generator.AudioSegment")
    def test_generate_openai_success(
        self, mock_audio_segment, openai_generator, mock_ai_client
    ):
        """Test successful OpenAI TTS generation."""
        # Setup mock
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=3000)  # 3 seconds
        mock_audio_segment.from_mp3.return_value = mock_audio

        # Generate
        path, duration = openai_generator.generate(
            text="Test narration",
            scene_number=2,
        )

        # Verify
        assert path.suffix == ".wav"
        assert duration == 3.0
        mock_ai_client.generate_speech.assert_called_once()


class TestVoiceGeneratorFallback:
    """Tests for fallback behavior."""

    @pytest.fixture
    def failing_openai_generator(self, tmp_path):
        """Create generator where OpenAI fails."""
        mock_client = MagicMock()
        mock_client.generate_speech.side_effect = Exception("API Error")

        config = VoiceConfig(
            provider=TTSProvider.OPENAI,
            fallback_enabled=True,
        )
        return VoiceGenerator(
            config=config,
            ai_client=mock_client,
            output_dir=tmp_path,
        )

    @patch("gtts.gTTS")
    @patch("src.pipeline.voice_generator.AudioSegment")
    def test_fallback_to_gtts_on_openai_failure(
        self, mock_audio_segment, mock_gtts, failing_openai_generator
    ):
        """Test that OpenAI failure falls back to gTTS."""
        # Setup mocks for gTTS fallback
        mock_tts = MagicMock()
        mock_gtts.return_value = mock_tts

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=2000)
        mock_audio_segment.from_mp3.return_value = mock_audio

        # Generate - should fall back to gTTS
        path, duration = failing_openai_generator.generate(
            text="Test text",
            scene_number=1,
        )

        # Verify gTTS was used as fallback
        mock_gtts.assert_called_once()
        assert path.suffix == ".wav"

    def test_fallback_disabled_raises_error(self, tmp_path):
        """Test that disabled fallback raises TTSFallbackError."""
        mock_client = MagicMock()
        mock_client.generate_speech.side_effect = Exception("API Error")

        config = VoiceConfig(
            provider=TTSProvider.OPENAI,
            fallback_enabled=False,  # Disable fallback
        )
        generator = VoiceGenerator(
            config=config,
            ai_client=mock_client,
            output_dir=tmp_path,
        )

        with pytest.raises(TTSFallbackError, match="Fallback is disabled"):
            generator.generate(text="Test", scene_number=1)


class TestTTSErrors:
    """Tests for TTS error classes."""

    def test_tts_error_is_exception(self):
        """Test TTSError is a proper exception."""
        error = TTSError("Something went wrong")
        assert isinstance(error, Exception)
        assert str(error) == "Something went wrong"

    def test_tts_fallback_error_is_tts_error(self):
        """Test TTSFallbackError inherits from TTSError."""
        error = TTSFallbackError("Fallback disabled")
        assert isinstance(error, TTSError)
        assert isinstance(error, Exception)


class TestCreateVoiceGenerator:
    """Tests for the create_voice_generator factory function."""

    def test_create_with_defaults(self, tmp_path):
        """Test factory with default settings."""
        generator = create_voice_generator(output_dir=tmp_path)

        assert generator.provider == TTSProvider.GTTS
        assert generator.config.fallback_enabled is True

    def test_create_with_custom_settings(self, tmp_path):
        """Test factory with custom settings."""
        generator = create_voice_generator(
            provider=TTSProvider.OPENAI,
            openai_voice="coral",
            openai_speed=1.2,
            openai_instructions="Be energetic",
            fallback_enabled=False,
            output_dir=tmp_path,
        )

        assert generator.provider == TTSProvider.OPENAI
        assert generator.config.openai_voice == "coral"
        assert generator.config.openai_speed == 1.2
        assert generator.config.openai_instructions == "Be energetic"
        assert generator.config.fallback_enabled is False

    def test_create_elevenlabs(self, tmp_path):
        """Test factory for ElevenLabs provider."""
        generator = create_voice_generator(
            provider=TTSProvider.ELEVENLABS,
            elevenlabs_api_key="test-key",
            elevenlabs_voice_id="custom-voice-id",
            output_dir=tmp_path,
        )

        assert generator.provider == TTSProvider.ELEVENLABS
        assert generator.config.elevenlabs_api_key == "test-key"
        assert generator.config.elevenlabs_voice_id == "custom-voice-id"


class TestVoiceGeneratorCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_removes_files(self, tmp_path):
        """Test that cleanup removes audio files."""
        generator = VoiceGenerator(output_dir=tmp_path)

        # Create some fake audio files
        (tmp_path / "voice_01.wav").touch()
        (tmp_path / "voice_02.wav").touch()
        (tmp_path / "voice_01.mp3").touch()

        # Cleanup
        generator.cleanup()

        # Verify files are removed
        assert not (tmp_path / "voice_01.wav").exists()
        assert not (tmp_path / "voice_02.wav").exists()
        assert not (tmp_path / "voice_01.mp3").exists()

    def test_cleanup_handles_missing_dir(self, tmp_path):
        """Test cleanup handles non-existent directory gracefully."""
        generator = VoiceGenerator(output_dir=tmp_path / "nonexistent")
        generator.output_dir.rmdir()  # Remove the directory

        # Should not raise
        generator.cleanup()
