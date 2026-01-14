"""
Tests for the configuration module.

These tests verify that:
1. AppConfig can be created from environment variables
2. Configuration validation works correctly
3. Sub-configs are properly structured
4. Default values are applied correctly
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    AIProviderConfig,
    AppConfig,
    PathConfig,
    SubtitleConfig,
    TTSConfig,
    VideoConfig,
    get_config,
    reset_config,
)


class TestAIProviderConfig:
    """Tests for AIProviderConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AIProviderConfig()
        assert config.provider == "openai"
        assert config.openai_model == "gpt-4o-mini"
        assert config.openai_image_model == "gpt-image-1"
        assert config.openai_image_quality == "low"
        assert config.gemini_model == "gemini-2.0-flash"

    def test_validate_openai_without_key(self):
        """Test validation fails without OpenAI key."""
        config = AIProviderConfig(provider="openai", openai_api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            config.validate()

    def test_validate_gemini_without_key(self):
        """Test validation fails without Gemini key."""
        config = AIProviderConfig(provider="gemini", google_api_key=None)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
            config.validate()

    def test_validate_openai_with_key(self):
        """Test validation passes with OpenAI key."""
        config = AIProviderConfig(provider="openai", openai_api_key="sk-test")
        config.validate()  # Should not raise

    def test_validate_gemini_with_key(self):
        """Test validation passes with Gemini key."""
        config = AIProviderConfig(provider="gemini", google_api_key="test-key")
        config.validate()  # Should not raise


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TTSConfig()
        assert config.provider == "openai"
        assert config.openai_voice == "marin"
        assert config.openai_speed == 1.0
        assert config.openai_model == "gpt-4o-mini-tts"

    def test_validate_invalid_provider(self):
        """Test validation fails with invalid provider."""
        config = TTSConfig(provider="invalid")
        with pytest.raises(ValueError, match="TTS provider must be one of"):
            config.validate()

    def test_validate_elevenlabs_without_key(self):
        """Test validation fails for ElevenLabs without key."""
        config = TTSConfig(provider="elevenlabs", elevenlabs_api_key=None)
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY is required"):
            config.validate()

    def test_validate_gtts(self):
        """Test gTTS validation passes without keys."""
        config = TTSConfig(provider="gtts")
        config.validate()  # Should not raise


class TestSubtitleConfig:
    """Tests for SubtitleConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SubtitleConfig()
        assert config.aligner == "wav2vec2"
        assert config.font == "Arial-Bold"
        assert config.font_size == 80

    def test_validate_invalid_aligner(self):
        """Test validation fails with invalid aligner."""
        config = SubtitleConfig(aligner="invalid")
        with pytest.raises(ValueError, match="Subtitle aligner must be one of"):
            config.validate()

    def test_validate_valid_aligners(self):
        """Test validation passes for valid aligners."""
        for aligner in ["whisper", "wav2vec2"]:
            config = SubtitleConfig(aligner=aligner)
            config.validate()  # Should not raise


class TestPathConfig:
    """Tests for PathConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PathConfig()
        assert config.output_dir == Path("output")
        assert config.assets_dir == Path("assets")
        assert config.temp_dir == Path("output/temp")

    def test_ensure_directories(self, tmp_path):
        """Test that directories are created."""
        config = PathConfig(
            output_dir=tmp_path / "output",
            assets_dir=tmp_path / "assets",
            temp_dir=tmp_path / "temp",
        )
        config.ensure_directories()

        assert (tmp_path / "output").exists()
        assert (tmp_path / "assets").exists()
        assert (tmp_path / "temp").exists()


class TestAppConfig:
    """Tests for AppConfig."""

    def test_from_environment_with_defaults(self):
        """Test creating config from environment with defaults."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = AppConfig.from_environment()

            assert config.ai.openai_api_key == "sk-test-key"
            assert config.ai.provider == "openai"
            assert config.tts.provider == "openai"
            assert config.subtitle.aligner == "wav2vec2"

    def test_from_environment_with_overrides(self):
        """Test creating config from environment with overrides."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key",
            "AI_PROVIDER": "openai",
            "TTS_PROVIDER": "gtts",
            "TTS_VOICE": "nova",
            "TTS_SPEED": "1.5",
            "SUBTITLE_ALIGNER": "whisper",
            "OUTPUT_DIR": "/custom/output",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = AppConfig.from_environment()

            assert config.ai.provider == "openai"
            assert config.tts.provider == "gtts"
            assert config.tts.openai_voice == "nova"
            assert config.tts.openai_speed == 1.5
            assert config.subtitle.aligner == "whisper"
            assert config.paths.output_dir == Path("/custom/output")

    def test_config_is_immutable(self):
        """Test that config objects are frozen (immutable)."""
        env_vars = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env_vars, clear=True):
            config = AppConfig.from_environment()

            # Attempting to modify should raise an error
            with pytest.raises(Exception):  # FrozenInstanceError
                config.ai.provider = "gemini"

    def test_validate_all(self):
        """Test that validate checks all sub-configs."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key",
            "TTS_PROVIDER": "gtts",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = AppConfig.from_environment()
            config.validate()  # Should not raise


class TestGetConfig:
    """Tests for get_config singleton function."""

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_same_instance(self):
        """Test that get_config returns the same instance."""
        env_vars = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env_vars, clear=True):
            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test that reset_config clears the singleton."""
        env_vars = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env_vars, clear=True):
            config1 = get_config()
            reset_config()
            config2 = get_config()

            # Should be equal but not the same instance
            assert config1 is not config2
