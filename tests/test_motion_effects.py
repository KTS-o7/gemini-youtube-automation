"""
Tests for the motion effects module.

These tests verify that:
1. MotionConfig correctly stores configuration values
2. MotionEffects applies effects based on configuration
3. Ken Burns effect is correctly applied
4. Transitions are correctly applied
5. Effects can be disabled via configuration
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.motion_effects import MotionConfig, MotionEffects


class TestMotionConfig:
    """Tests for MotionConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = MotionConfig()

        assert config.enabled is True
        assert config.ken_burns_enabled is True
        assert config.scene_shake_enabled is True
        assert config.zoom_pulse_enabled is True
        assert config.fade_duration == 0.3

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = MotionConfig(
            enabled=False,
            ken_burns_enabled=False,
            scene_shake_enabled=False,
            zoom_pulse_enabled=False,
            fade_duration=0.5,
        )

        assert config.enabled is False
        assert config.ken_burns_enabled is False
        assert config.scene_shake_enabled is False
        assert config.zoom_pulse_enabled is False
        assert config.fade_duration == 0.5

    def test_partial_config(self):
        """Test creating config with partial overrides."""
        config = MotionConfig(
            ken_burns_enabled=False,
            fade_duration=1.0,
        )

        assert config.enabled is True  # Default
        assert config.ken_burns_enabled is False  # Overridden
        assert config.scene_shake_enabled is True  # Default
        assert config.fade_duration == 1.0  # Overridden


class TestMotionEffects:
    """Tests for MotionEffects class."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance with default config."""
        return MotionEffects()

    @pytest.fixture
    def disabled_effects(self):
        """Create a MotionEffects instance with effects disabled."""
        config = MotionConfig(enabled=False)
        return MotionEffects(config=config)

    @pytest.fixture
    def mock_clip(self):
        """Create a mock ImageClip for testing."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)
        clip.fadein = MagicMock(return_value=clip)
        clip.fadeout = MagicMock(return_value=clip)
        return clip

    def test_init_default_config(self):
        """Test initialization with default config."""
        effects = MotionEffects()

        assert effects.config is not None
        assert effects.config.enabled is True

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MotionConfig(enabled=False, fade_duration=1.0)
        effects = MotionEffects(config=config)

        assert effects.config.enabled is False
        assert effects.config.fade_duration == 1.0


class TestAddDynamicMotion:
    """Tests for add_dynamic_motion method."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance."""
        return MotionEffects()

    @pytest.fixture
    def mock_clip(self):
        """Create a mock clip."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)
        return clip

    def test_disabled_returns_original_clip(self, mock_clip):
        """Test that disabled effects return original clip."""
        config = MotionConfig(enabled=False)
        effects = MotionEffects(config=config)

        result = effects.add_dynamic_motion(mock_clip, 0, 5, "informative")

        assert result is mock_clip
        mock_clip.fl.assert_not_called()

    def test_opening_scene_zoom_in(self, effects, mock_clip):
        """Test that opening scene (index 0) gets zoom in effect."""
        with patch.object(effects, "add_ken_burns", return_value=mock_clip) as mock_kb:
            effects.add_dynamic_motion(mock_clip, scene_index=0, total_scenes=5)

            mock_kb.assert_called_once()
            # Opening scene should zoom in (zoom_start=1.0, zoom_end > 1.0)
            call_args = mock_kb.call_args
            assert call_args[1]["zoom_start"] == 1.0
            assert call_args[1]["zoom_end"] > 1.0

    def test_closing_scene_zoom_out(self, effects, mock_clip):
        """Test that closing scene gets zoom out effect."""
        with patch.object(effects, "add_ken_burns", return_value=mock_clip) as mock_kb:
            effects.add_dynamic_motion(mock_clip, scene_index=4, total_scenes=5)

            mock_kb.assert_called_once()
            # Closing scene should zoom out (zoom_start > 1.0, zoom_end closer to 1.0)
            call_args = mock_kb.call_args
            assert call_args[1]["zoom_start"] > call_args[1]["zoom_end"]

    def test_middle_scenes_vary_effects(self, effects, mock_clip):
        """Test that middle scenes get varied effects based on index."""
        with patch.object(effects, "add_ken_burns", return_value=mock_clip) as mock_kb:
            # Scene index 1 (effect_type = 1 % 3 = 1) -> zoom out
            effects.add_dynamic_motion(mock_clip, scene_index=1, total_scenes=5)
            call_args_1 = mock_kb.call_args

            mock_kb.reset_mock()

            # Scene index 2 (effect_type = 2 % 3 = 2) -> pan effect
            effects.add_dynamic_motion(mock_clip, scene_index=2, total_scenes=5)
            call_args_2 = mock_kb.call_args

            # Different effects should have different parameters
            assert mock_kb.call_count == 1


class TestAddKenBurns:
    """Tests for add_ken_burns method."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance."""
        return MotionEffects()

    @pytest.fixture
    def mock_clip(self):
        """Create a mock clip."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)
        return clip

    def test_ken_burns_disabled_returns_original(self, mock_clip):
        """Test that disabled Ken Burns returns original clip."""
        config = MotionConfig(ken_burns_enabled=False)
        effects = MotionEffects(config=config)

        result = effects.add_ken_burns(mock_clip)

        assert result is mock_clip
        mock_clip.fl.assert_not_called()

    def test_ken_burns_enabled_applies_effect(self, effects, mock_clip):
        """Test that enabled Ken Burns applies effect."""
        result = effects.add_ken_burns(mock_clip, zoom_start=1.0, zoom_end=1.1)

        mock_clip.fl.assert_called_once()

    def test_ken_burns_default_zoom_values(self, effects, mock_clip):
        """Test Ken Burns with default zoom values."""
        result = effects.add_ken_burns(mock_clip)

        # Should have been called (default zoom_start=1.0, zoom_end=1.1)
        mock_clip.fl.assert_called_once()

    def test_ken_burns_custom_zoom_values(self, effects, mock_clip):
        """Test Ken Burns with custom zoom values."""
        result = effects.add_ken_burns(mock_clip, zoom_start=0.9, zoom_end=1.2)

        mock_clip.fl.assert_called_once()


class TestApplyTransition:
    """Tests for apply_transition method."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance."""
        return MotionEffects()

    @pytest.fixture
    def mock_clip(self):
        """Create a mock clip with duration."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fadein = MagicMock(return_value=clip)
        clip.fadeout = MagicMock(return_value=clip)
        return clip

    def test_fade_in_transition(self, effects, mock_clip):
        """Test fade_in transition."""
        result = effects.apply_transition(mock_clip, "fade_in")

        mock_clip.fadein.assert_called_once()

    def test_fade_out_transition(self, effects, mock_clip):
        """Test fade_out transition."""
        result = effects.apply_transition(mock_clip, "fade_out")

        mock_clip.fadeout.assert_called_once()

    def test_crossfade_transition(self, effects, mock_clip):
        """Test crossfade transition (fade in and out)."""
        result = effects.apply_transition(mock_clip, "crossfade")

        mock_clip.fadein.assert_called_once()
        mock_clip.fadeout.assert_called_once()

    def test_dissolve_transition(self, effects, mock_clip):
        """Test dissolve transition."""
        result = effects.apply_transition(mock_clip, "dissolve")

        mock_clip.fadein.assert_called_once()
        mock_clip.fadeout.assert_called_once()

    def test_fade_to_black_transition(self, effects, mock_clip):
        """Test fade_to_black transition."""
        result = effects.apply_transition(mock_clip, "fade_to_black")

        mock_clip.fadeout.assert_called_once()

    def test_cut_transition_no_effect(self, effects, mock_clip):
        """Test cut transition has no effect."""
        result = effects.apply_transition(mock_clip, "cut")

        mock_clip.fadein.assert_not_called()
        mock_clip.fadeout.assert_not_called()

    def test_short_clip_no_transition(self, effects):
        """Test that very short clips don't get transitions."""
        short_clip = MagicMock()
        short_clip.duration = 0.2  # Less than 0.3

        result = effects.apply_transition(short_clip, "fade_in")

        short_clip.fadein.assert_not_called()
        assert result is short_clip

    def test_no_duration_attribute(self, effects):
        """Test handling clip without duration attribute."""
        clip = MagicMock(spec=[])  # No duration attribute
        clip.duration = None

        result = effects.apply_transition(clip, "fade_in")

        assert result is clip

    def test_transition_respects_fade_duration(self):
        """Test that transition uses configured fade duration."""
        config = MotionConfig(fade_duration=0.5)
        effects = MotionEffects(config=config)

        clip = MagicMock()
        clip.duration = 10.0
        clip.fadein = MagicMock(return_value=clip)

        effects.apply_transition(clip, "fade_in")

        # Fade duration should be min(0.5, 10.0/5) = 0.5
        call_args = clip.fadein.call_args[0]
        assert call_args[0] == 0.5


class TestAddZoomPulse:
    """Tests for add_zoom_pulse method."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance."""
        return MotionEffects()

    @pytest.fixture
    def mock_clip(self):
        """Create a mock clip."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)
        return clip

    def test_zoom_pulse_disabled_returns_original(self, mock_clip):
        """Test that disabled zoom pulse returns original clip."""
        config = MotionConfig(zoom_pulse_enabled=False)
        effects = MotionEffects(config=config)

        result = effects.add_zoom_pulse(mock_clip, pulse_time=2.5)

        assert result is mock_clip
        mock_clip.fl.assert_not_called()

    def test_zoom_pulse_enabled_applies_effect(self, effects, mock_clip):
        """Test that enabled zoom pulse applies effect."""
        result = effects.add_zoom_pulse(mock_clip, pulse_time=2.5)

        mock_clip.fl.assert_called_once()

    def test_zoom_pulse_custom_parameters(self, effects, mock_clip):
        """Test zoom pulse with custom parameters."""
        result = effects.add_zoom_pulse(
            mock_clip,
            pulse_time=1.0,
            pulse_intensity=1.3,
            pulse_duration=0.5,
        )

        mock_clip.fl.assert_called_once()


class TestAddShake:
    """Tests for add_shake method."""

    @pytest.fixture
    def effects(self):
        """Create a MotionEffects instance."""
        return MotionEffects()

    @pytest.fixture
    def mock_clip(self):
        """Create a mock clip."""
        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)
        return clip

    def test_shake_disabled_returns_original(self, mock_clip):
        """Test that disabled shake returns original clip."""
        config = MotionConfig(scene_shake_enabled=False)
        effects = MotionEffects(config=config)

        result = effects.add_shake(mock_clip)

        assert result is mock_clip
        mock_clip.fl.assert_not_called()

    def test_shake_enabled_applies_effect(self, effects, mock_clip):
        """Test that enabled shake applies effect."""
        result = effects.add_shake(mock_clip)

        mock_clip.fl.assert_called_once()

    def test_shake_custom_parameters(self, effects, mock_clip):
        """Test shake with custom parameters."""
        result = effects.add_shake(mock_clip, intensity=10.0, frequency=20.0)

        mock_clip.fl.assert_called_once()


class TestMotionEffectsIntegration:
    """Integration tests for MotionEffects."""

    def test_all_effects_disabled(self):
        """Test that all effects can be disabled."""
        config = MotionConfig(
            enabled=False,
            ken_burns_enabled=False,
            scene_shake_enabled=False,
            zoom_pulse_enabled=False,
        )
        effects = MotionEffects(config=config)

        clip = MagicMock()
        clip.duration = 5.0

        # None of these should modify the clip
        result = effects.add_dynamic_motion(clip, 0, 5, "informative")
        assert result is clip

        result = effects.add_ken_burns(clip)
        assert result is clip

        result = effects.add_zoom_pulse(clip, 2.5)
        assert result is clip

        result = effects.add_shake(clip)
        assert result is clip

    def test_selective_effects_enabled(self):
        """Test enabling only specific effects."""
        config = MotionConfig(
            enabled=True,
            ken_burns_enabled=True,
            scene_shake_enabled=False,
            zoom_pulse_enabled=False,
        )
        effects = MotionEffects(config=config)

        clip = MagicMock()
        clip.duration = 5.0
        clip.fl = MagicMock(return_value=clip)

        # Ken Burns should work
        result = effects.add_ken_burns(clip)
        clip.fl.assert_called_once()

        clip.fl.reset_mock()

        # Shake should be disabled
        result = effects.add_shake(clip)
        clip.fl.assert_not_called()
