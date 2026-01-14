"""
Motion Effects Module - Handles video motion and transition effects.

This module extracts motion-related functionality from VideoComposer,
including Ken Burns effects, transitions, and dynamic motion.

FEATURES:
- 🎬 Ken Burns effect (slow zoom in/out)
- 🔄 Various transition effects (fade, dissolve, etc.)
- 📱 Dynamic motion based on scene position and mood
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from moviepy.editor import ImageClip, vfx
from PIL import Image


@dataclass
class MotionConfig:
    """Configuration for motion effects."""

    enabled: bool = True
    ken_burns_enabled: bool = True
    scene_shake_enabled: bool = True
    zoom_pulse_enabled: bool = True
    fade_duration: float = 0.3


class MotionEffects:
    """
    Applies motion and transition effects to video clips.

    Supports Ken Burns effect (slow zoom), various transitions,
    and dynamic motion based on scene context.
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        """
        Initialize motion effects handler.

        Args:
            config: Motion effect configuration
        """
        self.config = config or MotionConfig()

    def add_dynamic_motion(
        self,
        clip: ImageClip,
        scene_index: int,
        total_scenes: int,
        mood: str = "informative",
    ) -> ImageClip:
        """
        Add dynamic motion effects based on scene position and mood.

        Effects include:
        - Ken Burns (slow zoom in/out)
        - Quick zoom pulses
        - Subtle position shifts

        Args:
            clip: The image clip to apply effects to
            scene_index: Index of the current scene
            total_scenes: Total number of scenes
            mood: Scene mood (e.g., "informative", "exciting")

        Returns:
            Clip with motion effects applied
        """
        if not self.config.enabled:
            return clip

        # Determine effect type based on scene position
        if scene_index == 0:
            # Opening scene: zoom in to draw attention
            return self.add_ken_burns(clip, zoom_start=1.0, zoom_end=1.15)
        elif scene_index == total_scenes - 1:
            # Closing scene: zoom out for finale
            return self.add_ken_burns(clip, zoom_start=1.1, zoom_end=1.0)
        else:
            # Middle scenes: variety of effects
            effect_type = scene_index % 3

            if effect_type == 0:
                # Slow zoom in
                return self.add_ken_burns(clip, zoom_start=1.0, zoom_end=1.1)
            elif effect_type == 1:
                # Slow zoom out
                return self.add_ken_burns(clip, zoom_start=1.1, zoom_end=1.0)
            else:
                # Pan effect (via different zoom centers)
                return self.add_ken_burns(clip, zoom_start=1.05, zoom_end=1.1)

    def add_ken_burns(
        self,
        clip: ImageClip,
        zoom_start: float = 1.0,
        zoom_end: float = 1.1,
    ) -> ImageClip:
        """
        Add Ken Burns effect (slow zoom) to a clip.

        Args:
            clip: The image clip
            zoom_start: Starting zoom level (1.0 = original size)
            zoom_end: Ending zoom level

        Returns:
            Clip with zoom effect applied
        """
        if not self.config.ken_burns_enabled:
            return clip

        def zoom_effect(get_frame, t):
            """Apply zoom effect at time t."""
            frame = get_frame(t)
            duration = clip.duration

            # Smooth easing function for more natural motion
            progress = t / duration
            # Ease in-out cubic
            if progress < 0.5:
                eased = 4 * progress * progress * progress
            else:
                eased = 1 - pow(-2 * progress + 2, 3) / 2

            zoom = zoom_start + (zoom_end - zoom_start) * eased

            # Get original dimensions
            h, w = frame.shape[:2]

            # Calculate new dimensions
            new_w = int(w * zoom)
            new_h = int(h * zoom)

            # Calculate crop coordinates to keep center
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2

            # Resize frame
            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = img.crop((x_start, y_start, x_start + w, y_start + h))

            return np.array(img)

        return clip.fl(zoom_effect)

    def apply_transition(self, clip, transition: str):
        """
        Apply transition effects to a clip.
        Faster transitions for viral content.

        Args:
            clip: The video clip to apply transition to
            transition: Type of transition ("fade_in", "fade_out", "crossfade",
                       "dissolve", "fade_to_black", "cut")

        Returns:
            Clip with transition applied
        """
        clip_duration = getattr(clip, "duration", None)
        if clip_duration is None or clip_duration < 0.3:
            return clip

        # Use faster fade for viral content
        safe_fade = min(self.config.fade_duration, clip_duration / 5)

        if transition == "fade_in":
            clip = clip.fadein(safe_fade)
        elif transition == "fade_out":
            clip = clip.fadeout(safe_fade)
        elif transition == "crossfade":
            clip = clip.fadein(safe_fade).fadeout(safe_fade)
        elif transition == "dissolve":
            dissolve_fade = min(safe_fade * 1.2, clip_duration / 5)
            clip = clip.fadein(dissolve_fade).fadeout(dissolve_fade)
        elif transition == "fade_to_black":
            black_fade = min(safe_fade * 1.5, clip_duration / 4)
            clip = clip.fadeout(black_fade)
        # "cut" transition has no effect

        return clip

    def add_zoom_pulse(
        self,
        clip: ImageClip,
        pulse_time: float,
        pulse_intensity: float = 1.15,
        pulse_duration: float = 0.3,
    ) -> ImageClip:
        """
        Add a quick zoom pulse at a specific time.

        Args:
            clip: The image clip
            pulse_time: Time in seconds when pulse should occur
            pulse_intensity: How much to zoom (1.15 = 15% zoom)
            pulse_duration: Duration of the pulse effect

        Returns:
            Clip with zoom pulse applied
        """
        if not self.config.zoom_pulse_enabled:
            return clip

        def pulse_effect(get_frame, t):
            """Apply pulse effect at time t."""
            frame = get_frame(t)

            # Calculate if we're in the pulse window
            pulse_start = pulse_time - pulse_duration / 2
            pulse_end = pulse_time + pulse_duration / 2

            if pulse_start <= t <= pulse_end:
                # Calculate pulse progress (0 to 1 to 0)
                pulse_progress = (t - pulse_start) / pulse_duration
                # Smooth pulse using sine wave
                pulse_factor = np.sin(pulse_progress * np.pi)
                zoom = 1.0 + (pulse_intensity - 1.0) * pulse_factor
            else:
                zoom = 1.0

            if zoom == 1.0:
                return frame

            # Get original dimensions
            h, w = frame.shape[:2]

            # Calculate new dimensions
            new_w = int(w * zoom)
            new_h = int(h * zoom)

            # Calculate crop coordinates to keep center
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2

            # Resize frame
            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = img.crop((x_start, y_start, x_start + w, y_start + h))

            return np.array(img)

        return clip.fl(pulse_effect)

    def add_shake(
        self,
        clip: ImageClip,
        intensity: float = 5.0,
        frequency: float = 15.0,
    ) -> ImageClip:
        """
        Add subtle shake effect to a clip.

        Args:
            clip: The image clip
            intensity: Maximum pixels of shake
            frequency: Shake frequency in Hz

        Returns:
            Clip with shake effect applied
        """
        if not self.config.scene_shake_enabled:
            return clip

        def shake_effect(get_frame, t):
            """Apply shake effect at time t."""
            frame = get_frame(t)

            # Calculate shake offset using sine waves
            x_offset = int(intensity * np.sin(2 * np.pi * frequency * t))
            y_offset = int(intensity * np.sin(2 * np.pi * frequency * t + np.pi / 4))

            # Get dimensions
            h, w = frame.shape[:2]

            # Create shifted frame with padding
            img = Image.fromarray(frame)

            # Calculate crop box
            left = max(0, x_offset)
            top = max(0, y_offset)
            right = min(w, w + x_offset)
            bottom = min(h, h + y_offset)

            # If offset is negative, we need to shift the crop
            if x_offset < 0:
                left = -x_offset
                right = w
            if y_offset < 0:
                top = -y_offset
                bottom = h

            img = img.crop((left, top, right, bottom))
            img = img.resize((w, h), Image.Resampling.LANCZOS)

            return np.array(img)

        return clip.fl(shake_effect)
