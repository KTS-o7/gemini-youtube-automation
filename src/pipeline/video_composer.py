"""
Video Composer Module - Stitches scenes into final video with viral-optimized subtitles.

This module handles the final video composition, combining scene images
and audio with transitions, background music, and dynamic karaoke-style subtitles.

REFACTORED: Subtitle rendering and motion effects are now delegated to
specialized modules for better maintainability.

VIRAL SHORTS FEATURES:
- 🔥 Karaoke-style captions with word-by-word highlighting
- 🎤 Whisper-powered audio alignment for precise word sync
- 🎨 Color emphasis for key words
- 📱 Larger/bolder subtitles optimized for mobile
- 🎬 Dynamic motion effects (zooms, shakes)
- 🔊 Sound effects at transitions
- 😀 Emoji support in captions
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Lazy ImageMagick configuration state
_imagemagick_configured: bool = False


def configure_imagemagick(silent: bool = False) -> bool:
    """
    Find and configure ImageMagick binary for MoviePy.

    This function is idempotent - it will only configure once.
    Call this explicitly before using TextClip features.

    Args:
        silent: If True, suppress output messages

    Returns:
        True if ImageMagick was found and configured, False otherwise
    """
    global _imagemagick_configured

    if _imagemagick_configured:
        return True

    from moviepy.config import change_settings

    # Common ImageMagick paths on macOS
    possible_paths = [
        "/opt/homebrew/bin/magick",  # Apple Silicon Homebrew
        "/usr/local/bin/magick",  # Intel Homebrew
        "/opt/homebrew/bin/convert",  # Apple Silicon Homebrew (legacy)
        "/usr/local/bin/convert",  # Intel Homebrew (legacy)
        "/usr/bin/convert",  # System install
    ]

    # Try to find ImageMagick via 'which' command
    try:
        result = subprocess.run(
            ["which", "magick"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            possible_paths.insert(0, result.stdout.strip())
    except Exception:
        pass

    # Find first existing path
    for path in possible_paths:
        if os.path.exists(path):
            change_settings({"IMAGEMAGICK_BINARY": path})
            _imagemagick_configured = True
            if not silent:
                print(f"  ✓ ImageMagick configured: {path}")
            return True

    if not silent:
        print("  ⚠️ ImageMagick not found - subtitles may not work")
    return False


from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
    vfx,
)

from .models import PlannedScene, VideoRequest
from .motion_effects import MotionConfig, MotionEffects
from .subtitle_renderer import SubtitleConfig, SubtitleRenderer, SubtitleStyle


class VideoComposer:
    """Composes final video from scene assets with viral-optimized subtitles."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        background_music_path: Optional[Path] = None,
        sound_effects_dir: Optional[Path] = None,
        subtitle_aligner: Optional[str] = None,
    ):
        """
        Initialize the video composer.

        Args:
            output_dir: Directory to save output video
            background_music_path: Path to background music file
            sound_effects_dir: Directory containing sound effect files
            subtitle_aligner: Alignment method ("wav2vec2" or "whisper").
                             If not provided, defaults to "wav2vec2".
        """
        # Configure ImageMagick on first use (lazy initialization)
        configure_imagemagick()

        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default paths
        self.background_music_path = background_music_path or Path(
            "assets/music/bg_music.mp3"
        )
        self.sound_effects_dir = sound_effects_dir or Path("assets/sfx")

        # Font settings - prefer bold fonts for viral content
        self.font_path = Path("assets/fonts/ariblk.ttf")  # Arial Black - bolder
        if not self.font_path.exists():
            self.font_path = Path("assets/fonts/arialbd.ttf")  # Arial Bold fallback
        if not self.font_path.exists():
            self.font_path = Path("assets/fonts/arial.ttf")

        # Video settings - HIGH QUALITY
        self.fps = 30  # Smooth motion
        self.codec = "libx264"
        self.audio_codec = "aac"
        self.audio_bitrate = "192k"  # High quality audio
        self.preset = "slow"  # Better quality (slower encoding)
        self.threads = 0  # Use all available CPU cores

        # Audio settings
        self.fade_duration = 0.3  # Faster transitions for viral content
        self.background_music_volume = 0.12
        self.narration_volume = 1.3  # Louder narration
        self.sfx_volume = 0.25

        # Initialize subtitle renderer with default style
        subtitle_style = SubtitleStyle(
            font_size=70,
            color="white",
            highlight_color="#FFD700",
            stroke_color="black",
            stroke_width=4,
            position=("center", 0.78),
            max_chars_per_line=20,
            words_per_group=4,
            min_word_duration=0.15,
        )
        subtitle_config = SubtitleConfig(
            enabled=True,
            karaoke_mode=True,
            emoji_enabled=False,
            color_emphasis=True,
            aligner=(subtitle_aligner or "wav2vec2").lower(),
            save_ass_files=True,
        )
        self.subtitle_renderer = SubtitleRenderer(
            style=subtitle_style,
            config=subtitle_config,
            font_path=self.font_path,
            output_dir=self.output_dir,
        )

        # Initialize motion effects
        motion_config = MotionConfig(
            enabled=True,
            ken_burns_enabled=True,
            scene_shake_enabled=True,
            zoom_pulse_enabled=True,
            fade_duration=self.fade_duration,
        )
        self.motion_effects = MotionEffects(config=motion_config)

    def compose(
        self,
        scenes: list[PlannedScene],
        request: VideoRequest,
        output_filename: Optional[str] = None,
        enable_subtitles: bool = True,
        viral_mode: bool = True,
    ) -> Path:
        """
        Compose final video from scenes with viral-optimized subtitles.

        Args:
            scenes: List of planned scenes with image and audio paths
            request: Video request with format info
            output_filename: Optional custom output filename
            enable_subtitles: Whether to add subtitles to the video
            viral_mode: Enable all viral optimizations (karaoke, emoji, etc.)

        Returns:
            Path to the generated video file
        """
        print(f"🎥 Composing viral video from {len(scenes)} scenes...")

        if viral_mode:
            print("  🔥 Viral mode ENABLED: karaoke captions, emoji, motion effects")

        # Configure viral settings based on mode
        self.subtitle_renderer.config.karaoke_mode = viral_mode
        self.subtitle_renderer.config.color_emphasis = viral_mode
        self.motion_effects.config.enabled = viral_mode

        # Validate scenes have required assets
        self._validate_scenes(scenes)

        # Adjust subtitle settings based on video format
        is_short = request.format.value == "short"
        self.subtitle_renderer.adjust_style_for_format(is_short, viral_mode)

        # Get video dimensions
        video_width, video_height = request.get_video_dimensions()

        # Create video clips for each scene
        clips = []
        for i, scene in enumerate(scenes):
            print(f"  🎬 Processing scene {scene.scene_number}/{len(scenes)}...")
            clip = self._create_scene_clip(
                scene,
                video_width,
                video_height,
                enable_subtitles=enable_subtitles,
                scene_index=i,
                total_scenes=len(scenes),
            )
            clips.append(clip)

        print("  📎 Concatenating clips...")

        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")

        # Add background music if available
        if self.background_music_path.exists():
            final_video = self._add_background_music(final_video)

        # Determine output path
        if output_filename is None:
            format_suffix = "short" if is_short else "long"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"video_{format_suffix}_{timestamp}.mp4"

        output_path = self.output_dir / output_filename

        # Export the video
        print(f"  💾 Exporting video to {output_path.name}...")
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=self.codec,
            audio_codec=self.audio_codec,
            audio_bitrate=self.audio_bitrate,
            preset=self.preset,
            threads=self.threads,
            logger=None,  # Suppress moviepy's verbose logging
        )

        # Close clips to free memory
        final_video.close()
        for clip in clips:
            clip.close()

        print(f"✅ Viral video composed successfully: {output_path}")
        return output_path

    def _validate_scenes(self, scenes: list[PlannedScene]) -> None:
        """Validate that all scenes have required assets."""
        for scene in scenes:
            if not scene.image_path or not Path(scene.image_path).exists():
                raise ValueError(
                    f"Scene {scene.scene_number} missing image: {scene.image_path}"
                )
            if not scene.audio_path or not Path(scene.audio_path).exists():
                raise ValueError(
                    f"Scene {scene.scene_number} missing audio: {scene.audio_path}"
                )

    def _create_scene_clip(
        self,
        scene: PlannedScene,
        video_width: int,
        video_height: int,
        enable_subtitles: bool = True,
        scene_index: int = 0,
        total_scenes: int = 1,
    ) -> CompositeVideoClip:
        """
        Create a video clip for a single scene with viral subtitles.

        Args:
            scene: The planned scene with assets
            video_width: Width of the video
            video_height: Height of the video
            enable_subtitles: Whether to add subtitles
            scene_index: Index of this scene (for motion variety)
            total_scenes: Total number of scenes

        Returns:
            CompositeVideoClip with image, audio, and subtitles
        """
        # Load audio to get exact duration
        audio_clip = AudioFileClip(str(scene.audio_path))
        audio_duration = audio_clip.duration
        duration = audio_duration  # Use exact audio duration

        # Create image clip with same duration as audio
        image_clip = ImageClip(str(scene.image_path)).set_duration(duration)

        # Apply dynamic motion effects
        if self.motion_effects.config.enabled:
            image_clip = self.motion_effects.add_dynamic_motion(
                image_clip, scene_index, total_scenes, scene.mood
            )
        elif self.motion_effects.config.ken_burns_enabled:
            image_clip = self.motion_effects.add_ken_burns(image_clip)

        # Create list of clips to composite
        clips_to_composite = [image_clip]

        # Add subtitles if enabled
        if enable_subtitles and scene.narration:
            subtitle_clips = self.subtitle_renderer.create_subtitles(
                scene.narration,
                duration,
                video_width,
                video_height,
                audio_duration=audio_duration,
                audio_path=scene.audio_path,
            )
            clips_to_composite.extend(subtitle_clips)

        # Composite all clips
        composite_clip = CompositeVideoClip(
            clips_to_composite,
            size=(video_width, video_height),
        )

        # Set audio - ensure durations match exactly
        composite_clip = composite_clip.set_duration(duration)
        composite_clip = composite_clip.set_audio(audio_clip.set_duration(duration))

        # Apply transitions based on scene type
        composite_clip = self.motion_effects.apply_transition(
            composite_clip, scene.transition
        )

        return composite_clip

    # =========================================================================
    # BACKGROUND MUSIC & SOUND EFFECTS
    # =========================================================================

    def _add_background_music(self, video) -> any:
        """Add background music to the video with ducking support."""
        print("  🎵 Adding background music...")

        try:
            bg_music = AudioFileClip(str(self.background_music_path))
            video_duration = video.duration

            # Adjust volume (lower for viral - narration is king)
            bg_music = bg_music.volumex(self.background_music_volume)

            # Loop or trim music to match video duration
            if bg_music.duration < video_duration:
                bg_music = bg_music.fx(vfx.loop, duration=video_duration)
            else:
                bg_music = bg_music.subclip(0, video_duration)

            # Fade in at start, fade out at end
            fade_in = min(1.0, video_duration / 6)
            fade_out = min(2.0, video_duration / 4)
            bg_music = bg_music.audio_fadein(fade_in).audio_fadeout(fade_out)
            bg_music = bg_music.set_duration(video_duration)

            # Combine with existing audio
            if video.audio:
                narration = video.audio.volumex(self.narration_volume)
                narration = narration.set_duration(video_duration)

                composite_audio = CompositeAudioClip([narration, bg_music])
                composite_audio = composite_audio.set_duration(video_duration)
                video = video.set_audio(composite_audio)
            else:
                video = video.set_audio(bg_music)

            return video

        except Exception as e:
            print(f"  ⚠️ Could not add background music: {e}")
            return video

    def _add_sound_effect(self, clip, sfx_name: str, position: float = 0.0):
        """
        Add a sound effect to a clip at a specific position.

        Args:
            clip: The video clip
            sfx_name: Name of the sound effect file (without extension)
            position: Time position in seconds to play the effect

        Returns:
            Clip with sound effect added
        """
        try:
            # Try common audio formats
            for ext in [".mp3", ".wav", ".ogg"]:
                sfx_path = self.sound_effects_dir / f"{sfx_name}{ext}"
                if sfx_path.exists():
                    sfx = AudioFileClip(str(sfx_path))
                    sfx = sfx.volumex(self.sfx_volume)
                    sfx = sfx.set_start(position)

                    if clip.audio:
                        composite = CompositeAudioClip([clip.audio, sfx])
                        return clip.set_audio(composite)
                    else:
                        return clip.set_audio(sfx)

            return clip

        except Exception as e:
            print(f"  ⚠️ Could not add sound effect {sfx_name}: {e}")
            return clip

    # =========================================================================
    # SIMPLE COMPOSITION METHOD
    # =========================================================================

    def compose_simple(
        self,
        image_paths: list[Path],
        audio_paths: list[Path],
        output_path: Path,
        narrations: Optional[list[str]] = None,
        video_width: int = 1920,
        video_height: int = 1080,
        viral_mode: bool = True,
        cached_timestamps: Optional[list] = None,
    ) -> Path:
        """
        Simple composition method for direct image/audio lists.

        Args:
            image_paths: List of image file paths
            audio_paths: List of audio file paths (same length as images)
            output_path: Output video path
            narrations: Optional list of narration texts for subtitles
            video_width: Video width
            video_height: Video height
            viral_mode: Enable viral subtitle features
            cached_timestamps: Pre-loaded word timestamps (from generate_subtitles.py)

        Returns:
            Path to the generated video
        """
        if len(image_paths) != len(audio_paths):
            raise ValueError("Number of images and audio files must match")

        print(
            f"🎥 Simple compose: {len(image_paths)} slides (viral_mode={viral_mode})..."
        )
        print(f"   📊 Progress will be shown below:")

        # Configure for viral mode
        self.subtitle_renderer.config.karaoke_mode = viral_mode
        self.subtitle_renderer.config.color_emphasis = viral_mode
        self.motion_effects.config.enabled = viral_mode

        # Adjust settings for video dimensions
        is_vertical = video_height > video_width
        self.subtitle_renderer.adjust_style_for_format(is_vertical, viral_mode)

        clips = []
        total_scenes = len(image_paths)
        for i, (img_path, audio_path) in enumerate(zip(image_paths, audio_paths)):
            print(f"   🎬 [{i + 1}/{total_scenes}] Processing scene {i + 1}...")

            # Load audio
            audio_clip = AudioFileClip(str(audio_path))
            print(f"      🔊 Audio loaded: {audio_clip.duration:.1f}s")
            duration = audio_clip.duration + 0.3  # Smaller padding

            # Create image clip and resize to fill frame (no black bars)
            img_clip = ImageClip(str(img_path))

            # Resize image to fill the video frame
            img_w, img_h = img_clip.size
            scale_w = video_width / img_w
            scale_h = video_height / img_h
            scale = max(scale_w, scale_h)  # Use max to fill (crop edges if needed)

            img_clip = img_clip.resize(scale)

            # Center crop if needed
            new_w, new_h = img_clip.size
            if new_w > video_width or new_h > video_height:
                x_center = (new_w - video_width) // 2
                y_center = (new_h - video_height) // 2
                img_clip = img_clip.crop(
                    x1=x_center,
                    y1=y_center,
                    x2=x_center + video_width,
                    y2=y_center + video_height,
                )

            img_clip = img_clip.set_duration(duration)

            # Add motion effects
            if self.motion_effects.config.enabled:
                print(f"      🎬 Adding motion effects...")
                img_clip = self.motion_effects.add_dynamic_motion(
                    img_clip, i, len(image_paths), "informative"
                )

            # Create composite with subtitles if narrations provided
            clips_to_composite = [img_clip]

            if narrations and i < len(narrations) and narrations[i]:
                # Check for cached timestamps first
                scene_timestamps = None
                if cached_timestamps and i < len(cached_timestamps):
                    scene_timestamps = cached_timestamps[i]

                if self.subtitle_renderer.config.karaoke_mode:
                    if scene_timestamps:
                        # Use cached timestamps - no Whisper API call needed
                        print(
                            f"      🎤 Using cached timestamps ({len(scene_timestamps)} words)"
                        )
                        subtitle_clips = (
                            self.subtitle_renderer.create_karaoke_from_timestamps(
                                scene_timestamps,
                                video_width,
                                video_height,
                                audio_clip.duration,
                            )
                        )
                    else:
                        # Fall back to alignment or estimated timing
                        subtitle_clips = self.subtitle_renderer.create_subtitles(
                            narrations[i],
                            duration,
                            video_width,
                            video_height,
                            audio_duration=audio_clip.duration,
                            audio_path=str(audio_path),
                        )
                else:
                    subtitle_clips = self.subtitle_renderer.create_subtitles(
                        narrations[i],
                        duration,
                        video_width,
                        video_height,
                        audio_duration=audio_clip.duration,
                    )
                clips_to_composite.extend(subtitle_clips)

            # Composite
            composite = CompositeVideoClip(
                clips_to_composite,
                size=(video_width, video_height),
            )
            composite = (
                composite.set_duration(duration)
                .set_audio(audio_clip)
                .fadein(self.fade_duration)
                .fadeout(self.fade_duration)
            )

            clips.append(composite)
            print(f"      ✅ Scene {i + 1} complete")

        print(f"\n   📎 Concatenating {len(clips)} clips...")
        # Concatenate
        final_video = concatenate_videoclips(clips, method="compose")

        # Add background music
        if self.background_music_path.exists():
            final_video = self._add_background_music(final_video)

        # Export - optimized for speed
        print(f"   💾 Encoding video (optimized for speed)...")
        print(
            f"      Settings: {self.fps}fps, {self.codec}, {self.preset} preset, all CPU cores"
        )
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=self.codec,
            audio_codec=self.audio_codec,
            audio_bitrate=self.audio_bitrate,
            preset=self.preset,
            threads=self.threads,
            logger="bar",  # Show progress bar
            ffmpeg_params=[
                "-crf",
                "18",  # High quality (18 is visually lossless)
                "-profile:v",
                "high",  # High profile for better compression
                "-level",
                "4.2",  # Compatibility level
                "-pix_fmt",
                "yuv420p",  # Best compatibility
            ],
        )

        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()

        print(f"✅ Viral video created: {output_path}")
        return output_path

    def get_video_duration(self, video_path: Path) -> float:
        """Get the duration of a video file."""
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        return duration
