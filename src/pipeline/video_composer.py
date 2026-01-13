"""
Video Composer Module - Stitches scenes into final video with subtitles.

This module handles the final video composition, combining scene images
and audio with transitions, background music, and dynamic subtitles.
"""

import os
import subprocess
import textwrap
from pathlib import Path
from typing import Optional


# Configure ImageMagick for MoviePy before importing TextClip
def _configure_imagemagick():
    """Find and configure ImageMagick binary for MoviePy."""
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
            print(f"  ✓ ImageMagick configured: {path}")
            return True

    print("  ⚠️ ImageMagick not found - subtitles may not work")
    return False


# Configure on module load
_configure_imagemagick()

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
    vfx,
)

from .models import PlannedScene, VideoRequest


class VideoComposer:
    """Composes final video from scene assets with subtitles."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        background_music_path: Optional[Path] = None,
    ):
        """
        Initialize the video composer.

        Args:
            output_dir: Directory to save output video
            background_music_path: Path to background music file
        """
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default background music path
        self.background_music_path = background_music_path or Path(
            "assets/music/bg_music.mp3"
        )

        # Font settings
        self.font_path = Path("assets/fonts/arialbd.ttf")
        if not self.font_path.exists():
            self.font_path = Path("assets/fonts/arial.ttf")

        # Video settings
        self.fps = 24
        self.codec = "libx264"
        self.audio_codec = "aac"
        self.audio_bitrate = "192k"
        self.preset = "medium"
        self.threads = 4

        # Transition settings
        self.fade_duration = 0.5
        self.background_music_volume = 0.15
        self.narration_volume = 1.2

        # Subtitle settings
        self.subtitle_enabled = True
        self.subtitle_font_size = 45  # Will be adjusted based on format
        self.subtitle_color = "white"
        self.subtitle_stroke_color = "black"
        self.subtitle_stroke_width = 2
        self.subtitle_bg_color = None  # Set to 'black' for background box
        self.subtitle_position = ("center", 0.85)  # Relative position (x, y ratio)
        self.max_chars_per_line = 40  # For word wrapping

    def compose(
        self,
        scenes: list[PlannedScene],
        request: VideoRequest,
        output_filename: Optional[str] = None,
        enable_subtitles: bool = True,
    ) -> Path:
        """
        Compose final video from scenes with subtitles.

        Args:
            scenes: List of planned scenes with image and audio paths
            request: Video request with format info
            output_filename: Optional custom output filename
            enable_subtitles: Whether to add subtitles to the video

        Returns:
            Path to the generated video file
        """
        print(f"🎥 Composing video from {len(scenes)} scenes...")

        # Validate scenes have required assets
        self._validate_scenes(scenes)

        # Adjust subtitle settings based on video format
        self._adjust_subtitle_settings(request)

        # Get video dimensions
        video_width, video_height = request.get_video_dimensions()

        # Create video clips for each scene
        clips = []
        for scene in scenes:
            clip = self._create_scene_clip(
                scene,
                video_width,
                video_height,
                enable_subtitles=enable_subtitles,
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
            format_suffix = "short" if request.format.value == "short" else "long"
            output_filename = f"video_{format_suffix}.mp4"

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

        print(f"✅ Video composed successfully: {output_path}")
        return output_path

    def _adjust_subtitle_settings(self, request: VideoRequest) -> None:
        """Adjust subtitle settings based on video format."""
        if request.format.value == "short":
            # Vertical video (9:16) - larger text, more centered
            self.subtitle_font_size = 55
            self.max_chars_per_line = 25
            self.subtitle_position = ("center", 0.80)
        else:
            # Horizontal video (16:9) - standard settings
            self.subtitle_font_size = 45
            self.max_chars_per_line = 50
            self.subtitle_position = ("center", 0.85)

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
    ) -> CompositeVideoClip:
        """
        Create a video clip for a single scene with subtitles.

        Args:
            scene: The planned scene with assets
            video_width: Width of the video
            video_height: Height of the video
            enable_subtitles: Whether to add subtitles

        Returns:
            CompositeVideoClip with image, audio, and subtitles
        """
        # Load audio to get exact duration
        audio_clip = AudioFileClip(str(scene.audio_path))
        audio_duration = audio_clip.duration
        duration = audio_duration  # Use exact audio duration, no padding

        # Create image clip with same duration as audio
        image_clip = ImageClip(str(scene.image_path)).set_duration(duration)

        # Apply Ken Burns effect (subtle zoom)
        image_clip = self._add_ken_burns(image_clip)

        # Create list of clips to composite
        clips_to_composite = [image_clip]

        # Add subtitles if enabled
        if enable_subtitles and scene.narration:
            subtitle_clips = self._create_subtitle_clips(
                scene.narration,
                duration,
                video_width,
                video_height,
                audio_duration=audio_duration,
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
        composite_clip = self._apply_transition(composite_clip, scene.transition)

        return composite_clip

    def _create_subtitle_clips(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
    ) -> list[TextClip]:
        """
        Create subtitle clips for the narration text.

        This method creates word-by-word or phrase-by-phrase subtitles
        that appear synchronized with the audio.

        Args:
            narration: The full narration text
            total_duration: Total duration of the scene
            video_width: Width of the video
            video_height: Height of the video
            audio_duration: Duration of the audio

        Returns:
            List of TextClip objects for subtitles
        """
        subtitle_clips = []

        # Split narration into phrases/chunks for better readability
        phrases = self._split_into_phrases(narration)

        if not phrases:
            return subtitle_clips

        # Calculate timing for each phrase
        # Use audio_duration for timing, leave a small buffer at the end
        available_duration = audio_duration * 0.95  # Use 95% of audio to avoid overflow
        time_per_phrase = available_duration / len(phrases)

        # Calculate y position
        y_pos = int(video_height * self.subtitle_position[1])

        current_time = 0.0
        for phrase in phrases:
            # Wrap text if too long
            wrapped_text = self._wrap_text(phrase)

            try:
                # Create text clip
                txt_clip = TextClip(
                    wrapped_text,
                    fontsize=self.subtitle_font_size,
                    color=self.subtitle_color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.subtitle_stroke_color,
                    stroke_width=self.subtitle_stroke_width,
                    method="caption",
                    size=(video_width - 100, None),  # Width with padding
                    align="center",
                )

                # Set timing and position
                # Ensure we don't exceed audio duration
                phrase_duration = min(
                    time_per_phrase, audio_duration - current_time - 0.1
                )
                if phrase_duration <= 0:
                    break  # Stop if we've run out of time

                txt_clip = (
                    txt_clip.set_start(current_time)
                    .set_duration(phrase_duration)
                    .set_position(("center", y_pos))
                )

                # Add fade in/out for smooth transitions (only if duration allows)
                fade_duration = min(0.1, phrase_duration / 4)
                if fade_duration > 0.05:
                    txt_clip = txt_clip.crossfadein(fade_duration).crossfadeout(
                        fade_duration
                    )

                subtitle_clips.append(txt_clip)

            except Exception as e:
                print(f"  ⚠️ Failed to create subtitle clip: {e}")
                # Try with default font as fallback
                try:
                    phrase_duration = min(
                        time_per_phrase, audio_duration - current_time - 0.1
                    )
                    if phrase_duration <= 0:
                        break
                    txt_clip = TextClip(
                        wrapped_text,
                        fontsize=self.subtitle_font_size,
                        color=self.subtitle_color,
                        stroke_color=self.subtitle_stroke_color,
                        stroke_width=self.subtitle_stroke_width,
                        method="label",
                    )
                    txt_clip = (
                        txt_clip.set_start(current_time)
                        .set_duration(phrase_duration)
                        .set_position(("center", y_pos))
                    )
                    subtitle_clips.append(txt_clip)
                except Exception as e2:
                    print(f"  ⚠️ Subtitle fallback also failed: {e2}")

            current_time += time_per_phrase

        return subtitle_clips

    def _split_into_phrases(self, text: str) -> list[str]:
        """
        Split text into readable phrases for subtitles.

        Args:
            text: Full narration text

        Returns:
            List of phrases
        """
        # Clean up the text
        text = text.strip()

        # Split by sentence-ending punctuation first
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        phrases = []
        for sentence in sentences:
            # If sentence is short enough, use as-is
            words = sentence.split()
            if len(words) <= 8:
                phrases.append(sentence.strip())
            else:
                # Split longer sentences by commas or into chunks
                if "," in sentence:
                    parts = sentence.split(",")
                    for part in parts:
                        part = part.strip()
                        if part:
                            # Further split if still too long
                            part_words = part.split()
                            if len(part_words) <= 10:
                                phrases.append(part)
                            else:
                                # Split into chunks of ~6-8 words
                                chunk_size = 7
                                for i in range(0, len(part_words), chunk_size):
                                    chunk = " ".join(part_words[i : i + chunk_size])
                                    if chunk:
                                        phrases.append(chunk)
                else:
                    # Split into chunks of ~6-8 words
                    chunk_size = 7
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i : i + chunk_size])
                        if chunk:
                            phrases.append(chunk)

        return phrases

    def _wrap_text(self, text: str) -> str:
        """
        Wrap text to fit within max characters per line.

        Args:
            text: Text to wrap

        Returns:
            Wrapped text with newlines
        """
        return "\n".join(textwrap.wrap(text, width=self.max_chars_per_line))

    def _add_ken_burns(
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

        def zoom_effect(get_frame, t):
            """Apply zoom effect at time t."""
            frame = get_frame(t)
            duration = clip.duration
            zoom = zoom_start + (zoom_end - zoom_start) * (t / duration)

            # Get original dimensions
            h, w = frame.shape[:2]

            # Calculate new dimensions
            new_w = int(w * zoom)
            new_h = int(h * zoom)

            # Calculate crop coordinates to keep center
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2

            # Resize frame
            import numpy as np
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = img.crop((x_start, y_start, x_start + w, y_start + h))

            return np.array(img)

        return clip.fl(zoom_effect)

    def _apply_transition(self, clip, transition: str):
        """
        Apply transition effects to a clip.

        Args:
            clip: The video clip
            transition: Transition type

        Returns:
            Clip with transitions applied
        """
        # Get clip duration and calculate safe fade duration
        clip_duration = getattr(clip, "duration", None)
        if clip_duration is None or clip_duration < 0.5:
            # Too short for transitions, return as-is
            return clip

        # Ensure fade duration doesn't exceed half the clip duration
        safe_fade = min(self.fade_duration, clip_duration / 4)

        if transition == "fade_in":
            clip = clip.fadein(safe_fade)
        elif transition == "fade_out":
            clip = clip.fadeout(safe_fade)
        elif transition == "crossfade":
            clip = clip.fadein(safe_fade).fadeout(safe_fade)
        elif transition == "dissolve":
            dissolve_fade = min(safe_fade * 1.5, clip_duration / 4)
            clip = clip.fadein(dissolve_fade).fadeout(dissolve_fade)
        elif transition == "fade_to_black":
            black_fade = min(safe_fade * 2, clip_duration / 3)
            clip = clip.fadeout(black_fade)
        # "cut" transition has no effect

        return clip

    def _add_background_music(self, video) -> any:
        """
        Add background music to the video.

        Args:
            video: The video clip

        Returns:
            Video with background music added
        """
        print("  🎵 Adding background music...")

        try:
            bg_music = AudioFileClip(str(self.background_music_path))

            # Get the exact video duration
            video_duration = video.duration

            # Adjust volume
            bg_music = bg_music.volumex(self.background_music_volume)

            # Loop or trim music to match video duration
            if bg_music.duration < video_duration:
                # Loop the music
                bg_music = bg_music.fx(vfx.loop, duration=video_duration)
            else:
                # Trim to video length
                bg_music = bg_music.subclip(0, video_duration)

            # Fade out the music at the end (use smaller fade for short videos)
            fade_duration = min(2.0, video_duration / 4)
            bg_music = bg_music.audio_fadeout(fade_duration)

            # Set explicit duration on background music
            bg_music = bg_music.set_duration(video_duration)

            # Combine with existing audio
            if video.audio:
                # Get narration and adjust volume
                narration = video.audio.volumex(self.narration_volume)
                # Ensure narration has explicit duration
                narration = narration.set_duration(video_duration)

                # Composite both audio tracks
                composite_audio = CompositeAudioClip([narration, bg_music])
                composite_audio = composite_audio.set_duration(video_duration)
                video = video.set_audio(composite_audio)
            else:
                video = video.set_audio(bg_music)

            return video

        except Exception as e:
            print(f"  ⚠️ Could not add background music: {e}")
            return video

    def compose_simple(
        self,
        image_paths: list[Path],
        audio_paths: list[Path],
        output_path: Path,
        narrations: Optional[list[str]] = None,
        video_width: int = 1920,
        video_height: int = 1080,
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

        Returns:
            Path to the generated video
        """
        if len(image_paths) != len(audio_paths):
            raise ValueError("Number of images and audio files must match")

        print(f"🎥 Simple compose: {len(image_paths)} slides...")

        # Adjust subtitle position for the video dimensions
        y_pos = int(video_height * 0.85)

        clips = []
        for i, (img_path, audio_path) in enumerate(zip(image_paths, audio_paths)):
            # Load audio
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration + 0.5

            # Create image clip
            img_clip = ImageClip(str(img_path)).set_duration(duration)

            # Create composite with subtitles if narrations provided
            clips_to_composite = [img_clip]

            if narrations and i < len(narrations) and narrations[i]:
                subtitle_clips = self._create_subtitle_clips(
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
                .fadein(0.5)
                .fadeout(0.5)
            )

            clips.append(composite)

        # Concatenate
        final_video = concatenate_videoclips(clips, method="compose")

        # Add background music
        if self.background_music_path.exists():
            final_video = self._add_background_music(final_video)

        # Export
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=self.codec,
            audio_codec=self.audio_codec,
            audio_bitrate=self.audio_bitrate,
            preset=self.preset,
            threads=self.threads,
            logger=None,
        )

        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()

        print(f"✅ Video created: {output_path}")
        return output_path

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get the duration of a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Duration in seconds
        """
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        return duration
