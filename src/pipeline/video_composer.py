"""
Video Composer Module - Stitches scenes into final video with viral-optimized subtitles.

This module handles the final video composition, combining scene images
and audio with transitions, background music, and dynamic karaoke-style subtitles.

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
import random
import re
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

# =============================================================================
# VIRAL SHORTS CONFIGURATION
# =============================================================================

# Emphasis words that get highlighted in different colors
EMPHASIS_KEYWORDS = {
    # Strong emphasis (yellow/gold)
    "strong": [
        "secret",
        "amazing",
        "incredible",
        "powerful",
        "essential",
        "critical",
        "important",
        "key",
        "game-changer",
        "revolutionary",
        "breakthrough",
        "must",
        "never",
        "always",
        "best",
        "worst",
        "biggest",
        "most",
        "first",
        "last",
        "only",
        "ultimate",
        "proven",
        "guaranteed",
        "free",
        "new",
        "now",
        "today",
        "instantly",
        "immediately",
    ],
    # Action words (cyan/blue)
    "action": [
        "learn",
        "discover",
        "master",
        "unlock",
        "build",
        "create",
        "start",
        "stop",
        "try",
        "use",
        "get",
        "make",
        "find",
        "see",
        "watch",
        "click",
        "subscribe",
        "follow",
        "share",
        "comment",
        "like",
    ],
    # Numbers and stats (green)
    "stats": [
        "percent",
        "%",
        "million",
        "billion",
        "thousand",
        "hundred",
        "twice",
        "triple",
        "double",
        "half",
        "10x",
        "100x",
    ],
}

# Emoji mappings for common concepts
EMOJI_MAPPINGS = {
    # Tech/coding
    "code": "💻",
    "coding": "💻",
    "programming": "💻",
    "developer": "👨‍💻",
    "software": "🖥️",
    "app": "📱",
    "website": "🌐",
    "database": "🗄️",
    "ai": "🤖",
    "artificial intelligence": "🤖",
    "machine learning": "🧠",
    "robot": "🤖",
    "automation": "⚙️",
    "api": "🔌",
    # Business/money
    "money": "💰",
    "profit": "💵",
    "revenue": "📈",
    "growth": "📈",
    "success": "🏆",
    "win": "🏆",
    "goal": "🎯",
    "target": "🎯",
    "business": "💼",
    "startup": "🚀",
    "company": "🏢",
    # Learning/education
    "learn": "📚",
    "study": "📖",
    "knowledge": "🧠",
    "idea": "💡",
    "tip": "💡",
    "trick": "🎩",
    "hack": "⚡",
    "secret": "🤫",
    "mistake": "❌",
    "error": "⚠️",
    "wrong": "❌",
    "correct": "✅",
    "right": "✅",
    "yes": "✅",
    "no": "❌",
    # Emotions/reactions
    "amazing": "🤯",
    "incredible": "😱",
    "wow": "😮",
    "cool": "😎",
    "love": "❤️",
    "hate": "😤",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "excited": "🎉",
    "surprised": "😲",
    # Time
    "fast": "⚡",
    "quick": "⚡",
    "slow": "🐌",
    "time": "⏰",
    "now": "⏰",
    "today": "📅",
    "tomorrow": "📆",
    # General
    "fire": "🔥",
    "hot": "🔥",
    "trending": "📈",
    "viral": "🔥",
    "warning": "⚠️",
    "important": "❗",
    "attention": "👀",
    "question": "❓",
    "answer": "💬",
    "point": "👉",
}


class VideoComposer:
    """Composes final video from scene assets with viral-optimized subtitles."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        background_music_path: Optional[Path] = None,
        sound_effects_dir: Optional[Path] = None,
    ):
        """
        Initialize the video composer.

        Args:
            output_dir: Directory to save output video
            background_music_path: Path to background music file
            sound_effects_dir: Directory containing sound effect files
        """
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

        # Transition settings
        self.fade_duration = 0.3  # Faster transitions for viral content
        self.background_music_volume = 0.12
        self.narration_volume = 1.3  # Louder narration
        self.sfx_volume = 0.25

        # ===========================================
        # VIRAL SUBTITLE SETTINGS
        # ===========================================
        self.subtitle_enabled = True
        self.karaoke_mode = True  # Word-by-word highlighting
        self.emoji_enabled = False  # Add emojis to captions (disabled by default)
        self.color_emphasis = True  # Highlight key words
        self.dynamic_motion = True  # Enable zoom/shake effects

        # Subtitle appearance (will be adjusted based on format)
        self.subtitle_font_size = 70  # Larger for mobile
        self.subtitle_color = "white"
        self.subtitle_highlight_color = "#FFD700"  # Gold for current word
        self.subtitle_emphasis_colors = {
            "strong": "#FFD700",  # Gold/yellow
            "action": "#00D4FF",  # Cyan
            "stats": "#00FF88",  # Green
        }
        self.subtitle_stroke_color = "black"
        self.subtitle_stroke_width = 4  # Thicker stroke for readability
        self.subtitle_bg_color = None
        self.subtitle_position = ("center", 0.78)  # Slightly higher for mobile
        self.max_chars_per_line = 20  # Shorter lines for impact

        # Karaoke timing settings
        self.words_per_group = 3  # Show 3 words at a time
        self.min_word_duration = 0.15  # Minimum time per word
        self.use_whisper_alignment = True  # Use Whisper for precise word timing
        self.save_ass_files = True  # Save .ass subtitle files to output dir

        # Motion effect settings
        self.enable_ken_burns = True
        self.enable_scene_shake = True  # Subtle shake on emphasis
        self.enable_zoom_pulse = True  # Quick zoom on key moments

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
        self.karaoke_mode = viral_mode
        # emoji_enabled stays as initialized (False by default)
        self.color_emphasis = viral_mode
        self.dynamic_motion = viral_mode

        # Validate scenes have required assets
        self._validate_scenes(scenes)

        # Adjust subtitle settings based on video format
        self._adjust_subtitle_settings(request, viral_mode)

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
            format_suffix = "short" if request.format.value == "short" else "long"
            from datetime import datetime

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

    def _adjust_subtitle_settings(
        self, request: VideoRequest, viral_mode: bool = True
    ) -> None:
        """Adjust subtitle settings based on video format and viral mode."""
        is_short = request.format.value == "short"

        if is_short:
            # Vertical video (9:16) - MAXIMUM IMPACT for shorts
            if viral_mode:
                self.subtitle_font_size = 75  # Extra large
                self.subtitle_stroke_width = 5
                self.max_chars_per_line = 18  # Very short lines
                self.subtitle_position = ("center", 0.75)
                self.words_per_group = 2  # Faster word reveal
            else:
                self.subtitle_font_size = 55
                self.subtitle_stroke_width = 3
                self.max_chars_per_line = 25
                self.subtitle_position = ("center", 0.80)
        else:
            # Horizontal video (16:9)
            if viral_mode:
                self.subtitle_font_size = 55
                self.subtitle_stroke_width = 4
                self.max_chars_per_line = 35
                self.subtitle_position = ("center", 0.82)
                self.words_per_group = 4
            else:
                self.subtitle_font_size = 45
                self.subtitle_stroke_width = 2
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
        if self.dynamic_motion:
            image_clip = self._add_dynamic_motion(
                image_clip, scene_index, total_scenes, scene.mood
            )
        elif self.enable_ken_burns:
            image_clip = self._add_ken_burns(image_clip)

        # Create list of clips to composite
        clips_to_composite = [image_clip]

        # Add subtitles if enabled
        if enable_subtitles and scene.narration:
            if self.karaoke_mode:
                subtitle_clips = self._create_karaoke_subtitles(
                    scene.narration,
                    duration,
                    video_width,
                    video_height,
                    audio_duration=audio_duration,
                    audio_path=scene.audio_path,  # Pass audio for Whisper alignment
                )
            else:
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

    # =========================================================================
    # KARAOKE-STYLE SUBTITLES (VIRAL FEATURE)
    # =========================================================================

    def _create_karaoke_subtitles(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
        audio_path: str = None,
    ) -> list:
        """
        Create karaoke-style subtitles with word-by-word highlighting.

        TRUE KARAOKE: Single line where current word is highlighted in color,
        other words stay in default color. No duplicate text.

        If audio_path is provided and use_whisper_alignment is True,
        uses Whisper for precise word-level timing.
        """
        subtitle_clips = []

        # Preprocess: add emojis if enabled
        if self.emoji_enabled:
            narration = self._add_emojis_to_text(narration)

        y_pos = int(video_height * self.subtitle_position[1])

        # Try to use Whisper alignment for precise timing
        word_timestamps = None
        if self.use_whisper_alignment and audio_path:
            word_timestamps = self._get_whisper_timestamps(audio_path)

        if word_timestamps:
            # Use precise Whisper timestamps
            print(f"      🎤 Using Whisper alignment ({len(word_timestamps)} words)")

            # Save .ass file if enabled
            if self.save_ass_files and audio_path:
                self._save_ass_file(word_timestamps, audio_path)

            return self._create_karaoke_from_timestamps(
                word_timestamps, video_width, y_pos, audio_duration
            )

        # Fallback to estimated timing
        print(f"      ⏱️ Using estimated timing (no Whisper)")

        # Split into words while preserving punctuation
        words = self._tokenize_words(narration)

        if not words:
            return subtitle_clips

        # Calculate timing
        available_duration = audio_duration * 0.95
        time_per_word = available_duration / len(words)

        # Ensure minimum word duration
        time_per_word = max(time_per_word, self.min_word_duration)

        # Group words for display
        word_groups = self._group_words(words, self.words_per_group)

        # Calculate timing per group
        words_processed = 0

        for group_idx, word_group in enumerate(word_groups):
            group_start = words_processed * time_per_word
            group_duration = len(word_group) * time_per_word

            # Ensure we don't exceed audio duration
            if group_start + group_duration > audio_duration:
                group_duration = max(0.1, audio_duration - group_start - 0.05)

            if group_duration <= 0:
                break

            # Create clips for this word group with highlighting
            group_clips = self._create_word_group_clips(
                word_group,
                group_start,
                group_duration,
                video_width,
                y_pos,
            )
            subtitle_clips.extend(group_clips)

            words_processed += len(word_group)

        return subtitle_clips

    def _get_whisper_timestamps(self, audio_path: str) -> list:
        """
        Get word-level timestamps from audio using OpenAI Whisper.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of dicts with 'word', 'start', 'end' keys, or None if failed
        """
        try:
            from .audio_aligner import AudioAligner

            aligner = AudioAligner()
            result = aligner.get_word_timestamps(audio_path)

            # Store the result for .ass file generation
            self._last_alignment_result = result

            return [
                {"word": w.word, "start": w.start, "end": w.end} for w in result.words
            ]
        except Exception as e:
            print(f"      ⚠️ Whisper alignment failed: {e}")
            return None

    def _save_ass_file(self, word_timestamps: list, audio_path: str) -> None:
        """
        Save .ass subtitle file from word timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'
            audio_path: Path to the audio file (used for naming)
        """
        try:
            from pathlib import Path

            from .audio_aligner import AlignmentResult, AudioAligner, WordTimestamp

            # Convert back to AlignmentResult for .ass generation
            words = [
                WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
                for w in word_timestamps
            ]

            # Create result object
            result = AlignmentResult(
                words=words,
                full_text=" ".join(w["word"] for w in word_timestamps),
                duration=word_timestamps[-1]["end"] if word_timestamps else 0,
                language="en",
            )

            # Generate output path
            audio_path = Path(audio_path)
            ass_path = self.output_dir / "subtitles" / f"{audio_path.stem}.ass"
            ass_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate .ass file
            aligner = AudioAligner()
            aligner.generate_ass_file(
                result,
                ass_path,
                words_per_line=self.words_per_group,
                font_size=self.subtitle_font_size,
            )
            print(f"      💾 Saved: {ass_path}")

        except Exception as e:
            print(f"      ⚠️ Failed to save .ass file: {e}")

    def _create_karaoke_from_timestamps(
        self,
        word_timestamps: list,
        video_width: int,
        y_pos: int,
        audio_duration: float,
    ) -> list:
        """
        Create karaoke clips using precise Whisper timestamps.

        Args:
            word_timestamps: List of dicts with 'word', 'start', 'end'
            video_width: Width of video
            y_pos: Y position for subtitles
            audio_duration: Total audio duration

        Returns:
            List of subtitle clips
        """
        clips = []

        # Group words into phrases
        groups = []
        current_group = []

        for wt in word_timestamps:
            current_group.append(wt)
            word = wt["word"]

            ends_sentence = any(word.rstrip().endswith(p) for p in [".", "!", "?"])
            ends_clause = any(word.rstrip().endswith(p) for p in [",", ";", ":"])

            if len(current_group) >= self.words_per_group or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= self.words_per_group - 1:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        # Create clips for each group with precise timing
        for group in groups:
            group_start = group[0]["start"]
            group_end = group[-1]["end"]
            words = [wt["word"] for wt in group]

            # Create clips for each word highlight in this group
            for i, wt in enumerate(group):
                word_start = wt["start"]
                word_end = wt["end"]
                word_duration = word_end - word_start

                if word_duration <= 0:
                    continue

                # Get highlight color for current word
                word_lower = wt["word"].lower().strip(".,!?;:")
                highlight_color = self._get_word_color(word_lower, is_highlighted=True)

                try:
                    # Create clips for each word in the group
                    word_clips = self._create_word_group_clips_precise(
                        words,
                        i,
                        word_start,
                        word_duration,
                        video_width,
                        y_pos,
                        highlight_color,
                    )
                    clips.extend(word_clips)
                except Exception as e:
                    print(f"      ⚠️ Failed to create clip for '{wt['word']}': {e}")

        return clips

    def _create_word_group_clips_precise(
        self,
        words: list,
        highlight_idx: int,
        start_time: float,
        duration: float,
        video_width: int,
        y_pos: int,
        highlight_color: str,
    ) -> list:
        """
        Create text clips for a word group with precise timing from Whisper.
        """
        clips = []

        try:
            # Calculate total width for centering
            temp_clips = []
            for word in words:
                tc = TextClip(
                    word + " ",
                    fontsize=self.subtitle_font_size,
                    color=self.subtitle_color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.subtitle_stroke_color,
                    stroke_width=self.subtitle_stroke_width,
                    method="label",
                )
                temp_clips.append(tc)

            total_width = sum(tc.w for tc in temp_clips)

            for tc in temp_clips:
                tc.close()

            # Create the actual clips
            x_offset = (video_width - total_width) // 2

            for i, word in enumerate(words):
                is_current = i == highlight_idx
                word_color = highlight_color if is_current else self.subtitle_color
                font_size = int(self.subtitle_font_size * (1.1 if is_current else 1.0))

                word_clip = TextClip(
                    word + (" " if i < len(words) - 1 else ""),
                    fontsize=font_size,
                    color=word_color,
                    font=str(self.font_path)
                    if self.font_path.exists()
                    else "Arial-Bold",
                    stroke_color=self.subtitle_stroke_color,
                    stroke_width=self.subtitle_stroke_width,
                    method="label",
                )

                word_y = y_pos - (
                    int(self.subtitle_font_size * 0.05) if is_current else 0
                )

                word_clip = (
                    word_clip.set_start(start_time)
                    .set_duration(duration)
                    .set_position((x_offset, word_y))
                )

                clips.append(word_clip)
                x_offset += word_clip.w

        except Exception as e:
            print(f"      ⚠️ Precise clip creation failed: {e}")

        return clips

    def _tokenize_words(self, text: str) -> list[str]:
        """Split text into words while preserving emojis and punctuation."""
        # Clean up text
        text = text.strip()

        # Split by whitespace but keep words with their punctuation
        words = text.split()

        # Filter out empty strings
        words = [w for w in words if w.strip()]

        return words

    def _group_words(self, words: list[str], group_size: int) -> list[list[str]]:
        """Group words for display, respecting punctuation boundaries."""
        groups = []
        current_group = []

        for word in words:
            current_group.append(word)

            # End group at punctuation or when size reached
            ends_sentence = any(word.endswith(p) for p in [".", "!", "?", ":"])
            ends_clause = any(word.endswith(p) for p in [",", ";", "-"])

            if len(current_group) >= group_size or ends_sentence:
                groups.append(current_group)
                current_group = []
            elif ends_clause and len(current_group) >= group_size - 1:
                groups.append(current_group)
                current_group = []

        # Add remaining words
        if current_group:
            groups.append(current_group)

        return groups

    def _create_word_group_clips(
        self,
        words: list[str],
        start_time: float,
        duration: float,
        video_width: int,
        y_pos: int,
    ) -> list:
        """
        Create text clips for a group of words with karaoke highlighting.

        TRUE KARAOKE: Single line where current word is highlighted in color,
        other words stay in default color. No duplicate text.
        """
        clips = []
        time_per_word = duration / len(words)

        # For each word position, show the ENTIRE phrase but with current word highlighted
        for highlight_idx in range(len(words)):
            word_start = start_time + (highlight_idx * time_per_word)
            word_duration = time_per_word

            # Determine highlight color for current word
            current_word = words[highlight_idx]
            word_lower = current_word.lower().strip(".,!?;:")
            highlight_color = self._get_word_color(word_lower, is_highlighted=True)

            try:
                # Build clips for each word - non-highlighted words in white, highlighted in color
                word_clips_for_frame = []

                # Calculate total width to center the phrase
                # We'll position words side by side
                temp_clips = []
                for word in words:
                    tc = TextClip(
                        word + " ",
                        fontsize=self.subtitle_font_size,
                        color=self.subtitle_color,
                        font=str(self.font_path)
                        if self.font_path.exists()
                        else "Arial-Bold",
                        stroke_color=self.subtitle_stroke_color,
                        stroke_width=self.subtitle_stroke_width,
                        method="label",
                    )
                    temp_clips.append(tc)

                total_width = sum(tc.w for tc in temp_clips)

                # Clean up temp clips
                for tc in temp_clips:
                    tc.close()

                # Now create the actual clips with proper positioning
                x_offset = (video_width - total_width) // 2

                for i, word in enumerate(words):
                    is_current = i == highlight_idx
                    word_color = highlight_color if is_current else self.subtitle_color
                    # Highlighted word is 10% bigger (proportional to font size)
                    font_size = int(
                        self.subtitle_font_size * (1.1 if is_current else 1.0)
                    )

                    word_clip = TextClip(
                        word + (" " if i < len(words) - 1 else ""),
                        fontsize=font_size,
                        color=word_color,
                        font=str(self.font_path)
                        if self.font_path.exists()
                        else "Arial-Bold",
                        stroke_color=self.subtitle_stroke_color,
                        stroke_width=self.subtitle_stroke_width,
                        method="label",
                    )

                    # Adjust y position for larger highlighted word (proportional)
                    word_y = y_pos - (
                        int(self.subtitle_font_size * 0.05) if is_current else 0
                    )

                    word_clip = (
                        word_clip.set_start(word_start)
                        .set_duration(word_duration)
                        .set_position((x_offset, word_y))
                    )

                    word_clips_for_frame.append(word_clip)
                    x_offset += word_clip.w

                clips.extend(word_clips_for_frame)

            except Exception as e:
                print(f"  ⚠️ Failed to create karaoke clip: {e}")
                # Fallback - just show full phrase in highlight color
                try:
                    full_phrase = " ".join(words)
                    txt_clip = TextClip(
                        full_phrase,
                        fontsize=self.subtitle_font_size,
                        color=self.subtitle_highlight_color,
                        stroke_color=self.subtitle_stroke_color,
                        stroke_width=self.subtitle_stroke_width,
                        method="label",
                    )
                    txt_clip = (
                        txt_clip.set_start(word_start)
                        .set_duration(word_duration)
                        .set_position(("center", y_pos))
                    )
                    clips.append(txt_clip)
                except Exception:
                    pass

        return clips

    def _get_word_color(self, word: str, is_highlighted: bool = False) -> str:
        """Determine the color for a word based on emphasis rules."""
        word_lower = word.lower().strip(".,!?;:\"'")

        if not self.color_emphasis:
            return (
                self.subtitle_highlight_color if is_highlighted else self.subtitle_color
            )

        # Check emphasis categories
        for category, keywords in EMPHASIS_KEYWORDS.items():
            if word_lower in keywords:
                return self.subtitle_emphasis_colors.get(
                    category, self.subtitle_highlight_color
                )

        # Check if it's a number
        if any(c.isdigit() for c in word):
            return self.subtitle_emphasis_colors.get("stats", "#00FF88")

        # Default highlight color
        return self.subtitle_highlight_color if is_highlighted else self.subtitle_color

    def _add_emojis_to_text(self, text: str) -> str:
        """Add relevant emojis to text based on keyword mappings."""
        words = text.split()
        result_words = []

        # Track which emojis we've added to avoid repetition
        added_emojis = set()

        for i, word in enumerate(words):
            result_words.append(word)

            # Check if this word (or the previous word + this word) triggers an emoji
            word_lower = word.lower().strip(".,!?;:")

            # Check single word
            if word_lower in EMOJI_MAPPINGS and word_lower not in added_emojis:
                emoji = EMOJI_MAPPINGS[word_lower]
                # Add emoji after the word (max 3 emojis per text)
                if len(added_emojis) < 3:
                    result_words.append(emoji)
                    added_emojis.add(word_lower)

            # Check two-word phrases
            if i > 0:
                prev_word = words[i - 1].lower().strip(".,!?;:")
                phrase = f"{prev_word} {word_lower}"
                if phrase in EMOJI_MAPPINGS and phrase not in added_emojis:
                    emoji = EMOJI_MAPPINGS[phrase]
                    if len(added_emojis) < 3:
                        result_words.append(emoji)
                        added_emojis.add(phrase)

        return " ".join(result_words)

    # =========================================================================
    # STANDARD SUBTITLES (NON-KARAOKE)
    # =========================================================================

    def _create_subtitle_clips(
        self,
        narration: str,
        total_duration: float,
        video_width: int,
        video_height: int,
        audio_duration: float,
    ) -> list[TextClip]:
        """
        Create standard subtitle clips (phrase-based, non-karaoke).

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

        # Add emojis if enabled
        if self.emoji_enabled:
            narration = self._add_emojis_to_text(narration)

        # Split narration into phrases/chunks for better readability
        phrases = self._split_into_phrases(narration)

        if not phrases:
            return subtitle_clips

        # Calculate timing for each phrase
        available_duration = audio_duration * 0.95
        time_per_phrase = available_duration / len(phrases)

        # Calculate y position
        y_pos = int(video_height * self.subtitle_position[1])

        current_time = 0.0
        for phrase in phrases:
            # Wrap text if too long
            wrapped_text = self._wrap_text(phrase)

            try:
                # Create text clip with enhanced styling
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
                    size=(video_width - 80, None),
                    align="center",
                )

                # Set timing and position
                phrase_duration = min(
                    time_per_phrase, audio_duration - current_time - 0.1
                )
                if phrase_duration <= 0:
                    break

                txt_clip = (
                    txt_clip.set_start(current_time)
                    .set_duration(phrase_duration)
                    .set_position(("center", y_pos))
                )

                # Add fade in/out for smooth transitions
                fade_duration = min(0.08, phrase_duration / 4)
                if fade_duration > 0.03:
                    txt_clip = txt_clip.crossfadein(fade_duration).crossfadeout(
                        fade_duration
                    )

                subtitle_clips.append(txt_clip)

            except Exception as e:
                print(f"  ⚠️ Failed to create subtitle clip: {e}")
                # Fallback
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
        Shorter phrases for more impact.
        """
        text = text.strip()

        # Split by sentence-ending punctuation first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        phrases = []
        # Shorter chunk size for viral content
        chunk_size = 5 if self.karaoke_mode else 6

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= chunk_size + 2:
                phrases.append(sentence.strip())
            else:
                # Split by commas or into chunks
                if "," in sentence:
                    parts = sentence.split(",")
                    for part in parts:
                        part = part.strip()
                        if part:
                            part_words = part.split()
                            if len(part_words) <= chunk_size + 3:
                                phrases.append(part)
                            else:
                                for i in range(0, len(part_words), chunk_size):
                                    chunk = " ".join(part_words[i : i + chunk_size])
                                    if chunk:
                                        phrases.append(chunk)
                else:
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i : i + chunk_size])
                        if chunk:
                            phrases.append(chunk)

        return phrases

    def _wrap_text(self, text: str) -> str:
        """Wrap text to fit within max characters per line."""
        return "\n".join(textwrap.wrap(text, width=self.max_chars_per_line))

    # =========================================================================
    # DYNAMIC MOTION EFFECTS (VIRAL FEATURE)
    # =========================================================================

    def _add_dynamic_motion(
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
        """
        duration = clip.duration

        # Determine effect type based on scene position
        if scene_index == 0:
            # Opening scene: zoom in to draw attention
            return self._add_ken_burns(clip, zoom_start=1.0, zoom_end=1.15)
        elif scene_index == total_scenes - 1:
            # Closing scene: zoom out for finale
            return self._add_ken_burns(clip, zoom_start=1.1, zoom_end=1.0)
        else:
            # Middle scenes: variety of effects
            effect_type = scene_index % 3

            if effect_type == 0:
                # Slow zoom in
                return self._add_ken_burns(clip, zoom_start=1.0, zoom_end=1.1)
            elif effect_type == 1:
                # Slow zoom out
                return self._add_ken_burns(clip, zoom_start=1.1, zoom_end=1.0)
            else:
                # Pan effect (via different zoom centers)
                return self._add_ken_burns(clip, zoom_start=1.05, zoom_end=1.1)

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
        Faster transitions for viral content.
        """
        clip_duration = getattr(clip, "duration", None)
        if clip_duration is None or clip_duration < 0.3:
            return clip

        # Use faster fade for viral content
        safe_fade = min(self.fade_duration, clip_duration / 5)

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
        self.karaoke_mode = viral_mode
        # emoji_enabled stays as initialized (False by default)
        self.color_emphasis = viral_mode
        self.dynamic_motion = viral_mode

        # Adjust settings for video dimensions
        is_vertical = video_height > video_width
        if is_vertical and viral_mode:
            self.subtitle_font_size = 75
            self.subtitle_stroke_width = 5
            self.max_chars_per_line = 18
            self.subtitle_position = ("center", 0.75)
        elif viral_mode:
            self.subtitle_font_size = 55
            self.max_chars_per_line = 35

        y_pos = int(video_height * self.subtitle_position[1])

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
            if self.dynamic_motion:
                print(f"      🎬 Adding motion effects...")
                img_clip = self._add_dynamic_motion(
                    img_clip, i, len(image_paths), "informative"
                )

            # Create composite with subtitles if narrations provided
            clips_to_composite = [img_clip]

            if narrations and i < len(narrations) and narrations[i]:
                # Check for cached timestamps first
                scene_timestamps = None
                if cached_timestamps and i < len(cached_timestamps):
                    scene_timestamps = cached_timestamps[i]

                if self.karaoke_mode:
                    if scene_timestamps:
                        # Use cached timestamps - no Whisper API call needed
                        print(
                            f"      🎤 Using cached timestamps ({len(scene_timestamps)} words)"
                        )
                        subtitle_clips = self._create_karaoke_from_timestamps(
                            scene_timestamps,
                            video_width,
                            int(video_height * self.subtitle_position[1]),
                            audio_clip.duration,
                        )
                    else:
                        # Fall back to Whisper or estimated timing
                        subtitle_clips = self._create_karaoke_subtitles(
                            narrations[i],
                            duration,
                            video_width,
                            video_height,
                            audio_duration=audio_clip.duration,
                            audio_path=str(
                                audio_path
                            ),  # Pass audio for Whisper alignment
                        )
                else:
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
