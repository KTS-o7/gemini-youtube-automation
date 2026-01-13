#!/usr/bin/env python3
"""
Recompose Video - Regenerate video from existing images and audio with viral subtitles.

Usage:
    source .venv/bin/activate && python recompose_video.py

This script uses existing assets in the output/ folder:
- output/images/scene_01.png, etc.
- output/audio/voice_01.wav, etc.
- output/planned_scenes.json (for narration text)
- output/subtitles/timestamps/*.json (pre-generated word timestamps - OPTIONAL)

And recomposes them with the new viral subtitle features:
- Karaoke-style word-by-word captions (using cached Whisper timestamps)
- Color emphasis for key words
- Dynamic motion effects
- Faster scene pacing

If timestamps JSON files exist, Whisper API is NOT called again (saves time & money).
Run `python generate_subtitles.py` first to generate timestamps.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Verify API key is loaded
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("   Make sure you have a .env file with OPENAI_API_KEY=sk-...")
    sys.exit(1)


def load_cached_timestamps(audio_path: Path, timestamps_dir: Path) -> list:
    """
    Load cached word timestamps from JSON file if available.

    Args:
        audio_path: Path to the audio file
        timestamps_dir: Directory containing timestamp JSON files

    Returns:
        List of word timestamps or None if not found
    """
    json_path = timestamps_dir / f"{audio_path.stem}.json"

    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                return data.get("words", [])
        except Exception as e:
            print(f"  ⚠️ Failed to load cached timestamps: {e}")

    return None


def main():
    """Recompose video from existing assets."""
    print("🔄 Recomposing video from existing assets...\n")

    output_dir = Path("output")
    images_dir = output_dir / "images"
    audio_dir = output_dir / "audio"
    timestamps_dir = output_dir / "subtitles" / "timestamps"
    scenes_file = output_dir / "planned_scenes.json"

    # Check if assets exist
    if not scenes_file.exists():
        print(f"❌ Error: {scenes_file} not found")
        sys.exit(1)

    # Load planned scenes for narration
    with open(scenes_file, "r") as f:
        scenes_data = json.load(f)

    print(f"📋 Found {len(scenes_data)} scenes in planned_scenes.json")

    # Check if cached timestamps exist
    has_cached_timestamps = timestamps_dir.exists() and any(
        timestamps_dir.glob("*.json")
    )
    if has_cached_timestamps:
        print(f"✅ Found cached timestamps in {timestamps_dir}")
        print(f"   (Whisper API will NOT be called - using cached data)")
    else:
        print(f"⚠️  No cached timestamps found in {timestamps_dir}")
        print(f"   Run 'python generate_subtitles.py' first to cache timestamps")
        print(f"   (Will call Whisper API for each audio file)")

    # Collect image and audio paths
    image_paths = []
    audio_paths = []
    narrations = []
    cached_timestamps = []  # Pre-loaded timestamps for each scene

    for scene in scenes_data:
        scene_num = scene["scene_number"]

        # Find image
        img_path = images_dir / f"scene_{scene_num:02d}.png"
        if not img_path.exists():
            print(f"❌ Error: Image not found: {img_path}")
            sys.exit(1)
        image_paths.append(img_path)

        # Find audio
        audio_path = audio_dir / f"voice_{scene_num:02d}.wav"
        if not audio_path.exists():
            # Try .mp3
            audio_path = audio_dir / f"voice_{scene_num:02d}.mp3"
            if not audio_path.exists():
                print(f"❌ Error: Audio not found: {audio_path}")
                sys.exit(1)
        audio_paths.append(audio_path)

        # Get narration
        narrations.append(scene.get("narration", ""))

        # Load cached timestamps if available
        timestamps = load_cached_timestamps(audio_path, timestamps_dir)
        cached_timestamps.append(timestamps)

    print(f"🖼️  Images: {len(image_paths)} found")
    print(f"🔊 Audio: {len(audio_paths)} found")
    print(f"📝 Narrations: {len([n for n in narrations if n])} found")
    print(f"⏱️  Cached timestamps: {len([t for t in cached_timestamps if t])} found")

    # Import video composer
    from src.pipeline.video_composer import VideoComposer

    # Create composer with viral settings
    composer = VideoComposer(output_dir=output_dir)

    # Configure viral mode
    composer.karaoke_mode = True
    composer.emoji_enabled = False  # Disabled as requested
    composer.color_emphasis = (
        True  # Enabled - different colors for different word types
    )
    composer.dynamic_motion = True

    # Only use Whisper if no cached timestamps (saves API calls)
    composer.use_whisper_alignment = not has_cached_timestamps
    composer.save_ass_files = False  # Don't regenerate .ass files

    # Pass cached timestamps to composer
    composer._cached_timestamps = cached_timestamps

    # Vertical video settings (short format) - 2K with 9:16 aspect ratio
    video_width = 1440  # 2K vertical width
    video_height = 2560  # 2K vertical height (9:16 aspect ratio)

    # Adjust for 2K vertical - LARGER font for crisp subtitles
    composer.subtitle_font_size = 140  # Much bigger for 2K sharpness
    composer.subtitle_stroke_width = 5  # Clean stroke (not too thick)
    composer.max_chars_per_line = 18
    composer.subtitle_position = ("center", 0.75)
    composer.words_per_group = 2

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"video_viral_{timestamp}.mp4"

    print(f"\n🎬 Composing viral video...")
    print(f"   - Karaoke mode: {composer.karaoke_mode}")
    print(f"   - Whisper alignment: {composer.use_whisper_alignment}")
    print(f"   - Color emphasis: {composer.color_emphasis}")
    print(f"   - Dynamic motion: {composer.dynamic_motion}")
    print(f"   - Font size: {composer.subtitle_font_size}px")
    print(f"   - Words per group: {composer.words_per_group}")
    print(f"   - Resolution: {video_width}x{video_height} (2K)")
    print(f"\n⏳ Processing scenes (this may take 1-3 minutes)...")

    # Calculate estimated time
    total_words = sum(len(n.split()) for n in narrations if n)
    print(f"   - Total words to process: {total_words}")
    print(f"   - Estimated subtitle clips: ~{total_words} clips")
    print(f"\n📊 Progress:")

    # Compose the video with progress logging
    import time

    start_time = time.time()

    result_path = composer.compose_simple(
        image_paths=image_paths,
        audio_paths=audio_paths,
        output_path=output_path,
        narrations=narrations,
        video_width=video_width,
        video_height=video_height,
        viral_mode=True,
        cached_timestamps=cached_timestamps,  # Pass pre-loaded timestamps
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Video recomposed successfully!")
    print(f"📹 Output: {result_path}")
    print(f"⏱️  Time taken: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
