#!/usr/bin/env python3
"""
Generate Subtitles - Create .ass subtitle files for all audio files.

This script uses OpenAI Whisper to get precise word-level timestamps
and generates karaoke-style .ass subtitle files.

Usage:
    source .venv/bin/activate && python generate_subtitles.py

    # Or specify a different audio directory:
    python generate_subtitles.py --audio-dir output/audio --output-dir output/subtitles

Output:
    - output/subtitles/voice_01.ass
    - output/subtitles/voice_02.ass
    - etc.
    - output/subtitles/timestamps/ (JSON files with word timings)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Verify API key
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("   Make sure you have a .env file with OPENAI_API_KEY=sk-...")
    sys.exit(1)


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in the directory."""
    audio_extensions = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    # Sort by name
    audio_files.sort(key=lambda p: p.name)
    return audio_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate .ass subtitle files from audio using Whisper"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("output/audio"),
        help="Directory containing audio files (default: output/audio)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/subtitles"),
        help="Directory to save .ass files (default: output/subtitles)",
    )
    parser.add_argument(
        "--words-per-line",
        type=int,
        default=3,
        help="Number of words per subtitle line (default: 3)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=48,
        help="Font size for subtitles (default: 48)",
    )
    parser.add_argument(
        "--font-name",
        type=str,
        default="Arial Black",
        help="Font name for subtitles (default: Arial Black)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Also save timestamps as JSON files (default: True)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for Whisper (e.g., 'en'). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Validate audio directory
    if not args.audio_dir.exists():
        print(f"❌ Error: Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    # Find audio files
    audio_files = find_audio_files(args.audio_dir)

    if not audio_files:
        print(f"❌ Error: No audio files found in {args.audio_dir}")
        sys.exit(1)

    print(f"\n🎤 Generate Subtitles - Whisper Word-Level Alignment")
    print(f"=" * 55)
    print(f"📁 Audio directory: {args.audio_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎵 Audio files found: {len(audio_files)}")
    print(f"📝 Words per line: {args.words_per_line}")
    print(f"🔤 Font: {args.font_name} @ {args.font_size}px")
    print(f"=" * 55)

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_json:
        json_dir = args.output_dir / "timestamps"
        json_dir.mkdir(parents=True, exist_ok=True)

    # Import audio aligner
    from src.pipeline.audio_aligner import AudioAligner

    aligner = AudioAligner()

    # Process each audio file
    successful = 0
    failed = 0
    total_duration = 0
    total_words = 0

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        try:
            # Get word timestamps
            result = aligner.get_word_timestamps(audio_file, language=args.language)

            total_duration += result.duration
            total_words += len(result.words)

            # Generate .ass file
            ass_path = args.output_dir / f"{audio_file.stem}.ass"
            aligner.generate_ass_file(
                result,
                ass_path,
                words_per_line=args.words_per_line,
                font_name=args.font_name,
                font_size=args.font_size,
            )

            # Save JSON if requested
            if args.save_json:
                json_path = json_dir / f"{audio_file.stem}.json"
                aligner.save_timestamps_json(result, json_path)

            print(
                f"  ✅ Generated: {ass_path.name} ({len(result.words)} words, {result.duration:.1f}s)"
            )
            successful += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1

    # Summary
    print(f"\n{'=' * 55}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 55}")
    print(f"✅ Successful: {successful}/{len(audio_files)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(audio_files)}")
    print(f"⏱️  Total duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
    print(f"📝 Total words aligned: {total_words}")
    print(f"💰 Estimated Whisper cost: ${total_duration / 60 * 0.006:.4f}")
    print(f"\n📁 Output saved to: {args.output_dir}")

    if args.save_json:
        print(f"📁 Timestamps saved to: {args.output_dir / 'timestamps'}")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
