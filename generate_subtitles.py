#!/usr/bin/env python3
"""
Generate Subtitles - Create .ass subtitle files for all audio files.

Compatible with Python 3.9+

This script supports two alignment methods:
1. OpenAI Whisper (default) - Requires API key, costs ~$0.006/minute, auto-transcribes
2. Wav2Vec2 (free) - Runs locally, requires transcript text, completely free

Usage:
    # Using Whisper (auto-transcribes audio)
    python generate_subtitles.py

    # Using Wav2Vec2 with transcripts from script JSON
    python generate_subtitles.py --aligner wav2vec2 --script-json output/script.json

    # Using Wav2Vec2 with transcript text file
    python generate_subtitles.py --aligner wav2vec2 --transcript-dir output/transcripts

Output:
    - output/subtitles/voice_01.ass
    - output/subtitles/voice_02.ass
    - etc.
    - output/subtitles/timestamps/ (JSON files with word timings)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in the directory."""
    audio_extensions = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    # Sort by name
    audio_files.sort(key=lambda p: p.name)
    return audio_files


def load_transcripts_from_script(script_json_path: Path) -> dict[str, str]:
    """
    Load transcripts from a script JSON file.

    Returns dict mapping audio filename stem to transcript text.
    """
    transcripts = {}

    with open(script_json_path, "r") as f:
        data = json.load(f)

    # Handle different script JSON formats
    if "scenes" in data:
        # Format from scene planner
        for i, scene in enumerate(data["scenes"], 1):
            key = f"voice_{i:02d}"
            if isinstance(scene, dict):
                text = (
                    scene.get("narration")
                    or scene.get("text")
                    or scene.get("script", "")
                )
            else:
                text = str(scene)
            transcripts[key] = text
    elif "segments" in data:
        # Alternative format
        for i, segment in enumerate(data["segments"], 1):
            key = f"voice_{i:02d}"
            text = (
                segment.get("text", "") if isinstance(segment, dict) else str(segment)
            )
            transcripts[key] = text
    elif isinstance(data, list):
        # Simple list of texts
        for i, item in enumerate(data, 1):
            key = f"voice_{i:02d}"
            text = item.get("narration", item) if isinstance(item, dict) else str(item)
            transcripts[key] = text

    return transcripts


def load_transcript_from_file(transcript_dir: Path, audio_stem: str) -> Optional[str]:
    """Load transcript from a text file matching the audio filename."""
    # Try different extensions
    for ext in [".txt", ".transcript", ""]:
        transcript_path = transcript_dir / f"{audio_stem}{ext}"
        if transcript_path.exists():
            with open(transcript_path, "r") as f:
                return f.read().strip()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate .ass subtitle files from audio using Whisper or Wav2Vec2"
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
        "--aligner",
        type=str,
        choices=["whisper", "wav2vec2"],
        default="whisper",
        help="Alignment method: whisper (API, auto-transcribes) or wav2vec2 (free, needs transcript)",
    )
    parser.add_argument(
        "--script-json",
        type=Path,
        default=None,
        help="Path to script JSON file containing transcripts (for wav2vec2)",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=None,
        help="Directory containing transcript .txt files matching audio filenames (for wav2vec2)",
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

    # Check if wav2vec2 is available (requires torch)
    wav2vec2_available = True
    if args.aligner == "wav2vec2":
        try:
            import torch
            import torchaudio
        except ImportError:
            wav2vec2_available = False
            print("❌ Error: wav2vec2 requires PyTorch and torchaudio")
            print("   Install with: pip install torch torchaudio")
            print("   Or use --aligner whisper instead (requires OPENAI_API_KEY)")
            sys.exit(1)

    # Load transcripts if using wav2vec2
    transcripts = {}
    if args.aligner == "wav2vec2":
        if args.script_json:
            if not args.script_json.exists():
                print(f"❌ Error: Script JSON not found: {args.script_json}")
                sys.exit(1)
            transcripts = load_transcripts_from_script(args.script_json)
            print(f"📝 Loaded {len(transcripts)} transcripts from {args.script_json}")
        elif args.transcript_dir:
            if not args.transcript_dir.exists():
                print(
                    f"❌ Error: Transcript directory not found: {args.transcript_dir}"
                )
                sys.exit(1)
            # Transcripts will be loaded per-file
            print(f"📝 Will load transcripts from {args.transcript_dir}")
        else:
            print("❌ Error: wav2vec2 requires --script-json or --transcript-dir")
            print("   Provide the text that was spoken in each audio file.")
            sys.exit(1)
    else:
        # Check for Whisper API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment")
            print("   Options:")
            print(
                "   1. Set OPENAI_API_KEY in .env file for Whisper (costs ~$0.006/min)"
            )
            print("   2. Use --aligner wav2vec2 with --script-json for FREE alignment")
            sys.exit(1)

    aligner_name = (
        "Whisper (OpenAI API)" if args.aligner == "whisper" else "Wav2Vec2 (FREE local)"
    )

    print(f"\n🎤 Generate Subtitles - {aligner_name}")
    print(f"=" * 60)
    print(f"📁 Audio directory: {args.audio_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎵 Audio files found: {len(audio_files)}")
    print(f"🔧 Aligner: {args.aligner}")
    print(f"📝 Words per line: {args.words_per_line}")
    print(f"🔤 Font: {args.font_name} @ {args.font_size}px")
    if args.aligner == "wav2vec2":
        print(f"💰 Cost: $0.00 (runs locally)")
    else:
        print(f"💰 Cost: ~$0.006/minute of audio")
    print(f"=" * 60)

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_json:
        json_dir = args.output_dir / "timestamps"
        json_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the appropriate aligner
    # Handle imports - works with pip install -e . or direct script execution
    try:
        if args.aligner == "wav2vec2":
            from src.pipeline.wav2vec2_aligner import Wav2Vec2Aligner

            aligner = Wav2Vec2Aligner()
        else:
            from src.pipeline.audio_aligner import AudioAligner

            aligner = AudioAligner()
    except ImportError:
        # Fallback for running without package installation
        sys.path.insert(0, str(Path(__file__).parent))
        if args.aligner == "wav2vec2":
            from src.pipeline.wav2vec2_aligner import Wav2Vec2Aligner

            aligner = Wav2Vec2Aligner()
        else:
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
            # Get transcript for wav2vec2
            transcript = None
            if args.aligner == "wav2vec2":
                audio_stem = audio_file.stem

                # Try script JSON first
                if transcripts and audio_stem in transcripts:
                    transcript = transcripts[audio_stem]
                # Then try transcript directory
                elif args.transcript_dir:
                    transcript = load_transcript_from_file(
                        args.transcript_dir, audio_stem
                    )

                if not transcript:
                    print(f"  ⚠️ No transcript found for {audio_stem}, skipping...")
                    failed += 1
                    continue

                print(f"  📝 Transcript: {transcript[:60]}...")

            # Get word timestamps
            if args.aligner == "wav2vec2":
                result = aligner.align(audio_file, transcript)
            else:
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
            import traceback

            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {successful}/{len(audio_files)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(audio_files)}")
    print(f"⏱️  Total duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
    print(f"📝 Total words aligned: {total_words}")

    if args.aligner == "whisper":
        print(f"💰 Estimated Whisper cost: ${total_duration / 60 * 0.006:.4f}")
    else:
        print(f"💰 Total cost: $0.00 (wav2vec2 is free!)")

    print(f"\n📁 Output saved to: {args.output_dir}")

    if args.save_json:
        print(f"📁 Timestamps saved to: {args.output_dir / 'timestamps'}")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
