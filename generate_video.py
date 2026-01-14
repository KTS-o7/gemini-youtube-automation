#!/usr/bin/env python3
"""
Video Generation CLI - Command-line interface for AI video generation.

Usage:
    # Install the package first (recommended):
    pip install -e .

    # Then use the CLI:
    generate-video --topic "How does machine learning work?" --audience "beginners" --format long

    # Or run this script directly:
    python generate_video.py \
        --topic "How does machine learning work?" \
        --audience "beginners new to AI" \
        --format long

    # Generate a short-form video
    python generate_video.py \
        --topic "What is quantum computing?" \
        --audience "general audience" \
        --format short \
        --style "casual with humor"

    # Using environment variables for API keys
    export OPENAI_API_KEY=sk-...
    export AI_PROVIDER=openai  # or "gemini"
    export TTS_PROVIDER=gtts   # or "openai" or "elevenlabs"

Note: For best results, install the package with `pip install -e .` to avoid import issues.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Video Generator - Create videos from topics using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Long-form educational video
  python generate_video.py --topic "How do neural networks learn?" --audience "developers" --format long

  # Short-form quick tip
  python generate_video.py --topic "What is Docker?" --audience "beginners" --format short

  # With custom style
  python generate_video.py --topic "Python best practices" --audience "junior developers" --format long --style "tutorial with code examples"

Environment Variables:
  AI_PROVIDER       - AI provider: "openai" or "gemini" (default: openai)
  OPENAI_API_KEY    - OpenAI API key (required for openai provider)
  GOOGLE_API_KEY    - Google API key (required for gemini provider)
  PEXELS_API_KEY    - Pexels API key for fallback images (optional)
        """,
    )

    parser.add_argument(
        "--topic",
        "-t",
        required=True,
        help="The main topic or question for the video",
    )
    parser.add_argument(
        "--audience",
        "-a",
        required=True,
        help='Target audience (e.g., "beginners", "developers", "general audience")',
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["short", "long"],
        required=True,
        help="Video format: short (<60s, vertical) or long (3-10min, horizontal)",
    )
    parser.add_argument(
        "--style",
        "-s",
        default="educational",
        help='Video style (default: "educational")',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for generated files (default: output/)",
    )
    parser.add_argument(
        "--tts",
        choices=["gtts", "openai", "elevenlabs"],
        default="openai",
        help="TTS provider: gtts (free), openai ($0.015/min), elevenlabs (default: openai)",
    )
    parser.add_argument(
        "--voice",
        default="marin",
        help="Voice for OpenAI TTS: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar (default: marin)",
    )
    parser.add_argument(
        "--voice-instructions",
        default=None,
        help='Instructions for voice style, e.g. "Speak in a calm, professional tone" (OpenAI TTS only)',
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after generation",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to YouTube after generation (requires credentials)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate API keys
    ai_provider = os.environ.get("AI_PROVIDER", "openai").lower()
    if ai_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    elif ai_provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable is not set")
        print("   Set it with: export GOOGLE_API_KEY=...")
        sys.exit(1)

    # Override TTS settings if specified
    if args.tts:
        os.environ["TTS_PROVIDER"] = args.tts
    if args.voice:
        os.environ["TTS_VOICE"] = args.voice
    if args.voice_instructions:
        os.environ["TTS_INSTRUCTIONS"] = args.voice_instructions

    # Import pipeline components
    # These imports work when the package is installed with `pip install -e .`
    # or when running from the project root directory
    try:
        from src.pipeline import TTSProvider, VideoFormat, VideoPipeline, VideoRequest
        from src.pipeline.voice_generator import VoiceConfig
    except ImportError:
        # Fallback for running script directly without installation
        sys.path.insert(0, str(Path(__file__).parent))
        from src.pipeline import TTSProvider, VideoFormat, VideoPipeline, VideoRequest
        from src.pipeline.voice_generator import VoiceConfig

    # Determine TTS provider
    tts_env = os.environ.get("TTS_PROVIDER", "gtts").lower()
    tts_provider = (
        TTSProvider(tts_env)
        if tts_env in ["gtts", "openai", "elevenlabs"]
        else TTSProvider.GTTS
    )

    # Build voice configuration explicitly (no env var reads in constructor)
    voice_config = VoiceConfig(
        provider=tts_provider,
        openai_voice=args.voice,
        openai_speed=float(os.environ.get("TTS_SPEED", "1.0")),
        openai_instructions=args.voice_instructions,
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get(
            "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
        ),
        fallback_enabled=True,  # Enable gTTS fallback
    )

    # Create video request
    request = VideoRequest(
        topic=args.topic,
        target_audience=args.audience,
        format=VideoFormat(args.format),
        style=args.style,
    )

    # Initialize pipeline with explicit configuration
    pipeline = VideoPipeline(
        output_dir=args.output_dir,
        voice_config=voice_config,
        subtitle_aligner=os.environ.get("SUBTITLE_ALIGNER", "wav2vec2"),
    )

    try:
        # Generate video
        print("\n🚀 Starting video generation pipeline...\n")
        output = pipeline.generate_video_sync(request)

        if output.success:
            print("\n" + "=" * 60)
            print("✅ VIDEO GENERATION SUCCESSFUL!")
            print("=" * 60)
            print(f"\n📹 Video:     {output.video_path}")
            print(f"🖼️  Thumbnail: {output.thumbnail_path}")

            if output.metadata:
                print(f"\n📋 Title:     {output.metadata.title}")
                print(f"⏱️  Duration:  {output.metadata.duration_seconds:.1f}s")
                print(f"🏷️  Tags:      {', '.join(output.metadata.tags[:5])}")

            # Upload to YouTube if requested
            if args.upload:
                print("\n📤 Uploading to YouTube...")
                try:
                    from src.uploader import upload_to_youtube

                    video_id = upload_to_youtube(
                        video_path=output.video_path,
                        title=output.metadata.title,
                        description=output.metadata.youtube_description(),
                        tags=output.metadata.youtube_tags(),
                        thumbnail_path=output.thumbnail_path,
                    )
                    if video_id:
                        print(f"✅ Uploaded! Video ID: {video_id}")
                        print(f"🔗 URL: https://www.youtube.com/watch?v={video_id}")
                except Exception as e:
                    print(f"❌ Upload failed: {e}")

            # Cleanup if requested
            if args.cleanup:
                pipeline.cleanup()

        else:
            print("\n" + "=" * 60)
            print("❌ VIDEO GENERATION FAILED")
            print("=" * 60)
            print(f"\nError: {output.error_message}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Generation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
