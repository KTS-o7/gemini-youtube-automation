"""
Command-line interface for video generation.

This module provides proper entry points for the video generation
pipeline without relying on sys.path hacks.

Usage:
    # After installing the package:
    generate-video --topic "How does AI work?" --audience "beginners"

    # Or run directly:
    python -m src.cli --topic "How does AI work?" --audience "beginners"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def main(args: Optional[list[str]] = None) -> int:
    """
    Main CLI entry point for video generation.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="AI Video Generator - Create videos from topics using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Long-form educational video
  generate-video --topic "How do neural networks learn?" --audience "developers" --format long

  # Short-form quick tip
  generate-video --topic "What is Docker?" --audience "beginners" --format short

  # With specific TTS settings
  generate-video --topic "Python tips" --audience "programmers" --tts openai --voice marin

Environment Variables:
  OPENAI_API_KEY      - Required for OpenAI features
  GOOGLE_API_KEY      - Required for Gemini features
  TTS_PROVIDER        - Default TTS provider (gtts, openai, elevenlabs)
  TTS_VOICE           - Default voice for OpenAI TTS
  SUBTITLE_ALIGNER    - Subtitle alignment method (wav2vec2, whisper)
""",
    )

    parser.add_argument(
        "--topic",
        "-t",
        required=True,
        help="Topic or question for the video",
    )

    parser.add_argument(
        "--audience",
        "-a",
        required=True,
        help="Target audience description",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["short", "long"],
        default="long",
        help="Video format: short (< 60s, vertical) or long (3-10 min, horizontal)",
    )

    parser.add_argument(
        "--style",
        "-s",
        default="educational",
        help="Video style (e.g., educational, casual, professional)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for generated files",
    )

    parser.add_argument(
        "--tts",
        choices=["gtts", "openai", "elevenlabs"],
        default=None,
        help="Text-to-speech provider (overrides TTS_PROVIDER env var)",
    )

    parser.add_argument(
        "--voice",
        default=None,
        help="Voice for TTS (e.g., 'marin' for OpenAI)",
    )

    parser.add_argument(
        "--no-viral",
        action="store_true",
        help="Disable viral mode features (karaoke captions, etc.)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after generation",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (defaults to auto-discovery)",
    )

    parsed = parser.parse_args(args)

    # Load environment variables
    if parsed.env_file:
        load_dotenv(parsed.env_file)
    else:
        load_dotenv()

    # Import after dotenv is loaded
    from .config import AppConfig
    from .pipeline import TTSProvider, VideoFormat, VideoPipeline, VideoRequest
    from .pipeline.voice_generator import VoiceConfig

    try:
        # Build configuration
        config = AppConfig.from_environment()

        # Override TTS settings if provided via CLI
        tts_provider = (
            TTSProvider(parsed.tts) if parsed.tts else TTSProvider(config.tts.provider)
        )

        voice_config = VoiceConfig(
            provider=tts_provider,
            openai_voice=parsed.voice or config.tts.openai_voice,
            openai_speed=config.tts.openai_speed,
            openai_instructions=config.tts.openai_instructions,
            elevenlabs_api_key=config.tts.elevenlabs_api_key,
            elevenlabs_voice_id=config.tts.elevenlabs_voice_id,
            fallback_enabled=config.fallback_tts_enabled,
        )

        # Create request
        request = VideoRequest(
            topic=parsed.topic,
            target_audience=parsed.audience,
            format=VideoFormat(parsed.format),
            style=parsed.style,
        )

        # Create pipeline
        pipeline = VideoPipeline(
            output_dir=parsed.output,
            voice_config=voice_config,
            subtitle_aligner=config.subtitle.aligner,
        )

        # Generate video
        output = pipeline.generate_video_sync(request)

        if output.success:
            print(f"\n✅ Video generated successfully!")
            print(f"   Video: {output.video_path}")
            if output.thumbnail_path:
                print(f"   Thumbnail: {output.thumbnail_path}")
            if output.metadata:
                print(f"   Title: {output.metadata.title}")
                print(f"   Duration: {output.metadata.duration_seconds:.1f}s")

            if parsed.cleanup:
                pipeline.cleanup()
                print("\n🧹 Temporary files cleaned up")

            return 0
        else:
            print(f"\n❌ Video generation failed: {output.error_message}")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Generation cancelled by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def run_subtitles() -> int:
    """
    CLI entry point for subtitle regeneration.

    This allows regenerating subtitles for existing videos.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate subtitles for an existing video",
    )

    parser.add_argument(
        "--video",
        "-v",
        type=Path,
        required=True,
        help="Path to the video file",
    )

    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        help="Path to audio file (if separate from video)",
    )

    parser.add_argument(
        "--transcript",
        "-t",
        type=Path,
        help="Path to transcript file (required for wav2vec2)",
    )

    parser.add_argument(
        "--aligner",
        choices=["wav2vec2", "whisper"],
        default="wav2vec2",
        help="Alignment method to use",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for video with subtitles",
    )

    parsed = parser.parse_args()

    load_dotenv()

    print("⚠️ Subtitle regeneration CLI not fully implemented yet")
    print(f"   Video: {parsed.video}")
    print(f"   Aligner: {parsed.aligner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
