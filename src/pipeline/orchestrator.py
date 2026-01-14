"""
Pipeline Orchestrator - Main entry point for video generation.

This module coordinates all stages of the video generation pipeline,
from input processing to final video output.

REFACTORED: Now uses ComposablePipeline for skip/resume functionality.
Maintains backward compatibility with existing API.

Prefer explicit configuration via dependency injection:

    from src.config import AppConfig
    from src.pipeline import VideoPipeline

    config = AppConfig.from_environment()
    pipeline = VideoPipeline.from_config(config)
    output = pipeline.generate_video_sync(request)

New composable API:

    from src.pipeline import VideoPipeline, PipelineConfig

    pipeline = VideoPipeline.from_config(config)

    # Skip specific stages
    output = pipeline.run_composable(request, skip_stages=["research"])

    # Resume from checkpoint
    output = pipeline.resume_from_checkpoint(checkpoint_path)

    # Run only specific stages
    output = pipeline.run_composable(request, only_stages=["validation", "research"])
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..utils.ai_client import AIClient, create_ai_client
from .image_generator import ImageGenerator
from .models import (
    PipelineState,
    PlannedScene,
    TTSProvider,
    VideoFormat,
    VideoOutput,
    VideoRequest,
)
from .pipeline import ComposablePipeline, PipelineConfig, PipelineStage
from .researcher import ContentResearcher
from .scene_planner import ScenePlanner
from .script_writer import ScriptWriter
from .stages import PipelineContext, create_default_stages
from .video_composer import VideoComposer
from .voice_generator import VoiceConfig, VoiceGenerator

if TYPE_CHECKING:
    from ..config import AppConfig


class VideoPipeline:
    """
    Main orchestrator for the video generation pipeline.

    Coordinates all stages:
    1. Input validation
    2. Web research
    3. Script generation
    4. Scene planning
    5. Asset generation (images + voice)
    6. Video composition
    7. Output & metadata

    Supports two execution modes:
    - Legacy mode: `generate_video()` / `generate_video_sync()`
    - Composable mode: `run_composable()` with skip/resume support
    """

    def __init__(
        self,
        ai_client: Optional[AIClient] = None,
        output_dir: Optional[Path] = None,
        tts_provider: TTSProvider = TTSProvider.OPENAI,
        voice_config: Optional[VoiceConfig] = None,
        subtitle_aligner: Optional[str] = None,
    ):
        """
        Initialize the video pipeline.

        Args:
            ai_client: AI client for text/image generation. If not provided,
                      a new client will be created.
            output_dir: Base directory for all outputs
            tts_provider: Text-to-speech provider to use (ignored if voice_config provided)
            voice_config: Explicit voice generation configuration. If provided,
                         tts_provider is ignored.
            subtitle_aligner: Subtitle alignment method ("wav2vec2" or "whisper")
        """
        self.ai_client = ai_client or create_ai_client()
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build voice config if not provided
        if voice_config is None:
            voice_config = VoiceConfig(provider=tts_provider)

        # Initialize pipeline components
        self.researcher = ContentResearcher(ai_client=self.ai_client)
        self.script_writer = ScriptWriter(ai_client=self.ai_client)
        self.scene_planner = ScenePlanner(ai_client=self.ai_client)
        self.image_generator = ImageGenerator(
            ai_client=self.ai_client,
            output_dir=self.output_dir / "images",
        )
        self.voice_generator = VoiceGenerator(
            config=voice_config,
            ai_client=self.ai_client,
            output_dir=self.output_dir / "audio",
        )
        self.video_composer = VideoComposer(
            output_dir=self.output_dir,
            subtitle_aligner=subtitle_aligner,
        )

        # Create pipeline context for composable mode
        self._context = PipelineContext(
            researcher=self.researcher,
            script_writer=self.script_writer,
            scene_planner=self.scene_planner,
            image_generator=self.image_generator,
            voice_generator=self.voice_generator,
            video_composer=self.video_composer,
            output_dir=self.output_dir,
        )

        # Initialize composable pipeline
        self._composable_pipeline: Optional[ComposablePipeline] = None

    @classmethod
    def from_config(cls, config: "AppConfig") -> "VideoPipeline":
        """
        Create a VideoPipeline from an AppConfig.

        This is the preferred way to create a pipeline with explicit
        configuration from environment variables.

        Args:
            config: Application configuration (from AppConfig.from_environment())

        Returns:
            Configured VideoPipeline instance
        """
        # Create AI client from config
        from ..utils.ai_client import AIConfig

        ai_config = AIConfig(
            provider=config.ai.provider,
            openai_model=config.ai.openai_model,
            image_model=config.ai.openai_image_model,
            image_quality=config.ai.openai_image_quality,
        )
        ai_client = create_ai_client(ai_config)

        # Create voice config from AppConfig
        voice_config = VoiceConfig(
            provider=TTSProvider(config.tts.provider),
            openai_voice=config.tts.openai_voice,
            openai_speed=config.tts.openai_speed,
            openai_instructions=config.tts.openai_instructions,
            elevenlabs_api_key=config.tts.elevenlabs_api_key,
            elevenlabs_voice_id=config.tts.elevenlabs_voice_id,
            fallback_enabled=config.fallback_tts_enabled,
        )

        return cls(
            ai_client=ai_client,
            output_dir=config.paths.output_dir,
            voice_config=voice_config,
            subtitle_aligner=config.subtitle.aligner,
        )

    def _get_composable_pipeline(self) -> ComposablePipeline:
        """Get or create the composable pipeline."""
        if self._composable_pipeline is None:
            self._composable_pipeline = ComposablePipeline(context=self._context)
            for stage in create_default_stages():
                self._composable_pipeline.add_stage(stage)
        return self._composable_pipeline

    # =========================================================================
    # COMPOSABLE PIPELINE API (NEW)
    # =========================================================================

    def run_composable(
        self,
        request: VideoRequest,
        skip_stages: Optional[list[str]] = None,
        only_stages: Optional[list[str]] = None,
        stop_after: Optional[str] = None,
        save_checkpoints: bool = True,
    ) -> VideoOutput:
        """
        Run the pipeline with composable options.

        This is the new recommended API that supports skip/resume functionality.

        Args:
            request: Video generation request
            skip_stages: List of stage names to skip (e.g., ["research"])
            only_stages: If set, only run these stages (and their dependencies)
            stop_after: Stop pipeline after this stage completes
            save_checkpoints: Whether to save checkpoints after each stage

        Returns:
            VideoOutput with paths to generated files

        Example:
            # Skip research stage (use cached data)
            output = pipeline.run_composable(request, skip_stages=["research"])

            # Only run validation and research
            output = pipeline.run_composable(
                request,
                only_stages=["validation", "research"],
                stop_after="research"
            )
        """
        config = PipelineConfig(
            skip_stages=skip_stages or [],
            only_stages=only_stages,
            stop_after=stop_after,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=self.output_dir / "checkpoints",
        )

        pipeline = self._get_composable_pipeline()

        print(f"\n{'=' * 60}")
        print(f"🎬 Starting video generation (Composable Mode)")
        print(f"📌 Topic: {request.topic}")
        print(f"👥 Audience: {request.target_audience}")
        print(f"📐 Format: {request.format.value}")
        print(f"{'=' * 60}")

        state = pipeline.run(request, config=config)

        if state.output and state.output.success:
            print(f"\n{'=' * 60}")
            print("🎉 VIDEO GENERATION COMPLETE!")
            print(f"📹 Video: {state.output.video_path}")
            if state.output.thumbnail_path:
                print(f"🖼️ Thumbnail: {state.output.thumbnail_path}")
            if state.output.metadata:
                print(f"📋 Title: {state.output.metadata.title}")
            print(f"{'=' * 60}\n")

        return state.output or VideoOutput(
            video_path=Path(""),
            success=False,
            error_message="Pipeline did not produce output",
        )

    def resume_from_checkpoint(
        self,
        checkpoint_path: Path,
        skip_stages: Optional[list[str]] = None,
    ) -> VideoOutput:
        """
        Resume pipeline from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file (.pkl)
            skip_stages: Additional stages to skip

        Returns:
            VideoOutput with paths to generated files

        Example:
            # Resume from script_generation checkpoint
            output = pipeline.resume_from_checkpoint(
                Path("output/checkpoints/checkpoint_script_generation.pkl")
            )
        """
        config = PipelineConfig(
            skip_stages=skip_stages or [],
            save_checkpoints=True,
            checkpoint_dir=self.output_dir / "checkpoints",
        )

        pipeline = self._get_composable_pipeline()
        state = pipeline.resume(checkpoint_path, config=config)

        return state.output or VideoOutput(
            video_path=Path(""),
            success=False,
            error_message="Pipeline did not produce output after resume",
        )

    def list_stages(self) -> list[str]:
        """
        Get the list of available pipeline stages.

        Returns:
            List of stage names in execution order
        """
        return self._get_composable_pipeline().list_stages()

    def get_stage_info(self, stage_name: str) -> Optional[dict]:
        """
        Get information about a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Dictionary with stage info or None if not found
        """
        stage = self._get_composable_pipeline().get_stage(stage_name)
        if stage:
            return {
                "name": stage.name,
                "description": stage.description,
                "depends_on": stage.depends_on,
                "optional": stage.optional,
            }
        return None

    # =========================================================================
    # LEGACY API (BACKWARD COMPATIBLE)
    # =========================================================================

    async def generate_video(self, request: VideoRequest) -> VideoOutput:
        """
        Generate a complete video from a request.

        This is the legacy API maintained for backward compatibility.
        For new code, prefer `run_composable()` which supports skip/resume.

        Args:
            request: Video generation request

        Returns:
            VideoOutput with paths to generated files and metadata
        """
        # Use the composable pipeline internally
        return self.run_composable(request, save_checkpoints=True)

    def generate_video_sync(self, request: VideoRequest) -> VideoOutput:
        """
        Synchronous wrapper for generate_video.

        Args:
            request: Video generation request

        Returns:
            VideoOutput with paths to generated files
        """
        return asyncio.run(self.generate_video(request))

    def cleanup(self) -> None:
        """Clean up temporary files."""
        print("🧹 Cleaning up temporary files...")
        self.image_generator.cleanup()
        self.voice_generator.cleanup()
        print("✅ Cleanup complete")


def create_video(
    topic: str,
    target_audience: str,
    format: str = "long",
    style: str = "educational",
    skip_stages: Optional[list[str]] = None,
) -> VideoOutput:
    """
    Convenience function to create a video with minimal setup.

    Args:
        topic: The video topic
        target_audience: Who the video is for
        format: "short" or "long"
        style: Video style (e.g., "educational", "casual")
        skip_stages: Optional list of stages to skip

    Returns:
        VideoOutput with paths to generated files
    """
    request = VideoRequest(
        topic=topic,
        target_audience=target_audience,
        format=VideoFormat(format),
        style=style,
    )

    pipeline = VideoPipeline()

    if skip_stages:
        return pipeline.run_composable(request, skip_stages=skip_stages)
    else:
        return pipeline.generate_video_sync(request)
