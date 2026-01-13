"""
Pipeline Orchestrator - Main entry point for video generation.

This module coordinates all stages of the video generation pipeline,
from input processing to final video output.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .image_generator import ImageGenerator
from .models import (
    PipelineState,
    PlannedScene,
    TTSProvider,
    VideoFormat,
    VideoMetadata,
    VideoOutput,
    VideoRequest,
)
from .researcher import ContentResearcher
from .scene_planner import ScenePlanner
from .script_writer import ScriptWriter
from .video_composer import VideoComposer
from .voice_generator import VoiceGenerator


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
    """

    def __init__(
        self,
        ai_client: Optional[AIClient] = None,
        output_dir: Optional[Path] = None,
        tts_provider: TTSProvider = TTSProvider.OPENAI,
    ):
        """
        Initialize the video pipeline.

        Args:
            ai_client: AI client for text/image generation
            output_dir: Base directory for all outputs
            tts_provider: Text-to-speech provider to use
        """
        self.ai_client = ai_client or get_ai_client()
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline components
        self.researcher = ContentResearcher(ai_client=self.ai_client)
        self.script_writer = ScriptWriter(ai_client=self.ai_client)
        self.scene_planner = ScenePlanner(ai_client=self.ai_client)
        self.image_generator = ImageGenerator(
            ai_client=self.ai_client,
            output_dir=self.output_dir / "images",
        )
        self.voice_generator = VoiceGenerator(
            provider=tts_provider,
            ai_client=self.ai_client,
            output_dir=self.output_dir / "audio",
        )
        self.video_composer = VideoComposer(output_dir=self.output_dir)

    async def generate_video(self, request: VideoRequest) -> VideoOutput:
        """
        Generate a complete video from a request.

        This is the main entry point that orchestrates all pipeline stages.

        Args:
            request: Video generation request

        Returns:
            VideoOutput with paths to generated files and metadata
        """
        print(f"\n{'=' * 60}")
        print(f"🎬 Starting video generation")
        print(f"📌 Topic: {request.topic}")
        print(f"👥 Audience: {request.target_audience}")
        print(f"📐 Format: {request.format.value}")
        print(f"{'=' * 60}\n")

        # Initialize pipeline state
        state = PipelineState(request=request)

        try:
            # Stage 1: Validate input
            state.current_stage = "validation"
            request.validate()
            print("✅ Stage 1: Input validated\n")

            # Stage 2: Research
            state.current_stage = "research"
            print("🔍 Stage 2: Researching topic...")
            state.research = self.researcher.research_sync(request)
            self._save_research(state.research)
            print(
                f"✅ Research complete: {len(state.research.key_points)} key points\n"
            )

            # Stage 3: Generate script
            state.current_stage = "script_generation"
            print("📝 Stage 3: Generating script...")
            state.script = self.script_writer.generate_script(request, state.research)
            self._save_script(state.script)
            print(f"✅ Script generated: {state.script.scene_count()} scenes\n")

            # Stage 4: Plan scenes
            state.current_stage = "scene_planning"
            print("🎬 Stage 4: Planning scenes...")
            state.planned_scenes = self.scene_planner.plan_scenes(state.script, request)
            self._save_planned_scenes(state.planned_scenes)
            print(f"✅ Scenes planned: {len(state.planned_scenes)} scenes\n")

            # Stage 5: Generate assets
            state.current_stage = "asset_generation"
            print("🎨 Stage 5: Generating assets...")

            # Generate voice first to get accurate timing
            print("  🎤 Generating voice narration...")
            self.voice_generator.batch_generate(state.planned_scenes)

            # Generate images
            print("  🖼️ Generating scene images...")
            self.image_generator.batch_generate(state.planned_scenes, request)

            print("✅ All assets generated\n")

            # Stage 6: Compose video
            state.current_stage = "video_composition"
            print("🎥 Stage 6: Composing video...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"video_{request.format.value}_{timestamp}.mp4"
            video_path = self.video_composer.compose(
                state.planned_scenes, request, output_filename
            )
            print(f"✅ Video composed: {video_path}\n")

            # Stage 7: Generate thumbnail and metadata
            state.current_stage = "finalization"
            print("🖼️ Stage 7: Generating thumbnail and metadata...")
            thumbnail_path = self.image_generator.generate_thumbnail(
                title=state.script.title,
                request=request,
                prompt=state.script.thumbnail_prompt,
            )

            metadata = self._generate_metadata(state, request)
            print("✅ Thumbnail and metadata generated\n")

            # Create final output
            state.output = VideoOutput(
                video_path=video_path,
                thumbnail_path=thumbnail_path,
                metadata=metadata,
                success=True,
            )

            state.current_stage = "complete"

            print(f"\n{'=' * 60}")
            print("🎉 VIDEO GENERATION COMPLETE!")
            print(f"📹 Video: {video_path}")
            print(f"🖼️ Thumbnail: {thumbnail_path}")
            print(f"📋 Title: {metadata.title}")
            print(f"⏱️ Duration: {metadata.duration_seconds:.1f}s")
            print(f"{'=' * 60}\n")

            return state.output

        except Exception as e:
            error_msg = f"Pipeline failed at stage '{state.current_stage}': {str(e)}"
            print(f"\n❌ {error_msg}")
            state.add_error(error_msg)

            return VideoOutput(
                video_path=Path(""),
                success=False,
                error_message=error_msg,
            )

    def generate_video_sync(self, request: VideoRequest) -> VideoOutput:
        """
        Synchronous wrapper for generate_video.

        Args:
            request: Video generation request

        Returns:
            VideoOutput with paths to generated files
        """
        return asyncio.run(self.generate_video(request))

    def _generate_metadata(
        self, state: PipelineState, request: VideoRequest
    ) -> VideoMetadata:
        """Generate video metadata from pipeline state."""
        script = state.script
        research = state.research

        # Calculate total duration
        total_duration = sum(s.duration_seconds for s in state.planned_scenes)

        # Build description
        description_parts = [script.description or f"Learn about {request.topic}."]

        if research and research.key_points:
            description_parts.append("\n\nIn this video, we cover:")
            for point in research.key_points[:5]:
                description_parts.append(f"• {point}")

        description = "\n".join(description_parts)

        # Build tags
        tags = list(script.hashtags) if script.hashtags else []
        tags.extend([request.topic.split()[0], "education", "learning"])
        tags = list(set(tags))  # Remove duplicates

        return VideoMetadata(
            title=script.title,
            description=description,
            tags=tags,
            hashtags=script.hashtags,
            duration_seconds=total_duration,
            format=request.format.value,
            sources=research.sources if research else [],
        )

    def _save_research(self, research) -> None:
        """Save research results to JSON file."""
        research_path = self.output_dir / "research.json"
        research_data = {
            "topic": research.topic,
            "key_points": research.key_points,
            "facts": research.facts,
            "examples": research.examples,
            "analogies": research.analogies,
            "sources": research.sources,
            "related_topics": research.related_topics,
        }
        with open(research_path, "w") as f:
            json.dump(research_data, f, indent=2)
        print(f"  💾 Research saved: {research_path}")

    def _save_script(self, script) -> None:
        """Save script to JSON file."""
        script_path = self.output_dir / "script.json"
        script_data = {
            "title": script.title,
            "hook": script.hook,
            "description": script.description,
            "hashtags": script.hashtags,
            "thumbnail_prompt": script.thumbnail_prompt,
            "total_duration_seconds": script.total_duration_seconds,
            "scenes": [
                {
                    "scene_number": s.scene_number,
                    "narration": s.narration,
                    "visual_description": s.visual_description,
                    "duration_seconds": s.duration_seconds,
                    "mood": s.mood,
                    "key_visual_elements": s.key_visual_elements,
                }
                for s in script.scenes
            ],
        }
        with open(script_path, "w") as f:
            json.dump(script_data, f, indent=2)
        print(f"  💾 Script saved: {script_path}")

    def _save_planned_scenes(self, planned_scenes: list[PlannedScene]) -> None:
        """Save planned scenes to JSON file."""
        scenes_path = self.output_dir / "planned_scenes.json"
        scenes_data = [
            {
                "scene_number": s.scene_number,
                "narration": s.narration,
                "visual_description": s.visual_description,
                "image_prompt": s.image_prompt,
                "duration_seconds": s.duration_seconds,
                "mood": s.mood,
                "transition": s.transition,
            }
            for s in planned_scenes
        ]
        with open(scenes_path, "w") as f:
            json.dump(scenes_data, f, indent=2)
        print(f"  💾 Planned scenes saved: {scenes_path}")

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
) -> VideoOutput:
    """
    Convenience function to create a video with minimal setup.

    Args:
        topic: The video topic
        target_audience: Who the video is for
        format: "short" or "long"
        style: Video style (e.g., "educational", "casual")

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
    return pipeline.generate_video_sync(request)
