"""
Pipeline Stages - Individual stage functions for composable pipeline.

This module defines standalone functions for each pipeline stage.
Each function takes a PipelineState and optional context, and returns
an updated PipelineState.

Stages:
1. validation - Validate input request
2. research - Research the topic
3. script_generation - Generate video script
4. scene_planning - Plan scenes with assets
5. asset_generation - Generate images and voice (parallel)
6. video_composition - Compose final video
7. finalization - Generate thumbnail and metadata

Usage:
    from src.pipeline.stages import STAGE_DEFINITIONS, create_default_stages
    from src.pipeline.pipeline import ComposablePipeline

    pipeline = ComposablePipeline(context=my_context)
    for stage in create_default_stages():
        pipeline.add_stage(stage)
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .models import (
    PipelineState,
    PlannedScene,
    VideoMetadata,
    VideoOutput,
)
from .pipeline import PipelineStage

if TYPE_CHECKING:
    from .image_generator import ImageGenerator
    from .researcher import ContentResearcher
    from .scene_planner import ScenePlanner
    from .script_writer import ScriptWriter
    from .video_composer import VideoComposer
    from .voice_generator import VoiceGenerator


@dataclass
class PipelineContext:
    """
    Shared context for all pipeline stages.

    Contains all the components needed by stages to execute.
    """

    researcher: Optional["ContentResearcher"] = None
    script_writer: Optional["ScriptWriter"] = None
    scene_planner: Optional["ScenePlanner"] = None
    image_generator: Optional["ImageGenerator"] = None
    voice_generator: Optional["VoiceGenerator"] = None
    video_composer: Optional["VideoComposer"] = None
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STAGE FUNCTIONS
# =============================================================================


def stage_validation(state: PipelineState, context: Any) -> PipelineState:
    """
    Stage 1: Validate the input request.

    Validates that the request has all required fields and valid values.
    """
    print("   🔍 Validating input request...")
    state.request.validate()
    print(f"      Topic: {state.request.topic}")
    print(f"      Audience: {state.request.target_audience}")
    print(f"      Format: {state.request.format.value}")
    return state


def stage_research(state: PipelineState, context: PipelineContext) -> PipelineState:
    """
    Stage 2: Research the topic.

    Gathers information about the topic from various sources.
    """
    print("   🔍 Researching topic...")

    if context.researcher is None:
        raise ValueError("Researcher not configured in context")

    state.research = context.researcher.research_sync(state.request)

    # Save research results
    _save_research(state.research, context.output_dir)

    print(f"      Found {len(state.research.key_points)} key points")
    print(f"      Found {len(state.research.facts)} facts")
    return state


def stage_script_generation(
    state: PipelineState, context: PipelineContext
) -> PipelineState:
    """
    Stage 3: Generate the video script.

    Creates a complete script with scenes, narration, and visual descriptions.
    """
    print("   📝 Generating script...")

    if context.script_writer is None:
        raise ValueError("ScriptWriter not configured in context")

    if state.research is None:
        raise ValueError("Research must be completed before script generation")

    state.script = context.script_writer.generate_script(state.request, state.research)

    # Save script
    _save_script(state.script, context.output_dir)

    print(f"      Title: {state.script.title}")
    print(f"      Scenes: {state.script.scene_count()}")
    return state


def stage_scene_planning(
    state: PipelineState, context: PipelineContext
) -> PipelineState:
    """
    Stage 4: Plan scenes with detailed asset specifications.

    Creates detailed plans for each scene including image prompts and transitions.
    """
    print("   🎬 Planning scenes...")

    if context.scene_planner is None:
        raise ValueError("ScenePlanner not configured in context")

    if state.script is None:
        raise ValueError("Script must be generated before scene planning")

    state.planned_scenes = context.scene_planner.plan_scenes(
        state.script, state.request
    )

    # Save planned scenes
    _save_planned_scenes(state.planned_scenes, context.output_dir)

    print(f"      Planned {len(state.planned_scenes)} scenes")
    return state


def stage_asset_generation(
    state: PipelineState, context: PipelineContext
) -> PipelineState:
    """
    Stage 5: Generate assets (images and voice) in parallel.

    Generates all scene images and voice narrations concurrently for speed.
    """
    print("   🎨 Generating assets in parallel...")

    if context.image_generator is None:
        raise ValueError("ImageGenerator not configured in context")
    if context.voice_generator is None:
        raise ValueError("VoiceGenerator not configured in context")

    if not state.planned_scenes:
        raise ValueError("Scenes must be planned before asset generation")

    def generate_voices():
        print("      🎤 Generating voice narration...")
        context.voice_generator.batch_generate(state.planned_scenes)
        return "voice"

    def generate_images():
        print("      🖼️ Generating scene images...")
        context.image_generator.batch_generate(state.planned_scenes, state.request)
        return "images"

    # Run voice and image generation in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(generate_voices),
            executor.submit(generate_images),
        ]
        for future in as_completed(futures):
            result = future.result()
            print(f"      ✅ {result.capitalize()} generation complete")

    return state


def stage_video_composition(
    state: PipelineState, context: PipelineContext
) -> PipelineState:
    """
    Stage 6: Compose the final video.

    Combines all assets into the final video with subtitles and effects.
    """
    print("   🎥 Composing video...")

    if context.video_composer is None:
        raise ValueError("VideoComposer not configured in context")

    if not state.planned_scenes:
        raise ValueError("Scenes must be planned before video composition")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"video_{state.request.format.value}_{timestamp}.mp4"

    video_path = context.video_composer.compose(
        state.planned_scenes, state.request, output_filename
    )

    # Store video path in output (will be completed in finalization)
    state.output = VideoOutput(video_path=video_path, success=True)

    print(f"      Video saved: {video_path}")
    return state


def stage_finalization(state: PipelineState, context: PipelineContext) -> PipelineState:
    """
    Stage 7: Generate thumbnail and finalize metadata.

    Creates the video thumbnail and compiles all metadata.
    """
    print("   🖼️ Generating thumbnail and metadata...")

    if context.image_generator is None:
        raise ValueError("ImageGenerator not configured in context")

    if state.script is None:
        raise ValueError("Script required for finalization")

    # Generate thumbnail
    thumbnail_path = context.image_generator.generate_thumbnail(
        title=state.script.title,
        request=state.request,
        prompt=state.script.thumbnail_prompt,
    )

    # Generate metadata
    metadata = _generate_metadata(state)

    # Update output
    if state.output:
        state.output.thumbnail_path = thumbnail_path
        state.output.metadata = metadata
    else:
        state.output = VideoOutput(
            video_path=Path(""),
            thumbnail_path=thumbnail_path,
            metadata=metadata,
            success=False,
            error_message="Video composition was skipped",
        )

    print(f"      Thumbnail: {thumbnail_path}")
    print(f"      Title: {metadata.title}")
    return state


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _save_research(research, output_dir: Path) -> None:
    """Save research results to JSON file."""
    research_path = output_dir / "research.json"
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
    print(f"      💾 Research saved: {research_path}")


def _save_script(script, output_dir: Path) -> None:
    """Save script to JSON file."""
    script_path = output_dir / "script.json"
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
    print(f"      💾 Script saved: {script_path}")


def _save_planned_scenes(planned_scenes: list[PlannedScene], output_dir: Path) -> None:
    """Save planned scenes to JSON file."""
    scenes_path = output_dir / "planned_scenes.json"
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
    print(f"      💾 Planned scenes saved: {scenes_path}")


def _generate_metadata(state: PipelineState) -> VideoMetadata:
    """Generate video metadata from pipeline state."""
    script = state.script
    research = state.research
    request = state.request

    # Calculate total duration
    total_duration = sum(s.duration_seconds for s in state.planned_scenes)

    if script is None:
        return VideoMetadata(
            title=f"Video about {request.topic}",
            description=f"Learn about {request.topic}.",
            tags=[request.topic.split()[0], "education", "learning"],
            hashtags=[],
            duration_seconds=total_duration,
            format=request.format.value,
            sources=[],
        )

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
    tags = list(set(tags))

    return VideoMetadata(
        title=script.title,
        description=description,
        tags=tags,
        hashtags=script.hashtags or [],
        duration_seconds=total_duration,
        format=request.format.value,
        sources=research.sources if research else [],
    )


# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

# Stage definitions that can be used to create a pipeline
STAGE_DEFINITIONS = {
    "validation": {
        "execute": stage_validation,
        "description": "Validate input request",
        "depends_on": [],
        "optional": False,
    },
    "research": {
        "execute": stage_research,
        "description": "Research topic from web sources",
        "depends_on": ["validation"],
        "optional": False,
    },
    "script_generation": {
        "execute": stage_script_generation,
        "description": "Generate video script with scenes",
        "depends_on": ["research"],
        "optional": False,
    },
    "scene_planning": {
        "execute": stage_scene_planning,
        "description": "Plan scenes with image prompts and transitions",
        "depends_on": ["script_generation"],
        "optional": False,
    },
    "asset_generation": {
        "execute": stage_asset_generation,
        "description": "Generate images and voice narration in parallel",
        "depends_on": ["scene_planning"],
        "optional": False,
    },
    "video_composition": {
        "execute": stage_video_composition,
        "description": "Compose final video with subtitles and effects",
        "depends_on": ["asset_generation"],
        "optional": False,
    },
    "finalization": {
        "execute": stage_finalization,
        "description": "Generate thumbnail and video metadata",
        "depends_on": ["video_composition"],
        "optional": True,  # Can run even if video composition is skipped
    },
}


def create_default_stages() -> list[PipelineStage]:
    """
    Create the default list of pipeline stages.

    Returns:
        List of PipelineStage objects in execution order
    """
    stages = []
    for name in [
        "validation",
        "research",
        "script_generation",
        "scene_planning",
        "asset_generation",
        "video_composition",
        "finalization",
    ]:
        definition = STAGE_DEFINITIONS[name]
        stages.append(
            PipelineStage(
                name=name,
                execute=definition["execute"],
                description=definition["description"],
                depends_on=definition["depends_on"],
                optional=definition.get("optional", False),
            )
        )
    return stages


def create_pipeline_with_stages(context: PipelineContext) -> "ComposablePipeline":
    """
    Create a composable pipeline with all default stages.

    Args:
        context: Pipeline context with all components

    Returns:
        Configured ComposablePipeline ready to run
    """
    from .pipeline import ComposablePipeline

    pipeline = ComposablePipeline(context=context)
    for stage in create_default_stages():
        pipeline.add_stage(stage)
    return pipeline
