"""
Video Generation Pipeline Package

This package provides a modular, scalable pipeline for generating
AI-powered videos from text topics.

Pipeline Stages:
1. Input Processing - Validate and process video requests
2. Web Research - Gather relevant content from the web
3. Script Generation - Create engaging video scripts
4. Scene Planning - Break scripts into detailed scene plans
5. Asset Generation - Generate images and voice for each scene
6. Video Composition - Stitch scenes into final video
7. Output & Metadata - Generate thumbnails and video metadata

Usage:
    from src.pipeline import VideoPipeline, VideoRequest, VideoFormat

    request = VideoRequest(
        topic="How does machine learning work?",
        target_audience="beginners",
        format=VideoFormat.LONG,
        style="educational"
    )

    pipeline = VideoPipeline()
    output = pipeline.generate_video_sync(request)

    print(f"Video saved to: {output.video_path}")
"""

from .image_generator import ImageGenerator
from .models import (
    GeneratedAssets,
    PipelineState,
    PlannedScene,
    ResearchResult,
    Scene,
    Script,
    TTSProvider,
    VideoFormat,
    VideoMetadata,
    VideoOutput,
    VideoRequest,
)
from .orchestrator import VideoPipeline, create_video
from .researcher import ContentResearcher
from .scene_planner import ScenePlanner
from .script_writer import ScriptWriter
from .video_composer import VideoComposer
from .voice_generator import VoiceGenerator

__all__ = [
    # Main pipeline
    "VideoPipeline",
    "create_video",
    # Models
    "VideoRequest",
    "VideoFormat",
    "VideoOutput",
    "VideoMetadata",
    "ResearchResult",
    "Script",
    "Scene",
    "PlannedScene",
    "GeneratedAssets",
    "PipelineState",
    "TTSProvider",
    # Components
    "ContentResearcher",
    "ScriptWriter",
    "ScenePlanner",
    "ImageGenerator",
    "VoiceGenerator",
    "VideoComposer",
]
