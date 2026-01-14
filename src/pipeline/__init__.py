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

Modules:
- video_composer: Main video composition orchestrator
- subtitle_renderer: Karaoke-style and standard subtitle rendering
- motion_effects: Ken Burns, transitions, and dynamic motion effects
- pipeline: Composable pipeline infrastructure with skip/resume
- stages: Individual stage functions for composable pipeline
- api_models: Pydantic models for API boundaries with conversion utilities

TYPE SYSTEM DESIGN:
- Pydantic models (in api_models.py) are used ONLY at API boundaries
- Internal data flow uses plain dataclasses (in models.py)
- Conversion functions bridge the two type systems

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

from .alignment_types import AlignmentResult, SubtitleStyle, WordTimestamp
from .api_models import (
    PlannedSceneAPIModel,
    ResearchAPIModel,
    ResearchFactAPIModel,
    SceneAPIModel,
    ScriptAPIModel,
    to_api_research,
    to_api_scene,
    to_api_script,
    to_internal_research,
    to_internal_scene,
    to_internal_script,
)
from .audio_aligner import AudioAligner

# Try to import wav2vec2 components (requires torch)
try:
    from .aligner_factory import AlignerType, UnifiedAligner, create_aligner
    from .wav2vec2_aligner import Wav2Vec2Aligner, align_with_wav2vec2

    _WAV2VEC2_AVAILABLE = True
except ImportError:
    _WAV2VEC2_AVAILABLE = False
    AlignerType = None
    UnifiedAligner = None
    create_aligner = None
    Wav2Vec2Aligner = None
    align_with_wav2vec2 = None
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
from .motion_effects import MotionConfig, MotionEffects
from .orchestrator import VideoPipeline, create_video
from .pipeline import (
    ComposablePipeline,
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
    StageResult,
    StageStatus,
    create_pipeline,
)
from .researcher import ContentResearcher
from .scene_planner import ScenePlanner
from .script_writer import ScriptWriter
from .subtitle_renderer import SubtitleConfig, SubtitleRenderer, SubtitleStyle
from .video_composer import VideoComposer, configure_imagemagick
from .voice_generator import (
    TTSError,
    TTSFallbackError,
    VoiceConfig,
    VoiceGenerator,
    create_voice_generator,
)

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
    "VoiceConfig",
    "TTSError",
    "TTSFallbackError",
    "create_voice_generator",
    "VideoComposer",
    "configure_imagemagick",
    # API Models (Pydantic - for API boundaries only)
    "ScriptAPIModel",
    "SceneAPIModel",
    "ResearchAPIModel",
    "ResearchFactAPIModel",
    "PlannedSceneAPIModel",
    # API Model Conversion Functions
    "to_internal_script",
    "to_internal_scene",
    "to_internal_research",
    "to_api_script",
    "to_api_scene",
    "to_api_research",
    # Composable Pipeline
    "ComposablePipeline",
    "PipelineStage",
    "PipelineConfig",
    "PipelineCheckpoint",
    "StageResult",
    "StageStatus",
    "create_pipeline",
    # Subtitle Rendering
    "SubtitleRenderer",
    "SubtitleStyle",
    "SubtitleConfig",
    # Motion Effects
    "MotionEffects",
    "MotionConfig",
    # Audio Alignment (subtitles)
    "AudioAligner",  # OpenAI Whisper - auto-transcribes, costs ~$0.006/min
    # Shared alignment types
    "WordTimestamp",
    "AlignmentResult",
    "SubtitleStyle",
]

# Add wav2vec2 exports only if torch is available
if _WAV2VEC2_AVAILABLE:
    __all__.extend(
        [
            "Wav2Vec2Aligner",  # FREE local alignment - requires transcript
            "align_with_wav2vec2",  # Convenience function for wav2vec2
            # Aligner Factory (recommended)
            "AlignerType",  # Enum: WHISPER or WAV2VEC2
            "UnifiedAligner",  # Unified interface for both aligners
            "create_aligner",  # Factory function to create aligner
        ]
    )
