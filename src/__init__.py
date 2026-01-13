"""
AI Video Generation Source Package

This package contains:
- pipeline/: Modular video generation pipeline
- utils/: Utility modules and AI client wrappers
- generator.py: Legacy generator functions
- uploader.py: YouTube upload functionality
"""

from .pipeline import (
    VideoFormat,
    VideoOutput,
    VideoPipeline,
    VideoRequest,
    create_video,
)

__all__ = [
    "VideoPipeline",
    "VideoRequest",
    "VideoFormat",
    "VideoOutput",
    "create_video",
]
