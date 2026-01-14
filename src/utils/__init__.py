"""
Utility modules for the video generation pipeline.

TYPE SYSTEM DESIGN:
- Pydantic models are used ONLY at API boundaries (external services)
- Internal data flow uses plain dataclasses (from pipeline/models.py)
- For Pydantic API models and conversion utilities, import from:
  `from src.pipeline.api_models import ScriptAPIModel, to_internal_script`

The model aliases below (SceneModel, ScriptModel, etc.) are maintained for
backward compatibility but are deprecated. New code should import directly
from src.pipeline.api_models.
"""

from .ai_client import (
    AIClient,
    AIConfig,
    # Deprecated aliases - use src.pipeline.api_models instead
    ResearchFactModel,
    ResearchModel,
    SceneModel,
    ScriptModel,
    create_ai_client,
    get_ai_client,
    reset_ai_client,
)

__all__ = [
    # Client classes
    "AIClient",
    "AIConfig",
    # Factory functions
    "create_ai_client",
    "get_ai_client",
    "reset_ai_client",
    # Deprecated model aliases (use src.pipeline.api_models instead)
    "ResearchModel",
    "ResearchFactModel",
    "ScriptModel",
    "SceneModel",
]
