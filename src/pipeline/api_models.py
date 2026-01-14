"""
API Models - Pydantic models for external API boundaries.

This module contains Pydantic models used specifically for:
1. OpenAI Structured Outputs (requires Pydantic for schema validation)
2. External API request/response validation

DESIGN PATTERN:
- Pydantic models are used ONLY at API boundaries (external services)
- Internal data flow uses plain dataclasses (from models.py)
- Conversion functions bridge the two type systems

This separation provides:
- Type safety at API boundaries via Pydantic validation
- Lightweight internal data structures via dataclasses
- Clear separation of concerns
- Easier testing (dataclasses don't require Pydantic)

Usage:
    from src.pipeline.api_models import ScriptAPIModel, to_internal_script

    # Receive from OpenAI Structured Output
    api_response = ai_client.generate_structured(prompt, ScriptAPIModel)

    # Convert to internal dataclass for pipeline use
    internal_script = to_internal_script(api_response)
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# Import internal dataclass models for conversion
from .models import ResearchResult, Scene, Script

# =============================================================================
# PYDANTIC MODELS FOR OPENAI STRUCTURED OUTPUTS
# =============================================================================


class SceneAPIModel(BaseModel):
    """
    Pydantic model for a single scene (API boundary).

    Used with OpenAI's Structured Outputs to guarantee valid JSON responses.
    Convert to internal `Scene` dataclass using `to_internal_scene()`.
    """

    scene_number: int = Field(description="Sequential scene number starting from 1")
    narration: str = Field(description="Text for voice narration/TTS")
    visual_description: str = Field(description="Description for image generation")
    duration_seconds: float = Field(
        default=0.0, description="Estimated duration in seconds"
    )
    mood: str = Field(default="informative", description="Emotional tone of the scene")
    key_visual_elements: List[str] = Field(
        default_factory=list, description="Key elements to include in the visual"
    )


class ScriptAPIModel(BaseModel):
    """
    Pydantic model for a complete video script (API boundary).

    Used with OpenAI's Structured Outputs to guarantee valid JSON responses.
    Convert to internal `Script` dataclass using `to_internal_script()`.
    """

    title: str = Field(description="Catchy video title")
    hook: str = Field(description="Opening hook to grab attention")
    scenes: List[SceneAPIModel] = Field(description="List of scenes in order")
    total_duration_seconds: float = Field(
        default=0.0, description="Total estimated duration"
    )
    hashtags: List[str] = Field(
        default_factory=list, description="Relevant hashtags for the video"
    )
    thumbnail_prompt: str = Field(
        default="", description="Prompt for generating thumbnail image"
    )
    description: str = Field(default="", description="Video description for platforms")


class ResearchFactAPIModel(BaseModel):
    """
    Pydantic model for a single research fact (API boundary).

    Used with OpenAI's Structured Outputs.
    """

    fact: str = Field(description="The factual statement")
    source: str = Field(default="general knowledge", description="Source of the fact")


class ResearchAPIModel(BaseModel):
    """
    Pydantic model for research results (API boundary).

    Used with OpenAI's Structured Outputs to guarantee valid JSON responses.
    Convert to internal `ResearchResult` dataclass using `to_internal_research()`.
    """

    key_points: List[str] = Field(
        default_factory=list, description="Main points about the topic"
    )
    facts: List[ResearchFactAPIModel] = Field(
        default_factory=list, description="Verified facts with sources"
    )
    examples: List[str] = Field(default_factory=list, description="Real-world examples")
    analogies: List[str] = Field(
        default_factory=list, description="Helpful analogies for explanation"
    )
    related_topics: List[str] = Field(
        default_factory=list, description="Related topics for further exploration"
    )


class PlannedSceneAPIModel(BaseModel):
    """
    Pydantic model for a planned scene with image prompt (API boundary).

    Used when AI generates detailed scene plans with image prompts.
    """

    scene_number: int
    narration: str
    visual_description: str
    image_prompt: str = Field(description="Detailed prompt for image generation")
    duration_seconds: float = 0.0
    mood: str = "informative"
    transition: str = Field(
        default="crossfade",
        description="Transition type: crossfade, cut, fade_to_black",
    )


# =============================================================================
# CONVERSION FUNCTIONS: API Models -> Internal Dataclasses
# =============================================================================


def to_internal_scene(api_scene: SceneAPIModel) -> Scene:
    """
    Convert a Pydantic SceneAPIModel to internal Scene dataclass.

    Args:
        api_scene: Pydantic model from API response

    Returns:
        Internal Scene dataclass for pipeline use
    """
    return Scene(
        scene_number=api_scene.scene_number,
        narration=api_scene.narration,
        visual_description=api_scene.visual_description,
        duration_seconds=api_scene.duration_seconds,
        mood=api_scene.mood,
        key_visual_elements=list(api_scene.key_visual_elements),
    )


def to_internal_script(api_script: ScriptAPIModel) -> Script:
    """
    Convert a Pydantic ScriptAPIModel to internal Script dataclass.

    Args:
        api_script: Pydantic model from API response

    Returns:
        Internal Script dataclass for pipeline use
    """
    return Script(
        title=api_script.title,
        hook=api_script.hook,
        scenes=[to_internal_scene(s) for s in api_script.scenes],
        total_duration_seconds=api_script.total_duration_seconds,
        hashtags=list(api_script.hashtags),
        thumbnail_prompt=api_script.thumbnail_prompt,
        description=api_script.description,
    )


def to_internal_research(api_research: ResearchAPIModel, topic: str) -> ResearchResult:
    """
    Convert a Pydantic ResearchAPIModel to internal ResearchResult dataclass.

    Args:
        api_research: Pydantic model from API response
        topic: The research topic (not included in API model)

    Returns:
        Internal ResearchResult dataclass for pipeline use
    """
    return ResearchResult(
        topic=topic,
        key_points=list(api_research.key_points),
        facts=[{"fact": f.fact, "source": f.source} for f in api_research.facts],
        examples=list(api_research.examples),
        analogies=list(api_research.analogies),
        related_topics=list(api_research.related_topics),
    )


# =============================================================================
# CONVERSION FUNCTIONS: Internal Dataclasses -> API Models
# =============================================================================


def to_api_scene(scene: Scene) -> SceneAPIModel:
    """
    Convert an internal Scene dataclass to Pydantic SceneAPIModel.

    Useful when you need to serialize internal data for API calls.

    Args:
        scene: Internal Scene dataclass

    Returns:
        Pydantic SceneAPIModel for API use
    """
    return SceneAPIModel(
        scene_number=scene.scene_number,
        narration=scene.narration,
        visual_description=scene.visual_description,
        duration_seconds=scene.duration_seconds,
        mood=scene.mood,
        key_visual_elements=list(scene.key_visual_elements),
    )


def to_api_script(script: Script) -> ScriptAPIModel:
    """
    Convert an internal Script dataclass to Pydantic ScriptAPIModel.

    Useful when you need to serialize internal data for API calls.

    Args:
        script: Internal Script dataclass

    Returns:
        Pydantic ScriptAPIModel for API use
    """
    return ScriptAPIModel(
        title=script.title,
        hook=script.hook,
        scenes=[to_api_scene(s) for s in script.scenes],
        total_duration_seconds=script.total_duration_seconds,
        hashtags=list(script.hashtags),
        thumbnail_prompt=script.thumbnail_prompt,
        description=script.description,
    )


def to_api_research(research: ResearchResult) -> ResearchAPIModel:
    """
    Convert an internal ResearchResult dataclass to Pydantic ResearchAPIModel.

    Args:
        research: Internal ResearchResult dataclass

    Returns:
        Pydantic ResearchAPIModel for API use
    """
    return ResearchAPIModel(
        key_points=list(research.key_points),
        facts=[
            ResearchFactAPIModel(fact=f["fact"], source=f.get("source", "unknown"))
            for f in research.facts
        ],
        examples=list(research.examples),
        analogies=list(research.analogies),
        related_topics=list(research.related_topics),
    )


# =============================================================================
# TYPE ALIASES FOR CLARITY
# =============================================================================

# These aliases make it clear which type system is being used
APIScript = ScriptAPIModel
APIScene = SceneAPIModel
APIResearch = ResearchAPIModel

InternalScript = Script
InternalScene = Scene
InternalResearch = ResearchResult
