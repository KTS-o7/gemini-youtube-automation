"""
Prompts Module - Centralized prompt management for video generation pipeline.

This module provides easy access to all system prompts and user prompt templates
used throughout the pipeline. Prompts are stored as separate text files for
easy tweaking without modifying code.

Usage:
    from src.pipeline.prompts import (
        get_system_prompt,
        get_user_prompt_template,
        ScriptPrompts,
        ScenePlannerPrompts,
        ResearcherPrompts,
    )

    # Get a system prompt
    system_prompt = get_system_prompt("script_writer_short")

    # Get a user prompt template and format it
    template = get_user_prompt_template("script_writer_short")
    prompt = template.format(topic="Black Holes", target_audience="teenagers", ...)
"""

from pathlib import Path
from typing import Optional

# Directory where prompt files are stored
PROMPTS_DIR = Path(__file__).parent


class PromptLoader:
    """Loads and caches prompts from text files."""

    _cache: dict[str, str] = {}

    @classmethod
    def load(cls, filename: str) -> str:
        """
        Load a prompt from a text file.

        Args:
            filename: Name of the prompt file (without .txt extension)

        Returns:
            The prompt text content

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        if filename in cls._cache:
            return cls._cache[filename]

        filepath = PROMPTS_DIR / f"{filename}.md"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {filepath}\n"
                f"Available prompts: {cls.list_available()}"
            )

        content = filepath.read_text(encoding="utf-8").strip()
        cls._cache[filename] = content
        return content

    @classmethod
    def load_with_fallback(cls, filename: str, fallback: str) -> str:
        """
        Load a prompt, returning fallback if file doesn't exist.

        Args:
            filename: Name of the prompt file
            fallback: Fallback text if file not found

        Returns:
            The prompt text or fallback
        """
        try:
            return cls.load(filename)
        except FileNotFoundError:
            return fallback

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available prompt files."""
        return [f.stem for f in PROMPTS_DIR.glob("*.md")]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the prompt cache (useful for hot-reloading during development)."""
        cls._cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_system_prompt(name: str) -> str:
    """
    Get a system prompt by name.

    Available system prompts:
    - script_writer_short_system
    - script_writer_long_system
    - researcher_system
    - scene_planner_system

    Args:
        name: Name of the system prompt (without _system suffix is also accepted)

    Returns:
        The system prompt text
    """
    # Allow both "script_writer_short" and "script_writer_short_system"
    if not name.endswith("_system"):
        name = f"{name}_system"
    return PromptLoader.load(name)


def get_user_prompt_template(name: str) -> str:
    """
    Get a user prompt template by name.

    Available templates:
    - script_writer_short_user
    - script_writer_long_user
    - researcher_user
    - scene_planner_image_prompt

    Args:
        name: Name of the user prompt template

    Returns:
        The user prompt template (with {placeholders} for formatting)
    """
    if not name.endswith("_user") and not name.endswith("_prompt"):
        name = f"{name}_user"
    return PromptLoader.load(name)


def reload_prompts() -> None:
    """
    Reload all prompts from disk.

    Useful during development when tweaking prompts.
    """
    PromptLoader.clear_cache()
    print(
        f"🔄 Prompts cache cleared. Available prompts: {PromptLoader.list_available()}"
    )


# =============================================================================
# PROMPT CLASSES FOR ORGANIZED ACCESS
# =============================================================================


class ScriptPrompts:
    """Prompts for script generation."""

    @staticmethod
    def short_system() -> str:
        """System prompt for short-form script generation."""
        return PromptLoader.load("script_writer_short_system")

    @staticmethod
    def short_user() -> str:
        """User prompt template for short-form scripts."""
        return PromptLoader.load("script_writer_short_user")

    @staticmethod
    def long_system() -> str:
        """System prompt for long-form script generation."""
        return PromptLoader.load("script_writer_long_system")

    @staticmethod
    def long_user() -> str:
        """User prompt template for long-form scripts."""
        return PromptLoader.load("script_writer_long_user")


class ResearcherPrompts:
    """Prompts for content research."""

    @staticmethod
    def system(target_audience: str = "", style: str = "") -> str:
        """
        System prompt for research generation.

        Args:
            target_audience: Target audience for the content
            style: Content style preference

        Returns:
            Formatted system prompt
        """
        template = PromptLoader.load("researcher_system")
        return template.format(
            target_audience=target_audience or "general audience",
            style=style or "educational",
        )

    @staticmethod
    def user() -> str:
        """User prompt template for research."""
        return PromptLoader.load("researcher_user")


class ScenePlannerPrompts:
    """Prompts for scene planning and image generation."""

    @staticmethod
    def image_prompt_template() -> str:
        """Template for generating cohesive image prompts."""
        return PromptLoader.load("scene_planner_image_prompt")

    @staticmethod
    def refinement_system() -> str:
        """System prompt for AI-based image prompt refinement."""
        return PromptLoader.load("scene_planner_refinement_system")


class ImageGeneratorPrompts:
    """Prompts for image generation."""

    @staticmethod
    def thumbnail() -> str:
        """Template for thumbnail generation prompts."""
        return PromptLoader.load("image_generator_thumbnail")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PromptLoader",
    "get_system_prompt",
    "get_user_prompt_template",
    "reload_prompts",
    "ScriptPrompts",
    "ResearcherPrompts",
    "ScenePlannerPrompts",
    "ImageGeneratorPrompts",
    "PROMPTS_DIR",
]
