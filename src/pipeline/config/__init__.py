"""
Configuration Loader - Load YAML configuration files for video composition.

This module provides utilities for loading external configuration data
like emphasis keywords and emoji mappings from YAML files.

Usage:
    from src.pipeline.config import load_emphasis_keywords, load_emoji_mappings

    keywords = load_emphasis_keywords()
    emojis = load_emoji_mappings()
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import yaml, fall back to json parsing if not available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


CONFIG_DIR = Path(__file__).parent


def _load_yaml_file(filename: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        filename: Name of the YAML file in the config directory

    Returns:
        Parsed YAML content as a dictionary

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ImportError: If PyYAML is not installed
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load configuration files. "
            "Install with: pip install pyyaml"
        )

    filepath = CONFIG_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_emphasis_keywords() -> Dict[str, List[str]]:
    """
    Load emphasis keywords from YAML configuration.

    Returns a dictionary mapping emphasis categories to word lists:
    - 'strong': High-impact words (yellow/gold)
    - 'action': Call-to-action words (cyan/blue)
    - 'stats': Numbers and statistics (green)

    Returns:
        Dictionary of category -> word list

    Example:
        >>> keywords = load_emphasis_keywords()
        >>> 'secret' in keywords['strong']
        True
    """
    try:
        return _load_yaml_file("emphasis_keywords.yaml")
    except (ImportError, FileNotFoundError):
        # Fall back to hardcoded defaults
        return _get_default_emphasis_keywords()


@lru_cache(maxsize=1)
def load_emoji_mappings() -> Dict[str, str]:
    """
    Load emoji mappings from YAML configuration.

    Returns a flat dictionary mapping words to emojis,
    combining all categories from the YAML file.

    Returns:
        Dictionary of word -> emoji

    Example:
        >>> emojis = load_emoji_mappings()
        >>> emojis.get('code')
        '💻'
    """
    try:
        data = _load_yaml_file("emoji_mappings.yaml")

        # Flatten the categorized structure into a single dict
        flat_mappings = {}
        for category, mappings in data.items():
            if isinstance(mappings, dict):
                flat_mappings.update(mappings)

        return flat_mappings
    except (ImportError, FileNotFoundError):
        # Fall back to hardcoded defaults
        return _get_default_emoji_mappings()


def clear_config_cache() -> None:
    """Clear cached configuration data. Useful for testing or hot-reloading."""
    load_emphasis_keywords.cache_clear()
    load_emoji_mappings.cache_clear()


def _get_default_emphasis_keywords() -> Dict[str, List[str]]:
    """Return default emphasis keywords when YAML is unavailable."""
    return {
        "strong": [
            "secret",
            "amazing",
            "incredible",
            "powerful",
            "essential",
            "critical",
            "important",
            "key",
            "game-changer",
            "revolutionary",
            "breakthrough",
            "must",
            "never",
            "always",
            "best",
            "worst",
            "biggest",
            "most",
            "first",
            "last",
            "only",
            "ultimate",
            "proven",
            "guaranteed",
            "free",
            "new",
            "now",
            "today",
            "instantly",
            "immediately",
        ],
        "action": [
            "learn",
            "discover",
            "master",
            "unlock",
            "build",
            "create",
            "start",
            "stop",
            "try",
            "use",
            "get",
            "make",
            "find",
            "see",
            "watch",
            "click",
            "subscribe",
            "follow",
            "share",
            "comment",
            "like",
        ],
        "stats": [
            "percent",
            "%",
            "million",
            "billion",
            "thousand",
            "hundred",
            "twice",
            "triple",
            "double",
            "half",
            "10x",
            "100x",
        ],
    }


def _get_default_emoji_mappings() -> Dict[str, str]:
    """Return default emoji mappings when YAML is unavailable."""
    return {
        # Tech
        "code": "💻",
        "coding": "💻",
        "programming": "💻",
        "developer": "👨‍💻",
        "software": "🖥️",
        "app": "📱",
        "ai": "🤖",
        "machine learning": "🧠",
        # Business
        "money": "💰",
        "success": "🏆",
        "goal": "🎯",
        "startup": "🚀",
        "growth": "📈",
        # Learning
        "learn": "📚",
        "idea": "💡",
        "tip": "💡",
        "secret": "🤫",
        "mistake": "❌",
        "correct": "✅",
        # Emotions
        "amazing": "🤯",
        "love": "❤️",
        "happy": "😊",
        "excited": "🎉",
        # General
        "fire": "🔥",
        "warning": "⚠️",
        "important": "❗",
        "question": "❓",
    }


__all__ = [
    "load_emphasis_keywords",
    "load_emoji_mappings",
    "clear_config_cache",
]
