"""
AI Client Utility - Unified interface for OpenAI and Gemini APIs.

Updated with latest OpenAI models (2025):
- Text: gpt-5-mini ($0.25/$2.00 per 1M tokens) - 400K context, supports structured outputs
- Image: gpt-image-1-mini (low: $0.005/image), gpt-image-1 (low: $0.011/image)
- TTS: gpt-4o-mini-tts ($0.015/min), tts-1, tts-1-hd
- STT: gpt-4o-mini-transcribe ($0.003/min), gpt-4o-transcribe ($0.006/min)

Uses Structured Outputs with Pydantic for guaranteed valid JSON responses.
"""

import os
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================


class SceneModel(BaseModel):
    """A single scene in the video script."""

    scene_number: int
    narration: str
    visual_description: str
    duration_seconds: float
    mood: str
    key_visual_elements: list[str] = []


class ScriptModel(BaseModel):
    """Complete script for the video."""

    title: str
    hook: str
    scenes: list[SceneModel]
    total_duration_seconds: float
    hashtags: list[str]
    thumbnail_prompt: str
    description: str


class ResearchFactModel(BaseModel):
    """A single fact with source."""

    fact: str
    source: str = "general knowledge"


class ResearchModel(BaseModel):
    """Research results for a topic."""

    key_points: list[str]
    facts: list[ResearchFactModel]
    examples: list[str]
    analogies: list[str]
    related_topics: list[str] = []


# Type variable for generic structured output
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# AI Configuration
# =============================================================================


@dataclass
class AIConfig:
    """Configuration for AI providers."""

    provider: str = "openai"  # "openai" or "gemini"

    # Text generation models - gpt-5-mini is cost-efficient with 400K context
    # Pricing (per 1M tokens): gpt-5-mini $0.25/$2.00, gpt-5-nano $0.05/?, gpt-5 $1.25/?
    # Supports: Structured Outputs, Function Calling, Streaming, Reasoning
    openai_model: str = (
        "gpt-5-mini"  # Best cost/quality balance, supports structured outputs
    )
    gemini_model: str = "gemini-2.0-flash"

    # Image generation - gpt-image-1-mini with low quality is most cost-effective ($0.005/image)
    image_model: str = "gpt-image-1-mini"  # Cost-efficient - alternative: gpt-image-1
    image_quality: str = "low"  # low ($0.005), medium ($0.011), high ($0.036) per image

    # Text-to-speech - gpt-4o-mini-tts is $0.015/minute
    tts_model: str = "gpt-4o-mini-tts"  # alternatives: tts-1, tts-1-hd
    tts_voice: str = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer


# =============================================================================
# AI Client
# =============================================================================


class AIClient:
    """Unified AI client for text generation, image generation, and TTS."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self._openai_client = None
        self._gemini_client = None

    @property
    def provider(self) -> str:
        """Get the current AI provider from config or environment."""
        return os.environ.get("AI_PROVIDER", self.config.provider).lower()

    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    @property
    def gemini_client(self):
        """Lazy-load Gemini client."""
        if self._gemini_client is None:
            from google import genai

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generate structured output using Pydantic model.

        This uses OpenAI's Structured Outputs feature to guarantee
        valid JSON that matches the schema.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class for the response
            system_prompt: Optional system/context prompt

        Returns:
            Parsed Pydantic model instance
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.openai_client.responses.parse(
            model=self.config.openai_model,
            input=messages,
            text_format=response_model,
        )

        return response.output_parsed

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate text using the configured AI provider.

        Args:
            prompt: The user prompt
            system_prompt: Optional system/context prompt
            temperature: Creativity level (0-1)
            max_tokens: Maximum response tokens

        Returns:
            Generated text response
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate text using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.openai_client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate text using Gemini."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.gemini_client.models.generate_content(
            model=self.config.gemini_model,
            contents=full_prompt,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return response.text

    async def web_search(self, query: str) -> dict:
        """
        Perform web search using OpenAI's web search capability.

        Args:
            query: Search query

        Returns:
            Search results with sources
        """
        try:
            response = self.openai_client.responses.create(
                model=self.config.openai_model,
                tools=[{"type": "web_search"}],
                input=f"Search for accurate, recent information about: {query}",
            )
            return self._parse_web_search_response(response)
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            return {"results": [], "error": str(e)}

    def _parse_web_search_response(self, response) -> dict:
        """Parse web search response into structured format."""
        results = {"content": "", "sources": [], "raw": response}

        if hasattr(response, "output"):
            results["content"] = response.output
        if hasattr(response, "citations"):
            results["sources"] = response.citations

        return results

    def generate_image(
        self,
        prompt: str,
        size: str = "1536x1024",
        quality: Optional[str] = None,
    ) -> bytes:
        """
        Generate an image using OpenAI's image generation API.

        Args:
            prompt: Image description prompt
            size: Image dimensions (e.g., "1536x1024", "1024x1536")
            quality: Image quality ("low", "medium", "high") - defaults to config

        Returns:
            Image as bytes (PNG format)
        """
        import base64

        quality = quality or self.config.image_quality

        result = self.openai_client.images.generate(
            model=self.config.image_model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

        # Decode base64 image
        image_base64 = result.data[0].b64_json
        return base64.b64decode(image_base64)

    def generate_speech(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0
    ) -> bytes:
        """
        Generate speech using OpenAI's TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speech speed (0.25 to 4.0)

        Returns:
            Audio as bytes (MP3 format)
        """
        voice = voice or self.config.tts_voice

        response = self.openai_client.audio.speech.create(
            model=self.config.tts_model, voice=voice, input=text, speed=speed
        )

        return response.content


# =============================================================================
# Singleton and Factory
# =============================================================================


_default_client: Optional[AIClient] = None


def get_ai_client(config: Optional[AIConfig] = None) -> AIClient:
    """Get or create the default AI client instance."""
    global _default_client
    if _default_client is None or config is not None:
        _default_client = AIClient(config)
    return _default_client
