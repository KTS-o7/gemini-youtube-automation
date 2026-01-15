"""
Web Research Module - Searches and curates content for video generation.

This module handles research to gather accurate information about the video topic.
Uses OpenAI Structured Outputs with Pydantic for guaranteed valid JSON.

TYPE SYSTEM:
- Pydantic models (ResearchAPIModel) are used at API boundary
- Internal dataclasses (ResearchResult) are used for pipeline data flow
- Conversion is handled by to_internal_research() from api_models.py

PROMPTS:
- System and user prompts are loaded from external files in ./prompts/
- Edit those files to tweak prompt behavior without changing code
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .api_models import ResearchAPIModel, to_internal_research
from .models import ResearchResult, VideoRequest
from .prompts import PromptLoader


class ContentResearcher:
    """Researches topics using AI and returns structured content."""

    def __init__(self, ai_client: Optional[AIClient] = None):
        self.ai_client = ai_client or get_ai_client()

    async def research(self, request: VideoRequest) -> ResearchResult:
        """
        Research a topic and return curated content (async version).

        Args:
            request: Video request with topic and audience info

        Returns:
            ResearchResult with key points, facts, examples, etc.

        Raises:
            Exception: If research generation fails
        """
        return self.research_sync(request)

    def research_sync(self, request: VideoRequest) -> ResearchResult:
        """
        Research a topic using AI-powered content generation.

        Uses Structured Outputs to guarantee valid response format.

        Args:
            request: Video request with topic and audience info

        Returns:
            ResearchResult with key points, facts, examples, etc.

        Raises:
            Exception: If research generation fails
        """
        print(f"🔍 Researching topic: {request.topic}")

        # Load prompts from external files
        user_prompt_template = PromptLoader.load("researcher_user")
        system_prompt_template = PromptLoader.load("researcher_system")

        # Format the prompts with variables
        prompt = user_prompt_template.format(
            topic=request.topic,
            target_audience=request.target_audience,
            format=request.format.value,
            style=request.style,
        )

        system_prompt = system_prompt_template.format(
            target_audience=request.target_audience,
            style=request.style,
        )

        result = self.ai_client.generate_structured(
            prompt=prompt,
            response_model=ResearchAPIModel,
            system_prompt=system_prompt,
        )

        # Convert Pydantic API model to internal ResearchResult using standard converter
        research_result = to_internal_research(result, topic=request.topic)

        print(
            f"✅ Research complete: {len(research_result.key_points)} key points found"
        )
        return research_result
