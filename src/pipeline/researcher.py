"""
Web Research Module - Searches and curates content for video generation.

This module handles research to gather accurate information about the video topic.
Uses OpenAI Structured Outputs with Pydantic for guaranteed valid JSON.

TYPE SYSTEM:
- Pydantic models (ResearchAPIModel) are used at API boundary
- Internal dataclasses (ResearchResult) are used for pipeline data flow
- Conversion is handled by to_internal_research() from api_models.py
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .api_models import ResearchAPIModel, to_internal_research
from .models import ResearchResult, VideoRequest


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

        prompt = f"""Research the topic "{request.topic}" for a video targeting {request.target_audience}.

Provide comprehensive, accurate information that will help create an engaging {request.format.value}-form video.

Requirements:
- key_points: 5-7 main points to cover, ordered logically to build understanding
- facts: 3-5 interesting, accurate facts with sources (use "general knowledge" if no specific source)
- examples: 2-4 concrete, real-world examples that {request.target_audience} can relate to
- analogies: 2-3 simple analogies that explain complex concepts using familiar things
- related_topics: 2-3 topics for potential follow-up videos

Make sure:
- Content is appropriate for {request.target_audience}
- Key points flow logically and build on each other
- Examples are specific and relatable, not generic
- Analogies simplify without oversimplifying
- Facts are accurate and interesting"""

        system_prompt = f"""You are an expert researcher and educator specializing in creating content for {request.target_audience}.

Your research is known for:
- Accuracy and depth
- Clear, logical organization
- Relatable examples that resonate with the audience
- Analogies that make complex topics accessible

Style preference: {request.style}"""

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
