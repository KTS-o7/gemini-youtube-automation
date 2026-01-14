"""
Script Writer Module - Generates video scripts with strong narrative continuity.

This module transforms research content into engaging, well-structured
video scripts with smooth transitions and cohesive storytelling.

Uses OpenAI Structured Outputs with Pydantic for guaranteed valid JSON.

TYPE SYSTEM:
- Pydantic models (ScriptAPIModel) are used at API boundary
- Internal dataclasses (Script) are used for pipeline data flow
- Conversion is handled by to_internal_script() from api_models.py
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .api_models import ScriptAPIModel, to_internal_script
from .models import ResearchResult, Script, VideoFormat, VideoRequest


class ScriptWriter:
    """Generates video scripts with narrative continuity from research content."""

    def __init__(self, ai_client: Optional[AIClient] = None):
        self.ai_client = ai_client or get_ai_client()

    def generate_script(
        self, request: VideoRequest, research: ResearchResult
    ) -> Script:
        """
        Generate a complete video script from research.

        Args:
            request: Video request with format and style info
            research: Research results with key points and facts

        Returns:
            Script with scenes, narration, and metadata

        Raises:
            Exception: If script generation fails
        """
        print(f"📝 Generating {request.format.value}-form script...")

        if request.format == VideoFormat.SHORT:
            return self._generate_short_script(request, research)
        else:
            return self._generate_long_script(request, research)

    def _generate_short_script(
        self, request: VideoRequest, research: ResearchResult
    ) -> Script:
        """Generate a short-form script (30-60 seconds) with strong narrative flow."""

        # Build key points string for the prompt
        key_points_text = "\n".join(f"- {p}" for p in research.key_points[:5])

        examples_text = ""
        if research.examples:
            examples_text = "EXAMPLES TO USE:\n" + "\n".join(
                f"- {e}" for e in research.examples[:3]
            )

        analogies_text = ""
        if research.analogies:
            analogies_text = "ANALOGIES:\n" + "\n".join(
                f"- {a}" for a in research.analogies[:2]
            )

        prompt = f"""Create a viral-worthy 45-60 second video script about "{request.topic}".

TARGET AUDIENCE: {request.target_audience}
STYLE: {request.style}

KEY POINTS TO COVER:
{key_points_text}

{examples_text}
{analogies_text}

=== YOUR TASK ===

You are the creative director. Analyze the topic and decide:
1. How many scenes are needed to explain this topic well (could be 2, 3, 4, 5, or more)
2. How to structure the narrative for maximum impact
3. How long each scene should be based on content complexity

There is NO fixed structure. You decide what works best for THIS specific topic.

=== NARRATIVE GUIDELINES ===

- Start with a hook that grabs attention immediately (no "Hey guys" or "Welcome to")
- Build a cohesive story with natural flow between scenes
- Each scene should connect logically to the next
- End with a memorable takeaway or callback to the opening

=== VISUAL CONTINUITY ===

- Choose ONE visual theme/metaphor and maintain it across ALL scenes
- Visual descriptions should feel like they're from the SAME video
- Describe visuals that naturally evolve and flow from one to the next
- Each visual_description should specify: subject, composition, lighting, color palette

=== SCENE GUIDELINES ===

For each scene you create:
- Duration: 3-15 seconds depending on content (you decide)
- Narration: As many words as needed to explain clearly
- Smooth transitions between scenes

=== OUTPUT REQUIREMENTS ===

- Total duration: 45-60 seconds
- Total narration: 120-180 words
- Number of scenes: YOU DECIDE based on topic complexity
- Write ACTUAL educational content, not placeholders
- Be specific and informative, not generic
- key_visual_elements should include recurring motifs across scenes"""

        system_prompt = """You are an expert short-form video scriptwriter who creates viral educational content with exceptional narrative flow.

Your scripts are known for:
- Hooks that stop people from scrolling AND set up the entire video
- Dense, valuable information delivered in a cohesive story
- Visual continuity that makes the video feel professionally produced
- Memorable endings that callback to the opening
- Smooth transitions that make complex topics feel like natural conversations

You ALWAYS think about how scenes connect before writing. You NEVER write disconnected, random scenes.
Every visual description you write could be a frame from the SAME animated explainer video."""

        result = self.ai_client.generate_structured(
            prompt=prompt,
            response_model=ScriptAPIModel,
            system_prompt=system_prompt,
        )
        return self._convert_and_validate(result)

    def _generate_long_script(
        self, request: VideoRequest, research: ResearchResult
    ) -> Script:
        """Generate a long-form script (3-10 minutes) with strong narrative continuity."""
        min_duration, max_duration = request.get_target_duration()

        # Build research content for prompt
        key_points_text = "\n".join(f"- {p}" for p in research.key_points)

        facts_text = ""
        if research.facts:
            facts_text = "FACTS:\n" + "\n".join(
                f"- {f.get('fact', f)}" for f in research.facts[:5]
            )

        examples_text = ""
        if research.examples:
            examples_text = "EXAMPLES:\n" + "\n".join(
                f"- {e}" for e in research.examples
            )

        prompt = f"""Create a {min_duration // 60}-{max_duration // 60} minute educational video script about "{request.topic}".

TARGET AUDIENCE: {request.target_audience}
STYLE: {request.style}

KEY POINTS TO COVER:
{key_points_text}

{facts_text}
{examples_text}

=== YOUR TASK ===

You are the creative director. Analyze the topic and decide:
1. How many scenes are needed to explain this topic thoroughly (could be 4, 5, 6, 8, 10, or more)
2. How to structure the narrative for maximum clarity and engagement
3. How long each scene should be based on content complexity
4. What visual metaphor/world will tie everything together

There is NO fixed number of scenes. You decide what works best for THIS specific topic.

=== NARRATIVE GUIDELINES ===

Your script must tell ONE cohesive story:
- Establish a CENTRAL QUESTION or PROBLEM early that gets answered by the end
- Each scene must build on the previous one - no random jumps
- Use transitional sentences that reference what came before
- Create a "golden thread" narrative that connects everything
- The conclusion must clearly resolve the opening question/problem

=== VISUAL CONTINUITY ===

Create a consistent visual world:
- Choose a PRIMARY VISUAL METAPHOR or SETTING that appears throughout
- Use a CONSISTENT COLOR PALETTE across all visual descriptions
- Recurring visual motifs should appear in multiple scenes
- Visuals should EVOLVE and BUILD, not randomly change
- Think of this as storyboarding an animated explainer video

=== SCENE GUIDELINES ===

For each scene you create:
- Duration: 10-60 seconds depending on content (you decide)
- Narration: As many sentences as needed to explain clearly
- Start with a transition referencing what we just learned
- End with a bridge to the next concept
- Visual: Continue evolving the same visual metaphor

=== OUTPUT REQUIREMENTS ===

- Total duration: {min_duration // 60}-{max_duration // 60} minutes
- Total word count: 500-800 words
- Number of scenes: YOU DECIDE based on topic complexity
- Write FULL narration for each scene (not placeholders)
- Use analogies appropriate for {request.target_audience}
- Include specific examples and clear explanations
- Visual descriptions must describe ONE consistent visual style/world
- key_visual_elements should include recurring motifs across scenes"""

        system_prompt = """You are an expert educational video scriptwriter known for creating content with exceptional narrative flow and visual consistency.

Your scripts are known for:
- Clear story arcs that make complex topics feel like engaging journeys
- Smooth transitions that feel conversational, not choppy
- Visual continuity that makes videos feel professionally produced
- Central questions that get satisfyingly answered
- Content that builds progressively - each scene depends on the last

Before writing, you ALWAYS:
1. Identify the central question/problem the video will answer
2. Choose a visual metaphor/world that will persist throughout
3. Map out how each scene connects to the next
4. Plan callbacks and references that tie the narrative together

You NEVER write disconnected scenes. Every scene you write is part of ONE cohesive story."""

        result = self.ai_client.generate_structured(
            prompt=prompt,
            response_model=ScriptAPIModel,
            system_prompt=system_prompt,
        )
        return self._convert_and_validate(result)

    def _convert_and_validate(self, api_model: ScriptAPIModel) -> Script:
        """
        Convert Pydantic API model to internal Script dataclass and validate.

        Uses the centralized conversion function from api_models.py.
        """
        # Convert using the standard conversion function
        script = to_internal_script(api_model)

        # Validate content quality - check average words per scene
        total_words = sum(len(scene.narration.split()) for scene in script.scenes)
        avg_words_per_scene = total_words / len(script.scenes) if script.scenes else 0
        min_avg_words = 15  # At least 15 words per scene on average

        if avg_words_per_scene < min_avg_words:
            raise ValueError(
                f"Script content too thin: {avg_words_per_scene:.0f} words/scene (minimum: {min_avg_words}). "
                f"AI generated insufficient content."
            )

        # Log continuity metrics
        self._log_continuity_check(script)

        print(f"✅ Script generated: {len(script.scenes)} scenes, {total_words} words")
        return script

    def _log_continuity_check(self, script: Script) -> None:
        """Log a quick continuity check for debugging."""
        # Check for transition words at scene starts
        transition_indicators = [
            "so",
            "now",
            "but",
            "this",
            "that",
            "remember",
            "building",
            "with",
            "and",
            "here",
            "as",
            "once",
            "after",
            "before",
        ]

        scenes_with_transitions = 0
        for i, scene in enumerate(script.scenes):
            if i == 0:
                continue  # First scene doesn't need transition
            first_word = scene.narration.split()[0].lower().rstrip(",.:;")
            if first_word in transition_indicators:
                scenes_with_transitions += 1

        if len(script.scenes) > 1:
            transition_score = scenes_with_transitions / (len(script.scenes) - 1)
            if transition_score < 0.5:
                print(
                    f"  ⚠️ Low transition score: {transition_score:.0%} of scenes have clear transitions"
                )
            else:
                print(
                    f"  ✓ Good transition flow: {transition_score:.0%} of scenes have clear transitions"
                )
