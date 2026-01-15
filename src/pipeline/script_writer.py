"""
Script Writer Module - Generates video scripts with strong narrative continuity.

This module transforms research content into engaging, well-structured
video scripts with smooth transitions and cohesive storytelling.

Uses OpenAI Structured Outputs with Pydantic for guaranteed valid JSON.

TYPE SYSTEM:
- Pydantic models (ScriptAPIModel) are used at API boundary
- Internal dataclasses (Script) are used for pipeline data flow
- Conversion is handled by to_internal_script() from api_models.py

PROMPTS:
- System and user prompts are loaded from external files in ./prompts/
- Edit those files to tweak prompt behavior without changing code
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .api_models import ScriptAPIModel, to_internal_script
from .models import ResearchResult, Script, VideoFormat, VideoRequest
from .prompts import PromptLoader


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

        # Load prompts from external files
        user_prompt_template = PromptLoader.load("script_writer_short_user")
        system_prompt = PromptLoader.load("script_writer_short_system")

        # Format the user prompt with variables
        prompt = user_prompt_template.format(
            topic=request.topic,
            target_audience=request.target_audience,
            style=request.style,
            key_points_text=key_points_text,
            examples_text=examples_text,
            analogies_text=analogies_text,
        )

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

        # Load prompts from external files
        user_prompt_template = PromptLoader.load("script_writer_long_user")
        system_prompt = PromptLoader.load("script_writer_long_system")

        # Format the user prompt with variables
        prompt = user_prompt_template.format(
            topic=request.topic,
            target_audience=request.target_audience,
            style=request.style,
            key_points_text=key_points_text,
            facts_text=facts_text,
            examples_text=examples_text,
            min_duration=min_duration // 60,
            max_duration=max_duration // 60,
        )

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
        # Check for transition words/phrases at scene starts
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
            "which",
            "because",
            "however",
            "meanwhile",
            "in",
            "the",
            "what",
            "how",
            "why",
            "let",
            "to",
            "it",
        ]

        # Also check for common transition phrases
        transition_phrases = [
            "but here",
            "so how",
            "now that",
            "this is",
            "that's",
            "here's",
            "let's",
            "which brings",
            "which means",
            "and that",
            "and this",
            "remember when",
            "back to",
            "earlier we",
            "as we",
            "to understand",
            "in other words",
            "think of",
            "imagine",
            "picture",
        ]

        scenes_with_transitions = 0
        for i, scene in enumerate(script.scenes):
            if i == 0:
                continue  # First scene doesn't need transition

            narration_lower = scene.narration.lower()
            first_word = narration_lower.split()[0].rstrip(",.:;")

            # Check first word
            has_transition = first_word in transition_indicators

            # Check for phrase matches in first 50 chars
            if not has_transition:
                first_part = narration_lower[:50]
                has_transition = any(
                    phrase in first_part for phrase in transition_phrases
                )

            if has_transition:
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

        # Check for animation words in visual descriptions (should be avoided)
        animation_words = [
            "animat",
            "moving",
            "flies",
            "flying",
            "zooms",
            "zooming",
            "transforms",
            "morphs",
            "transitions",
            "camera moves",
            "camera pushes",
            "camera pulls",
            "pans to",
            "tilts to",
        ]
        scenes_with_animation_words = 0
        for scene in script.scenes:
            visual_lower = scene.visual_description.lower()
            if any(word in visual_lower for word in animation_words):
                scenes_with_animation_words += 1

        if scenes_with_animation_words > 0:
            print(
                f"  ⚠️ {scenes_with_animation_words} scene(s) have animation words in visual descriptions (should be static)"
            )
        else:
            print(f"  ✓ All visual descriptions are static-image friendly")
