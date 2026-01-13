"""
Script Writer Module - Generates video scripts with strong narrative continuity.

This module transforms research content into engaging, well-structured
video scripts with smooth transitions and cohesive storytelling.

Uses OpenAI Structured Outputs with Pydantic for guaranteed valid JSON.
"""

from typing import Optional

from ..utils.ai_client import AIClient, ScriptModel, get_ai_client
from .models import ResearchResult, Scene, Script, VideoFormat, VideoRequest


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

=== NARRATIVE CONTINUITY RULES (CRITICAL) ===

Your script must tell ONE cohesive story with a clear narrative arc:
1. SETUP → EXPLORATION → PAYOFF structure
2. Each scene must directly connect to the previous one
3. Use transitional phrases that link ideas naturally
4. Maintain a consistent tone and perspective throughout
5. The ending must clearly resolve or callback to the opening hook

=== VISUAL CONTINUITY RULES (CRITICAL) ===

Choose ONE visual theme/metaphor and maintain it across ALL scenes:
- If Scene 1 uses a "journey" metaphor, all scenes should continue that journey
- If Scene 1 shows a specific environment (office, lab, etc.), stay in that world
- Visual descriptions should feel like they're from the SAME video, not random clips
- Describe visuals that naturally flow from one to the next

=== SCRIPT STRUCTURE ===

SCENE 1 - HOOK (5-8 seconds):
- Start with a relatable problem, surprising fact, or bold question
- NO "Hey guys" or "Welcome to" - jump straight into value
- Establish the visual world/metaphor that continues throughout
- Visual: Set the scene that will evolve across the video
- Example hook styles: "Ever wondered why...", "Here's a secret that...", "What if I told you..."

SCENE 2 - CORE CONTENT (35-45 seconds):
- This is the MAIN educational content
- Explain the topic clearly in 5-8 sentences
- Use ONE concrete example or analogy (from the established visual world)
- Break down complex ideas into simple terms
- Make every sentence add real value
- TRANSITION: Use a phrase that connects back to the hook
- Visual: Continue/evolve the visual metaphor from Scene 1
- Minimum 80 words for this scene

SCENE 3 - MEMORABLE ENDING (8-12 seconds):
- Callback to the opening hook or problem
- Summarize the key takeaway in 1-2 sentences
- End with a thought-provoking question OR call to action
- Visual: Complete the visual journey with a satisfying conclusion

=== TRANSITION EXAMPLES ===
- "So how does this solve our problem?" (connects back to hook)
- "And that's exactly why..." (links cause to effect)
- "Building on that..." (continues the thread)
- "Here's where it gets interesting..." (escalates engagement)
- "Remember when we said...? Well..." (explicit callback)

=== OUTPUT REQUIREMENTS ===

- Total narration: 120-180 words
- Write ACTUAL educational content, not placeholders
- Be specific and informative, not generic
- Visual descriptions must describe ONE consistent visual style/world
- Each visual_description should specify: subject, composition, lighting, color palette
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
            response_model=ScriptModel,
            system_prompt=system_prompt,
        )
        return self._convert_to_script(result)

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

=== NARRATIVE CONTINUITY RULES (CRITICAL) ===

Your script must tell ONE cohesive story:
1. Establish a CENTRAL QUESTION or PROBLEM in Scene 1 that gets answered by the end
2. Each scene must build on the previous one - no random jumps
3. Use transitional sentences at the START of each scene that reference the previous scene
4. Create a "golden thread" narrative that connects everything
5. The conclusion must clearly resolve the opening question/problem

=== VISUAL CONTINUITY RULES (CRITICAL) ===

Create a consistent visual world:
1. Choose a PRIMARY VISUAL METAPHOR or SETTING that appears throughout
   - Examples: "a journey through a digital landscape", "building blocks assembling",
     "a day in the life scenario", "an evolving diagram/flowchart"
2. Use a CONSISTENT COLOR PALETTE across all visual descriptions
3. Recurring visual motifs should appear in multiple scenes
4. Visuals should EVOLVE and BUILD, not randomly change
5. Think of this as storyboarding an animated explainer video

=== SCRIPT STRUCTURE (6-8 scenes) ===

SCENE 1 - HOOK + CENTRAL QUESTION (10-15 seconds):
- Start with a compelling question, surprising fact, or relatable scenario
- This question/problem should be ANSWERED by the end
- Establish the visual world that will continue throughout
- Transition: Set up what we'll explore
- 3-4 sentences

SCENE 2 - CONTEXT & FOUNDATION (20-30 seconds):
- Set context for the topic - why it matters NOW
- Preview the journey we'll take
- Establish foundational concept #1
- Transition: "With that foundation, let's explore..."
- Visual: Expand on Scene 1's world
- 4-6 sentences

SCENES 3-5 - MAIN CONTENT (2-6 minutes total):
Each scene should:
- Cover ONE main concept that builds on previous scenes
- START with a transition referencing what we just learned
- Use examples/analogies from within our visual world
- END with a bridge to the next concept
- Visual: Continue evolving the same visual metaphor
- 6-10 sentences each

Scene 3: "Building on [Scene 2 concept], now let's look at..."
Scene 4: "This connects to [previous idea] because..."
Scene 5: "Now we can see how all of this comes together..."

SCENE 6 - PRACTICAL APPLICATION (30-45 seconds):
- "So how do you actually USE this?"
- Real-world application that ties back to opening scenario
- Visual: Show the concepts in action
- 4-6 sentences

SCENE 7 - SYNTHESIS & ANSWER (20-30 seconds):
- Explicitly ANSWER the central question from Scene 1
- Connect all the dots - show how everything we learned fits together
- Visual: Complete visual metaphor (journey ends, building complete, etc.)
- 3-5 sentences

SCENE 8 - CALL TO ACTION (10-15 seconds):
- Brief recap of the transformation (before → after)
- Engagement prompt (comment, subscribe)
- Tease related content
- 2-3 sentences

=== TRANSITION TOOLKIT ===

Use these patterns to maintain flow:
- Referential: "Remember when we said X? Here's why that matters..."
- Building: "Now that we understand X, we can explore Y..."
- Contrasting: "But here's where it gets interesting..."
- Questioning: "So you might be wondering... how does X work with Y?"
- Connecting: "This directly connects to what we saw earlier..."
- Revealing: "And this is exactly why [callback to hook]..."

=== OUTPUT REQUIREMENTS ===

- Total word count: 500-800 words
- Write FULL narration for each scene (not placeholders)
- Use analogies appropriate for {request.target_audience}
- Include specific examples and clear explanations
- Visual descriptions must describe ONE consistent visual style/world
- Specify in each visual_description: subject, action, lighting, color tone
- key_visual_elements should include 2-3 elements, with at least 1 recurring across scenes
- Each scene's mood should flow naturally from the previous (no jarring shifts)"""

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
            response_model=ScriptModel,
            system_prompt=system_prompt,
        )
        return self._convert_to_script(result)

    def _convert_to_script(self, model: ScriptModel) -> Script:
        """Convert Pydantic ScriptModel to internal Script dataclass."""
        scenes = []
        for scene_model in model.scenes:
            scene = Scene(
                scene_number=scene_model.scene_number,
                narration=scene_model.narration,
                visual_description=scene_model.visual_description,
                duration_seconds=scene_model.duration_seconds,
                mood=scene_model.mood,
                key_visual_elements=scene_model.key_visual_elements,
            )
            scenes.append(scene)

        script = Script(
            title=model.title,
            hook=model.hook,
            scenes=scenes,
            total_duration_seconds=model.total_duration_seconds,
            hashtags=model.hashtags,
            thumbnail_prompt=model.thumbnail_prompt,
            description=model.description,
        )

        # Validate content quality
        total_words = sum(len(scene.narration.split()) for scene in script.scenes)
        min_words = 80 if len(script.scenes) <= 4 else 300

        if total_words < min_words:
            raise ValueError(
                f"Script content too thin: {total_words} words (minimum: {min_words}). "
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
