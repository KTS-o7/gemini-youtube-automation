"""
Scene Planner Module - Creates visually cohesive scene plans with continuity.

This module takes a script and creates detailed plans for each scene,
ensuring visual consistency across all scenes through:
- Consistent visual themes and color palettes
- Recurring visual motifs
- Coherent style progression
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .models import PlannedScene, Script, VideoFormat, VideoRequest


class ScenePlanner:
    """Plans scenes with visual continuity and cohesive image prompts."""

    # Predefined visual themes for consistency
    VISUAL_THEMES = {
        "tech": {
            "color_palette": "deep blues, electric cyan, and subtle purple accents on dark backgrounds",
            "lighting": "soft neon glow with dramatic rim lighting",
            "style": "modern, sleek, futuristic aesthetic with clean lines",
            "recurring_elements": [
                "glowing circuit patterns",
                "holographic interfaces",
                "geometric shapes",
            ],
        },
        "business": {
            "color_palette": "professional navy, warm gold accents, and clean whites",
            "lighting": "bright, professional studio lighting with soft shadows",
            "style": "corporate, polished, trustworthy aesthetic",
            "recurring_elements": [
                "rising charts",
                "connected nodes",
                "professional environments",
            ],
        },
        "education": {
            "color_palette": "friendly teal, warm orange accents, and soft cream backgrounds",
            "lighting": "warm, inviting natural light",
            "style": "approachable, clear, diagram-like aesthetic",
            "recurring_elements": [
                "flowing connections",
                "building blocks",
                "lightbulb moments",
            ],
        },
        "creative": {
            "color_palette": "vibrant gradients, bold primary colors, and artistic splashes",
            "lighting": "dynamic, dramatic lighting with bold contrasts",
            "style": "artistic, expressive, energetic aesthetic",
            "recurring_elements": [
                "paint splashes",
                "creative tools",
                "evolving patterns",
            ],
        },
        "nature": {
            "color_palette": "natural greens, earth tones, and sky blues",
            "lighting": "golden hour sunlight, natural outdoor lighting",
            "style": "organic, flowing, natural aesthetic",
            "recurring_elements": [
                "growing plants",
                "flowing water",
                "natural landscapes",
            ],
        },
        "minimal": {
            "color_palette": "monochromatic with single accent color, lots of white space",
            "lighting": "clean, even, shadowless lighting",
            "style": "minimalist, modern, uncluttered aesthetic",
            "recurring_elements": ["simple icons", "clean shapes", "centered subjects"],
        },
    }

    def __init__(self, ai_client: Optional[AIClient] = None):
        self.ai_client = ai_client or get_ai_client()

    def plan_scenes(self, script: Script, request: VideoRequest) -> list[PlannedScene]:
        """
        Create detailed scene plans with visual continuity.

        Args:
            script: The video script with scenes
            request: Video request with format info

        Returns:
            List of planned scenes with cohesive image prompts
        """
        print(f"🎬 Planning {len(script.scenes)} scenes with visual continuity...")

        # Step 1: Analyze script to determine best visual theme
        visual_theme = self._determine_visual_theme(script, request)
        print(f"  🎨 Visual theme: {visual_theme['name']}")

        # Step 2: Extract recurring motifs from all scenes
        recurring_motifs = self._extract_recurring_motifs(script, visual_theme)
        print(f"  🔄 Recurring motifs: {', '.join(recurring_motifs[:3])}")

        # Step 3: Plan each scene with continuity context
        planned_scenes = []
        previous_scene_context = None

        for i, scene in enumerate(script.scenes):
            print(f"  📍 Planning scene {scene.scene_number}/{len(script.scenes)}...")

            # Create detailed image prompt with continuity
            image_prompt = self._create_cohesive_image_prompt(
                scene=scene,
                scene_index=i,
                total_scenes=len(script.scenes),
                visual_theme=visual_theme,
                recurring_motifs=recurring_motifs,
                previous_context=previous_scene_context,
                request=request,
            )

            # Determine transition type
            transition = self._determine_transition(i, len(script.scenes), scene.mood)

            planned_scene = PlannedScene(
                scene_number=scene.scene_number,
                narration=scene.narration,
                visual_description=scene.visual_description,
                image_prompt=image_prompt,
                duration_seconds=scene.duration_seconds,
                mood=scene.mood,
                transition=transition,
            )

            planned_scenes.append(planned_scene)

            # Update context for next scene
            previous_scene_context = {
                "visual_description": scene.visual_description,
                "mood": scene.mood,
                "key_elements": scene.key_visual_elements,
            }

        print(f"✅ All {len(planned_scenes)} scenes planned with visual continuity")
        return planned_scenes

    def _determine_visual_theme(self, script: Script, request: VideoRequest) -> dict:
        """
        Analyze the script and request to determine the best visual theme.

        Returns a theme dictionary with color palette, lighting, style, etc.
        """
        topic_lower = request.topic.lower()
        style_lower = request.style.lower()

        # Match topic/style to predefined themes
        theme_name = "education"  # Default

        if any(
            word in topic_lower
            for word in [
                "code",
                "programming",
                "software",
                "api",
                "docker",
                "tech",
                "computer",
                "data",
                "ai",
                "machine learning",
            ]
        ):
            theme_name = "tech"
        elif any(
            word in topic_lower
            for word in [
                "business",
                "marketing",
                "sales",
                "finance",
                "startup",
                "entrepreneur",
            ]
        ):
            theme_name = "business"
        elif any(
            word in topic_lower
            for word in ["art", "design", "creative", "music", "video", "photo"]
        ):
            theme_name = "creative"
        elif any(
            word in topic_lower
            for word in ["nature", "environment", "health", "fitness", "outdoor"]
        ):
            theme_name = "nature"
        elif any(word in style_lower for word in ["minimal", "simple", "clean"]):
            theme_name = "minimal"

        theme = self.VISUAL_THEMES[theme_name].copy()
        theme["name"] = theme_name
        theme["topic"] = request.topic

        return theme

    def _extract_recurring_motifs(
        self, script: Script, visual_theme: dict
    ) -> list[str]:
        """
        Extract and determine recurring visual motifs from the script.

        These will appear across multiple scenes for visual continuity.
        """
        motifs = []

        # Start with theme's recurring elements
        motifs.extend(visual_theme.get("recurring_elements", []))

        # Extract common elements from scene visual descriptions
        all_elements = []
        for scene in script.scenes:
            all_elements.extend(scene.key_visual_elements)

        # Find elements that appear multiple times or are particularly important
        element_counts = {}
        for element in all_elements:
            element_lower = element.lower()
            element_counts[element_lower] = element_counts.get(element_lower, 0) + 1

        # Add frequently mentioned elements
        for element, count in sorted(element_counts.items(), key=lambda x: -x[1]):
            if count >= 1 and element not in [m.lower() for m in motifs]:
                motifs.append(element)
            if len(motifs) >= 5:
                break

        return motifs[:5]

    def _create_cohesive_image_prompt(
        self,
        scene,
        scene_index: int,
        total_scenes: int,
        visual_theme: dict,
        recurring_motifs: list[str],
        previous_context: Optional[dict],
        request: VideoRequest,
    ) -> str:
        """
        Create a detailed, cohesive image prompt that maintains visual continuity.
        """
        prompt_parts = []

        # === SCENE POSITION CONTEXT ===
        if scene_index == 0:
            position_context = "Opening scene - establish the visual world"
            composition_hint = "Wide establishing shot or centered focal point"
        elif scene_index == total_scenes - 1:
            position_context = "Final scene - visual conclusion/resolution"
            composition_hint = "Satisfying closure, completed journey"
        else:
            progress = (scene_index + 1) / total_scenes
            if progress < 0.4:
                position_context = "Early scene - building/expanding the visual world"
                composition_hint = "Building complexity, adding layers"
            elif progress < 0.7:
                position_context = "Middle scene - peak visual complexity"
                composition_hint = "Full visual richness, dynamic composition"
            else:
                position_context = "Late scene - converging toward resolution"
                composition_hint = "Coming together, simplifying toward conclusion"

        # === MAIN SUBJECT ===
        prompt_parts.append(
            f"SCENE {scene_index + 1}/{total_scenes}: {scene.visual_description}"
        )

        # === VISUAL CONTINUITY INSTRUCTIONS ===
        prompt_parts.append(f"\n--- VISUAL CONTINUITY ({position_context}) ---")

        # Describe what should persist from previous scene
        if previous_context:
            prompt_parts.append(
                f"Continue from previous scene's visual style. "
                f"Previous mood: {previous_context['mood']}."
            )

        # === STYLE REQUIREMENTS ===
        prompt_parts.append(f"\n--- CONSISTENT STYLE ---")
        prompt_parts.append(f"Color Palette: {visual_theme['color_palette']}")
        prompt_parts.append(f"Lighting: {visual_theme['lighting']}")
        prompt_parts.append(f"Overall Style: {visual_theme['style']}")

        # === RECURRING MOTIFS ===
        if recurring_motifs:
            # Select 1-2 motifs to include in this scene
            motifs_for_scene = recurring_motifs[:2]
            prompt_parts.append(
                f"\n--- RECURRING VISUAL ELEMENTS (include at least one) ---"
            )
            prompt_parts.append(f"Visual motifs: {', '.join(motifs_for_scene)}")

        # === KEY ELEMENTS FROM SCRIPT ===
        if scene.key_visual_elements:
            prompt_parts.append(f"\n--- KEY ELEMENTS FOR THIS SCENE ---")
            prompt_parts.append(f"Must include: {', '.join(scene.key_visual_elements)}")

        # === MOOD ===
        mood_visual_guide = self._get_mood_visual_guide(scene.mood)
        prompt_parts.append(f"\n--- MOOD: {scene.mood.upper()} ---")
        prompt_parts.append(mood_visual_guide)

        # === COMPOSITION ===
        prompt_parts.append(f"\n--- COMPOSITION ---")
        prompt_parts.append(composition_hint)

        if request.format == VideoFormat.SHORT:
            prompt_parts.append("Vertical composition (9:16 aspect ratio)")
            prompt_parts.append("Subject centered for mobile viewing")
            prompt_parts.append(
                "Bold, eye-catching composition visible on small screens"
            )
        else:
            prompt_parts.append("Horizontal composition (16:9 aspect ratio)")
            prompt_parts.append("Cinematic widescreen framing")
            prompt_parts.append("Room for visual storytelling across the frame")

        # === TECHNICAL REQUIREMENTS ===
        prompt_parts.append(f"\n--- TECHNICAL REQUIREMENTS ---")
        prompt_parts.append("High quality, sharp details")
        prompt_parts.append("Professional photography/illustration quality")
        prompt_parts.append("Clean, uncluttered composition with clear focal point")
        prompt_parts.append(
            "Suitable as video background (subtle enough for text overlay)"
        )
        prompt_parts.append(
            "ABSOLUTELY NO TEXT, WORDS, LETTERS, OR NUMBERS IN THE IMAGE"
        )

        return "\n".join(prompt_parts)

    def _get_mood_visual_guide(self, mood: str) -> str:
        """Get visual guidance based on mood."""
        mood_guides = {
            "intriguing": "Mysterious atmosphere, subtle shadows, sense of discovery waiting",
            "curious": "Open, inviting composition, elements that draw the eye deeper",
            "exciting": "Dynamic angles, sense of motion, energetic composition",
            "urgent": "High contrast, dramatic lighting, forward momentum",
            "calm": "Balanced composition, soft gradients, peaceful atmosphere",
            "informative": "Clear visual hierarchy, organized elements, professional clarity",
            "educational": "Diagram-like clarity, logical arrangement, learning-friendly",
            "serious": "Authoritative framing, stable composition, professional tone",
            "friendly": "Warm tones, approachable composition, inviting atmosphere",
            "energetic": "Vibrant colors, dynamic lines, sense of movement",
            "inspiring": "Uplifting composition, aspirational imagery, hopeful lighting",
            "motivational": "Empowering framing, forward-looking perspective, triumphant feel",
            "conclusive": "Resolved composition, satisfying visual closure, complete feeling",
            "challenging": "Bold framing, direct perspective, empowering composition",
        }

        mood_lower = mood.lower()
        for key, guide in mood_guides.items():
            if key in mood_lower:
                return guide

        return "Professional, polished aesthetic appropriate for the content"

    def _determine_transition(
        self, scene_index: int, total_scenes: int, mood: str
    ) -> str:
        """
        Determine the appropriate transition for a scene.
        """
        # First scene - fade in
        if scene_index == 0:
            return "fade_in"

        # Last scene - fade out
        if scene_index == total_scenes - 1:
            return "fade_out"

        # Mood-based transitions for middle scenes
        mood_lower = mood.lower()

        if any(word in mood_lower for word in ["exciting", "energetic", "urgent"]):
            return "cut"  # Quick cuts for energy
        elif any(word in mood_lower for word in ["calm", "peaceful", "conclusive"]):
            return "dissolve"  # Smooth dissolves for calm
        elif any(word in mood_lower for word in ["serious", "dramatic"]):
            return "fade_to_black"  # Dramatic fades

        return "crossfade"  # Default smooth transition

    def refine_image_prompt_with_ai(
        self, base_prompt: str, request: VideoRequest
    ) -> str:
        """
        Use AI to further refine and enhance an image prompt.
        Optional enhancement step.
        """
        prompt = f"""You are an expert at creating prompts for AI image generation.

Enhance this image prompt to create a more visually striking and cohesive result:

Original prompt:
{base_prompt}

Context:
- This is for a {request.format.value}-form video about "{request.topic}"
- Target audience: {request.target_audience}
- Style: {request.style}

Requirements:
- Maintain all the style/continuity instructions from the original
- Add more specific, vivid visual details
- Enhance the atmospheric description
- Keep it under 250 words
- Ensure NO text/words/letters will appear in the image

Return ONLY the enhanced prompt text, nothing else.
"""

        try:
            enhanced = self.ai_client.generate_text(prompt, temperature=0.5)
            return enhanced.strip()
        except Exception as e:
            print(f"⚠️ Prompt refinement failed: {e}")
            return base_prompt

    def estimate_scene_duration(
        self, narration: str, words_per_minute: int = 150
    ) -> float:
        """
        Estimate scene duration based on narration length.
        """
        word_count = len(narration.split())
        duration = (word_count / words_per_minute) * 60

        # Add small buffer for natural pauses
        duration *= 1.1

        # Minimum 3 seconds, maximum 120 seconds per scene
        return max(3.0, min(120.0, duration))

    def adjust_scene_timing(
        self, scenes: list[PlannedScene], target_duration: tuple[int, int]
    ) -> list[PlannedScene]:
        """
        Adjust scene timing to fit within target duration.
        """
        min_duration, max_duration = target_duration
        current_total = sum(s.duration_seconds for s in scenes)

        if min_duration <= current_total <= max_duration:
            return scenes

        # Calculate adjustment factor
        target = (min_duration + max_duration) / 2
        factor = target / current_total if current_total > 0 else 1.0

        # Adjust each scene proportionally
        for scene in scenes:
            scene.duration_seconds = round(scene.duration_seconds * factor, 1)

        return scenes
