"""
Scene Planner Module - Creates visually cohesive scene plans with continuity.

This module takes a script and creates detailed plans for each scene,
ensuring visual consistency across all scenes through:
- Consistent visual themes and color palettes
- Recurring visual motifs
- Coherent style progression

KEN BURNS OPTIMIZATION:
- 🖼️ Static image descriptions optimized for zoom/pan effects
- 📐 Layered compositions with foreground/midground/background
- 🎯 Clear focal points that work with slow reveals

VIRAL SHORTS OPTIMIZATION:
- ⚡ Fast scene pacing (max 5-8 seconds per scene for shorts)
- 🎬 Quick transitions for engagement
- 📱 Mobile-optimized visual compositions
"""

from typing import Optional

from ..utils.ai_client import AIClient, get_ai_client
from .models import PlannedScene, Script, VideoFormat, VideoRequest
from .prompts import PromptLoader


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
        is_short = request.format == VideoFormat.SHORT

        if is_short:
            print(
                f"🎬 Planning {len(script.scenes)} scenes for VIRAL SHORT (max 8s/scene)..."
            )
        else:
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

            # Determine transition type (faster for shorts)
            transition = self._determine_transition(i, len(script.scenes), scene.mood)

            # For viral shorts, prefer faster transitions
            if is_short and transition in ["dissolve", "fade_to_black"]:
                transition = "crossfade"

            # Calculate duration with viral optimization for shorts
            duration = self.estimate_scene_duration(scene.narration, is_short=is_short)

            planned_scene = PlannedScene(
                scene_number=scene.scene_number,
                narration=scene.narration,
                visual_description=scene.visual_description,
                image_prompt=image_prompt,
                duration_seconds=duration,
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

        # Apply viral optimization for shorts
        if is_short:
            planned_scenes = self.optimize_for_viral(planned_scenes)

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
        Create a detailed, cohesive image prompt optimized for Ken Burns effects.

        Ken Burns Optimization:
        - Describes STATIC images only (no animation language)
        - Includes depth layers for zoom effects to reveal
        - Clear focal points for pan effects
        - Composition that works with slow zoom in OR zoom out

        Prompts are loaded from external template files for easy tweaking.
        """
        # === CLEAN THE VISUAL DESCRIPTION ===
        # Remove animation words from the script's visual description
        cleaned_visual = self._clean_animation_words(scene.visual_description)

        # === SCENE POSITION CONTEXT (affects Ken Burns direction) ===
        if scene_index == 0:
            position_context = "Opening scene"
            composition_hint = "Wide establishing shot with rich background detail - zoom IN will reveal the main subject"
            ken_burns_hint = "Composition should work with ZOOM IN: interesting background that leads eye to central focal point"
        elif scene_index == total_scenes - 1:
            position_context = "Final scene"
            composition_hint = "Centered focal point with context around it - zoom OUT will reveal the bigger picture"
            ken_burns_hint = "Composition should work with ZOOM OUT: strong center that reveals surrounding context"
        else:
            progress = (scene_index + 1) / total_scenes
            if progress < 0.4:
                position_context = "Early scene"
                composition_hint = "Building complexity with layered depth"
                ken_burns_hint = "Include foreground, midground, and background elements for zoom to traverse"
            elif progress < 0.7:
                position_context = "Middle scene"
                composition_hint = (
                    "Peak visual richness with multiple points of interest"
                )
                ken_burns_hint = "Rich detail throughout - zoom effect will slowly reveal different areas"
            else:
                position_context = "Late scene"
                composition_hint = (
                    "Converging toward resolution, simplified but detailed"
                )
                ken_burns_hint = (
                    "Clear visual hierarchy - zoom will emphasize the key element"
                )

        # === BUILD CONTINUITY INSTRUCTION ===
        if previous_context:
            continuity_instruction = (
                f"Must feel like the NEXT FRAME from the same video. "
                f"Previous scene mood: {previous_context['mood']}."
            )
        else:
            continuity_instruction = "Establish the visual world for this video series."

        # === FORMAT-SPECIFIC SETTINGS ===
        if request.format == VideoFormat.SHORT:
            aspect_ratio = "9:16 (vertical/portrait)"
            format_specific_instructions = (
                "Subject centered for mobile viewing\n"
                "Bold, eye-catching focal point visible on small screens\n"
                "Leave some headroom for text overlay"
            )
        else:
            aspect_ratio = "16:9 (horizontal/landscape)"
            format_specific_instructions = (
                "Cinematic widescreen framing with rule of thirds\n"
                "Visual weight balanced across the frame\n"
                "Lower third area suitable for text overlay"
            )

        # === LOAD AND FORMAT TEMPLATE ===
        try:
            template = PromptLoader.load("scene_planner_image_prompt")
            return template.format(
                scene_number=scene_index + 1,
                total_scenes=total_scenes,
                cleaned_visual_description=cleaned_visual,
                position_context=position_context,
                ken_burns_hint=ken_burns_hint,
                continuity_instruction=continuity_instruction,
                color_palette=visual_theme["color_palette"],
                lighting=visual_theme["lighting"],
                visual_style=visual_theme["style"],
                recurring_motifs=", ".join(recurring_motifs[:2])
                if recurring_motifs
                else "None specified",
                key_visual_elements=", ".join(scene.key_visual_elements)
                if scene.key_visual_elements
                else "None specified",
                mood=scene.mood.upper(),
                mood_visual_guide=self._get_mood_visual_guide(scene.mood),
                composition_hint=composition_hint,
                aspect_ratio=aspect_ratio,
                format_specific_instructions=format_specific_instructions,
            )
        except (FileNotFoundError, KeyError):
            # Fallback to inline prompt if template not found or has missing keys
            return self._create_fallback_image_prompt(
                cleaned_visual=cleaned_visual,
                scene_index=scene_index,
                total_scenes=total_scenes,
                position_context=position_context,
                ken_burns_hint=ken_burns_hint,
                composition_hint=composition_hint,
                continuity_instruction=continuity_instruction,
                visual_theme=visual_theme,
                recurring_motifs=recurring_motifs,
                scene=scene,
                aspect_ratio=aspect_ratio,
                format_specific_instructions=format_specific_instructions,
            )

    def _create_fallback_image_prompt(
        self,
        cleaned_visual: str,
        scene_index: int,
        total_scenes: int,
        position_context: str,
        ken_burns_hint: str,
        composition_hint: str,
        continuity_instruction: str,
        visual_theme: dict,
        recurring_motifs: list[str],
        scene,
        aspect_ratio: str,
        format_specific_instructions: str,
    ) -> str:
        """Fallback prompt generation if template file is not available."""
        prompt_parts = []

        # === STATIC IMAGE INSTRUCTION (CRITICAL) ===
        prompt_parts.append(
            "Generate a SINGLE STATIC IMAGE (photograph or illustration)."
        )
        prompt_parts.append(
            "This is NOT an animation - describe only what is VISIBLE in ONE frozen moment."
        )
        prompt_parts.append("")

        # === MAIN SUBJECT ===
        prompt_parts.append(f"SCENE {scene_index + 1}/{total_scenes}: {cleaned_visual}")

        # === KEN BURNS OPTIMIZATION ===
        prompt_parts.append(
            f"\n--- COMPOSITION FOR SLOW ZOOM/PAN ({position_context}) ---"
        )
        prompt_parts.append(ken_burns_hint)
        prompt_parts.append("Include DEPTH LAYERS for zoom to reveal:")
        prompt_parts.append(
            "  • Foreground: Elements closest to viewer (can be slightly blurred)"
        )
        prompt_parts.append("  • Midground: Main subject with sharp focus")
        prompt_parts.append("  • Background: Context and atmosphere (softer focus)")

        # === VISUAL CONTINUITY INSTRUCTIONS ===
        prompt_parts.append(f"\n--- VISUAL CONTINUITY ---")
        prompt_parts.append(continuity_instruction)

        # === STYLE REQUIREMENTS ===
        prompt_parts.append(f"\n--- CONSISTENT STYLE (same across all scenes) ---")
        prompt_parts.append(f"Color Palette: {visual_theme['color_palette']}")
        prompt_parts.append(f"Lighting: {visual_theme['lighting']}")
        prompt_parts.append(f"Visual Style: {visual_theme['style']}")

        # === RECURRING MOTIFS ===
        if recurring_motifs:
            motifs_for_scene = recurring_motifs[:2]
            prompt_parts.append(f"\n--- RECURRING VISUAL ELEMENTS ---")
            prompt_parts.append(f"Include at least one: {', '.join(motifs_for_scene)}")

        # === KEY ELEMENTS FROM SCRIPT ===
        if scene.key_visual_elements:
            prompt_parts.append(f"\n--- SCENE-SPECIFIC ELEMENTS ---")
            prompt_parts.append(f"Must include: {', '.join(scene.key_visual_elements)}")

        # === MOOD ===
        mood_visual_guide = self._get_mood_visual_guide(scene.mood)
        prompt_parts.append(f"\n--- MOOD: {scene.mood.upper()} ---")
        prompt_parts.append(mood_visual_guide)

        # === FORMAT-SPECIFIC COMPOSITION ===
        prompt_parts.append(f"\n--- COMPOSITION ---")
        prompt_parts.append(composition_hint)
        prompt_parts.append(f"Aspect Ratio: {aspect_ratio}")
        prompt_parts.append(format_specific_instructions)

        # === TECHNICAL REQUIREMENTS ===
        prompt_parts.append(f"\n--- TECHNICAL REQUIREMENTS ---")
        prompt_parts.append("High resolution, sharp details on main subject")
        prompt_parts.append(
            "Professional quality (photography or digital illustration)"
        )
        prompt_parts.append("Subtle depth of field to separate layers")
        prompt_parts.append(
            "ABSOLUTELY NO TEXT, WORDS, LETTERS, NUMBERS, OR WATERMARKS"
        )
        prompt_parts.append("")
        prompt_parts.append(
            "REMEMBER: This is a STILL IMAGE, not a video frame or animation."
        )

        return "\n".join(prompt_parts)

    def _clean_animation_words(self, visual_description: str) -> str:
        """
        Remove or replace animation-related words from visual descriptions.

        Converts dynamic descriptions into static image descriptions.
        """
        import re

        # Mapping of animation phrases to static equivalents
        replacements = [
            # Camera movements -> static compositions
            (
                r"camera zooms? (in|out|into|to)",
                "close-up view of" if "in" else "wide shot of",
            ),
            (r"camera (pushes?|pulls?|moves?|pans?|tilts?|tracks?)", "view showing"),
            (r"zoom(s|ing)? (in|out|into)", "detailed view of"),
            (r"pan(s|ning)? (to|across|over)", "wide shot showing"),
            # Animation words -> static equivalents
            (r"animat(ed|ion|ing)", "illustrated"),
            (r"(flies|flying|floats|floating) (out|in|around|through)", "positioned"),
            (r"transform(s|ing|ation)", "shown as"),
            (r"morph(s|ing)", "transitioning between"),
            (r"(moves?|moving) (to|toward|across|through)", "positioned"),
            (r"(spins?|spinning|rotates?|rotating)", "angled view of"),
            (r"(fades?|fading) (in|out)", ""),
            (r"(grows?|growing|shrinks?|shrinking)", ""),
            (r"(appears?|appearing|disappears?|disappearing)", "visible"),
            (r"(unfolds?|unfolding|opens?|opening)", "open"),
            (r"(flows?|flowing)", "arranged in flowing pattern"),
            (r"(pulses?|pulsing|glows? and fades?)", "softly glowing"),
            # Sequence words -> single moment
            (r"(then|next|after that|suddenly)", ""),
            (r"(begins? to|starts? to)", ""),
            (r"(continues? to|keeps?)", ""),
        ]

        result = visual_description
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def _get_mood_visual_guide(self, mood: str) -> str:
        """Get visual guidance based on mood, optimized for static images with Ken Burns."""
        mood_guides = {
            "intriguing": "Mysterious atmosphere with subtle shadows; depth layers that invite exploration; sense of hidden details waiting to be revealed by zoom",
            "curious": "Open, inviting composition with visual pathways that lead the eye; layered elements that reward closer inspection",
            "exciting": "Dynamic diagonal lines and bold angles; high contrast; composition with visual tension even in stillness",
            "urgent": "High contrast dramatic lighting; strong foreground elements; visual weight pushing toward focal point",
            "calm": "Balanced symmetrical or rule-of-thirds composition; soft gradient backgrounds; peaceful negative space",
            "informative": "Clear visual hierarchy with organized elements; professional clarity; distinct foreground subject against contextual background",
            "educational": "Diagram-like clarity with logical spatial arrangement; labeled areas (visually, not with text); learning-friendly layers",
            "serious": "Authoritative centered framing; stable horizontal lines; professional muted tones with depth",
            "friendly": "Warm color temperature; approachable eye-level perspective; inviting soft lighting",
            "energetic": "Vibrant saturated colors; dynamic compositional lines; visual rhythm through repeated elements",
            "inspiring": "Upward-looking perspective or expansive view; aspirational scale; hopeful warm lighting",
            "motivational": "Empowering low-angle or eye-level framing; forward-facing subject; triumphant golden lighting",
            "conclusive": "Resolved centered composition; satisfying visual closure; complete feeling with clear focal point",
            "challenging": "Bold direct framing; confrontational perspective; strong subject presence",
            "playful": "Bright cheerful colors; whimsical arrangements; fun visual surprises in background details",
            "wonder": "Awe-inspiring scale or detail; magical lighting effects; dreamlike atmosphere with depth",
            "satisfying": "Harmonious balanced composition; resolved visual tension; completeness in arrangement",
        }

        mood_lower = mood.lower()
        for key, guide in mood_guides.items():
            if key in mood_lower:
                return guide

        return "Professional, polished aesthetic with clear focal point and layered depth for visual interest"

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
        self, narration: str, words_per_minute: int = 150, is_short: bool = False
    ) -> float:
        """
        Estimate scene duration based on narration length.

        For viral shorts, enforces faster pacing (max 5-8 seconds per scene).

        Args:
            narration: The narration text
            words_per_minute: Speaking rate (higher = faster for shorts)
            is_short: Whether this is a short-form video
        """
        word_count = len(narration.split())

        # Faster speaking rate for shorts (more energetic delivery)
        if is_short:
            words_per_minute = 180  # Faster pace for viral content

        duration = (word_count / words_per_minute) * 60

        # Add small buffer for natural pauses (smaller for shorts)
        if is_short:
            duration *= 1.05  # Tighter timing for shorts
        else:
            duration *= 1.1

        # VIRAL SHORTS: Enforce max 5-8 seconds per scene
        if is_short:
            # Minimum 2 seconds, maximum 8 seconds per scene for shorts
            return max(2.0, min(8.0, duration))
        else:
            # Minimum 3 seconds, maximum 120 seconds per scene for long-form
            return max(3.0, min(120.0, duration))

    def estimate_scene_duration_viral(self, narration: str) -> float:
        """
        Estimate scene duration optimized for viral shorts.

        Enforces strict 5-8 second max per scene for maximum engagement.
        """
        return self.estimate_scene_duration(narration, is_short=True)

    def adjust_scene_timing(
        self,
        scenes: list[PlannedScene],
        target_duration: tuple[int, int],
        is_short: bool = False,
    ) -> list[PlannedScene]:
        """
        Adjust scene timing to fit within target duration.

        For viral shorts, enforces faster pacing with strict per-scene limits.

        Args:
            scenes: List of planned scenes
            target_duration: Tuple of (min_seconds, max_seconds)
            is_short: Whether this is a short-form video
        """
        min_duration, max_duration = target_duration
        current_total = sum(s.duration_seconds for s in scenes)

        # VIRAL SHORTS: Enforce strict per-scene limits
        if is_short:
            max_scene_duration = 8.0  # Maximum 8 seconds per scene
            min_scene_duration = 2.0  # Minimum 2 seconds per scene

            for scene in scenes:
                # Clamp each scene to viral-friendly duration
                scene.duration_seconds = max(
                    min_scene_duration, min(max_scene_duration, scene.duration_seconds)
                )

            # Recalculate total after clamping
            current_total = sum(s.duration_seconds for s in scenes)

            print(
                f"  ⚡ Viral pacing: {len(scenes)} scenes, avg {current_total / len(scenes):.1f}s each"
            )

        if min_duration <= current_total <= max_duration:
            return scenes

        # Calculate adjustment factor
        target = (min_duration + max_duration) / 2
        factor = target / current_total if current_total > 0 else 1.0

        # Adjust each scene proportionally
        for scene in scenes:
            new_duration = scene.duration_seconds * factor

            # For shorts, still respect the max limit
            if is_short:
                new_duration = max(2.0, min(8.0, new_duration))

            scene.duration_seconds = round(new_duration, 1)

        return scenes

    def optimize_for_viral(self, scenes: list[PlannedScene]) -> list[PlannedScene]:
        """
        Optimize scene list for viral short-form content.

        - Enforces max 8 seconds per scene
        - Ensures punchy, fast-paced timing
        - Adjusts transitions for quick cuts
        """
        print("  🔥 Optimizing for viral shorts...")

        for scene in scenes:
            # Cap scene duration at 8 seconds
            if scene.duration_seconds > 8.0:
                scene.duration_seconds = 8.0

            # Ensure minimum 2 seconds
            if scene.duration_seconds < 2.0:
                scene.duration_seconds = 2.0

            # Use faster transitions for viral content
            if scene.transition in ["dissolve", "fade_to_black"]:
                scene.transition = "crossfade"  # Faster transition

        total = sum(s.duration_seconds for s in scenes)
        avg = total / len(scenes) if scenes else 0
        print(
            f"  ⚡ Viral optimization complete: {total:.1f}s total, {avg:.1f}s avg per scene"
        )

        return scenes
