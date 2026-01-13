"""
Image Generator Module - Creates AI-generated images for video scenes.

This module handles image generation using AI models (OpenAI DALL-E/GPT-Image)
for creating scene backgrounds and thumbnails.

Fails loud on errors - no fallback images.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

from ..utils.ai_client import AIClient, get_ai_client
from .models import PlannedScene, VideoFormat, VideoRequest


class ImageGenerator:
    """Generates AI images for video scenes."""

    def __init__(
        self, ai_client: Optional[AIClient] = None, output_dir: Optional[Path] = None
    ):
        self.ai_client = ai_client or get_ai_client()
        self.output_dir = output_dir or Path("output/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_scene_image(
        self,
        scene: PlannedScene,
        request: VideoRequest,
        apply_overlay: bool = False,
    ) -> Path:
        """
        Generate an AI image for a scene.

        Args:
            scene: The planned scene with image prompt
            request: Video request with format info
            apply_overlay: Whether to apply darkening overlay (default False for vibrant images)

        Returns:
            Path to the generated image

        Raises:
            Exception: If image generation fails
        """
        print(f"  🎨 Generating image for scene {scene.scene_number}...")

        # Get appropriate size
        size = request.get_image_size()

        # Generate image using AI
        image_bytes = self.ai_client.generate_image(
            prompt=scene.image_prompt,
            size=size,
            quality="medium",
        )

        # Process the image
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # Apply light processing (minimal - subtitles now have their own stroke)
        if apply_overlay:
            image = self._apply_overlay(image, opacity=40)  # Much lighter than before

        # Resize to exact video dimensions
        width, height = request.get_video_dimensions()
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Save the image
        output_path = self.output_dir / f"scene_{scene.scene_number:02d}.png"
        image.convert("RGB").save(output_path, quality=95)

        print(f"  ✅ Image saved: {output_path.name}")
        return output_path

    def generate_thumbnail(
        self,
        title: str,
        request: VideoRequest,
        prompt: Optional[str] = None,
    ) -> Path:
        """
        Generate a thumbnail image for the video.

        Args:
            title: Video title
            request: Video request with format info
            prompt: Optional custom prompt for thumbnail

        Returns:
            Path to the generated thumbnail

        Raises:
            Exception: If thumbnail generation fails
        """
        print("🖼️ Generating thumbnail...")

        # Create thumbnail prompt
        if not prompt:
            prompt = self._create_thumbnail_prompt(title, request)

        size = request.get_image_size()

        # Generate image
        image_bytes = self.ai_client.generate_image(
            prompt=prompt,
            size=size,
            quality="high",
        )

        # Process the image
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # Resize to exact dimensions
        width, height = request.get_video_dimensions()
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Save
        output_path = self.output_dir / "thumbnail.png"
        image.convert("RGB").save(output_path, quality=95)

        print(f"✅ Thumbnail saved: {output_path.name}")
        return output_path

    def _create_thumbnail_prompt(self, title: str, request: VideoRequest) -> str:
        """Create an effective thumbnail prompt."""
        return f"""Create an eye-catching YouTube thumbnail image for a video titled "{title}".

Topic: {request.topic}
Target Audience: {request.target_audience}

Requirements:
- Bold, vibrant colors that stand out
- High contrast and dramatic lighting
- Professional, polished look
- Suitable for {"vertical mobile" if request.format == VideoFormat.SHORT else "horizontal desktop"} viewing
- NO text, NO words, NO letters, NO numbers in the image
- Central focal point that draws the eye
- Clean composition with space for text overlay
"""

    def _apply_overlay(self, image: Image.Image, opacity: int = 40) -> Image.Image:
        """
        Apply a very light darkening overlay for subtle text readability boost.

        Args:
            image: Input image
            opacity: Overlay opacity (0-255). Default 40 (~15%) for vibrant images.
                     Subtitles now have stroke/outline so heavy overlay not needed.

        Returns:
            Image with light overlay applied
        """
        # Skip blur - keep images sharp and vibrant
        # image = image.filter(ImageFilter.GaussianBlur(radius=2))

        # Very light overlay - just enough to slightly improve subtitle contrast
        # without making images gloomy
        if opacity > 0:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, opacity))
            return Image.alpha_composite(image, overlay)

        return image

    def batch_generate(
        self,
        scenes: list[PlannedScene],
        request: VideoRequest,
    ) -> dict[int, Path]:
        """
        Generate images for multiple scenes.

        Args:
            scenes: List of planned scenes
            request: Video request

        Returns:
            Dictionary mapping scene numbers to image paths

        Raises:
            Exception: If any image generation fails
        """
        print(f"🎨 Generating images for {len(scenes)} scenes...")

        results = {}
        for scene in scenes:
            path = self.generate_scene_image(scene, request)
            results[scene.scene_number] = path
            # Update scene with path
            scene.image_path = path

        print(f"✅ Generated {len(results)} images")
        return results

    def cleanup(self) -> None:
        """Remove all generated images."""
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.png"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"⚠️ Failed to delete {file}: {e}")
