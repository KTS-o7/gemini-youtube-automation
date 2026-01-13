# AI Video Generation Pipeline - Implementation Plan

## Overview

A flexible, AI-powered video generation pipeline that accepts a topic/question, researches the web for relevant content, generates scripts, creates scenes with AI-generated visuals and voiceovers, and produces polished videos.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO GENERATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   INPUT     │───▶│   RESEARCH  │───▶│   SCRIPT    │───▶│   SCENES    │      │
│  │             │    │             │    │             │    │             │      │
│  │ • Topic     │    │ • Web Search│    │ • Structure │    │ • Breakdown │      │
│  │ • Audience  │    │ • Curation  │    │ • Narration │    │ • Timing    │      │
│  │ • Format    │    │ • Summary   │    │ • Flow      │    │ • Visuals   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                                 │
│                                         │                                       │
│                                         ▼                                       │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐     │
│  │   OUTPUT    │◀───│   STITCH    │◀───│      SCENE GENERATION           │     │
│  │             │    │             │    │                                 │     │
│  │ • MP4 File  │    │ • Concat    │    │  For each scene:                │     │
│  │ • Metadata  │    │ • Audio Mix │    │  • Generate AI Image            │     │
│  │ • Thumbnail │    │ • Transitions│   │  • Generate Voice (TTS)         │     │
│  └─────────────┘    └─────────────┘    │  • Sync timing                  │     │
│                                        └─────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Input Processing

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | Yes | The main topic or question to cover |
| `target_audience` | string | Yes | Who the video is for (e.g., "beginners", "developers", "kids") |
| `format` | enum | Yes | `"short"` (< 60s, 9:16) or `"long"` (3-10 min, 16:9) |
| `style` | string | No | Visual/tone style (e.g., "professional", "casual", "educational") |
| `language` | string | No | Output language (default: "en") |

### Example Input

```python
{
    "topic": "How do transformers work in AI?",
    "target_audience": "software developers new to machine learning",
    "format": "long",
    "style": "educational with analogies",
    "language": "en"
}
```

### Implementation

```python
# src/pipeline/input_processor.py

@dataclass
class VideoRequest:
    topic: str
    target_audience: str
    format: Literal["short", "long"]
    style: str = "educational"
    language: str = "en"
    
    def validate(self) -> bool:
        """Validate input parameters"""
        pass
    
    def get_video_dimensions(self) -> tuple[int, int]:
        """Return (width, height) based on format"""
        return (1920, 1080) if self.format == "long" else (1080, 1920)
    
    def get_target_duration(self) -> tuple[int, int]:
        """Return (min_seconds, max_seconds)"""
        return (180, 600) if self.format == "long" else (30, 60)
```

---

## Stage 2: Web Research & Content Curation

### Purpose
Search the web for relevant, up-to-date information about the topic to ensure accuracy and depth.

### Data Sources

1. **Web Search API** (primary)
   - OpenAI Web Search tool
   - Or: SerpAPI, Tavily, Brave Search API

2. **Fallback Sources**
   - Wikipedia API
   - ArXiv (for technical topics)
   - YouTube transcript extraction

### Research Pipeline

```python
# src/pipeline/researcher.py

class ContentResearcher:
    def __init__(self, search_provider: str = "openai"):
        self.search_provider = search_provider
    
    async def research(self, request: VideoRequest) -> ResearchResult:
        """
        1. Generate search queries from topic
        2. Execute web searches
        3. Extract and summarize relevant content
        4. Fact-check key claims
        5. Return curated research
        """
        
        # Step 1: Generate smart search queries
        queries = self._generate_search_queries(request.topic, request.target_audience)
        
        # Step 2: Execute searches
        raw_results = await self._execute_searches(queries)
        
        # Step 3: Extract key information
        extracted = self._extract_content(raw_results)
        
        # Step 4: Summarize and structure
        curated = self._curate_content(extracted, request)
        
        return ResearchResult(
            topic=request.topic,
            key_points=curated.key_points,
            facts=curated.facts,
            examples=curated.examples,
            sources=curated.sources
        )
```

### Output Structure

```python
@dataclass
class ResearchResult:
    topic: str
    key_points: list[str]          # Main points to cover
    facts: list[dict]              # Verified facts with sources
    examples: list[str]            # Real-world examples
    analogies: list[str]           # Helpful analogies for the audience
    sources: list[str]             # Source URLs for attribution
    related_topics: list[str]      # For potential follow-up content
```

### OpenAI Web Search Implementation

```python
async def search_with_openai(self, query: str) -> dict:
    """Use OpenAI's web search tool"""
    response = self.client.responses.create(
        model="gpt-5-mini",
        tools=[{"type": "web_search"}],
        input=f"Search for accurate, recent information about: {query}"
    )
    return self._parse_search_response(response)
```

---

## Stage 3: Script Generation

### Purpose
Transform research into a compelling, well-structured script tailored to the audience and format.

### Script Structure

#### Long-form (3-10 minutes)
```
1. Hook (5-10 seconds) - Grab attention
2. Introduction (15-30 seconds) - Set context
3. Main Content (2-8 minutes) - Core information in sections
4. Summary (15-30 seconds) - Recap key points
5. Call to Action (5-10 seconds) - Subscribe/like/comment
```

#### Short-form (30-60 seconds)
```
1. Hook (2-3 seconds) - Immediate attention grabber
2. Core Message (20-45 seconds) - One key insight
3. Punchline/CTA (5-10 seconds) - Memorable ending
```

### Implementation

```python
# src/pipeline/script_writer.py

class ScriptWriter:
    def generate_script(self, request: VideoRequest, research: ResearchResult) -> Script:
        """Generate a complete script from research"""
        
        prompt = f"""
        Create a {request.format}-form video script about "{request.topic}".
        
        Target Audience: {request.target_audience}
        Style: {request.style}
        
        Research/Key Points:
        {research.key_points}
        
        Facts to include:
        {research.facts}
        
        Examples to use:
        {research.examples}
        
        Requirements:
        - Write in a conversational, engaging tone
        - Use analogies appropriate for the audience
        - Include natural pauses and emphasis markers
        - Break into clear scenes/sections
        - Each scene should have: narration text, visual description, duration estimate
        
        Output as JSON with this structure:
        {{
            "title": "...",
            "hook": "...",
            "scenes": [
                {{
                    "scene_number": 1,
                    "narration": "What the narrator says...",
                    "visual_description": "What should be shown...",
                    "duration_seconds": 15,
                    "mood": "intriguing/exciting/calm/etc"
                }}
            ],
            "total_duration_seconds": ...,
            "hashtags": ["...", "..."]
        }}
        """
        
        return self._parse_script_response(generate_content(prompt))
```

### Script Output

```python
@dataclass
class Scene:
    scene_number: int
    narration: str                 # Text for TTS
    visual_description: str        # Description for image generation
    duration_seconds: float        # Estimated duration
    mood: str                      # Emotional tone
    key_visual_elements: list[str] # Specific elements to include

@dataclass
class Script:
    title: str
    hook: str
    scenes: list[Scene]
    total_duration_seconds: int
    hashtags: list[str]
    thumbnail_prompt: str
```

---

## Stage 4: Scene Breakdown

### Purpose
Refine each scene with specific visual prompts and timing based on the generated narration audio.

### Implementation

```python
# src/pipeline/scene_planner.py

class ScenePlanner:
    def plan_scenes(self, script: Script, request: VideoRequest) -> list[PlannedScene]:
        """
        For each scene:
        1. Generate TTS audio to get exact duration
        2. Create detailed image prompt
        3. Determine transitions
        """
        
        planned_scenes = []
        
        for scene in script.scenes:
            # Generate audio first to get exact timing
            audio_path, duration = self.tts_generator.generate(scene.narration)
            
            # Generate detailed image prompt based on visual description
            image_prompt = self._create_image_prompt(
                scene.visual_description,
                scene.mood,
                request.format
            )
            
            planned_scenes.append(PlannedScene(
                scene_number=scene.scene_number,
                narration=scene.narration,
                audio_path=audio_path,
                duration_seconds=duration,
                image_prompt=image_prompt,
                transition="crossfade"  # or "cut", "fade_to_black", etc.
            ))
        
        return planned_scenes
```

---

## Stage 5: Asset Generation (Per Scene)

### 5.1 Image Generation

```python
# src/pipeline/image_generator.py

class SceneImageGenerator:
    def generate_scene_image(self, scene: PlannedScene, request: VideoRequest) -> str:
        """Generate AI image for a scene"""
        
        size = "1536x1024" if request.format == "long" else "1024x1536"
        
        # Enhance prompt for better results
        enhanced_prompt = f"""
        {scene.image_prompt}
        
        Style: Cinematic, high quality, {scene.mood} mood
        Lighting: Professional, dramatic
        NO text, NO words, NO letters in the image
        Suitable for video background with potential text overlay
        """
        
        result = self.openai_client.images.generate(
            model="gpt-image-1-mini",
            prompt=enhanced_prompt,
            size=size,
            quality="medium",
            n=1
        )
        
        # Save and return path
        image_path = self._save_image(result.data[0].b64_json, scene.scene_number)
        return image_path
```

### 5.2 Voice Generation

```python
# src/pipeline/voice_generator.py

class VoiceGenerator:
    def __init__(self, provider: str = "gtts"):
        """
        Providers:
        - "gtts": Free, decent quality
        - "openai": OpenAI TTS (gpt-4o-mini-tts)
        - "elevenlabs": Premium quality
        """
        self.provider = provider
    
    def generate(self, text: str, output_path: Path) -> tuple[Path, float]:
        """Generate voice audio and return (path, duration_seconds)"""
        
        if self.provider == "openai":
            return self._generate_openai_tts(text, output_path)
        elif self.provider == "elevenlabs":
            return self._generate_elevenlabs(text, output_path)
        else:
            return self._generate_gtts(text, output_path)
    
    def _generate_openai_tts(self, text: str, output_path: Path) -> tuple[Path, float]:
        """Use OpenAI's TTS model"""
        response = self.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",  # or: echo, fable, onyx, nova, shimmer
            input=text
        )
        response.stream_to_file(output_path)
        return output_path, self._get_audio_duration(output_path)
```

---

## Stage 6: Video Stitching

### Purpose
Combine all scene images and audio into a final video with transitions and background music.

### Implementation

```python
# src/pipeline/video_composer.py

class VideoComposer:
    def compose(
        self,
        scenes: list[PlannedScene],
        request: VideoRequest,
        output_path: Path
    ) -> Path:
        """Stitch all scenes into final video"""
        
        clips = []
        
        for scene in scenes:
            # Create image clip with duration matching audio
            image_clip = (
                ImageClip(scene.image_path)
                .set_duration(scene.duration_seconds)
                .set_audio(AudioFileClip(scene.audio_path))
            )
            
            # Add Ken Burns effect (subtle zoom/pan)
            image_clip = self._add_ken_burns(image_clip)
            
            # Add fade transitions
            image_clip = image_clip.fadein(0.5).fadeout(0.5)
            
            clips.append(image_clip)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add background music
        if self.background_music_path.exists():
            final_video = self._add_background_music(final_video)
        
        # Export
        final_video.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac"
        )
        
        return output_path
    
    def _add_ken_burns(self, clip: ImageClip) -> ImageClip:
        """Add subtle zoom effect for more dynamic visuals"""
        # Implement slow zoom from 100% to 110% over clip duration
        pass
```

---

## Stage 7: Output & Metadata

### Final Outputs

```python
@dataclass
class VideoOutput:
    video_path: Path               # Final MP4 file
    thumbnail_path: Path           # Generated thumbnail
    metadata: VideoMetadata        # Title, description, tags

@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: list[str]
    hashtags: list[str]
    duration_seconds: int
    format: str
    sources: list[str]             # Attribution for research
```

### Thumbnail Generation

```python
def generate_thumbnail(self, script: Script, request: VideoRequest) -> Path:
    """Generate eye-catching thumbnail"""
    
    prompt = f"""
    Create a YouTube thumbnail for a video titled "{script.title}".
    
    Requirements:
    - Bold, eye-catching visuals
    - High contrast colors
    - Suitable for {request.format}-form content
    - Topic: {request.topic}
    - NO text in the image (text will be added separately)
    """
    
    # Generate and return thumbnail
    pass
```

---

## Complete Pipeline Orchestration

```python
# src/pipeline/orchestrator.py

class VideoPipeline:
    def __init__(self):
        self.researcher = ContentResearcher()
        self.script_writer = ScriptWriter()
        self.scene_planner = ScenePlanner()
        self.image_generator = SceneImageGenerator()
        self.voice_generator = VoiceGenerator()
        self.video_composer = VideoComposer()
    
    async def generate_video(self, request: VideoRequest) -> VideoOutput:
        """Main pipeline orchestration"""
        
        print(f"🎬 Starting video generation for: {request.topic}")
        
        # Stage 1: Input validation
        request.validate()
        print("✅ Input validated")
        
        # Stage 2: Research
        print("🔍 Researching topic...")
        research = await self.researcher.research(request)
        print(f"✅ Found {len(research.key_points)} key points")
        
        # Stage 3: Script generation
        print("📝 Generating script...")
        script = self.script_writer.generate_script(request, research)
        print(f"✅ Script generated: {len(script.scenes)} scenes")
        
        # Stage 4: Scene planning
        print("🎬 Planning scenes...")
        planned_scenes = self.scene_planner.plan_scenes(script, request)
        print("✅ Scenes planned with timing")
        
        # Stage 5: Asset generation
        print("🎨 Generating assets...")
        for scene in planned_scenes:
            print(f"  Scene {scene.scene_number}: Generating image...")
            scene.image_path = self.image_generator.generate_scene_image(scene, request)
            
            print(f"  Scene {scene.scene_number}: Generating voice...")
            scene.audio_path, scene.duration = self.voice_generator.generate(
                scene.narration, 
                self._get_audio_path(scene)
            )
        print("✅ All assets generated")
        
        # Stage 6: Video composition
        print("🎥 Composing video...")
        output_path = self._get_output_path(request)
        video_path = self.video_composer.compose(planned_scenes, request, output_path)
        print("✅ Video composed")
        
        # Stage 7: Generate metadata and thumbnail
        print("🖼️ Generating thumbnail...")
        thumbnail_path = self._generate_thumbnail(script, request)
        metadata = self._generate_metadata(script, research, request)
        
        print(f"🎉 Video complete: {video_path}")
        
        return VideoOutput(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            metadata=metadata
        )
```

---

## CLI Interface

```python
# main.py

import argparse
from src.pipeline import VideoPipeline, VideoRequest

def main():
    parser = argparse.ArgumentParser(description="AI Video Generator")
    parser.add_argument("--topic", required=True, help="Video topic or question")
    parser.add_argument("--audience", required=True, help="Target audience")
    parser.add_argument("--format", choices=["short", "long"], required=True)
    parser.add_argument("--style", default="educational")
    
    args = parser.parse_args()
    
    request = VideoRequest(
        topic=args.topic,
        target_audience=args.audience,
        format=args.format,
        style=args.style
    )
    
    pipeline = VideoPipeline()
    result = asyncio.run(pipeline.generate_video(request))
    
    print(f"\n✅ Video saved to: {result.video_path}")
    print(f"📋 Metadata: {result.metadata}")

if __name__ == "__main__":
    main()
```

### Example Usage

```bash
# Generate a short-form video
python main.py \
  --topic "What is quantum computing?" \
  --audience "general audience, non-technical" \
  --format short

# Generate a long-form video
python main.py \
  --topic "How to build a REST API with FastAPI" \
  --audience "Python developers" \
  --format long \
  --style "tutorial with code examples"
```

---

## File Structure

```
gemini-youtube-automation/
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main pipeline coordination
│   │   ├── input_processor.py   # Input validation & processing
│   │   ├── researcher.py        # Web search & content curation
│   │   ├── script_writer.py     # Script generation
│   │   ├── scene_planner.py     # Scene breakdown & planning
│   │   ├── image_generator.py   # AI image generation
│   │   ├── voice_generator.py   # TTS audio generation
│   │   └── video_composer.py    # Video stitching & export
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py           # VideoRequest dataclass
│   │   ├── research.py          # ResearchResult dataclass
│   │   ├── script.py            # Script, Scene dataclasses
│   │   └── output.py            # VideoOutput dataclass
│   └── utils/
│       ├── __init__.py
│       ├── ai_client.py         # OpenAI/Gemini client wrapper
│       └── file_utils.py        # File handling utilities
├── assets/
│   ├── fonts/
│   └── music/
├── output/                      # Generated videos
├── main.py                      # CLI entry point
├── requirements.txt
└── .env
```

---

## Environment Variables

```bash
# .env

# AI Provider (openai or gemini)
AI_PROVIDER=openai

# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...        # For Gemini

# Optional
PEXELS_API_KEY=...        # Fallback images
ELEVENLABS_API_KEY=...    # Premium TTS

# Feature Flags
USE_AI_BACKGROUNDS=true
USE_WEB_SEARCH=true
TTS_PROVIDER=openai       # gtts, openai, or elevenlabs
```

---

## Dependencies

```txt
# requirements.txt

# AI APIs
openai>=1.0.0
google-genai

# Video Processing
moviepy==1.0.3
Pillow>=9.5.0

# Audio
gTTS
pydub

# Web & Utils
requests
python-dotenv
aiohttp                   # For async web requests

# CLI
argparse
rich                      # Pretty console output
```

---

## Future Enhancements

1. **Multiple voice options** - Different AI voices for variety
2. **B-roll integration** - Mix AI images with stock video
3. **Subtitle generation** - Auto-generate captions
4. **Multi-language support** - Translate and generate in different languages
5. **Template system** - Pre-defined styles and formats
6. **Batch processing** - Generate multiple videos from a topic list
7. **YouTube auto-upload** - Direct upload after generation
8. **Analytics feedback loop** - Learn from video performance

---

## Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| Phase 1 | Input processing, basic pipeline structure | 1 day |
| Phase 2 | Web research integration | 1-2 days |
| Phase 3 | Script generation & scene planning | 1 day |
| Phase 4 | Image & voice generation | 1 day |
| Phase 5 | Video composition & stitching | 1 day |
| Phase 6 | CLI, testing, polish | 1 day |

**Total: ~6-7 days**