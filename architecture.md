# AI Video Generation Pipeline - Architecture Documentation

## Overview

This project is an **autonomous AI-powered video generation pipeline** that creates engaging educational videos from any topic. It supports both **Google Gemini** and **OpenAI** APIs with a modular, scalable architecture designed for both local development and automated CI/CD deployment via GitHub Actions.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              VIDEO GENERATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   INPUT      │──▶│  RESEARCH    │──▶│   SCRIPT     │──▶│   SCENE      │         │
│  │              │   │              │   │              │   │   PLANNING   │         │
│  │ • Topic      │   │ • AI Search  │   │ • Structure  │   │              │         │
│  │ • Audience   │   │ • Key Points │   │ • Narration  │   │ • Breakdown  │         │
│  │ • Format     │   │ • Facts      │   │ • Scenes     │   │ • Prompts    │         │
│  │ • Style      │   │ • Examples   │   │ • Flow       │   │ • Timing     │         │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘         │
│                                                                                     │
│                                              │                                      │
│                                              ▼                                      │
│                                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────────────┐     │
│  │   OUTPUT     │◀──│   VIDEO      │◀──│        ASSET GENERATION             │     │
│  │              │   │   COMPOSER   │   │                                     │     │
│  │ • MP4 Video  │   │              │   │  For each scene:                    │     │
│  │ • Thumbnail  │   │ • Stitching  │   │  ├─ Generate AI Image (DALL-E)      │     │
│  │ • Metadata   │   │ • Subtitles  │   │  ├─ Generate Voice (TTS)            │     │
│  │ • YouTube    │   │ • Effects    │   │  └─ Word Alignment (Whisper)        │     │
│  └──────────────┘   └──────────────┘   └─────────────────────────────────────┘     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
gemini-youtube-automation/
├── .github/
│   └── workflows/
│       └── main.yml              # GitHub Actions CI/CD pipeline
├── .venv/                        # Python virtual environment
├── assets/
│   ├── fonts/                    # Font files (Arial, Arial Bold, Arial Black)
│   ├── music/                    # Background music files
│   ├── sfx/                      # Sound effects
│   └── fallback.jpg              # Fallback image if AI generation fails
├── output/                       # Generated outputs
│   ├── audio/                    # Voice narration files (*.wav)
│   ├── images/                   # Scene images (scene_01.png, etc.)
│   ├── subtitles/                # Generated subtitles
│   │   └── timestamps/           # Whisper word timestamps (*.json)
│   ├── research.json             # Cached research results
│   ├── script.json               # Generated script
│   ├── planned_scenes.json       # Scene planning data
│   └── video_*.mp4               # Final rendered videos
├── src/
│   ├── __init__.py
│   ├── pipeline/                 # Core video generation pipeline
│   │   ├── __init__.py           # Package exports
│   │   ├── audio_aligner.py      # Whisper word-level alignment
│   │   ├── image_generator.py    # AI image generation (DALL-E)
│   │   ├── models.py             # Data models & enums
│   │   ├── orchestrator.py       # Main pipeline coordinator
│   │   ├── researcher.py         # Content research module
│   │   ├── scene_planner.py      # Scene planning & prompts
│   │   ├── script_writer.py      # Script generation
│   │   ├── video_composer.py     # Video stitching & effects
│   │   └── voice_generator.py    # TTS generation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── ai_client.py          # Unified AI client (OpenAI/Gemini)
│   ├── generator.py              # Legacy generator (still functional)
│   └── uploader.py               # YouTube upload functionality
├── generate_video.py             # Main CLI entry point
├── generate_subtitles.py         # Standalone subtitle generator
├── recompose_video.py            # Recompose video from cached assets
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
├── architecture.md               # This file
└── scratchpad.md                 # Development notes
```

---

## Core Components

### 1. Pipeline Orchestrator (`src/pipeline/orchestrator.py`)

The **central coordinator** that manages the entire video generation workflow.

**Class: `VideoPipeline`**

```python
class VideoPipeline:
    """
    Coordinates all 7 stages of video generation:
    1. Input Validation
    2. Web Research
    3. Script Generation
    4. Scene Planning
    5. Asset Generation (Images + Voice)
    6. Video Composition
    7. Output & Metadata
    """
```

**Key Methods:**
- `generate_video(request: VideoRequest) -> VideoOutput` - Async main entry point
- `generate_video_sync(request: VideoRequest) -> VideoOutput` - Sync wrapper
- `cleanup()` - Remove temporary files

**Helper Function:**
- `create_video(topic, audience, format, style) -> VideoOutput` - Convenience wrapper

---

### 2. Data Models (`src/pipeline/models.py`)

Defines the data structures flowing through the pipeline.

#### Enums
| Enum | Values | Description |
|------|--------|-------------|
| `VideoFormat` | `SHORT`, `LONG` | Short (<60s, 9:16) or Long (3-10min, 16:9) |
| `TTSProvider` | `GTTS`, `OPENAI`, `ELEVENLABS` | Text-to-speech provider |

#### Core Dataclasses

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `VideoRequest` | Input request | `topic`, `target_audience`, `format`, `style`, `language` |
| `ResearchResult` | Research output | `key_points`, `facts`, `examples`, `analogies`, `sources` |
| `Scene` | Single script scene | `scene_number`, `narration`, `visual_description`, `mood` |
| `Script` | Complete script | `title`, `hook`, `scenes`, `hashtags`, `thumbnail_prompt` |
| `PlannedScene` | Scene with assets | `image_prompt`, `audio_path`, `image_path`, `transition` |
| `VideoMetadata` | Output metadata | `title`, `description`, `tags`, `duration_seconds` |
| `VideoOutput` | Final output | `video_path`, `thumbnail_path`, `metadata`, `success` |
| `PipelineState` | Pipeline tracker | `request`, `research`, `script`, `planned_scenes`, `errors` |

---

### 3. AI Client (`src/utils/ai_client.py`)

**Unified interface** for OpenAI and Google Gemini APIs with structured outputs.

#### Configuration (`AIConfig`)
```python
@dataclass
class AIConfig:
    provider: str = "openai"           # "openai" or "gemini"
    openai_model: str = "gpt-5-mini"   # Text generation model
    gemini_model: str = "gemini-2.0-flash"
    image_model: str = "gpt-image-1-mini"  # DALL-E model
    image_quality: str = "low"         # low/medium/high
    tts_model: str = "gpt-4o-mini-tts" # TTS model
    tts_voice: str = "alloy"           # Voice option
```

#### AIClient Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `generate_structured(prompt, response_model)` | Structured JSON output via Pydantic | `T (BaseModel)` |
| `generate_text(prompt, system_prompt)` | Free-form text generation | `str` |
| `generate_image(prompt, size, quality)` | AI image generation | `bytes` |
| `generate_speech(text, voice, speed)` | Text-to-speech | `bytes` |
| `web_search(query)` | Web search (OpenAI) | `dict` |

#### Pydantic Models for Structured Outputs
- `SceneModel` - Single scene structure
- `ScriptModel` - Complete script structure
- `ResearchModel` - Research results structure
- `ResearchFactModel` - Fact with source

---

### 4. Content Researcher (`src/pipeline/researcher.py`)

Generates comprehensive research about the video topic using AI.

**Class: `ContentResearcher`**

**Output Structure:**
```python
ResearchResult(
    topic="Machine Learning",
    key_points=["Point 1", "Point 2", ...],      # 5-7 main points
    facts=[{"fact": "...", "source": "..."}],    # 3-5 verified facts
    examples=["Example 1", "Example 2"],          # 2-4 real-world examples
    analogies=["Analogy 1", "Analogy 2"],         # 2-3 simple analogies
    related_topics=["Topic A", "Topic B"]         # 2-3 follow-up topics
)
```

---

### 5. Script Writer (`src/pipeline/script_writer.py`)

Transforms research into engaging video scripts with narrative continuity.

**Class: `ScriptWriter`**

**Key Features:**
- **Short-form scripts** (30-60 seconds): 3 scenes with hook → content → CTA structure
- **Long-form scripts** (3-10 minutes): 6-8 scenes with full narrative arc
- **Narrative continuity rules**: Transitions, visual consistency, callbacks
- **Visual continuity**: Single visual theme/metaphor throughout

**Script Structure (Short):**
1. **HOOK** (5-8s): Attention-grabbing opening
2. **CORE CONTENT** (35-45s): Main educational value
3. **MEMORABLE ENDING** (8-12s): Callback + CTA

**Script Structure (Long):**
1. **HOOK + CENTRAL QUESTION** (10-15s)
2. **CONTEXT & FOUNDATION** (20-30s)
3-5. **MAIN CONTENT** (2-6 min total)
6. **PRACTICAL APPLICATION** (30-45s)
7. **SYNTHESIS & ANSWER** (20-30s)
8. **CALL TO ACTION** (10-15s)

---

### 6. Scene Planner (`src/pipeline/scene_planner.py`)

Creates visually cohesive scene plans with consistent image prompts.

**Class: `ScenePlanner`**

**Visual Themes:**
| Theme | Color Palette | Style | Elements |
|-------|--------------|-------|----------|
| `tech` | Deep blues, cyan, purple | Futuristic, sleek | Circuits, holograms |
| `business` | Navy, gold, white | Corporate, polished | Charts, nodes |
| `education` | Teal, orange, cream | Approachable, clear | Connections, blocks |
| `creative` | Vibrant gradients | Artistic, energetic | Paint, patterns |
| `nature` | Greens, earth tones | Organic, flowing | Plants, water |
| `minimal` | Monochromatic | Minimalist, modern | Simple icons |

**Key Methods:**
- `plan_scenes(script, request)` - Create detailed scene plans
- `_determine_visual_theme(script)` - Auto-detect theme from content
- `_create_cohesive_image_prompt(scene, theme)` - Generate consistent prompts

---

### 7. Image Generator (`src/pipeline/image_generator.py`)

Generates AI images for scenes and thumbnails using DALL-E.

**Class: `ImageGenerator`**

**Image Sizes by Format:**
| Format | Size | Dimensions |
|--------|------|------------|
| LONG | `1536x1024` | 1920×1080 (16:9) |
| SHORT | `1024x1536` | 1080×1920 (9:16) |

**Key Methods:**
- `generate_scene_image(scene, request)` - Generate scene background
- `generate_thumbnail(title, request, prompt)` - Create thumbnail
- `batch_generate(scenes, request)` - Batch process all scenes

---

### 8. Voice Generator (`src/pipeline/voice_generator.py`)

Converts narration text to speech with multiple provider support.

**Class: `VoiceGenerator`**

**Supported Providers:**

| Provider | Cost | Quality | Notes |
|----------|------|---------|-------|
| `GTTS` | Free | Good | Google Text-to-Speech |
| `OPENAI` | $0.015/min | Excellent | 6 voice options |
| `ELEVENLABS` | Premium | Best | Natural voices |

**OpenAI Voice Options:**
- `alloy` - Neutral, balanced
- `echo` - Warm, conversational
- `fable` - Expressive, storytelling
- `onyx` - Deep, authoritative
- `nova` - Friendly, upbeat
- `shimmer` - Clear, professional

**Output Format:** WAV (converted from MP3 for moviepy compatibility)

---

### 9. Audio Aligner (`src/pipeline/audio_aligner.py`)

Uses OpenAI Whisper for precise word-level timestamp extraction.

**Class: `AudioAligner`**

**Purpose:** Enables karaoke-style subtitle synchronization with exact word timing.

**Key Classes:**
```python
@dataclass
class WordTimestamp:
    word: str
    start: float  # seconds
    end: float    # seconds

@dataclass
class AlignmentResult:
    words: list[WordTimestamp]
    full_text: str
    duration: float
    language: str
```

**Key Methods:**
- `get_word_timestamps(audio_path)` - Extract word-level timing
- `generate_ass_file(result, output_path)` - Create .ass subtitle file
- `save_timestamps_json(result, output_path)` - Cache timestamps

---

### 10. Video Composer (`src/pipeline/video_composer.py`)

Stitches scenes into final video with viral-optimized features.

**Class: `VideoComposer`**

**Video Settings:**
```python
fps = 30                    # Smooth motion
codec = "libx264"           # H.264 encoding
audio_codec = "aac"
audio_bitrate = "192k"      # High quality
preset = "slow"             # Better quality encoding
```

**Viral Subtitle Features:**
| Feature | Description | Default |
|---------|-------------|---------|
| `karaoke_mode` | Word-by-word highlighting | `True` |
| `emoji_enabled` | Add emojis to captions | `False` |
| `color_emphasis` | Highlight key words | `True` |
| `dynamic_motion` | Ken Burns + shake effects | `True` |

**Subtitle Appearance:**
- Font size: 70px (adjustable for format)
- Primary color: White
- Highlight color: Gold (#FFD700)
- Stroke: 4px black outline
- Position: Lower third (75-78% from top)

**Effects:**
- Ken Burns (pan/zoom on images)
- Crossfade transitions
- Background music mixing
- Sound effects support

---

### 11. YouTube Uploader (`src/uploader.py`)

Handles OAuth2 authentication and video upload to YouTube.

**Key Functions:**
- `get_authenticated_service()` - OAuth2 flow (local + CI/CD)
- `upload_to_youtube(video_path, title, description, tags, thumbnail_path)` - Upload video

**Authentication Files:**
- `client_secrets.json` - OAuth2 client credentials
- `credentials.json` - Stored access/refresh tokens

---

## Entry Points

### 1. `generate_video.py` - Main CLI

```bash
python generate_video.py \
    --topic "How does machine learning work?" \
    --audience "beginners" \
    --format long \
    --style "educational with analogies" \
    --tts openai \
    --upload
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--topic, -t` | Yes | Video topic |
| `--audience, -a` | Yes | Target audience |
| `--format, -f` | Yes | `short` or `long` |
| `--style, -s` | No | Video style (default: educational) |
| `--output-dir, -o` | No | Output directory |
| `--tts` | No | TTS provider: gtts/openai/elevenlabs |
| `--cleanup` | No | Clean temp files after |
| `--upload` | No | Upload to YouTube |
| `--verbose, -v` | No | Verbose output |

### 2. `generate_subtitles.py` - Subtitle Generator

Standalone tool to generate .ass subtitle files from audio.

```bash
python generate_subtitles.py \
    --audio-dir output/audio \
    --output-dir output/subtitles \
    --words-per-line 3 \
    --font-size 48
```

### 3. `recompose_video.py` - Video Recomposer

Regenerates video from cached assets with new subtitle styles.

```bash
python recompose_video.py
```

**Uses cached:**
- `output/images/scene_*.png`
- `output/audio/voice_*.wav`
- `output/planned_scenes.json`
- `output/subtitles/timestamps/*.json` (optional)

---

## CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/main.yml`)

**Trigger:**
- Scheduled: Daily at 7:00 UTC (12:30 PM IST)
- Manual: `workflow_dispatch`

**Pipeline Steps:**
1. ⬇️ Checkout repository
2. 🐍 Set up Python 3.11
3. 📦 Install dependencies + patch ImageMagick
4. 🔑 Restore API credentials from Base64 secrets
5. 🚀 Run autonomous production pipeline
6. 🎥 Upload video artifacts
7. 💾 Commit content plan updates

**Required Secrets:**
| Secret | Description |
|--------|-------------|
| `AI_PROVIDER` | `openai` or `gemini` |
| `GOOGLE_API_KEY` | Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `PEXELS_API_KEY` | Fallback images (optional) |
| `CLIENT_SECRET_B64` | YouTube OAuth client (base64) |
| `CREDENTIALS_B64` | YouTube refresh token (base64) |

---

## Data Flow

```
VideoRequest
    │
    ▼
┌───────────────────┐
│ ContentResearcher │ ──▶ ResearchResult
└───────────────────┘
    │
    ▼
┌───────────────────┐
│   ScriptWriter    │ ──▶ Script (with Scenes)
└───────────────────┘
    │
    ▼
┌───────────────────┐
│   ScenePlanner    │ ──▶ list[PlannedScene] (with image prompts)
└───────────────────┘
    │
    ├───────────────────────────────┐
    ▼                               ▼
┌───────────────────┐    ┌───────────────────┐
│  VoiceGenerator   │    │  ImageGenerator   │
│                   │    │                   │
│ PlannedScene      │    │ PlannedScene      │
│ + audio_path      │    │ + image_path      │
│ + duration        │    │                   │
└───────────────────┘    └───────────────────┘
    │                               │
    └───────────────┬───────────────┘
                    ▼
           ┌───────────────────┐
           │   AudioAligner    │ ──▶ Word Timestamps
           └───────────────────┘
                    │
                    ▼
           ┌───────────────────┐
           │   VideoComposer   │ ──▶ Final MP4 + Thumbnail
           └───────────────────┘
                    │
                    ▼
           ┌───────────────────┐
           │     Uploader      │ ──▶ YouTube Video ID
           └───────────────────┘
```

---

## API Cost Estimates

### OpenAI Pricing (per video)

| Component | Model | Cost Estimate |
|-----------|-------|---------------|
| Text Generation | gpt-5-mini | ~$0.01-0.05 |
| Image Generation | gpt-image-1-mini (low) | ~$0.005 × scenes |
| TTS | gpt-4o-mini-tts | ~$0.015/min × duration |
| Whisper Alignment | whisper-1 | ~$0.006/min × duration |

**Estimated total per video:**
- Short (60s, 3 scenes): ~$0.10-0.15
- Long (5min, 8 scenes): ~$0.30-0.50

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | AI provider: `openai` or `gemini` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `GOOGLE_API_KEY` | Google API key | Required for Gemini |
| `TTS_PROVIDER` | TTS: `gtts`, `openai`, `elevenlabs` | `gtts` |
| `TTS_VOICE` | OpenAI TTS voice | `alloy` |
| `TTS_SPEED` | TTS speed (0.25-4.0) | `1.0` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Optional |
| `PEXELS_API_KEY` | Pexels fallback images | Optional |
| `USE_AI_BACKGROUNDS` | Use AI-generated backgrounds | `true` |

---

## Dependencies

### Core Libraries
- `openai>=1.0.0` - OpenAI API client
- `google-genai` - Google Gemini client
- `moviepy==1.0.3` - Video processing
- `Pillow>=9.5.0` - Image processing
- `pydub` - Audio processing

### TTS
- `gTTS` - Google Text-to-Speech (free)

### YouTube Upload
- `google-api-python-client`
- `google-auth-httplib2`
- `google-auth-oauthlib`

### Utilities
- `python-dotenv` - Environment management
- `requests`, `aiohttp` - HTTP clients
- `rich` - Console output (optional)

### System Requirements
- **Python 3.10+**
- **FFmpeg** - Video encoding
- **ImageMagick** - Text rendering for moviepy

---

## Legacy System

The original curriculum-based generator is still available in `src/generator.py` and can be invoked via `main.py`. It uses a different approach:

1. Generates a curriculum of lessons
2. Produces both long and short videos per lesson
3. Auto-uploads to YouTube

This is maintained for backward compatibility but the new pipeline (`generate_video.py`) is recommended.

---

## Future Improvements

- [ ] Add support for more TTS providers
- [ ] Implement video caching for faster regeneration
- [ ] Add A/B testing for thumbnails
- [ ] Support for multiple languages
- [ ] Real-time preview during generation
- [ ] Web UI for easier configuration