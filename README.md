# AI Video Generation Pipeline

An autonomous AI-powered video generator that creates engaging videos from any topic. Supports both **Google Gemini** and **OpenAI** APIs with a modular, scalable architecture.

## ✨ Features

- **Topic-Driven Generation**: Just provide a topic and audience - the pipeline handles everything else
- **Dual AI Support**: Works with OpenAI GPT-4o or Google Gemini
- **Multiple TTS Providers**: gTTS (free), OpenAI TTS, or ElevenLabs (premium)
- **Two Video Formats**: 
  - Long-form (3-10 minutes, 16:9 horizontal)
  - Short-form (<60 seconds, 9:16 vertical)
- **Full Pipeline**: Research → Script → Images → Voice → Video → Thumbnail
- **YouTube Integration**: Optional auto-upload with metadata

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO GENERATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   INPUT     │───▶│   RESEARCH  │───▶│   SCRIPT    │───▶│   SCENES    │      │
│  │             │    │             │    │             │    │             │      │
│  │ • Topic     │    │ • AI Search │    │ • Structure │    │ • Breakdown │      │
│  │ • Audience  │    │ • Curation  │    │ • Narration │    │ • Timing    │      │
│  │ • Format    │    │ • Summary   │    │ • Flow      │    │ • Visuals   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                                 │
│                                         │                                       │
│                                         ▼                                       │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐     │
│  │   OUTPUT    │◀───│   STITCH    │◀───│      ASSET GENERATION           │     │
│  │             │    │             │    │                                 │     │
│  │ • MP4 File  │    │ • Concat    │    │  For each scene:                │     │
│  │ • Metadata  │    │ • Audio Mix │    │  • Generate AI Image            │     │
│  │ • Thumbnail │    │ • Transitions│   │  • Generate Voice (TTS)         │     │
│  └─────────────┘    └─────────────┘    └─────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
gemini-youtube-automation/
├── src/
│   ├── pipeline/                    # NEW: Modular video generation pipeline
│   │   ├── __init__.py              # Package exports
│   │   ├── orchestrator.py          # Main pipeline coordination
│   │   ├── models.py                # Data models (VideoRequest, Script, etc.)
│   │   ├── researcher.py            # Content research & curation
│   │   ├── script_writer.py         # Script generation
│   │   ├── scene_planner.py         # Scene breakdown & planning
│   │   ├── image_generator.py       # AI image generation
│   │   ├── voice_generator.py       # TTS audio generation
│   │   └── video_composer.py        # Video stitching & export
│   ├── utils/
│   │   ├── __init__.py
│   │   └── ai_client.py             # Unified AI client wrapper
│   ├── generator.py                 # Legacy generator (still works)
│   └── uploader.py                  # YouTube upload functionality
├── assets/
│   ├── fonts/                       # Font files for text rendering
│   └── music/                       # Background music files
├── output/                          # Generated videos and assets
├── generate_video.py                # NEW: CLI entry point
├── main.py                          # Legacy entry point
├── test_pipeline.py                 # Pipeline structure tests
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ChaituRajSagar/gemini-youtube-automation.git
cd gemini-youtube-automation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
```

**Required API Key (choose one):**
- `OPENAI_API_KEY` - For OpenAI provider
- `GOOGLE_API_KEY` - For Gemini provider

### 3. Generate a Video

```bash
# Long-form video (3-10 minutes)
python generate_video.py \
    --topic "How does machine learning actually work?" \
    --audience "developers new to AI" \
    --format long

# Short-form video (<60 seconds)
python generate_video.py \
    --topic "What is Docker in 60 seconds?" \
    --audience "beginners" \
    --format short
```

## 📖 Usage

### Command Line Interface

```bash
python generate_video.py [OPTIONS]

Required Arguments:
  --topic, -t       The main topic or question for the video
  --audience, -a    Target audience (e.g., "beginners", "developers")
  --format, -f      Video format: "short" or "long"

Optional Arguments:
  --style, -s       Video style (default: "educational")
  --output-dir, -o  Output directory (default: output/)
  --tts             TTS provider: gtts, openai, or elevenlabs
  --cleanup         Clean up temporary files after generation
  --upload          Upload to YouTube after generation
  --verbose, -v     Enable verbose output
```

### Examples

```bash
# Educational long-form video
python generate_video.py \
    --topic "How do transformers work in AI?" \
    --audience "software developers" \
    --format long \
    --style "educational with analogies"

# Quick tips short video
python generate_video.py \
    --topic "3 Python tricks you didn't know" \
    --audience "junior developers" \
    --format short \
    --style "casual and fun"

# With OpenAI TTS and YouTube upload
python generate_video.py \
    --topic "Understanding Kubernetes" \
    --audience "DevOps beginners" \
    --format long \
    --tts openai \
    --upload
```

### Python API

```python
from src.pipeline import VideoPipeline, VideoRequest, VideoFormat

# Create a request
request = VideoRequest(
    topic="How does blockchain work?",
    target_audience="non-technical audience",
    format=VideoFormat.LONG,
    style="simple with real-world examples"
)

# Generate video
pipeline = VideoPipeline()
output = pipeline.generate_video_sync(request)

if output.success:
    print(f"Video saved to: {output.video_path}")
    print(f"Thumbnail: {output.thumbnail_path}")
    print(f"Title: {output.metadata.title}")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | AI provider: `openai` or `gemini` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required for openai |
| `GOOGLE_API_KEY` | Google API key | Required for gemini |
| `TTS_PROVIDER` | TTS: `gtts`, `openai`, `elevenlabs` | `gtts` |
| `TTS_VOICE` | OpenAI TTS voice | `alloy` |
| `TTS_SPEED` | OpenAI TTS speed (0.25-4.0) | `1.0` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Optional |
| `PEXELS_API_KEY` | Pexels API key for fallback images | Optional |
| `USE_AI_BACKGROUNDS` | Use AI-generated backgrounds | `true` |

### TTS Voice Options (OpenAI)

- `alloy` - Neutral, balanced
- `echo` - Warm, conversational
- `fable` - Expressive, storytelling
- `onyx` - Deep, authoritative
- `nova` - Friendly, upbeat
- `shimmer` - Clear, professional

## 🎥 Output Files

After generation, you'll find in the `output/` directory:

```
output/
├── video_long_20240115_143022.mp4   # Final video
├── thumbnail.png                      # Video thumbnail
├── images/                            # Scene images
│   ├── scene_01.png
│   ├── scene_02.png
│   └── ...
└── audio/                             # Voice narration
    ├── voice_01.wav
    ├── voice_02.wav
    └── ...
```

## 🔧 Legacy Mode

The original curriculum-based system is still available:

```bash
# Run the legacy lesson generator
python main.py
```

This will:
1. Generate a curriculum if none exists
2. Produce lessons with long-form and short videos
3. Upload to YouTube automatically

## 🧪 Testing

```bash
# Run pipeline structure tests
python test_pipeline.py

# Expected output: All tests should pass
```

## 📋 Requirements

- Python 3.10+
- FFmpeg (for video processing)
- ImageMagick (for moviepy text rendering)

### API Keys Needed

| Feature | Required Key |
|---------|--------------|
| Text Generation | `OPENAI_API_KEY` or `GOOGLE_API_KEY` |
| Image Generation | `OPENAI_API_KEY` |
| TTS (OpenAI) | `OPENAI_API_KEY` |
| TTS (ElevenLabs) | `ELEVENLABS_API_KEY` |
| Fallback Images | `PEXELS_API_KEY` |
| YouTube Upload | OAuth credentials |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_pipeline.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT and DALL-E APIs
- Google for Gemini API
- The moviepy team for video processing
- gTTS for free text-to-speech