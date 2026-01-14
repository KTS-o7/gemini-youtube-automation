# AI Video Generation Pipeline

An autonomous AI-powered video generator that creates engaging videos from any topic. Features a **Streamlit web interface** and supports both **Google Gemini** and **OpenAI** APIs with a modular, scalable architecture.

## ✨ Features

- **🎬 Streamlit Web App**: Beautiful, interactive UI for video generation
- **✨ AI-Powered Topic Improver**: Auto-enhance your topics for better engagement
- **🎯 Voice Instruction Improver**: AI-generated voice style instructions for perfect narration
- **💡 Topic Suggestions**: Get AI-generated topic ideas based on your audience
- **Topic-Driven Generation**: Just provide a topic and audience - the pipeline handles everything else
- **Dual AI Support**: Works with OpenAI GPT-5 series or Google Gemini
- **Multiple TTS Providers**: gTTS (free), OpenAI TTS ($0.015/min), or ElevenLabs (premium)
- **Two Video Formats**: 
  - Long-form (3-10 minutes, 16:9 horizontal)
  - Short-form (<60 seconds, 9:16 vertical)
- **🔥 Viral Mode**: Karaoke-style captions, dynamic motion effects, emoji support
- **🎯 Two Subtitle Alignment Options**:
  - **Whisper (OpenAI)**: Auto-transcribes audio, ~$0.006/minute
  - **Wav2Vec2 (FREE)**: Local forced alignment, requires transcript
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
├── app.py                           # Streamlit web application
├── generate_video.py                # CLI entry point (legacy)
├── generate_subtitles.py            # Subtitle generator (Whisper or Wav2Vec2)
├── recompose_video.py               # Video recomposer utility
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

### 3. Run the Streamlit App (Recommended)

```bash
streamlit run app.py
```

This opens a web interface with:
- 🎬 **Generate Video** - Full video generation with real-time progress
- 📁 **View Output** - Browse and download generated content  
- 🔄 **Recompose Video** - Regenerate video from existing assets
- 📝 **Generate Subtitles** - Create word-level timestamps with Whisper
- ⚙️ **Settings** - Configure API keys and preferences

### 4. Or Use the CLI (Legacy)

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

### Streamlit Web App

The Streamlit app provides a user-friendly interface:

1. **Generate Video Page**
   - **AI Topic Improver** (NEW!): 
     - ✨ Click "Improve" to make your topic more engaging and specific
     - 💡 Click "Suggest" to get 5 AI-generated topic ideas (uses structured outputs)
   - **AI Voice Instructions** (NEW! OpenAI TTS only):
     - ✨ Click "Improve" to enhance your voice instructions with AI
     - 🎯 Click "Auto" to automatically generate optimal instructions
   - Choose TTS provider (gTTS free, OpenAI, or ElevenLabs) and voice
   - Enable/disable viral mode features (karaoke captions, motion effects)
   - Watch real-time progress with checkpoint system

2. **View Output Page**
   - Browse generated videos, images, audio
   - Preview and download files
   - View JSON data (research, scripts, scenes)

3. **Recompose Video Page**
   - Regenerate video with new settings
   - Use existing images and audio
   - Toggle viral mode, subtitles, alignment method (Wav2Vec2 or Whisper)

4. **Generate Subtitles Page**
   - Choose between Wav2Vec2 (FREE) or Whisper (OpenAI API)
   - Generate word-level timestamps for karaoke-style captions
   - See real-time caching of generated timestamps
   - Preview cached timestamp files

5. **Settings Page**
   - Configure API keys
   - Set default TTS voice and speed
   - View pricing information

### Command Line Interface (Legacy)

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

## 🎤 Subtitle Alignment (FREE Option!)

The pipeline supports two methods for generating word-level subtitle timestamps. Choose directly in the **Generate Subtitles** page in the Streamlit app, or use the CLI below.

### 1. Wav2Vec2 (FREE - Recommended)

Uses PyTorch's wav2vec2 model locally. **Completely free**, but requires the transcript text.

```bash
# From Streamlit app
# 1. Go to "Generate Subtitles" page
# 2. Select "Wav2Vec2" from dropdown
# 3. Click "Generate Timestamps"

# From command line with script JSON (contains transcripts)
python generate_subtitles.py --aligner wav2vec2 --script-json output/script.json

# From command line with transcript files
python generate_subtitles.py --aligner wav2vec2 --transcript-dir output/transcripts
```

```python
# From Python
from src.pipeline import Wav2Vec2Aligner

aligner = Wav2Vec2Aligner()
result = aligner.align("audio.wav", "Hello world this is a test")

# Generate .ass subtitle file
aligner.generate_ass_file(result, "output.ass", words_per_line=3)
```

### 2. Whisper (OpenAI API)

Uses OpenAI's Whisper API. **Auto-transcribes** audio (no transcript needed), but costs ~$0.006/minute.

```bash
# From command line
python generate_subtitles.py --aligner whisper

# Requires OPENAI_API_KEY in environment
```

```python
# From Python
from src.pipeline import AudioAligner

aligner = AudioAligner()
result = aligner.get_word_timestamps("audio.wav")
```

### Unified Interface

Use the factory function to easily switch between aligners:

```python
from src.pipeline import create_aligner, AlignerType

# Auto-select (checks SUBTITLE_ALIGNER env var, defaults to wav2vec2)
aligner = create_aligner()

# Or explicitly choose
aligner = create_aligner(AlignerType.WAV2VEC2)  # Free!
aligner = create_aligner(AlignerType.WHISPER)   # Auto-transcribes

# Align audio (wav2vec2 needs transcript, whisper doesn't)
result = aligner.align("audio.wav", transcript="Hello world")

# Generate subtitle file
aligner.generate_ass_file(result, "output.ass")
```

### Environment Variable

Set `SUBTITLE_ALIGNER` to control the default:

```bash
# Use free wav2vec2 by default
export SUBTITLE_ALIGNER=wav2vec2

# Or use Whisper (auto-transcribes but costs money)
export SUBTITLE_ALIGNER=whisper
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
| `SUBTITLE_ALIGNER` | Subtitle aligner: `wav2vec2` or `whisper` | `wav2vec2` |

### TTS Voice Options (OpenAI)

13 voices available with `gpt-4o-mini-tts`:

- ⭐ `marin` - Best quality (recommended)
- ⭐ `cedar` - Best quality (recommended)
- `alloy` - Neutral, balanced
- `ash` - Clear
- `ballad` - Melodic
- `coral` - Popular choice
- `echo` - Warm, conversational
- `fable` - Expressive, storytelling
- `nova` - Friendly, upbeat
- `onyx` - Deep, authoritative
- `sage` - Wise
- `shimmer` - Clear, professional
- `verse` - Poetic

### Voice Instructions (gpt-4o-mini-tts only)

Control voice style with natural language:
```bash
--voice-instructions "Speak in a calm, professional tone with clear enunciation."
```

Examples:
- "Speak cheerfully and with enthusiasm"
- "Speak slowly and deliberately"
- "Speak like an educational video narrator"

## 🎥 Output Files

After generation, you'll find in the `output/` directory:

```
output/
├── video_short_20240115_143022.mp4  # Final video
├── thumbnail.png                     # Video thumbnail
├── research.json                     # Research results
├── script.json                       # Generated script
├── planned_scenes.json               # Scene planning data
├── images/                           # Scene images
│   ├── scene_01.png
│   ├── scene_02.png
│   └── ...
├── audio/                            # Voice narration
│   ├── voice_01.wav
│   ├── voice_02.wav
│   └── ...
└── subtitles/                        # Subtitle files
    ├── timestamps/                   # Whisper word timestamps
    │   ├── voice_01.json
    │   └── ...
    └── *.ass                         # ASS subtitle files
```

## 💰 Cost Estimates

Using cost-optimized models:

| Component | Model | Cost |
|-----------|-------|------|
| Text | gpt-5-mini | $0.25 / $2.00 per 1M tokens |
| Images | gpt-image-1-mini (low) | $0.005 per image |
| TTS | gpt-4o-mini-tts | $0.015 per minute |
| Whisper | whisper-1 | $0.006 per minute |
| Wav2Vec2 | local model | **FREE** |

**Estimated cost per video:**
- Short (60s, 3 scenes): ~$0.05
- Long (5min, 8 scenes): ~$0.18

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
- PyTorch & torchaudio (for free wav2vec2 alignment)

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

## 📚 Recent Improvements (2025)

### Code Quality & Architecture
- ✅ **Complete refactoring** of 1700+ line god function
- ✅ **Modular pipeline** with composable stages
- ✅ **Type safety** with Pydantic models at API boundaries
- ✅ **249 test cases** covering all major components
- ✅ **Removed anti-patterns**: No more mutable global state, sys.path hacks, or silent failures

### New Features
- ✨ **AI Topic Improver**: Uses structured outputs to enhance topics for engagement
- 🎯 **Voice Instruction Generator**: AI creates optimal voice style instructions
- 💡 **Topic Suggestions**: Get 5 AI-generated topic ideas per audience/format
- 📊 **Timestamp Caching**: Smart caching of Wav2Vec2/Whisper alignments
- 🔄 **Checkpoint System**: Resume from any pipeline stage

### Testing & Reliability
- **Test Coverage**: 249 automated tests (pytest)
- **API Models**: Pydantic validation for all AI responses
- **Error Handling**: Proper exception handling with fallback mechanisms
- **Alignment Tests**: Comprehensive tests for Wav2Vec2 and Whisper

### Performance
- **Lazy Loading**: AI clients initialized only when needed
- **Caching**: 5-minute cache for AI improvement operations
- **Optimized Models**: Using gpt-5-mini for cost-efficiency
- **Free Alignment**: Wav2Vec2 option eliminates API costs for subtitles

## 🙏 Acknowledgments

- OpenAI for GPT and DALL-E APIs
- Google for Gemini API
- The moviepy team for video processing
- gTTS for free text-to-speech
- Facebook Research for Wav2Vec2 model