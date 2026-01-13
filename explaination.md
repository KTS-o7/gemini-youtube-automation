# Gemini YouTube Automation - Project Explanation

## Overview

This project is an **autonomous AI-powered YouTube content generator** that automatically creates educational video content about AI/Machine Learning topics, and uploads them directly to YouTube. It supports both **Google Gemini** and **OpenAI** APIs for content generation, Google Text-to-Speech for narration, and the YouTube Data API for automated uploads.

The system produces two types of videos per lesson:
1. **Long-form videos** (landscape 1920x1080) - Detailed educational content with multiple slides
2. **Short videos** (portrait 1080x1920) - Quick tip highlights for YouTube Shorts

---

## Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN WORKFLOW (main.py)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Load/Generate Content Plan (content_plan.json)                          │
│                    │                                                        │
│                    ▼                                                        │
│  2. Find Pending Lessons                                                    │
│                    │                                                        │
│                    ▼                                                        │
│  3. For Each Lesson:                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  a. Generate Lesson Content (Gemini AI or OpenAI)               │    │
│     │  b. Generate Audio (gTTS → WAV)                                 │    │
│     │  c. Generate Slide Images (PIL + Pexels API)                    │    │
│     │  d. Create Video (MoviePy)                                      │    │
│     │  e. Upload to YouTube (YouTube Data API)                        │    │
│     │  f. Update content_plan.json status                             │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                    │                                                        │
│                    ▼                                                        │
│  4. Cleanup temporary files (.wav)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure Explained

### Core Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point - orchestrates the entire workflow |
| `src/generator.py` | AI content generation (Gemini/OpenAI), TTS, image creation, video rendering |
| `src/uploader.py` | YouTube OAuth2 authentication and video upload |
| `content_plan.json` | Curriculum tracking - lessons and their completion status |
| `requirements.txt` | Python dependencies |

### Assets

| Directory/File | Purpose |
|----------------|---------|
| `assets/fonts/` | Arial font files for slide text rendering |
| `assets/music/bg_music.mp3` | Background music for videos |
| `assets/fallback.jpg` | Fallback image if Pexels API fails |

---

## Component Deep Dive

### 1. Content Plan Management (`main.py`)

The `content_plan.json` file tracks the curriculum:

```json
{
  "lessons": [
    {
      "chapter": 1,
      "part": 1,
      "title": "Understanding AI: From Magic to Math",
      "status": "complete",      // or "pending"
      "youtube_id": "EN-Q73PXuYk" // null if pending
    }
  ]
}
```

**Key Functions:**
- `get_content_plan()` - Loads existing plan or generates new one via Gemini
- `update_content_plan()` - Saves progress after each lesson
- When all lessons are complete, it generates a **new curriculum** continuing from where it left off

### 2. AI Content Generation (`src/generator.py`)

#### AI Provider Selection
```python
def get_ai_provider():
```
- Returns the configured AI provider based on `AI_PROVIDER` environment variable
- Supports `"gemini"` (default) or `"openai"`

#### Curriculum Generation
```python
def generate_curriculum(previous_titles=None):
```
- Uses **Gemini 2.0 Flash** or **GPT-4o** depending on configured provider
- Generates 20 lessons covering AI topics from beginner to advanced
- Can accept previous titles to continue series without repetition
- Returns structured JSON with chapters, parts, titles

#### Lesson Content Generation
```python
def generate_lesson_content(lesson_title):
```
- Generates for each lesson:
  - `long_form_slides`: 7-8 slides with title and content
  - `short_form_highlight`: 1-2 sentence summary for Shorts
  - `hashtags`: 5-7 relevant hashtags

### 3. Text-to-Speech (`src/generator.py`)

```python
def text_to_speech(text, output_path):
```
- Uses **gTTS (Google Text-to-Speech)**
- Converts script to MP3, then to WAV using **pydub**
- WAV format ensures compatibility with MoviePy

### 4. Visual Generation (`src/generator.py`)

#### Background Images
```python
def get_pexels_image(query, video_type):
```
- Fetches relevant images from **Pexels API**
- Automatically selects orientation based on video type
- Falls back to solid color if API unavailable

#### Slide/Thumbnail Creation
```python
def generate_visuals(output_dir, video_type, slide_content=None, ...):
```
- Creates professional PPT-style slides using **PIL (Pillow)**
- Features:
  - Dynamic text wrapping
  - Header with title
  - Footer with branding and slide numbers
  - Blurred background with dark overlay
  - Different dimensions for long (1920x1080) vs short (1080x1920)

### 5. Video Rendering (`src/generator.py`)

```python
def create_video(slide_paths, audio_paths, output_path, video_type):
```
- Uses **MoviePy** to composite videos
- Each slide duration = audio duration + 0.5s padding
- Adds fade in/out transitions
- Overlays background music at 15% volume
- Exports with H.264 codec and AAC audio

### 6. YouTube Upload (`src/uploader.py`)

#### Authentication
```python
def get_authenticated_service():
```
- OAuth2 flow with credentials caching
- First run: Opens browser for user consent
- Subsequent runs: Uses refresh token automatically
- Stores credentials in `credentials.json`

#### Upload Process
```python
def upload_to_youtube(video_path, title, description, tags, thumbnail_path):
```
- Uploads video with metadata (title, description, tags, category)
- Sets privacy to public
- Uploads custom thumbnail after video upload
- Returns video ID for linking in Shorts description

---

## Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `AI_PROVIDER` | AI provider to use: `"gemini"` (default) or `"openai"` |
| `GOOGLE_API_KEY` | Gemini AI API key for content generation (required if using Gemini) |
| `OPENAI_API_KEY` | OpenAI API key for content generation (required if using OpenAI) |
| `PEXELS_API_KEY` | Pexels API key for background images (optional) |

---

## Required Files for YouTube Upload

| File | Purpose | How to Obtain |
|------|---------|---------------|
| `client_secrets.json` | OAuth2 client credentials | Download from Google Cloud Console |
| `credentials.json` | User access/refresh tokens | Auto-generated on first run |

---

## Running Locally

### Step 1: Install Dependencies

```bash
cd gemini-youtube-automation
pip install -r requirements.txt
```

### Step 2: Install System Dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg imagemagick
```

### Step 3: Set Environment Variables

**For Gemini (default):**
```bash
export AI_PROVIDER="gemini"  # Optional, gemini is the default
export GOOGLE_API_KEY="your-gemini-api-key"
export PEXELS_API_KEY="your-pexels-api-key"  # Optional
```

**For OpenAI:**
```bash
export AI_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key"
export PEXELS_API_KEY="your-pexels-api-key"  # Optional
```

### Step 4: Set Up YouTube API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the **YouTube Data API v3**
4. Create OAuth 2.0 credentials (Desktop application)
5. Download and save as `client_secrets.json` in project root

### Step 5: Run the Application

```bash
python main.py
```

**First Run:**
- A browser window will open for YouTube authorization
- Grant permissions to upload videos
- Credentials will be saved for future runs

---

## Production Flow (Per Lesson)

1. **Load** pending lesson from `content_plan.json`
2. **Generate** lesson content via Gemini AI
3. **Create Long-Form Video:**
   - Generate intro + content + outro slides
   - Convert each slide script to audio (TTS)
   - Create slide images with Pexels backgrounds
   - Render video with MoviePy
   - Create thumbnail
4. **Upload** long-form video to YouTube
5. **Wait** 30 seconds (rate limiting)
6. **Create Short Video:**
   - Generate single slide with highlight
   - Convert highlight to audio
   - Render portrait video
   - Create portrait thumbnail
7. **Upload** short video with link to long-form
8. **Update** `content_plan.json` with completion status

---

## GitHub Actions Automation

The project includes a workflow (`.github/workflows/main.yml`) that:
- Runs daily at 7:00 AM UTC
- Processes one lesson per run (`LESSONS_PER_RUN = 1`)
- Uses stored secrets for API keys
- Auto-commits updated `content_plan.json`

---

## Key Design Decisions

1. **WAV over MP3**: Ensures audio compatibility with MoviePy
2. **Per-slide audio sync**: Each slide has its own audio for perfect timing
3. **Fallback mechanisms**: Solid color backgrounds if Pexels fails
4. **Curriculum continuation**: Avoids repetition by passing previous titles
5. **Credential caching**: OAuth tokens stored for automated runs
6. **Modular architecture**: Separate modules for generation vs uploading

---

## Error Handling

- Content plan regeneration if invalid/empty
- Network error handling for Pexels API
- Graceful fallback for missing fonts
- Automatic cleanup of temporary WAV files
- Exception logging with traceback for debugging

---

## Output Structure

```
output/
├── audio_slide_1_YYYYMMDD_C_P.mp3    # Audio files
├── audio_slide_2_YYYYMMDD_C_P.mp3
├── slides_long_YYYYMMDD_C_P/         # Long-form slides
│   ├── slide_01.png
│   ├── slide_02.png
│   └── ...
├── slides_short_YYYYMMDD_C_P/        # Short slides
│   └── slide_01.png
├── long_video_YYYYMMDD_C_P.mp4       # Final long video
├── short_video_YYYYMMDD_C_P.mp4      # Final short video
└── thumbnail.png                      # Thumbnails
```

Where `C` = Chapter number, `P` = Part number

---

## Dependencies Summary

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini AI for content generation |
| `openai` | OpenAI API for content generation |
| `google-api-python-client` | YouTube Data API |
| `google-auth-oauthlib` | OAuth2 authentication |
| `gTTS` | Text-to-speech conversion |
| `moviepy` | Video editing and rendering |
| `Pillow` | Image processing for slides |
| `pydub` | Audio format conversion |
| `requests` | HTTP requests for Pexels API |
| `python-dotenv` | Environment variable loading |

---

## Troubleshooting

### Quota Exceeded Error

If you see an error like:
```
google.api_core.exceptions.ResourceExhausted: 429 You exceeded your current quota
```

**Solutions:**
1. **Wait for quota reset** - Free tier limits reset per-minute and per-day
2. **Create a new API key** - Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a fresh key
3. **Enable billing** - Remove free tier limits by enabling billing on Google Cloud

### Model Not Found Error

If you see:
```
404 models/gemini-1.5-flash is not found
```

The model name may be outdated. Update `src/generator.py` to use a current model like `gemini-2.0-flash`.

### .env File Not Loading

Ensure you have `python-dotenv` installed and `load_dotenv()` is called at the start of `main.py`:
```bash
pip install python-dotenv
```

### YouTube Upload Authorization Failed

1. Ensure `client_secrets.json` exists in the project root
2. Delete `credentials.json` and re-run to trigger fresh OAuth flow
3. Make sure YouTube Data API v3 is enabled in Google Cloud Console

### FFmpeg Not Found

Install FFmpeg:
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Audio/Video Sync Issues

The project converts MP3 to WAV for better MoviePy compatibility. Ensure `pydub` is installed:
```bash
pip install pydub
```

### Switching Between AI Providers

To switch from Gemini to OpenAI:
1. Set the `AI_PROVIDER` environment variable to `"openai"`
2. Ensure `OPENAI_API_KEY` is set with a valid API key
3. The system will automatically use GPT-4o for content generation

To switch back to Gemini:
1. Set `AI_PROVIDER` to `"gemini"` or remove the variable (Gemini is default)
2. Ensure `GOOGLE_API_KEY` is set with a valid API key