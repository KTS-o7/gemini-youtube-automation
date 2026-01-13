# AI Video Pipeline - Implementation Summary

## Models Used (Cost-Optimized)
- **Text**: `gpt-5-mini` ($0.25/$2.00 per 1M tokens) - 400K context, supports structured outputs
- **Image**: `gpt-image-1-mini` with `low` quality ($0.005/image)
- **TTS**: `gpt-4o-mini-tts` ($0.015/min) or `gtts` (free)

## GPT-5 mini Specs
- **Context Window**: 400,000 tokens
- **Max Output**: 128,000 tokens
- **Knowledge Cutoff**: May 31, 2024
- **Structured Outputs**: ✅ Supported
- **Function Calling**: ✅ Supported
- **Reasoning**: ✅ Supported

## Pipeline Architecture
```
Input → Research → Script → Scene Planning → Assets (Image+Voice) → Video → Output
```

## Key Features
- **Structured Outputs**: Uses Pydantic models with `responses.parse()` for guaranteed valid JSON
- **No Fallbacks**: Pipeline fails loud on errors instead of producing garbage
- **Auto-save**: Research, scripts, and scenes saved to JSON for debugging

## Usage

### Short Video (30-60 seconds, vertical 9:16)
```bash
source .venv/bin/activate && python generate_video.py \
  --topic "What is Docker?" \
  --audience "developers" \
  --format short \
  --tts openai
```

### Long Video (3-10 minutes, horizontal 16:9)
```bash
source .venv/bin/activate && python generate_video.py \
  --topic "Machine Learning basics" \
  --audience "beginners" \
  --format long \
  --tts openai
```

### TTS Options
- `--tts openai` - Best quality ($0.015/min)
- `--tts gtts` - Free, decent quality
- `--tts elevenlabs` - Premium quality (requires API key)

## Output Files
After running, the `output/` directory contains:
- `research.json` - Research results with key points, facts, examples
- `script.json` - Generated script with scenes and narration
- `planned_scenes.json` - Scenes with image prompts and timing
- `images/` - Generated scene images (scene_01.png, scene_02.png, etc.)
- `audio/` - Voice narration files (voice_01.wav, voice_02.wav, etc.)
- `video_short_*.mp4` or `video_long_*.mp4` - Final video
- `thumbnail.png` - Video thumbnail

## Environment Variables
```bash
OPENAI_API_KEY=your_key_here      # Required for AI generation
AI_PROVIDER=openai                 # Default provider
TTS_PROVIDER=openai                # TTS provider (gtts, openai, elevenlabs)
TTS_VOICE=alloy                    # Voice for OpenAI TTS
```

## Files Structure
```
src/
├── pipeline/
│   ├── orchestrator.py    # Main pipeline coordinator
│   ├── researcher.py      # Topic research with structured output
│   ├── script_writer.py   # Script generation with Pydantic
│   ├── scene_planner.py   # Scene planning and image prompts
│   ├── image_generator.py # AI image generation
│   ├── voice_generator.py # TTS audio generation
│   ├── video_composer.py  # Final video composition
│   └── models.py          # Data models (VideoRequest, Script, etc.)
└── utils/
    └── ai_client.py       # Unified AI client with structured outputs
```

## Completed
- [x] Update to gpt-5-mini (better & cheaper than gpt-4o-mini)
- [x] Remove fallback scripts/images - fail loud on errors
- [x] Save research, scripts, scenes to JSON files
- [x] Test full pipeline with structured outputs

## TODO
- [ ] Add web search for research (optional)
- [ ] YouTube auto-upload integration
- [ ] Background music support
- [x] Text overlays on video (subtitles implemented!)
- [x] Script narrative continuity improvements
- [x] Visual continuity for image generation

### Viral Shorts Improvements (Priority) ✅ IMPLEMENTED
- [x] 🔥 Karaoke-style captions - Word-by-word highlight with color emphasis (HIGH IMPACT)
- [x] ⚡ Faster scene pacing - Force max 5-8 seconds per scene for shorts
- [x] 🎵 Background music integration - Add actual music files + auto-sync
- [x] 🔊 Sound effects - Whooshes, pops, emphasis sounds at transitions (infrastructure ready)
- [x] 📱 Larger/bolder subtitles - More visible on mobile (bigger font, yellow/white)
- [x] 🎬 Dynamic motion effects - Quick zooms, Ken Burns with easing
- [x] 😀 Emoji support in captions - Available but disabled by default (set `emoji_enabled=True` to enable)
- [x] 🎨 Color emphasis - Highlight key words in different colors (gold, cyan, green)

## Subtitles Feature
Subtitles are now automatically generated using MoviePy's `TextClip`:
- **Phrase-based**: Narration is split into readable phrases (6-8 words each)
- **Auto-timed**: Subtitles sync with audio duration
- **Format-aware**: Font size and positioning adjust for short (vertical) vs long (horizontal) videos
- **Styled**: White text with black stroke for readability on any background

### Disable Subtitles
```python
# In orchestrator or direct call:
video_composer.compose(scenes, request, enable_subtitles=False)
```

## Script & Visual Continuity (NEW)

### Script Writer Improvements
- **Narrative Arc**: Scripts now follow SETUP → EXPLORATION → PAYOFF structure
- **Transition Toolkit**: Explicit transitional phrases between scenes
- **Central Question**: Long-form scripts establish a question in Scene 1, answer it by the end
- **Callback System**: Endings reference opening hooks for satisfying closure
- **Continuity Check**: Auto-logs transition quality score after generation

### Scene Planner Improvements
- **Visual Themes**: 6 predefined themes (tech, business, education, creative, nature, minimal)
- **Auto Theme Detection**: Matches topic/style to appropriate visual theme
- **Consistent Style Elements**:
  - Color palette persists across all scenes
  - Lighting style maintained throughout
  - Recurring visual motifs appear in multiple scenes
- **Scene Position Context**: Prompts include position (opening/middle/closing) with appropriate composition hints
- **Previous Scene Reference**: Each prompt considers the previous scene's context

### Visual Theme Example (Tech)
```
Color Palette: deep blues, electric cyan, purple accents on dark backgrounds
Lighting: soft neon glow with dramatic rim lighting
Style: modern, sleek, futuristic aesthetic with clean lines
Recurring Elements: glowing circuits, holographic interfaces, geometric shapes
```

## Viral Mode Features (NEW)

The video composer now supports a `viral_mode=True` option that enables all viral optimizations:

### Karaoke-Style Captions
- Words appear in groups of 2-4 (configurable via `words_per_group`)
- Current word highlighted in gold (#FFD700)
- Key words get color emphasis based on category:
  - **Strong words** (secret, amazing, important): Gold/Yellow
  - **Action words** (learn, discover, build): Cyan/Blue  
  - **Numbers/stats** (percent, million, 10x): Green

### Emoji Support (Disabled by Default)
- Available but disabled by default
- Enable with `video_composer.emoji_enabled = True`
- 60+ keyword mappings available (tech, business, learning, etc.)
- Max 3 emojis per caption to avoid clutter

### Dynamic Motion Effects
- Opening scene: Zoom in (1.0 → 1.15) to draw attention
- Closing scene: Zoom out (1.1 → 1.0) for finale
- Middle scenes: Alternating zoom in/out effects
- Smooth easing (cubic ease in-out) for natural motion

### Fast Scene Pacing
- Short videos: Max 8 seconds per scene, min 2 seconds
- Faster speaking rate assumption (180 WPM vs 150 WPM)
- Quick crossfade transitions (0.3s vs 0.5s)

### Mobile-Optimized Subtitles
- Short (9:16): 75px font, 5px stroke, 18 chars/line
- Long (16:9): 55px font, 4px stroke, 35 chars/line

### Usage
```python
# Enable viral mode (default for shorts)
video_composer.compose(scenes, request, viral_mode=True)

# Or disable for traditional style
video_composer.compose(scenes, request, viral_mode=False)
```

## Error Handling Philosophy
- **No fallbacks**: If AI fails, the pipeline fails immediately
- **Fail loud**: Clear error messages instead of silent degradation
- **Quality over availability**: Better to fail than produce garbage content

## Recompose Video (Quick Regeneration)

When you already have images and audio, use `recompose_video.py` to regenerate video with new subtitle styles:

```bash
source .venv/bin/activate && python recompose_video.py
```

This uses existing assets in `output/` folder:
- `output/images/scene_01.png`, etc.
- `output/audio/voice_01.wav`, etc.
- `output/planned_scenes.json` for narration text

## Video Quality Settings

### Resolution Options (9:16 vertical)
| Resolution | Width | Height | Font Size | Stroke | Use Case |
|------------|-------|--------|-----------|--------|----------|
| 1080p | 1080 | 1920 | 75px | 5px | Standard, fast encode |
| 2K | 1440 | 2560 | 100px | 7px | Sharp subtitles, balanced |
| 4K | 2160 | 3840 | 150px | 10px | Maximum quality, slow |

### Encoding Presets
| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `ultrafast` | ⚡⚡⚡ | ⭐ | Testing, quick preview |
| `fast` | ⚡⚡ | ⭐⭐ | Development |
| `medium` | ⚡ | ⭐⭐⭐ | Balanced |
| `slow` | 🐌 | ⭐⭐⭐⭐ | Final export, high quality |

### CRF (Quality) Values
- `18` - Visually lossless (recommended for final)
- `23` - Good balance of quality/size
- `28` - Smaller file, visible compression

## Karaoke Subtitle Implementation

### How It Works
True karaoke style shows **all words in the phrase** with the **current word highlighted**:
- All words visible in white (default color)
- Current word highlighted in gold/cyan/green (based on word type)
- Words positioned inline (side by side, not stacked)
- Each word clip rendered separately for precise positioning

### Color Emphasis Categories
```python
EMPHASIS_KEYWORDS = {
    "strong": ["secret", "amazing", "important", "key", ...],  # Gold #FFD700
    "action": ["learn", "discover", "build", "create", ...],   # Cyan #00D4FF
    "stats": ["percent", "million", "10x", ...],               # Green #00FF88
}
```

### Key Learnings
1. **Don't stack text** - Creates duplicate/overlapping subtitles
2. **Render each word separately** - Allows per-word color control
3. **Calculate total width first** - Needed for centering the phrase
4. **Font size scales with resolution** - 75px@1080p → 100px@2K → 150px@4K
5. **Stroke width proportional to font** - ~5-7% of font size works well

### Subtitle Sharpness Tips
- **Resolution matters most** - Higher res = sharper text
- **Font size** - Bigger = sharper at any resolution
- **Stroke width** - Less stroke = cleaner look, but harder to read on light backgrounds
- Recommended ratio: `stroke_width = font_size * 0.07`

## Performance Optimizations

### Speed Tips
- Use `preset="ultrafast"` for testing
- Use `threads=0` to use all CPU cores
- Lower `fps` (24 vs 30) for faster encode
- Lower `crf` value only for final export

### Current High-Quality Settings
```python
self.fps = 30
self.preset = "slow"
self.threads = 0  # All cores
ffmpeg_params = ["-crf", "18", "-profile:v", "high"]
```

## Whisper Audio Alignment (NEW)

Precise word-level timestamps using OpenAI Whisper API for true karaoke sync.

### How It Works
1. **Whisper API** analyzes audio files and returns exact timing for each word
2. **Timestamps cached** as JSON files for reuse (no repeated API calls)
3. **Karaoke subtitles** use precise timing instead of estimated timing

### Workflow

**Step 1: Generate timestamps (one-time, costs ~$0.006/min)**
```bash
source .venv/bin/activate && python generate_subtitles.py
```

**Step 2: Recompose video (free, uses cached timestamps)**
```bash
source .venv/bin/activate && python recompose_video.py
```

### Output Files
```
output/subtitles/
├── voice_01.ass          # ASS subtitle file (karaoke format)
├── voice_02.ass
├── voice_03.ass
└── timestamps/
    ├── voice_01.json     # Word-level timestamps (cached)
    ├── voice_02.json
    └── voice_03.json
```

### Timestamp JSON Format
```json
{
  "full_text": "Ever had works on my machine...",
  "duration": 15.2,
  "language": "en",
  "words": [
    {"word": "Ever", "start": 0.0, "end": 0.32},
    {"word": "had", "start": 0.32, "end": 0.48},
    {"word": "works", "start": 0.48, "end": 0.72}
  ]
}
```

### Benefits
- ✅ **Precise sync** - Words appear exactly when spoken
- ✅ **Pay once** - Whisper API called only once per audio file
- ✅ **Fast regeneration** - Recompose uses cached timestamps (no API calls)
- ✅ **Debuggable** - JSON files show exact word timing
- ✅ **ASS files** - Industry-standard subtitle format

### Cost
- Whisper API: ~$0.006 per minute of audio
- For a 60-second video: ~$0.006 total
- Cached timestamps: FREE for all future regenerations

### Commands Reference
```bash
# Generate timestamps for all audio files
python generate_subtitles.py

# Generate with custom settings
python generate_subtitles.py --words-per-line 4 --font-size 60

# Recompose video (uses cached timestamps automatically)
python recompose_video.py
```

### Audio Aligner Module
Located at `src/pipeline/audio_aligner.py`:
- `AudioAligner` class - Main alignment interface
- `get_word_timestamps()` - Get word timings from audio
- `generate_ass_file()` - Create .ass subtitle file
- `save_timestamps_json()` - Cache timestamps as JSON