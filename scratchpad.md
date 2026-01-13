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

### Viral Shorts Improvements (Priority)
- [ ] 🔥 Karaoke-style captions - Word-by-word highlight with color emphasis (HIGH IMPACT)
- [ ] ⚡ Faster scene pacing - Force max 5-8 seconds per scene for shorts
- [ ] 🎵 Background music integration - Add actual music files + auto-sync
- [ ] 🔊 Sound effects - Whooshes, pops, emphasis sounds at transitions
- [ ] 📱 Larger/bolder subtitles - More visible on mobile (bigger font, maybe yellow/white)
- [ ] 🎬 Dynamic motion effects - Quick zooms, slight shakes for emphasis
- [ ] 😀 Emoji support in captions - Add relevant emojis to key phrases
- [ ] 🎨 Color emphasis - Highlight key words in different colors

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

## Error Handling Philosophy
- **No fallbacks**: If AI fails, the pipeline fails immediately
- **Fail loud**: Clear error messages instead of silent degradation
- **Quality over availability**: Better to fail than produce garbage content