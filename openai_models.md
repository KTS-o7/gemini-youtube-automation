# OpenAI Models Reference Guide

> **Last Updated:** Fetched from OpenAI documentation on January 2025
> **Latest Flagship Model:** GPT-5.2

---

## Table of Contents

1. [GPT-5 Series (Text Generation)](#gpt-5-series-text-generation)
2. [Image Generation Models](#image-generation-models)
3. [Audio Models (TTS & STT)](#audio-models-tts--stt)
4. [Video Generation Models](#video-generation-models)
5. [Pricing Reference](#pricing-reference)
6. [Structured Outputs](#structured-outputs)
7. [Images & Vision](#images--vision)
8. [Recommendations for This Project](#recommendations-for-this-project)

---

## GPT-5 Series (Text Generation)

### GPT-5.2 (Latest Flagship)
- **Model ID:** `gpt-5.2`
- **Description:** The best model for coding and agentic tasks across industries
- **Supports:** Structured Outputs, Function Calling, Streaming, Vision

### GPT-5.1
- **Model ID:** `gpt-5.1`
- **Description:** Best model for coding and agentic tasks with configurable reasoning effort
- **Supports:** Structured Outputs, Function Calling, Streaming, Vision

### GPT-5
- **Model ID:** `gpt-5`
- **Description:** Intelligent reasoning model for coding and agentic tasks
- **Supports:** Structured Outputs, Function Calling, Streaming, Vision

### GPT-5 mini
- **Model ID:** `gpt-5-mini`
- **Description:** A faster, cost-efficient version of GPT-5 for well-defined tasks
- **Best For:** Production workloads with good cost/performance balance
- **Supports:** Structured Outputs, Function Calling, Streaming, Vision

### GPT-5 nano
- **Model ID:** `gpt-5-nano`
- **Description:** Fastest, most cost-efficient version of GPT-5
- **Best For:** High-volume, latency-sensitive applications
- **Supports:** Structured Outputs, Function Calling, Streaming, Vision

### GPT-5 Pro Variants

| Model | Description |
|-------|-------------|
| `gpt-5.2-pro` | Version of GPT-5.2 with more compute for smarter, more precise responses |
| `gpt-5-pro` | Version of GPT-5 with more compute for better responses |

### Codex Variants (Specialized for Coding)

| Model | Description |
|-------|-------------|
| `gpt-5.1-codex` | Optimized for agentic coding in Codex |
| `gpt-5.1-codex-max` | Most intelligent coding model for long-horizon tasks |
| `gpt-5.1-codex-mini` | Smaller, cost-effective coding model |
| `gpt-5-codex` | GPT-5 optimized for agentic coding |

---

## Image Generation Models

### GPT Image 1.5 (Latest)
- **Model ID:** `gpt-image-1.5`
- **Description:** State-of-the-art image generation model
- **Capabilities:** 
  - Natively multimodal (understands text AND images)
  - Leverages world knowledge for realistic image generation
  - Better instruction following and contextual awareness

### GPT Image 1
- **Model ID:** `gpt-image-1`
- **Description:** Previous generation image model
- **Capabilities:** Both analyze and generate images

### GPT Image 1 Mini
- **Model ID:** `gpt-image-1-mini`
- **Description:** Cost-efficient version of GPT Image 1
- **Best For:** Budget-conscious image generation

### ChatGPT Image Latest
- **Model ID:** `chatgpt-image-latest`
- **Description:** Image model used in ChatGPT

### Image Sizes Supported

| Aspect Ratio | Dimensions |
|--------------|------------|
| Square | 1024 × 1024 |
| Portrait | 1024 × 1536, 1024 × 1792 |
| Landscape | 1536 × 1024, 1792 × 1024 |

### Image Quality Levels

| Quality | Description | Cost Impact |
|---------|-------------|-------------|
| `low` | Fastest, lowest cost | Cheapest |
| `medium` | Balanced quality/cost | Medium |
| `high` | Best quality | Most expensive |

---

## Audio Models (TTS & STT)

### Text-to-Speech (TTS)

| Model | Description | Cost |
|-------|-------------|------|
| `gpt-4o-mini-tts` | TTS powered by GPT-4o mini, supports `instructions` | $0.015/minute |
| `tts-1` | Optimized for speed | $15.00/1M characters |
| `tts-1-hd` | Optimized for quality | $30.00/1M characters |

#### API Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | `gpt-4o-mini-tts`, `tts-1`, or `tts-1-hd` |
| `input` | string | ✅ | Text to generate audio for (max 4096 characters) |
| `voice` | string | ✅ | Voice ID (see below) |
| `instructions` | string | ❌ | Control voice style (**only `gpt-4o-mini-tts`**) |
| `response_format` | string | ❌ | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | number | ❌ | Speed from `0.25` to `4.0` (default: `1.0`) |

#### Available Voices

**All 13 voices for `gpt-4o-mini-tts`:**

| Voice | Notes |
|-------|-------|
| `alloy` | Neutral, balanced |
| `ash` | |
| `ballad` | |
| `coral` | Popular choice |
| `echo` | Warm, conversational |
| `fable` | Expressive, storytelling |
| `nova` | Friendly, upbeat |
| `onyx` | Deep, authoritative |
| `sage` | |
| `shimmer` | Clear, professional |
| `verse` | |
| `marin` | ⭐ **Best quality (recommended)** |
| `cedar` | ⭐ **Best quality (recommended)** |

**Voices for `tts-1` and `tts-1-hd` (9 voices only):**
`alloy`, `ash`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`

#### The `instructions` Parameter (gpt-4o-mini-tts only)

Control speech characteristics with natural language instructions:

| Aspect | Example Instructions |
|--------|---------------------|
| **Tone** | `"Speak in a cheerful and positive tone."` |
| **Emotion** | `"Speak with excitement and enthusiasm."` |
| **Speed** | `"Speak slowly and deliberately."` |
| **Accent** | `"Speak with a British accent."` |
| **Style** | `"Speak like a news anchor."` |
| **Whispering** | `"Speak in a soft whisper."` |
| **Energy** | `"Speak with high energy like a sports commentator."` |

**Example with instructions:**
```python
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input="Today is a wonderful day to build something people love!",
    instructions="Speak in a cheerful and positive tone.",
)

# Save to file
with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

**Example for educational video narration:**
```python
response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="marin",  # Best quality voice
    input="Machine learning is a subset of artificial intelligence...",
    instructions="Speak clearly and professionally, like an educational video narrator. Use a warm, engaging tone that keeps viewers interested.",
    response_format="wav",  # Best for video editing
)
```

#### Output Formats

| Format | Use Case |
|--------|----------|
| `mp3` | Default, general use |
| `wav` | Video editing, low latency (recommended for this project) |
| `pcm` | Raw audio, lowest latency |
| `opus` | Streaming, low latency |
| `aac` | YouTube, iOS, Android |
| `flac` | Lossless archiving |

#### Custom Voices

Custom voices are available for eligible customers (contact sales@openai.com):
- Provide a 30-second audio sample
- Requires consent recording from voice actor
- Up to 20 custom voices per organization

### Speech-to-Text (STT)

| Model | Description | Cost |
|-------|-------------|------|
| `gpt-4o-transcribe` | STT powered by GPT-4o | $0.006/minute |
| `gpt-4o-mini-transcribe` | STT powered by GPT-4o mini | $0.003/minute |
| `gpt-4o-transcribe-diarize` | Identifies who's speaking | $0.006/minute |
| `whisper-1` | General-purpose speech recognition | $0.006/minute |

---

## Video Generation Models

### Sora 2
- **Model ID:** `sora-2`
- **Description:** Flagship video generation with synced audio
- **Resolutions:** 
  - Portrait: 720×1280
  - Landscape: 1280×720

### Sora 2 Pro
- **Model ID:** `sora-2-pro`
- **Description:** Most advanced synced-audio video generation
- **Resolutions:**
  - Standard: 720×1280 / 1280×720
  - High: 1024×1792 / 1792×1024

---

## Pricing Reference

### GPT-5 Text Tokens (Standard Tier, per 1M tokens)

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| **gpt-5.2** | $1.75 | $0.175 | $14.00 |
| **gpt-5.1** | $1.25 | $0.125 | $10.00 |
| **gpt-5** | $1.25 | $0.125 | $10.00 |
| **gpt-5-mini** | $0.25 | $0.025 | $2.00 |
| **gpt-5-nano** | $0.05 | $0.005 | $0.40 |
| gpt-5.2-pro | $21.00 | - | $168.00 |
| gpt-5-pro | $15.00 | - | $120.00 |

### Batch Tier (50% discount, per 1M tokens)

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| gpt-5.2 | $0.875 | $0.0875 | $7.00 |
| gpt-5.1 | $0.625 | $0.0625 | $5.00 |
| gpt-5 | $0.625 | $0.0625 | $5.00 |
| gpt-5-mini | $0.125 | $0.0125 | $1.00 |
| gpt-5-nano | $0.025 | $0.0025 | $0.20 |

### Image Generation (per image, 1024×1024)

| Model | Low | Medium | High |
|-------|-----|--------|------|
| **GPT Image 1.5** | $0.009 | $0.034 | $0.133 |
| **GPT Image Latest** | $0.009 | $0.034 | $0.133 |
| GPT Image 1 | $0.011 | $0.042 | $0.167 |
| **gpt-image-1-mini** | $0.005 | $0.011 | $0.036 |

### Image Generation (per image, 1024×1536 / 1536×1024)

| Model | Low | Medium | High |
|-------|-----|--------|------|
| GPT Image 1.5 | $0.013 | $0.05 | $0.20 |
| GPT Image 1 | $0.016 | $0.063 | $0.25 |
| **gpt-image-1-mini** | $0.006 | $0.015 | $0.052 |

### Video Generation (per second)

| Model | Resolution | Price/second |
|-------|------------|--------------|
| sora-2 | 720p | $0.10 |
| sora-2-pro | 720p | $0.30 |
| sora-2-pro | 1080p | $0.50 |

### Audio (TTS & STT)

| Model | Cost |
|-------|------|
| gpt-4o-mini-tts | $0.015/minute |
| gpt-4o-transcribe | $0.006/minute |
| gpt-4o-mini-transcribe | $0.003/minute |
| Whisper | $0.006/minute |
| TTS-1 | $15.00/1M characters |
| TTS-1-HD | $30.00/1M characters |

---

## Structured Outputs

Structured Outputs ensures the model generates JSON that adheres to your defined schema.

### Supported GPT-5 Models
- All GPT-5 variants (gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano)
- All Pro and Codex variants

### How to Use

#### Python with Pydantic
```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-5-mini",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed
```

### Supported Schema Types
- String (with `pattern`, `format` constraints)
- Number (with `minimum`, `maximum`, `multipleOf`)
- Boolean
- Integer
- Object
- Array (with `minItems`, `maxItems`)
- Enum
- anyOf

### Schema Constraints
- All fields must be `required` (use `null` union for optional)
- `additionalProperties: false` required on all objects
- Max 5000 object properties total
- Max 10 levels of nesting
- Max 1000 enum values across all properties
- Recursive schemas supported

---

## Images & Vision

### Image Input Methods
1. **URL:** Pass a fully qualified image URL
2. **Base64:** Pass as data URL (`data:image/jpeg;base64,...`)
3. **File ID:** Upload via Files API first

### Supported Image Formats
- PNG (.png)
- JPEG (.jpeg, .jpg)
- WEBP (.webp)
- Non-animated GIF (.gif)

### Size Limits
- Up to 50 MB total payload per request
- Up to 500 individual images per request

### Detail Levels

| Level | Description | Token Cost |
|-------|-------------|------------|
| `low` | 512×512 preview | ~85 tokens |
| `high` | Full resolution analysis | Variable by size |
| `auto` | Model decides | Variable |

### Image Token Calculation (GPT-5-mini, GPT-5-nano)

1. Calculate 32×32 patches needed to cover image
2. Scale down if patches > 1536
3. Apply model multiplier:
   - `gpt-5-mini`: 1.62×
   - `gpt-5-nano`: 2.46×

---

## Recommendations for This Project

### Text Generation
| Use Case | Model | Cost (per 1M tokens) |
|----------|-------|---------------------|
| **Best Quality** | `gpt-5.2` | $1.75 / $14.00 |
| **Balanced (Recommended)** | `gpt-5-mini` | $0.25 / $2.00 |
| **Cheapest** | `gpt-5-nano` | $0.05 / $0.40 |

### Image Generation
| Use Case | Model | Quality | Cost |
|----------|-------|---------|------|
| **Cheapest (Recommended)** | `gpt-image-1-mini` | low | $0.005/image |
| **Better Quality** | `gpt-image-1-mini` | medium | $0.011/image |
| **Best Quality** | `gpt-image-1.5` | medium | $0.034/image |

### Text-to-Speech
| Use Case | Model | Cost |
|----------|-------|------|
| **Recommended** | `gpt-4o-mini-tts` | $0.015/minute |
| **Free Alternative** | gTTS | $0.00 |

### Speech-to-Text (Whisper Alignment)
| Use Case | Model | Cost |
|----------|-------|------|
| **Cheapest** | `gpt-4o-mini-transcribe` | $0.003/minute |
| **Standard** | `whisper-1` | $0.006/minute |

### Estimated Cost Per Video

| Component | Model | Short (60s, 3 scenes) | Long (5min, 8 scenes) |
|-----------|-------|----------------------|----------------------|
| Text Gen | gpt-5-mini | ~$0.01 | ~$0.03 |
| Images | gpt-image-1-mini (low) | $0.015 | $0.040 |
| TTS | gpt-4o-mini-tts | $0.015 | $0.075 |
| Whisper | whisper-1 | $0.006 | $0.030 |
| **Total** | | **~$0.05** | **~$0.18** |

---

## Quick Reference Card

```
GPT-5 TEXT MODELS:
├── gpt-5.2      → Best quality ($1.75/$14.00 per 1M tokens)
├── gpt-5.1      → Great quality ($1.25/$10.00 per 1M tokens)
├── gpt-5        → Standard ($1.25/$10.00 per 1M tokens)
├── gpt-5-mini   → Balanced ⭐ ($0.25/$2.00 per 1M tokens)
└── gpt-5-nano   → Cheapest ($0.05/$0.40 per 1M tokens)

IMAGE GENERATION:
├── gpt-image-1.5      → Best quality ($0.009-$0.133/image)
├── gpt-image-1        → Good quality ($0.011-$0.167/image)
└── gpt-image-1-mini   → Cheapest ⭐ ($0.005-$0.036/image)

TTS:
├── gpt-4o-mini-tts    → Recommended ⭐ ($0.015/min)
├── tts-1              → Speed optimized ($15/1M chars)
└── tts-1-hd           → Quality optimized ($30/1M chars)

STT:
├── gpt-4o-mini-transcribe → Cheapest ($0.003/min)
├── whisper-1              → Standard ($0.006/min)
└── gpt-4o-transcribe      → Best ($0.006/min)

VIDEO:
├── sora-2             → Standard ($0.10/sec)
└── sora-2-pro         → Premium ($0.30-$0.50/sec)
```

---

## API Endpoints Reference

| Endpoint | Purpose |
|----------|---------|
| `/v1/responses` | New unified API (recommended) |
| `/v1/chat/completions` | Chat completions (legacy) |
| `/v1/images/generate` | Image generation |
| `/v1/audio/speech` | Text-to-speech |
| `/v1/audio/transcriptions` | Speech-to-text |