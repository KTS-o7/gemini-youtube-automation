# Sound Effects for Viral Shorts

This directory contains sound effects for video transitions and emphasis moments.

## Recommended Sound Effects

Add the following sound effect files to this directory:

### Transition Sounds
- `whoosh.mp3` - Quick whoosh for scene transitions
- `pop.mp3` - Pop sound for text/element appearance
- `swoosh.mp3` - Swoosh for sliding transitions

### Emphasis Sounds
- `ding.mp3` - Notification/highlight sound
- `impact.mp3` - Bass drop for dramatic moments
- `rise.mp3` - Rising tension sound
- `success.mp3` - Achievement/success sound

### Ambient
- `typing.mp3` - Keyboard typing sound
- `click.mp3` - UI click sound

## Where to Get Free Sound Effects

1. **Pixabay** (Free, no attribution required)
   - https://pixabay.com/sound-effects/

2. **Freesound** (Free, various licenses)
   - https://freesound.org/

3. **Mixkit** (Free for commercial use)
   - https://mixkit.co/free-sound-effects/

4. **YouTube Audio Library** (Free for YouTube videos)
   - https://studio.youtube.com/channel/UC/music

## File Format Requirements

- **Format**: MP3, WAV, or OGG
- **Duration**: 0.5 - 2 seconds recommended
- **Sample Rate**: 44.1kHz or 48kHz
- **Bitrate**: 128kbps+ for MP3

## Usage in Code

Sound effects are automatically detected by the `VideoComposer` when placed in this directory:

```python
# In video_composer.py
self.sound_effects_dir = Path("assets/sfx")

# To add a sound effect at a specific time:
composer._add_sound_effect(clip, "whoosh", position=0.5)
```

## Quick Setup Script

Run this to download some basic sound effects (requires curl):

```bash
# Example - download from Pixabay (check their current URLs)
# curl -o whoosh.mp3 "https://example.com/whoosh.mp3"
```

## Volume Levels

Sound effects are played at 25% volume by default (configurable via `self.sfx_volume` in VideoComposer).
Adjust individual effect volumes during editing if needed.