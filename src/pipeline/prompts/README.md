# Prompts Directory

This directory contains all system and user prompts used by the video generation pipeline. Prompts are stored as separate `.txt` files so you can easily tweak them without modifying code.

## 📁 File Structure

```
prompts/
├── __init__.py                      # Prompt loader utilities
├── README.md                        # This file
├── script_writer_short_system.md    # System prompt for short-form scripts
├── script_writer_short_user.md      # User prompt template for short-form scripts
├── script_writer_long_system.md     # System prompt for long-form scripts
├── script_writer_long_user.md       # User prompt template for long-form scripts
├── researcher_system.md             # System prompt for content research
├── researcher_user.md               # User prompt template for research
├── scene_planner_image_prompt.md    # Template for image generation prompts
└── image_generator_thumbnail.md     # Template for thumbnail generation
```

## 🚀 Quick Start

### Editing Prompts

1. Open any `.md` file in this directory
2. Edit the content as needed (with full markdown syntax highlighting!)
3. Save the file
4. Run your video generation - changes take effect immediately

### Hot Reloading (Development)

If you're actively tweaking prompts, you can reload them without restarting:

```python
from src.pipeline.prompts import reload_prompts

# Clear the cache to reload all prompts from disk
reload_prompts()
```

## 📝 Template Variables

Prompts use Python's `str.format()` syntax with `{variable_name}` placeholders.

### Script Writer (Short-form)
| Variable | Description |
|----------|-------------|
| `{topic}` | The video topic |
| `{target_audience}` | Who the video is for |
| `{style}` | Video style (educational, casual, etc.) |
| `{key_points_text}` | Bullet list of key points from research |
| `{examples_text}` | Examples section (may be empty) |
| `{analogies_text}` | Analogies section (may be empty) |

### Script Writer (Long-form)
Same as short-form, plus:
| Variable | Description |
|----------|-------------|
| `{facts_text}` | Facts section with sources |
| `{min_duration}` | Minimum duration in minutes |
| `{max_duration}` | Maximum duration in minutes |

### Researcher
| Variable | Description |
|----------|-------------|
| `{topic}` | The research topic |
| `{target_audience}` | Target audience |
| `{format}` | "short" or "long" |
| `{style}` | Content style preference |

### Scene Planner (Image Prompt)
| Variable | Description |
|----------|-------------|
| `{scene_number}` | Current scene number |
| `{total_scenes}` | Total number of scenes |
| `{cleaned_visual_description}` | Visual description (animation words removed) |
| `{position_context}` | "Opening scene", "Middle scene", etc. |
| `{ken_burns_hint}` | Hint for zoom/pan composition |
| `{continuity_instruction}` | Instructions for visual continuity |
| `{color_palette}` | Color palette for the visual theme |
| `{lighting}` | Lighting style |
| `{visual_style}` | Overall visual aesthetic |
| `{recurring_motifs}` | Visual elements to repeat |
| `{key_visual_elements}` | Must-include elements for this scene |
| `{mood}` | Scene mood (uppercase) |
| `{mood_visual_guide}` | Visual guidance based on mood |
| `{composition_hint}` | Composition guidance |
| `{aspect_ratio}` | "9:16 (vertical)" or "16:9 (horizontal)" |
| `{format_specific_instructions}` | Format-specific composition tips |

### Image Generator (Thumbnail)
| Variable | Description |
|----------|-------------|
| `{title}` | Video title |
| `{topic}` | Video topic |
| `{target_audience}` | Target audience |
| `{format_type}` | "short-form vertical" or "long-form horizontal" |
| `{orientation}` | "Vertical (9:16)" or "Horizontal (16:9)" |
| `{viewing_context}` | "mobile viewing" or "desktop viewing" |
| `{mood}` | Desired mood/feeling |

## 🎯 Best Practices

### For Better Scripts

1. **Emphasize continuity**: Add more examples of good vs. bad transitions
2. **Be specific**: Generic instructions produce generic results
3. **Include examples**: Show don't tell - provide concrete examples
4. **Test mentally**: Read your prompt aloud - does it flow naturally?

### For Better Images

1. **Static descriptions**: Always describe frozen moments, not animations
2. **Depth layers**: Include foreground, midground, background for Ken Burns
3. **No text**: Repeatedly emphasize NO TEXT in images
4. **Consistent style**: Reference the color palette and visual theme

### For Better Research

1. **Audience-specific**: Tailor complexity and examples to the audience
2. **Narrative flow**: Key points should tell a story, not just list facts
3. **Concrete examples**: Specific is better than generic

## 🔧 Programmatic Access

```python
from src.pipeline.prompts import (
    PromptLoader,
    get_system_prompt,
    get_user_prompt_template,
    ScriptPrompts,
    ResearcherPrompts,
    ScenePlannerPrompts,
)

# Load a prompt directly
template = PromptLoader.load("script_writer_short_user")

# Use convenience functions
system = get_system_prompt("script_writer_short")  # auto-adds _system suffix
user = get_user_prompt_template("script_writer_short")  # auto-adds _user suffix

# Use organized classes
system = ScriptPrompts.short_system()
user = ScriptPrompts.short_user()

# List all available prompts
print(PromptLoader.list_available())
```

## 🔄 Fallback Behavior

If a prompt file is missing or has formatting errors, the pipeline will fall back to hardcoded prompts built into the Python code. This ensures the pipeline doesn't break even if prompt files are corrupted.

## 📄 Why Markdown?

Prompt files use `.md` extension for:
- **Syntax highlighting** in editors and GitHub
- **Better readability** with headers, lists, and formatting
- **Easy diffing** in version control
- **Native preview** in most code editors

## 💡 Tips for Experimentation

1. **A/B Testing**: Create copies like `script_writer_short_user_v2.md` and manually switch between them
2. **Version Control**: These are just text files - use git to track changes
3. **Logging**: Add print statements in `__init__.py` to see which prompts are loaded
4. **Isolation**: Test one prompt change at a time to understand its impact

## 📊 Prompt Engineering Guidelines

### For Narrative Continuity
- Emphasize "ONE CONTINUOUS STORY" multiple times
- Require transition phrases (e.g., "But here's the thing...")
- Demand that the final scene resolves the opening hook
- Provide BAD and GOOD examples

### For Static Images (Ken Burns)
- Explicitly state "STATIC IMAGE" or "FROZEN MOMENT"
- Forbid animation words: moving, flying, zooming, transforming
- Require depth layers: foreground, midground, background
- Use photography terminology: close-up, wide shot, depth of field

### For Visual Consistency
- Define color palette explicitly
- Require recurring visual motifs
- Reference "same video" or "next frame"
- Maintain lighting and style consistency