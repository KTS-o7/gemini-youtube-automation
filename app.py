#!/usr/bin/env python3
"""
AI Video Generation Pipeline - Streamlit Application

A beautiful, interactive web interface for generating AI-powered videos.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Handle imports - works with pip install -e . or direct script execution
from pydantic import BaseModel

try:
    from src.pipeline import TTSProvider, VideoFormat, VideoPipeline, VideoRequest
    from src.pipeline.voice_generator import VoiceConfig
    from src.utils.ai_client import AIClient
except ImportError:
    # Fallback for running without package installation
    sys.path.insert(0, str(Path(__file__).parent))
    from src.pipeline import TTSProvider, VideoFormat, VideoPipeline, VideoRequest
    from src.pipeline.voice_generator import VoiceConfig
    from src.utils.ai_client import AIClient


# =============================================================================
# Pydantic Models for Structured AI Outputs
# =============================================================================
class ImprovedTopic(BaseModel):
    """Structured output for improved topic."""

    improved_topic: str
    why_better: str


class ImprovedVoiceInstructions(BaseModel):
    """Structured output for improved voice instructions."""

    instructions: str
    tone: str
    pacing: str


class TopicSuggestions(BaseModel):
    """Structured output for topic suggestions."""

    topics: list[str]


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
    }
    .error-box {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
    }
    .info-box {
        background-color: #D1ECF1;
        border: 1px solid #BEE5EB;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Session State Initialization
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "api_key_valid": False,
        "generation_in_progress": False,
        "generation_complete": False,
        "current_stage": "",
        "progress": 0,
        "output": None,
        "logs": [],
        "settings": {
            "ai_provider": os.environ.get("AI_PROVIDER", "openai"),
            "tts_provider": os.environ.get("TTS_PROVIDER", "openai"),
            "tts_voice": os.environ.get("TTS_VOICE", "marin"),
            "tts_speed": float(os.environ.get("TTS_SPEED", "1.0")),
            "viral_mode": True,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================
def validate_api_keys() -> tuple[bool, str]:
    """Validate that required API keys are set."""
    provider = st.session_state.settings["ai_provider"]

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "":
            return False, "OpenAI API key not set. Please configure in Settings."
        return True, "OpenAI API key configured ✓"
    elif provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "":
            return False, "Google API key not set. Please configure in Settings."
        return True, "Google API key configured ✓"
    return False, "Unknown provider"


def get_output_files() -> dict:
    """Get all output files organized by type."""
    output_dir = Path("output")
    files = {
        "videos": [],
        "images": [],
        "audio": [],
        "json": [],
        "subtitles": [],
    }

    if not output_dir.exists():
        return files

    # Videos
    files["videos"] = list(output_dir.glob("*.mp4"))

    # Images
    images_dir = output_dir / "images"
    if images_dir.exists():
        files["images"] = list(images_dir.glob("*.png")) + list(
            images_dir.glob("*.jpg")
        )

    # Audio
    audio_dir = output_dir / "audio"
    if audio_dir.exists():
        files["audio"] = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    # JSON files
    files["json"] = list(output_dir.glob("*.json"))

    # Subtitles
    subtitles_dir = output_dir / "subtitles"
    if subtitles_dir.exists():
        files["subtitles"] = list(subtitles_dir.glob("*.ass")) + list(
            subtitles_dir.glob("*.srt")
        )

    return files


def load_json_file(path: Path) -> dict:
    """Load and return JSON file contents."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


# =============================================================================
# AI Improver Functions
# =============================================================================
@st.cache_data(ttl=300)
def improve_topic(topic: str, audience: str, video_format: str) -> str:
    """Use AI to improve a topic/question for better video content."""
    try:
        client = AIClient()

        system_prompt = """You are an expert content strategist for viral video content.
Your job is to improve video topics/questions to make them more engaging, clear, and optimized for the target platform.

Guidelines:
- Make the topic specific and actionable
- Add curiosity hooks where appropriate
- Ensure it's clear what the viewer will learn
- Optimize for the video format (short = punchy, long = comprehensive)
- Keep it concise but compelling"""

        prompt = f"""Improve this video topic for maximum engagement:

Topic: {topic}
Target Audience: {audience}
Video Format: {video_format} ({"< 60 seconds, vertical, TikTok/Reels style" if video_format == "short" else "3-10 minutes, horizontal, YouTube style"})"""

        result = client.generate_structured(
            prompt=prompt,
            response_model=ImprovedTopic,
            system_prompt=system_prompt,
        )
        return result.improved_topic
    except Exception as e:
        st.error(f"Failed to improve topic: {e}")
        return topic


@st.cache_data(ttl=300)
def improve_voice_instructions(
    instructions: str, topic: str, audience: str, voice: str
) -> str:
    """Use AI to improve voice instructions for better TTS output."""
    try:
        client = AIClient()

        system_prompt = """You are an expert voice director for AI text-to-speech systems.
Your job is to write clear, effective voice instructions that will make the AI voice sound natural and engaging.

Guidelines:
- Be specific about tone, pace, and emotion
- Include guidance on emphasis and pauses
- Match the voice style to the content and audience
- Keep instructions concise but comprehensive
- Focus on what makes the delivery compelling"""

        current = instructions if instructions else "No current instructions"

        prompt = f"""Create optimal voice instructions for this video:

Current Instructions: {current}
Topic: {topic}
Target Audience: {audience}
Voice: {voice}

Write voice instructions that will make the narration engaging and professional."""

        result = client.generate_structured(
            prompt=prompt,
            response_model=ImprovedVoiceInstructions,
            system_prompt=system_prompt,
        )
        return result.instructions
    except Exception as e:
        st.error(f"Failed to improve voice instructions: {e}")
        return instructions


def get_topic_suggestions(audience: str, video_format: str) -> tuple[list[str], str]:
    """Generate topic suggestions based on audience and format.

    Returns:
        Tuple of (list of suggestions, error message if any)
    """
    try:
        client = AIClient()

        system_prompt = """You are a viral content strategist. Generate engaging video topic ideas that are specific, actionable, and optimized for engagement."""

        prompt = f"""Generate 5 engaging video topic ideas for:
Audience: {audience if audience else "general audience"}
Format: {video_format} ({"short viral content under 60 seconds" if video_format == "short" else "educational long-form 3-10 minutes"})

Each topic should be a compelling question or statement that would make someone want to watch."""

        result = client.generate_structured(
            prompt=prompt,
            response_model=TopicSuggestions,
            system_prompt=system_prompt,
        )
        return result.topics[:5], ""
    except Exception as e:
        return [], str(e)


# =============================================================================
# Page: Generate Video
# =============================================================================
def page_generate():
    """Video generation page."""
    st.markdown('<p class="main-header">🎬 Generate Video</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Create AI-powered videos from any topic</p>',
        unsafe_allow_html=True,
    )

    # Check API key
    is_valid, message = validate_api_keys()
    if not is_valid:
        st.warning(f"⚠️ {message}")
        st.info("Go to **Settings** in the sidebar to configure your API keys.")
        return

    # Initialize session state
    if "topic_value" not in st.session_state:
        st.session_state.topic_value = ""
    if "voice_instructions_value" not in st.session_state:
        st.session_state.voice_instructions_value = ""

    # Basic settings (outside form for AI improver to work)
    st.markdown("### 📝 Content Settings")

    col1, col2 = st.columns(2)

    with col1:
        format_option = st.selectbox(
            "📐 Video Format",
            options=["short", "long"],
            format_func=lambda x: (
                "Short (< 60s, 9:16 vertical)"
                if x == "short"
                else "Long (3-10 min, 16:9 horizontal)"
            ),
            help="Short videos are vertical (TikTok/Reels), Long videos are horizontal (YouTube)",
            key="format_select",
        )

        audience = st.text_input(
            "👥 Target Audience",
            placeholder="e.g., developers, beginners, general audience",
            help="Who is this video for?",
            key="audience_input",
        )

    with col2:
        tts_provider = st.selectbox(
            "🎤 Voice Provider",
            options=["openai", "gtts", "elevenlabs"],
            format_func=lambda x: {
                "openai": "OpenAI TTS ($0.015/min) - Best Quality",
                "gtts": "Google TTS (Free)",
                "elevenlabs": "ElevenLabs (Premium)",
            }.get(x, x),
            key="tts_select",
        )

        if tts_provider == "openai":
            voice = st.selectbox(
                "🗣️ Voice",
                options=[
                    "marin",
                    "cedar",
                    "alloy",
                    "ash",
                    "ballad",
                    "coral",
                    "echo",
                    "fable",
                    "nova",
                    "onyx",
                    "sage",
                    "shimmer",
                    "verse",
                ],
                help="marin and cedar are recommended for best quality",
                key="voice_select",
            )
        else:
            voice = "marin"

    st.markdown("---")

    # Topic with AI Improver
    st.markdown("### 📌 Topic / Question")

    topic_col1, topic_col2 = st.columns([4, 1])

    with topic_col1:
        topic = st.text_area(
            "Enter your topic",
            value=st.session_state.topic_value,
            placeholder="e.g., What is Docker and why should developers use it?",
            height=100,
            help="The main topic or question your video will answer",
            label_visibility="collapsed",
        )
        # Sync back to state
        st.session_state.topic_value = topic

    with topic_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        improve_topic_clicked = st.button(
            "✨ Improve", key="improve_topic_btn", help="Use AI to improve your topic"
        )
        suggest_topic_clicked = st.button(
            "💡 Suggest", key="suggest_topic_btn", help="Get AI-generated topic ideas"
        )

    # Handle improve button
    if improve_topic_clicked:
        if st.session_state.topic_value.strip():
            with st.spinner("Improving topic..."):
                improved = improve_topic(
                    st.session_state.topic_value, audience, format_option
                )
                st.session_state.topic_value = improved
                st.rerun()
        else:
            st.warning("Enter a topic first!")

    # Handle suggest button
    if suggest_topic_clicked:
        with st.spinner("Generating ideas..."):
            suggestions, error = get_topic_suggestions(audience, format_option)
            if suggestions:
                st.session_state.topic_suggestions = suggestions
            elif error:
                st.error(f"Failed to generate suggestions: {error}")
            else:
                st.warning("No suggestions generated. Try again.")

    # Show topic suggestions if available
    if "topic_suggestions" in st.session_state and st.session_state.topic_suggestions:
        st.markdown("**💡 Suggested Topics:**")
        for i, suggestion in enumerate(st.session_state.topic_suggestions):
            if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                st.session_state.topic_value = suggestion
                st.session_state.topic_suggestions = []
                st.rerun()
        if st.button("❌ Clear suggestions", key="clear_suggestions"):
            st.session_state.topic_suggestions = []
            st.rerun()

    st.markdown("---")

    # Voice Instructions with AI Improver (only for OpenAI)
    if tts_provider == "openai":
        st.markdown("### 🎭 Voice Instructions")

        voice_col1, voice_col2 = st.columns([4, 1])

        with voice_col1:
            voice_instructions = st.text_area(
                "Voice style instructions",
                value=st.session_state.voice_instructions_value,
                placeholder="e.g., Speak in a calm, professional tone with clear enunciation. Add slight enthusiasm when explaining key concepts.",
                height=100,
                help="Natural language instructions to control voice style",
                label_visibility="collapsed",
            )
            # Sync back to state
            st.session_state.voice_instructions_value = voice_instructions

        with voice_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            improve_voice_clicked = st.button(
                "✨ Improve",
                key="improve_voice_btn",
                help="Use AI to improve voice instructions",
            )
            auto_voice_clicked = st.button(
                "🎯 Auto", key="auto_voice_btn", help="Auto-generate voice instructions"
            )

        # Handle improve voice button
        if improve_voice_clicked:
            with st.spinner("Improving instructions..."):
                improved = improve_voice_instructions(
                    st.session_state.voice_instructions_value, topic, audience, voice
                )
                st.session_state.voice_instructions_value = improved
                st.rerun()

        # Handle auto voice button
        if auto_voice_clicked:
            with st.spinner("Generating instructions..."):
                auto = improve_voice_instructions("", topic, audience, voice)
                st.session_state.voice_instructions_value = auto
                st.rerun()
    else:
        voice_instructions = ""

    st.markdown("---")

    # Style input
    style = st.text_input(
        "🎨 Style",
        value="educational",
        placeholder="e.g., casual, tutorial, professional",
        help="The tone and style of the video",
        key="style_input",
    )

    # Generation form (simplified - just advanced options and submit)
    with st.form("generation_form"):
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        with st.expander("Configure advanced settings"):
            col3, col4 = st.columns(2)
            with col3:
                viral_mode = st.checkbox(
                    "🔥 Viral Mode",
                    value=True,
                    help="Enable karaoke-style captions, dynamic motion, and faster pacing",
                )
                upload_to_youtube = st.checkbox(
                    "📤 Upload to YouTube",
                    value=False,
                    help="Automatically upload to YouTube after generation",
                )
            with col4:
                cleanup_after = st.checkbox(
                    "🧹 Cleanup Temp Files",
                    value=False,
                    help="Remove temporary files after generation",
                )

        submitted = st.form_submit_button(
            "🚀 Generate Video", width="stretch", type="primary"
        )

    # Handle form submission
    if submitted:
        if not topic or not topic.strip():
            st.error("❌ Please enter a topic")
            return
        if not audience or not audience.strip():
            st.error("❌ Please enter a target audience")
            return

        # Store settings
        st.session_state.settings["tts_provider"] = tts_provider
        st.session_state.settings["tts_voice"] = voice
        st.session_state.settings["viral_mode"] = viral_mode

        # Set environment variables
        os.environ["TTS_PROVIDER"] = tts_provider
        os.environ["TTS_VOICE"] = voice
        if voice_instructions:
            os.environ["TTS_INSTRUCTIONS"] = voice_instructions

        # Run generation
        generate_video(
            topic=topic,
            audience=audience,
            format_option=format_option,
            style=style,
            viral_mode=viral_mode,
            upload=upload_to_youtube,
            cleanup=cleanup_after,
        )


# =============================================================================
# Video Generation Helper Functions (Extracted from God Function)
# =============================================================================


def _create_voice_config() -> VoiceConfig:
    """
    Create VoiceConfig from environment variables.

    This is the single place where TTS-related env vars are read.
    """
    tts_env = os.environ.get("TTS_PROVIDER", "openai").lower()
    tts_provider = (
        TTSProvider(tts_env)
        if tts_env in ["gtts", "openai", "elevenlabs"]
        else TTSProvider.OPENAI
    )

    return VoiceConfig(
        provider=tts_provider,
        openai_voice=os.environ.get("TTS_VOICE", "marin"),
        openai_speed=float(os.environ.get("TTS_SPEED", "1.0")),
        openai_instructions=os.environ.get("TTS_INSTRUCTIONS"),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=os.environ.get(
            "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
        ),
        fallback_enabled=True,
    )


def _create_video_request(
    topic: str,
    audience: str,
    format_option: str,
    style: str,
) -> VideoRequest:
    """Create a VideoRequest from form data."""
    return VideoRequest(
        topic=topic,
        target_audience=audience,
        format=VideoFormat(format_option),
        style=style,
    )


def _create_pipeline(voice_config: VoiceConfig) -> VideoPipeline:
    """Create and configure the video pipeline."""
    return VideoPipeline(
        output_dir=Path("output"),
        voice_config=voice_config,
        subtitle_aligner=os.environ.get("SUBTITLE_ALIGNER", "wav2vec2"),
    )


def _display_stage_progress(stage_container, stages: dict) -> dict:
    """
    Display stage progress indicators.

    Returns:
        Dictionary of stage placeholders for updating later
    """
    with stage_container:
        stage_cols = st.columns(len(stages))
        stage_placeholders = {}
        for i, (stage, (label, _)) in enumerate(stages.items()):
            with stage_cols[i]:
                stage_placeholders[stage] = st.empty()
                stage_placeholders[stage].markdown(f"⏳ {stage.split('_')[0].title()}")
    return stage_placeholders


def _display_video_results(output) -> None:
    """Display the generated video and metadata."""
    st.markdown("---")
    st.markdown("### 📹 Generated Video")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Video player
        if output.video_path and output.video_path.exists():
            st.video(str(output.video_path))

            # Download button
            with open(output.video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f.read(),
                    file_name=output.video_path.name,
                    mime="video/mp4",
                    width="stretch",
                )

    with col2:
        # Thumbnail
        if output.thumbnail_path and output.thumbnail_path.exists():
            st.image(
                str(output.thumbnail_path),
                caption="Thumbnail",
                width="stretch",
            )

        # Metadata
        if output.metadata:
            st.markdown("#### 📋 Video Info")
            st.markdown(f"**Title:** {output.metadata.title}")
            st.markdown(
                f"**Duration:** {format_duration(output.metadata.duration_seconds)}"
            )
            st.markdown(f"**Format:** {output.metadata.format}")

            if output.metadata.tags:
                st.markdown(f"**Tags:** {', '.join(output.metadata.tags[:5])}")


def _handle_youtube_upload(output) -> None:
    """Handle YouTube upload if metadata is available."""
    if not output.metadata:
        st.warning("⚠️ Cannot upload: Video metadata is missing")
        return

    st.markdown("---")
    st.markdown("### 📤 YouTube Upload")
    try:
        from src.uploader import upload_to_youtube

        with st.spinner("Uploading to YouTube..."):
            video_id = upload_to_youtube(
                video_path=output.video_path,
                title=output.metadata.title,
                description=output.metadata.youtube_description(),
                tags=output.metadata.youtube_tags(),
                thumbnail_path=output.thumbnail_path,
            )
            if video_id:
                st.success(f"✅ Uploaded! Video ID: {video_id}")
                st.markdown(
                    f"🔗 [Watch on YouTube](https://www.youtube.com/watch?v={video_id})"
                )
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")


def _mark_stages_complete(stage_placeholders: dict) -> None:
    """Mark all stage placeholders as complete."""
    for stage in stage_placeholders:
        stage_placeholders[stage].markdown("✅ Done")


def generate_video(
    topic: str,
    audience: str,
    format_option: str,
    style: str,
    viral_mode: bool,
    upload: bool,
    cleanup: bool,
):
    """
    Generate video with progress tracking.

    This function orchestrates the video generation process,
    delegating specific tasks to helper functions.
    """
    # Progress UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    stage_container = st.container()

    # Stage definitions (for progress display)
    stages = {
        "validation": ("✅ Validating input...", 5),
        "research": ("🔍 Researching topic...", 15),
        "script_generation": ("📝 Generating script...", 20),
        "scene_planning": ("🎬 Planning scenes...", 15),
        "asset_generation": ("🎨 Generating assets (images + voice)...", 30),
        "video_composition": ("🎥 Composing video...", 10),
        "finalization": ("🖼️ Generating thumbnail & metadata...", 5),
    }

    try:
        status_text.info("🚀 Starting video generation pipeline...")

        # Build configuration
        voice_config = _create_voice_config()
        request = _create_video_request(topic, audience, format_option, style)
        pipeline = _create_pipeline(voice_config)

        # Display progress stages
        stage_placeholders = _display_stage_progress(stage_container, stages)

        # Run pipeline
        status_text.info("🔄 Pipeline running... This may take a few minutes.")
        progress_bar.progress(10)

        output = pipeline.generate_video_sync(request)

        progress_bar.progress(100)

        if output.success:
            status_text.success("🎉 Video generation complete!")
            _mark_stages_complete(stage_placeholders)

            # Show results
            st.balloons()
            _display_video_results(output)

            # Handle YouTube upload if requested
            if upload and output.success and output.metadata:
                _handle_youtube_upload(output)

            # Cleanup if requested
            if cleanup:
                pipeline.cleanup()
                st.info("🧹 Temporary files cleaned up")

        else:
            status_text.error(f"❌ Generation failed: {output.error_message}")

    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Error: {str(e)}")
        st.exception(e)


# =============================================================================
# Page: View Output
# =============================================================================
def page_view_output():
    """View generated outputs page."""
    st.markdown('<p class="main-header">📁 View Output</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Browse and download generated content</p>',
        unsafe_allow_html=True,
    )

    files = get_output_files()

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🎥 Videos", len(files["videos"]))
    col2.metric("🖼️ Images", len(files["images"]))
    col3.metric("🎵 Audio", len(files["audio"]))
    col4.metric("📄 Data Files", len(files["json"]))
    col5.metric("📝 Subtitles", len(files["subtitles"]))

    st.markdown("---")

    # Tabs for different file types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎥 Videos", "🖼️ Images", "🎵 Audio", "📄 Data (JSON)", "📝 Subtitles"]
    )

    with tab1:
        if files["videos"]:
            for video_path in sorted(files["videos"], reverse=True):
                with st.expander(f"📹 {video_path.name}", expanded=True):
                    st.video(str(video_path))
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        with open(video_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name=video_path.name,
                                mime="video/mp4",
                            )
                    with col2:
                        size_mb = video_path.stat().st_size / (1024 * 1024)
                        st.caption(f"Size: {size_mb:.1f} MB")
        else:
            st.info("No videos generated yet. Go to **Generate Video** to create one!")

    with tab2:
        if files["images"]:
            # Grid layout for images
            cols = st.columns(4)
            for i, img_path in enumerate(sorted(files["images"])):
                with cols[i % 4]:
                    st.image(str(img_path), caption=img_path.name, width="stretch")
        else:
            st.info("No images generated yet.")

    with tab3:
        if files["audio"]:
            for audio_path in sorted(files["audio"]):
                with st.expander(f"🎵 {audio_path.name}"):
                    st.audio(str(audio_path))
        else:
            st.info("No audio files generated yet.")

    with tab4:
        if files["json"]:
            for json_path in sorted(files["json"]):
                with st.expander(f"📄 {json_path.name}"):
                    data = load_json_file(json_path)
                    st.json(data)
        else:
            st.info("No data files generated yet.")

    with tab5:
        if files["subtitles"]:
            for sub_path in sorted(files["subtitles"]):
                with st.expander(f"📝 {sub_path.name}"):
                    try:
                        with open(sub_path, "r") as f:
                            content = f.read()
                        st.code(content, language="text")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        else:
            st.info("No subtitle files generated yet.")


# =============================================================================
# Page: Recompose Video
# =============================================================================
def page_recompose():
    """Recompose video from existing assets."""
    st.markdown('<p class="main-header">🔄 Recompose Video</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Regenerate video from existing images and audio</p>',
        unsafe_allow_html=True,
    )

    output_dir = Path("output")
    scenes_path = output_dir / "planned_scenes.json"
    images_dir = output_dir / "images"
    audio_dir = output_dir / "audio"

    # Check for required files
    has_scenes = scenes_path.exists()
    has_images = images_dir.exists() and len(list(images_dir.glob("scene_*.png"))) > 0
    has_audio = audio_dir.exists() and len(list(audio_dir.glob("voice_*.wav"))) > 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Scenes Data", "✅" if has_scenes else "❌")
    col2.metric("🖼️ Images", "✅" if has_images else "❌")
    col3.metric("🎵 Audio", "✅" if has_audio else "❌")

    if not (has_scenes and has_images and has_audio):
        st.warning(
            "⚠️ Missing required assets. Generate a video first to have assets to recompose."
        )
        return

    st.markdown("---")

    # Load current scenes
    with open(scenes_path, "r") as f:
        scenes_data = json.load(f)

    st.markdown(f"**Found {len(scenes_data)} scenes**")

    # Preview scenes
    with st.expander("📋 Preview Scenes", expanded=False):
        for scene in scenes_data:
            st.markdown(
                f"**Scene {scene['scene_number']}:** {scene['narration'][:100]}..."
            )

    # Recompose options
    with st.form("recompose_form"):
        st.markdown("### ⚙️ Recompose Options")

        col1, col2 = st.columns(2)

        with col1:
            format_option = st.selectbox(
                "📐 Video Format",
                options=["short", "long"],
                format_func=lambda x: (
                    "Short (9:16 vertical)"
                    if x == "short"
                    else "Long (16:9 horizontal)"
                ),
            )

            viral_mode = st.checkbox(
                "🔥 Viral Mode (Karaoke Subtitles)",
                value=True,
                help="Enable karaoke-style captions with word highlighting",
            )

        with col2:
            enable_subtitles = st.checkbox(
                "📝 Enable Subtitles",
                value=True,
            )

            use_whisper = st.checkbox(
                "🎯 Use Whisper for Precise Timing",
                value=True,
                help="Use OpenAI Whisper API for word-level timestamp alignment",
            )

        submitted = st.form_submit_button(
            "🔄 Recompose Video", width="stretch", type="primary"
        )

    if submitted:
        recompose_video(
            scenes_data=scenes_data,
            format_option=format_option,
            viral_mode=viral_mode,
            enable_subtitles=enable_subtitles,
            use_whisper=use_whisper,
        )


def recompose_video(
    scenes_data: list,
    format_option: str,
    viral_mode: bool,
    enable_subtitles: bool,
    use_whisper: bool,
):
    """Recompose video from existing assets."""
    from src.pipeline import PlannedScene, VideoComposer, VideoFormat, VideoRequest

    output_dir = Path("output")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.info("🔄 Loading assets...")
        progress_bar.progress(10)

        # Reconstruct PlannedScene objects
        planned_scenes = []
        for scene in scenes_data:
            ps = PlannedScene(
                scene_number=scene["scene_number"],
                narration=scene["narration"],
                visual_description=scene["visual_description"],
                image_prompt=scene.get("image_prompt", ""),
                duration_seconds=scene["duration_seconds"],
                mood=scene.get("mood", "informative"),
                transition=scene.get("transition", "crossfade"),
                audio_path=output_dir
                / "audio"
                / f"voice_{scene['scene_number']:02d}.wav",
                image_path=output_dir
                / "images"
                / f"scene_{scene['scene_number']:02d}.png",
            )
            planned_scenes.append(ps)

        progress_bar.progress(30)

        # Create dummy request for video dimensions
        request = VideoRequest(
            topic="Recomposed Video",
            target_audience="general",
            format=VideoFormat(format_option),
            style="educational",
        )

        # Generate timestamps if using Whisper
        if use_whisper and enable_subtitles:
            status_text.info("🎯 Generating word-level timestamps with Whisper...")
            try:
                from src.pipeline.audio_aligner import AudioAligner

                aligner = AudioAligner()
                timestamps_dir = output_dir / "subtitles" / "timestamps"
                timestamps_dir.mkdir(parents=True, exist_ok=True)

                for scene in planned_scenes:
                    if scene.audio_path and scene.audio_path.exists():
                        result = aligner.get_word_timestamps(scene.audio_path)
                        # Save timestamps for reuse
                        json_path = timestamps_dir / f"{scene.audio_path.stem}.json"
                        aligner.save_timestamps_json(result, json_path)
                progress_bar.progress(50)
            except Exception as e:
                st.warning(f"⚠️ Whisper alignment failed: {e}. Using estimated timing.")

        # Compose video
        status_text.info("🎥 Composing video...")
        progress_bar.progress(60)

        composer = VideoComposer(output_dir=output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"video_recomposed_{timestamp}.mp4"

        video_path = composer.compose(
            scenes=planned_scenes,
            request=request,
            output_filename=output_filename,
            viral_mode=viral_mode,
            enable_subtitles=enable_subtitles,
        )

        progress_bar.progress(100)
        status_text.success("🎉 Video recomposed successfully!")

        # Show result
        st.balloons()
        st.markdown("---")
        st.markdown("### 📹 Recomposed Video")

        if video_path and video_path.exists():
            st.video(str(video_path))

            with open(video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f.read(),
                    file_name=video_path.name,
                    mime="video/mp4",
                    width="stretch",
                )

    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Error: {str(e)}")
        st.exception(e)


# =============================================================================
# Page: Generate Subtitles
# =============================================================================
def page_subtitles():
    """Generate subtitles page."""
    st.markdown(
        '<p class="main-header">📝 Generate Subtitles</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Create word-level timestamps for karaoke subtitles</p>',
        unsafe_allow_html=True,
    )

    output_dir = Path("output")
    audio_dir = output_dir / "audio"
    subtitles_dir = output_dir / "subtitles"
    timestamps_dir = subtitles_dir / "timestamps"

    # Check for audio files
    audio_files = list(audio_dir.glob("voice_*.wav")) if audio_dir.exists() else []

    if not audio_files:
        st.warning("⚠️ No audio files found. Generate a video first!")
        return

    st.success(f"✅ Found {len(audio_files)} audio files")

    # Check for existing timestamps
    existing_timestamps = (
        list(timestamps_dir.glob("*.json")) if timestamps_dir.exists() else []
    )
    if existing_timestamps:
        st.info(f"📋 {len(existing_timestamps)} timestamp files already cached")

    st.markdown("---")

    # Aligner selection (outside form for immediate feedback)
    col1, col2 = st.columns(2)
    with col1:
        aligner_choice = st.selectbox(
            "Alignment Method",
            options=["wav2vec2", "whisper"],
            index=0,
            help="wav2vec2 = FREE local processing, whisper = OpenAI API (~$0.006/min)",
        )
    with col2:
        if aligner_choice == "wav2vec2":
            st.info("💰 **FREE** - Runs locally using Wav2Vec2")
        else:
            st.warning("💵 **Paid** - Uses OpenAI Whisper API (~$0.006/min)")

    # Options form
    with st.form("subtitle_form"):
        col1, col2 = st.columns(2)

        with col1:
            words_per_line = st.slider(
                "Words per line",
                min_value=2,
                max_value=8,
                value=4,
                help="Number of words to show at a time",
            )

        with col2:
            force_regenerate = st.checkbox(
                "Force regenerate (ignore cache)",
                value=False,
                help="Regenerate timestamps even if cached",
            )

        submitted = st.form_submit_button(
            "🎯 Generate Timestamps", width="stretch", type="primary"
        )

    if submitted:
        generate_subtitles(
            audio_files, words_per_line, force_regenerate, aligner_choice
        )

    # Show existing timestamps
    if existing_timestamps:
        st.markdown("---")
        st.markdown("### 📋 Cached Timestamps")

        for ts_file in sorted(existing_timestamps):
            with st.expander(f"📄 {ts_file.name}"):
                data = load_json_file(ts_file)
                if "words" in data:
                    st.markdown(
                        f"**Full text:** {data.get('full_text', 'N/A')[:200]}..."
                    )
                    st.markdown(f"**Duration:** {data.get('duration', 0):.2f}s")
                    st.markdown(f"**Words:** {len(data.get('words', []))}")

                    # Show word table
                    if data.get("words"):
                        words_preview = data["words"][:20]
                        st.dataframe(
                            words_preview,
                            column_config={
                                "word": "Word",
                                "start": st.column_config.NumberColumn(
                                    "Start (s)", format="%.2f"
                                ),
                                "end": st.column_config.NumberColumn(
                                    "End (s)", format="%.2f"
                                ),
                            },
                            width="stretch",
                        )
                else:
                    st.json(data)


def generate_subtitles(
    audio_files: list,
    words_per_line: int,
    force_regenerate: bool,
    aligner_choice: str = "wav2vec2",
):
    """Generate word-level timestamps for audio files."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        output_dir = Path("output")

        # Initialize the appropriate aligner
        if aligner_choice == "wav2vec2":
            from src.pipeline.wav2vec2_aligner import Wav2Vec2Aligner

            aligner = Wav2Vec2Aligner()
            status_text.info("🔧 Using Wav2Vec2 (FREE local alignment)...")
        else:
            from src.pipeline.audio_aligner import AudioAligner

            aligner = AudioAligner()
            status_text.info("🔧 Using Whisper API (paid)...")

        # Load transcripts for wav2vec2 (it needs the text)
        transcripts = {}
        if aligner_choice == "wav2vec2":
            scenes_path = output_dir / "planned_scenes.json"
            if scenes_path.exists():
                with open(scenes_path, "r") as f:
                    scenes = json.load(f)
                    for scene in scenes:
                        scene_num = scene.get("scene_number", 0)
                        transcripts[f"voice_{scene_num:02d}"] = scene.get(
                            "narration", ""
                        )

        total_files = len(audio_files)

        for i, audio_path in enumerate(sorted(audio_files)):
            status_text.info(f"🎯 Processing {audio_path.name}...")
            progress = int(((i + 1) / total_files) * 100)
            progress_bar.progress(progress)

            # Check cache
            timestamps_dir = output_dir / "subtitles" / "timestamps"
            cache_file = timestamps_dir / f"{audio_path.stem}.json"

            if cache_file.exists() and not force_regenerate:
                status_text.info(f"📋 Using cached timestamps for {audio_path.name}")
                continue

            # Generate timestamps based on aligner type
            if aligner_choice == "wav2vec2":
                # Wav2Vec2 needs the transcript text
                transcript = transcripts.get(audio_path.stem, "")
                if not transcript:
                    status_text.warning(
                        f"⚠️ No transcript for {audio_path.name}, skipping..."
                    )
                    continue
                result = aligner.align(audio_path, transcript)
            else:
                # Whisper can auto-transcribe or use provided text
                scenes_path = output_dir / "planned_scenes.json"
                narration = ""
                if scenes_path.exists():
                    with open(scenes_path, "r") as f:
                        scenes = json.load(f)
                        scene_num = int(audio_path.stem.split("_")[-1])
                        for scene in scenes:
                            if scene["scene_number"] == scene_num:
                                narration = scene["narration"]
                                break

                if narration:
                    result = aligner.align_with_narration(audio_path, narration)
                else:
                    result = aligner.get_word_timestamps(audio_path)

            if result:
                # Save timestamps to cache
                timestamps_dir.mkdir(parents=True, exist_ok=True)
                aligner.save_timestamps_json(result, cache_file)
                status_text.success(f"✅ Generated timestamps for {audio_path.name}")

        progress_bar.progress(100)
        status_text.success(f"🎉 Generated timestamps for {total_files} audio files!")
        st.info(
            "💡 Now you can use **Recompose Video** to create a video with precise karaoke subtitles."
        )

    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Error: {str(e)}")
        st.exception(e)


# =============================================================================
# Page: Settings
# =============================================================================
def page_settings():
    """Settings page for API configuration."""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Configure API keys and preferences</p>',
        unsafe_allow_html=True,
    )

    # API Keys
    st.markdown("### 🔑 API Keys")
    st.info(
        "API keys are stored as environment variables and are not persisted. "
        "For persistent storage, add them to a `.env` file in the project root."
    )

    col1, col2 = st.columns(2)

    with col1:
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Required for text generation, images, and TTS",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    with col2:
        google_key = st.text_input(
            "Google API Key (Gemini)",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            help="Required if using Gemini as AI provider",
        )
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key

    # Validate keys
    if st.button("🔍 Validate API Keys"):
        is_valid, message = validate_api_keys()
        if is_valid:
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")

    # Provider Settings
    st.markdown("### 🤖 AI Provider Settings")

    col1, col2 = st.columns(2)

    with col1:
        ai_provider = st.selectbox(
            "AI Provider",
            options=["openai", "gemini"],
            index=0 if st.session_state.settings["ai_provider"] == "openai" else 1,
            help="Provider for text generation and image generation",
        )
        st.session_state.settings["ai_provider"] = ai_provider
        os.environ["AI_PROVIDER"] = ai_provider

    with col2:
        tts_provider = st.selectbox(
            "TTS Provider",
            options=["openai", "gtts", "elevenlabs"],
            index=["openai", "gtts", "elevenlabs"].index(
                st.session_state.settings["tts_provider"]
            ),
            help="Provider for text-to-speech",
        )
        st.session_state.settings["tts_provider"] = tts_provider
        os.environ["TTS_PROVIDER"] = tts_provider

    st.markdown("---")

    # TTS Settings
    st.markdown("### 🎤 Text-to-Speech Settings")

    col1, col2 = st.columns(2)

    with col1:
        voice = st.selectbox(
            "Default Voice (OpenAI)",
            options=[
                "marin",
                "cedar",
                "alloy",
                "ash",
                "ballad",
                "coral",
                "echo",
                "fable",
                "nova",
                "onyx",
                "sage",
                "shimmer",
                "verse",
            ],
            index=0,
            help="Default voice for OpenAI TTS. marin and cedar are recommended.",
        )
        st.session_state.settings["tts_voice"] = voice
        os.environ["TTS_VOICE"] = voice

    with col2:
        speed = st.slider(
            "Speech Speed",
            min_value=0.25,
            max_value=4.0,
            value=st.session_state.settings["tts_speed"],
            step=0.25,
            help="Speech speed (1.0 = normal)",
        )
        st.session_state.settings["tts_speed"] = speed
        os.environ["TTS_SPEED"] = str(speed)

    default_instructions = st.text_area(
        "Default Voice Instructions",
        value=os.environ.get(
            "TTS_INSTRUCTIONS",
            "Speak clearly and professionally, like an educational video narrator. Use a warm, engaging tone.",
        ),
        help="Default instructions for voice style (OpenAI TTS only)",
    )
    if default_instructions:
        os.environ["TTS_INSTRUCTIONS"] = default_instructions

    st.markdown("---")

    # Model Information
    st.markdown("### 📊 Model Information")

    with st.expander("💰 Pricing Reference", expanded=False):
        st.markdown(
            """
        | Component | Model | Cost |
        |-----------|-------|------|
        | **Text** | gpt-5-mini | $0.25 / $2.00 per 1M tokens |
        | **Images** | gpt-image-1-mini (low) | $0.005 per image |
        | **TTS** | gpt-4o-mini-tts | $0.015 per minute |
        | **Whisper** | whisper-1 | $0.006 per minute |

        **Estimated cost per video:**
        - Short (60s, 3 scenes): ~$0.05
        - Long (5min, 8 scenes): ~$0.18
        """
        )

    with st.expander("🎤 Available Voices", expanded=False):
        st.markdown(
            """
        | Voice | Notes |
        |-------|-------|
        | **marin** | ⭐ Best quality (recommended) |
        | **cedar** | ⭐ Best quality (recommended) |
        | alloy | Neutral, balanced |
        | ash | - |
        | ballad | - |
        | coral | Popular choice |
        | echo | Warm, conversational |
        | fable | Expressive, storytelling |
        | nova | Friendly, upbeat |
        | onyx | Deep, authoritative |
        | sage | - |
        | shimmer | Clear, professional |
        | verse | - |
        """
        )


# =============================================================================
# Page: About
# =============================================================================
def page_about():
    """About page with project information."""
    st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI Video Generation Pipeline</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ## 🎬 What is this?

    This is an **autonomous AI-powered video generation pipeline** that creates engaging
    educational videos from any topic. It supports both short-form (TikTok/Reels) and
    long-form (YouTube) content.

    ## 🔧 How it works

    1. **Research** - AI researches your topic to gather key points, facts, and examples
    2. **Script Writing** - Generates an engaging script with a hook and narrative arc
    3. **Scene Planning** - Breaks down the script into visual scenes with image prompts
    4. **Asset Generation** - Creates AI images and voice narration for each scene
    5. **Video Composition** - Stitches everything together with subtitles and effects
    6. **Output** - Final video with thumbnail and metadata

    ## ✨ Features

    - **Viral Mode** - Karaoke-style captions with word highlighting
    - **Dynamic Motion** - Ken Burns effects and zoom animations
    - **Whisper Alignment** - Precise word-level subtitle timing
    - **Multiple Voices** - 13 OpenAI TTS voices with custom instructions
    - **YouTube Upload** - Direct upload with metadata

    ## 💰 Cost Efficiency

    Using cost-optimized models:
    - **Text**: gpt-5-mini ($0.25/$2.00 per 1M tokens)
    - **Images**: gpt-image-1-mini ($0.005/image)
    - **TTS**: gpt-4o-mini-tts ($0.015/minute)

    **Typical cost per video: $0.05 - $0.20**

    ## 📁 Project Structure

    ```
    output/
    ├── audio/           # Voice narration files
    ├── images/          # Scene images
    ├── subtitles/       # Subtitle files
    │   └── timestamps/  # Whisper word timestamps
    ├── research.json    # Research results
    ├── script.json      # Generated script
    ├── planned_scenes.json
    └── video_*.mp4      # Final videos
    ```
    """
    )


# =============================================================================
# Main App Navigation
# =============================================================================
def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.markdown("# 🎬 AI Video Generator")
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "🎬 Generate Video": page_generate,
        "📁 View Output": page_view_output,
        "🔄 Recompose Video": page_recompose,
        "📝 Generate Subtitles": page_subtitles,
        "⚙️ Settings": page_settings,
        "ℹ️ About": page_about,
    }

    selection = st.sidebar.radio(
        "Navigation", list(pages.keys()), label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Quick status
    st.sidebar.markdown("### 📊 Status")
    is_valid, _ = validate_api_keys()
    st.sidebar.markdown(f"**API Key:** {'✅ Valid' if is_valid else '❌ Not Set'}")
    st.sidebar.markdown(
        f"**Provider:** {st.session_state.settings['ai_provider'].upper()}"
    )
    st.sidebar.markdown(f"**TTS:** {st.session_state.settings['tts_provider'].upper()}")

    # Output summary
    files = get_output_files()
    if any(len(v) > 0 for v in files.values()):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Output Files")
        st.sidebar.markdown(f"🎥 {len(files['videos'])} videos")
        st.sidebar.markdown(f"🖼️ {len(files['images'])} images")
        st.sidebar.markdown(f"🎵 {len(files['audio'])} audio")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            Made with ❤️ using Streamlit<br>
            Powered by OpenAI
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render selected page
    pages[selection]()


if __name__ == "__main__":
    main()
