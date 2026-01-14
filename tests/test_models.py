"""
Tests for the pipeline models module.

These tests verify that:
1. VideoRequest validates correctly
2. VideoFormat enum works correctly
3. Script and Scene dataclasses are properly structured
4. ResearchResult correctly generates summaries
5. PipelineState tracks state correctly
6. VideoOutput and VideoMetadata are properly structured
"""

from pathlib import Path

import pytest

from src.pipeline.models import (
    GeneratedAssets,
    PipelineState,
    PlannedScene,
    ResearchResult,
    Scene,
    Script,
    TTSProvider,
    VideoFormat,
    VideoMetadata,
    VideoOutput,
    VideoRequest,
)


class TestVideoFormat:
    """Tests for VideoFormat enum."""

    def test_short_format(self):
        """Test SHORT format value."""
        assert VideoFormat.SHORT.value == "short"

    def test_long_format(self):
        """Test LONG format value."""
        assert VideoFormat.LONG.value == "long"

    def test_format_from_string(self):
        """Test creating format from string."""
        assert VideoFormat("short") == VideoFormat.SHORT
        assert VideoFormat("long") == VideoFormat.LONG

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            VideoFormat("invalid")


class TestTTSProvider:
    """Tests for TTSProvider enum."""

    def test_gtts_provider(self):
        """Test GTTS provider value."""
        assert TTSProvider.GTTS.value == "gtts"

    def test_openai_provider(self):
        """Test OPENAI provider value."""
        assert TTSProvider.OPENAI.value == "openai"

    def test_elevenlabs_provider(self):
        """Test ELEVENLABS provider value."""
        assert TTSProvider.ELEVENLABS.value == "elevenlabs"


class TestVideoRequest:
    """Tests for VideoRequest dataclass."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample video request."""
        return VideoRequest(
            topic="How AI Works",
            target_audience="beginners",
            format=VideoFormat.LONG,
            style="educational",
        )

    def test_create_request(self, sample_request):
        """Test creating a video request."""
        assert sample_request.topic == "How AI Works"
        assert sample_request.target_audience == "beginners"
        assert sample_request.format == VideoFormat.LONG
        assert sample_request.style == "educational"
        assert sample_request.language == "en"

    def test_default_values(self):
        """Test default values."""
        request = VideoRequest(
            topic="Test",
            target_audience="everyone",
            format=VideoFormat.SHORT,
        )

        assert request.style == "educational"
        assert request.language == "en"

    def test_validate_success(self, sample_request):
        """Test validation passes with valid data."""
        assert sample_request.validate() is True

    def test_validate_empty_topic_raises(self):
        """Test validation fails with empty topic."""
        request = VideoRequest(
            topic="",
            target_audience="beginners",
            format=VideoFormat.SHORT,
        )

        with pytest.raises(ValueError, match="Topic cannot be empty"):
            request.validate()

    def test_validate_whitespace_topic_raises(self):
        """Test validation fails with whitespace-only topic."""
        request = VideoRequest(
            topic="   ",
            target_audience="beginners",
            format=VideoFormat.SHORT,
        )

        with pytest.raises(ValueError, match="Topic cannot be empty"):
            request.validate()

    def test_validate_empty_audience_raises(self):
        """Test validation fails with empty target audience."""
        request = VideoRequest(
            topic="Test Topic",
            target_audience="",
            format=VideoFormat.SHORT,
        )

        with pytest.raises(ValueError, match="Target audience cannot be empty"):
            request.validate()

    def test_get_video_dimensions_long(self):
        """Test video dimensions for long format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.LONG,
        )

        width, height = request.get_video_dimensions()

        assert width == 1920
        assert height == 1080

    def test_get_video_dimensions_short(self):
        """Test video dimensions for short format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.SHORT,
        )

        width, height = request.get_video_dimensions()

        assert width == 1080
        assert height == 1920

    def test_get_target_duration_long(self):
        """Test target duration for long format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.LONG,
        )

        min_dur, max_dur = request.get_target_duration()

        assert min_dur == 180  # 3 minutes
        assert max_dur == 600  # 10 minutes

    def test_get_target_duration_short(self):
        """Test target duration for short format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.SHORT,
        )

        min_dur, max_dur = request.get_target_duration()

        assert min_dur == 30
        assert max_dur == 60

    def test_get_image_size_long(self):
        """Test image size for long format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.LONG,
        )

        size = request.get_image_size()

        assert size == "1536x1024"

    def test_get_image_size_short(self):
        """Test image size for short format."""
        request = VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.SHORT,
        )

        size = request.get_image_size()

        assert size == "1024x1536"


class TestResearchResult:
    """Tests for ResearchResult dataclass."""

    @pytest.fixture
    def sample_research(self):
        """Create sample research result."""
        return ResearchResult(
            topic="Machine Learning",
            key_points=["Point 1", "Point 2", "Point 3"],
            facts=[
                {"fact": "ML is a subset of AI", "source": "Wikipedia"},
                {
                    "fact": "Deep learning uses neural networks",
                    "source": "Research paper",
                },
            ],
            examples=["Image recognition", "Speech recognition"],
            analogies=["ML is like teaching a child"],
            sources=["https://example.com"],
            related_topics=["Deep Learning", "Neural Networks"],
        )

    def test_create_research(self, sample_research):
        """Test creating research result."""
        assert sample_research.topic == "Machine Learning"
        assert len(sample_research.key_points) == 3
        assert len(sample_research.facts) == 2
        assert len(sample_research.examples) == 2

    def test_default_values(self):
        """Test default values."""
        research = ResearchResult(topic="Test")

        assert research.key_points == []
        assert research.facts == []
        assert research.examples == []
        assert research.analogies == []
        assert research.sources == []
        assert research.related_topics == []
        assert research.raw_content == ""

    def test_summary(self, sample_research):
        """Test summary generation."""
        summary = sample_research.summary()

        assert "Machine Learning" in summary
        assert "Point 1" in summary
        assert "ML is a subset of AI" in summary
        assert "Image recognition" in summary
        assert "ML is like teaching a child" in summary

    def test_summary_empty_research(self):
        """Test summary with minimal research."""
        research = ResearchResult(topic="Empty Topic")
        summary = research.summary()

        assert "Empty Topic" in summary


class TestScene:
    """Tests for Scene dataclass."""

    @pytest.fixture
    def sample_scene(self):
        """Create a sample scene."""
        return Scene(
            scene_number=1,
            narration="This is the narration for scene one.",
            visual_description="A beautiful landscape with mountains.",
            duration_seconds=10.0,
            mood="peaceful",
            key_visual_elements=["mountains", "sky", "trees"],
        )

    def test_create_scene(self, sample_scene):
        """Test creating a scene."""
        assert sample_scene.scene_number == 1
        assert "narration" in sample_scene.narration
        assert sample_scene.mood == "peaceful"

    def test_default_values(self):
        """Test default values."""
        scene = Scene(
            scene_number=1,
            narration="Test",
            visual_description="Test visual",
        )

        assert scene.duration_seconds == 0.0
        assert scene.mood == "informative"
        assert scene.key_visual_elements == []

    def test_word_count(self, sample_scene):
        """Test word count calculation."""
        word_count = sample_scene.word_count()

        assert word_count == 7  # "This is the narration for scene one."

    def test_estimate_duration(self, sample_scene):
        """Test duration estimation."""
        # 7 words at 150 words per minute = 7/150 * 60 = 2.8 seconds
        duration = sample_scene.estimate_duration(words_per_minute=150)

        assert duration == pytest.approx(2.8, rel=0.1)

    def test_estimate_duration_custom_wpm(self, sample_scene):
        """Test duration estimation with custom WPM."""
        # 7 words at 100 words per minute = 7/100 * 60 = 4.2 seconds
        duration = sample_scene.estimate_duration(words_per_minute=100)

        assert duration == pytest.approx(4.2, rel=0.1)


class TestScript:
    """Tests for Script dataclass."""

    @pytest.fixture
    def sample_script(self):
        """Create a sample script."""
        return Script(
            title="Introduction to AI",
            hook="Ever wondered how AI actually works?",
            scenes=[
                Scene(
                    scene_number=1,
                    narration="Scene 1",
                    visual_description="Visual 1",
                    duration_seconds=10.0,
                ),
                Scene(
                    scene_number=2,
                    narration="Scene 2",
                    visual_description="Visual 2",
                    duration_seconds=15.0,
                ),
                Scene(
                    scene_number=3,
                    narration="Scene 3",
                    visual_description="Visual 3",
                    duration_seconds=20.0,
                ),
            ],
            total_duration_seconds=45.0,
            hashtags=["AI", "Technology"],
            thumbnail_prompt="Futuristic AI brain",
            description="Learn about AI basics",
        )

    def test_create_script(self, sample_script):
        """Test creating a script."""
        assert sample_script.title == "Introduction to AI"
        assert len(sample_script.scenes) == 3
        assert sample_script.total_duration_seconds == 45.0

    def test_default_values(self):
        """Test default values."""
        script = Script(
            title="Test",
            hook="Test hook",
            scenes=[],
        )

        assert script.total_duration_seconds == 0.0
        assert script.hashtags == []
        assert script.thumbnail_prompt == ""
        assert script.description == ""

    def test_scene_count(self, sample_script):
        """Test scene count."""
        assert sample_script.scene_count() == 3

    def test_calculate_total_duration(self, sample_script):
        """Test total duration calculation."""
        sample_script.total_duration_seconds = 0  # Reset
        total = sample_script.calculate_total_duration()

        assert total == 45.0
        assert sample_script.total_duration_seconds == 45.0


class TestPlannedScene:
    """Tests for PlannedScene dataclass."""

    def test_create_planned_scene(self):
        """Test creating a planned scene."""
        scene = PlannedScene(
            scene_number=1,
            narration="Test narration",
            visual_description="Test visual",
            image_prompt="Detailed image prompt",
            duration_seconds=10.0,
            mood="informative",
            transition="crossfade",
        )

        assert scene.scene_number == 1
        assert scene.image_prompt == "Detailed image prompt"
        assert scene.transition == "crossfade"

    def test_default_values(self):
        """Test default values."""
        scene = PlannedScene(
            scene_number=1,
            narration="Test",
            visual_description="Test",
            image_prompt="Test prompt",
        )

        assert scene.duration_seconds == 0.0
        assert scene.mood == "informative"
        assert scene.transition == "crossfade"
        assert scene.audio_path is None
        assert scene.image_path is None

    def test_asset_paths(self):
        """Test setting asset paths."""
        scene = PlannedScene(
            scene_number=1,
            narration="Test",
            visual_description="Test",
            image_prompt="Test",
            audio_path=Path("/path/to/audio.wav"),
            image_path=Path("/path/to/image.png"),
        )

        assert scene.audio_path == Path("/path/to/audio.wav")
        assert scene.image_path == Path("/path/to/image.png")


class TestGeneratedAssets:
    """Tests for GeneratedAssets dataclass."""

    def test_create_assets(self):
        """Test creating generated assets."""
        assets = GeneratedAssets(
            scene_number=1,
            image_path=Path("/path/to/image.png"),
            audio_path=Path("/path/to/audio.wav"),
            duration_seconds=10.5,
        )

        assert assets.scene_number == 1
        assert assets.image_path == Path("/path/to/image.png")
        assert assets.audio_path == Path("/path/to/audio.wav")
        assert assets.duration_seconds == 10.5


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            title="Test Video",
            description="A test video description.",
            tags=["test", "video", "sample"],
            hashtags=["Test", "Video"],
            duration_seconds=120.0,
            format="long",
            sources=["https://example.com", "https://test.com"],
        )

    def test_create_metadata(self, sample_metadata):
        """Test creating video metadata."""
        assert sample_metadata.title == "Test Video"
        assert sample_metadata.duration_seconds == 120.0
        assert len(sample_metadata.tags) == 3

    def test_default_values(self):
        """Test default values."""
        metadata = VideoMetadata(
            title="Test",
            description="Test description",
        )

        assert metadata.tags == []
        assert metadata.hashtags == []
        assert metadata.duration_seconds == 0.0
        assert metadata.format == "long"
        assert metadata.sources == []

    def test_youtube_description(self, sample_metadata):
        """Test YouTube description generation."""
        desc = sample_metadata.youtube_description()

        assert "A test video description" in desc
        assert "#Test" in desc
        assert "#Video" in desc
        assert "https://example.com" in desc

    def test_youtube_tags(self, sample_metadata):
        """Test YouTube tags generation."""
        tags = sample_metadata.youtube_tags()

        assert tags == "test, video, sample"


class TestVideoOutput:
    """Tests for VideoOutput dataclass."""

    def test_create_successful_output(self):
        """Test creating a successful output."""
        output = VideoOutput(
            video_path=Path("/path/to/video.mp4"),
            thumbnail_path=Path("/path/to/thumbnail.png"),
            metadata=VideoMetadata(title="Test", description="Test"),
            success=True,
        )

        assert output.video_path == Path("/path/to/video.mp4")
        assert output.success is True
        assert output.error_message == ""

    def test_create_failed_output(self):
        """Test creating a failed output."""
        output = VideoOutput(
            video_path=Path(""),
            success=False,
            error_message="Generation failed",
        )

        assert output.success is False
        assert output.error_message == "Generation failed"

    def test_default_values(self):
        """Test default values."""
        output = VideoOutput(video_path=Path("/path/to/video.mp4"))

        assert output.thumbnail_path is None
        assert output.metadata is None
        assert output.success is True
        assert output.error_message == ""

    def test_str_successful(self):
        """Test string representation for successful output."""
        output = VideoOutput(
            video_path=Path("/path/to/video.mp4"),
            thumbnail_path=Path("/path/to/thumb.png"),
            success=True,
        )

        str_repr = str(output)

        assert "video.mp4" in str_repr
        assert "thumb.png" in str_repr

    def test_str_failed(self):
        """Test string representation for failed output."""
        output = VideoOutput(
            video_path=Path(""),
            success=False,
            error_message="Something went wrong",
        )

        str_repr = str(output)

        assert "Something went wrong" in str_repr


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample request."""
        return VideoRequest(
            topic="Test",
            target_audience="test",
            format=VideoFormat.SHORT,
        )

    @pytest.fixture
    def sample_state(self, sample_request):
        """Create a sample pipeline state."""
        return PipelineState(request=sample_request)

    def test_create_state(self, sample_state):
        """Test creating pipeline state."""
        assert sample_state.request is not None
        assert sample_state.current_stage == "initialized"

    def test_default_values(self, sample_state):
        """Test default values."""
        assert sample_state.research is None
        assert sample_state.script is None
        assert sample_state.planned_scenes == []
        assert sample_state.output is None
        assert sample_state.errors == []

    def test_add_error(self, sample_state):
        """Test adding an error."""
        sample_state.add_error("First error")
        sample_state.add_error("Second error")

        assert len(sample_state.errors) == 2
        assert "First error" in sample_state.errors
        assert "Second error" in sample_state.errors

    def test_is_successful_true(self, sample_state):
        """Test is_successful returns True when conditions met."""
        sample_state.output = VideoOutput(
            video_path=Path("/path/to/video.mp4"),
            success=True,
        )

        assert sample_state.is_successful() is True

    def test_is_successful_false_with_errors(self, sample_state):
        """Test is_successful returns False with errors."""
        sample_state.add_error("An error occurred")
        sample_state.output = VideoOutput(
            video_path=Path("/path/to/video.mp4"),
            success=True,
        )

        assert sample_state.is_successful() is False

    def test_is_successful_false_no_output(self, sample_state):
        """Test is_successful returns False without output."""
        assert sample_state.is_successful() is False

    def test_is_successful_false_failed_output(self, sample_state):
        """Test is_successful returns False with failed output."""
        sample_state.output = VideoOutput(
            video_path=Path(""),
            success=False,
            error_message="Failed",
        )

        assert sample_state.is_successful() is False
