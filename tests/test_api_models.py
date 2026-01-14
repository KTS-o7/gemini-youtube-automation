"""
Tests for the API models module.

These tests verify that:
1. Pydantic API models are correctly structured
2. Conversion functions correctly transform API models to internal dataclasses
3. Conversion functions correctly transform internal dataclasses to API models
4. Round-trip conversions preserve data integrity
"""

import pytest

from src.pipeline.api_models import (
    PlannedSceneAPIModel,
    ResearchAPIModel,
    ResearchFactAPIModel,
    SceneAPIModel,
    ScriptAPIModel,
    to_api_research,
    to_api_scene,
    to_api_script,
    to_internal_research,
    to_internal_scene,
    to_internal_script,
)
from src.pipeline.models import ResearchResult, Scene, Script


class TestSceneAPIModel:
    """Tests for SceneAPIModel Pydantic model."""

    def test_create_with_required_fields(self):
        """Test creating a scene with required fields only."""
        scene = SceneAPIModel(
            scene_number=1,
            narration="This is the narration.",
            visual_description="A beautiful landscape.",
        )

        assert scene.scene_number == 1
        assert scene.narration == "This is the narration."
        assert scene.visual_description == "A beautiful landscape."

    def test_default_values(self):
        """Test that default values are set correctly."""
        scene = SceneAPIModel(
            scene_number=1,
            narration="Test",
            visual_description="Test visual",
        )

        assert scene.duration_seconds == 0.0
        assert scene.mood == "informative"
        assert scene.key_visual_elements == []

    def test_custom_values(self):
        """Test creating a scene with all custom values."""
        scene = SceneAPIModel(
            scene_number=3,
            narration="Exciting narration here!",
            visual_description="Dynamic action scene",
            duration_seconds=15.5,
            mood="exciting",
            key_visual_elements=["explosion", "hero", "cityscape"],
        )

        assert scene.scene_number == 3
        assert scene.duration_seconds == 15.5
        assert scene.mood == "exciting"
        assert len(scene.key_visual_elements) == 3

    def test_json_serialization(self):
        """Test that the model can be serialized to JSON."""
        scene = SceneAPIModel(
            scene_number=1,
            narration="Test narration",
            visual_description="Test visual",
            duration_seconds=10.0,
        )

        json_data = scene.model_dump()

        assert json_data["scene_number"] == 1
        assert json_data["narration"] == "Test narration"
        assert json_data["duration_seconds"] == 10.0


class TestScriptAPIModel:
    """Tests for ScriptAPIModel Pydantic model."""

    @pytest.fixture
    def sample_scenes(self):
        """Create sample scenes for testing."""
        return [
            SceneAPIModel(
                scene_number=1,
                narration="Opening hook to grab attention.",
                visual_description="Dramatic opening shot",
                duration_seconds=5.0,
                mood="exciting",
            ),
            SceneAPIModel(
                scene_number=2,
                narration="Main content explanation.",
                visual_description="Informative diagram",
                duration_seconds=20.0,
                mood="informative",
            ),
            SceneAPIModel(
                scene_number=3,
                narration="Closing call to action.",
                visual_description="Logo and subscribe button",
                duration_seconds=5.0,
                mood="engaging",
            ),
        ]

    def test_create_script(self, sample_scenes):
        """Test creating a complete script."""
        script = ScriptAPIModel(
            title="How AI Works",
            hook="Ever wondered how AI actually thinks?",
            scenes=sample_scenes,
            total_duration_seconds=30.0,
            hashtags=["AI", "Technology", "Education"],
            thumbnail_prompt="Futuristic AI brain concept",
            description="Learn the basics of artificial intelligence.",
        )

        assert script.title == "How AI Works"
        assert len(script.scenes) == 3
        assert script.total_duration_seconds == 30.0
        assert "AI" in script.hashtags

    def test_default_values(self):
        """Test script default values."""
        script = ScriptAPIModel(
            title="Test",
            hook="Test hook",
            scenes=[],
        )

        assert script.total_duration_seconds == 0.0
        assert script.hashtags == []
        assert script.thumbnail_prompt == ""
        assert script.description == ""


class TestResearchAPIModel:
    """Tests for ResearchAPIModel Pydantic model."""

    def test_create_research(self):
        """Test creating research results."""
        research = ResearchAPIModel(
            key_points=["Point 1", "Point 2", "Point 3"],
            facts=[
                ResearchFactAPIModel(fact="Interesting fact", source="Wikipedia"),
                ResearchFactAPIModel(fact="Another fact"),
            ],
            examples=["Example 1", "Example 2"],
            analogies=["Like a car engine"],
            related_topics=["Machine Learning", "Data Science"],
        )

        assert len(research.key_points) == 3
        assert len(research.facts) == 2
        assert research.facts[0].source == "Wikipedia"
        assert research.facts[1].source == "general knowledge"  # Default

    def test_default_values(self):
        """Test research default values."""
        research = ResearchAPIModel()

        assert research.key_points == []
        assert research.facts == []
        assert research.examples == []
        assert research.analogies == []
        assert research.related_topics == []


class TestToInternalScene:
    """Tests for to_internal_scene conversion function."""

    def test_basic_conversion(self):
        """Test basic API to internal conversion."""
        api_scene = SceneAPIModel(
            scene_number=1,
            narration="Test narration",
            visual_description="Test visual",
            duration_seconds=10.0,
            mood="informative",
            key_visual_elements=["element1", "element2"],
        )

        internal_scene = to_internal_scene(api_scene)

        assert isinstance(internal_scene, Scene)
        assert internal_scene.scene_number == 1
        assert internal_scene.narration == "Test narration"
        assert internal_scene.visual_description == "Test visual"
        assert internal_scene.duration_seconds == 10.0
        assert internal_scene.mood == "informative"
        assert internal_scene.key_visual_elements == ["element1", "element2"]

    def test_empty_key_visual_elements(self):
        """Test conversion with empty key visual elements."""
        api_scene = SceneAPIModel(
            scene_number=1,
            narration="Test",
            visual_description="Test",
        )

        internal_scene = to_internal_scene(api_scene)

        assert internal_scene.key_visual_elements == []


class TestToInternalScript:
    """Tests for to_internal_script conversion function."""

    @pytest.fixture
    def api_script(self):
        """Create a sample API script."""
        return ScriptAPIModel(
            title="Test Video",
            hook="Attention grabbing hook",
            scenes=[
                SceneAPIModel(
                    scene_number=1,
                    narration="Scene 1 narration",
                    visual_description="Scene 1 visual",
                    duration_seconds=10.0,
                ),
                SceneAPIModel(
                    scene_number=2,
                    narration="Scene 2 narration",
                    visual_description="Scene 2 visual",
                    duration_seconds=15.0,
                ),
            ],
            total_duration_seconds=25.0,
            hashtags=["test", "video"],
            thumbnail_prompt="Thumbnail description",
            description="Video description",
        )

    def test_full_conversion(self, api_script):
        """Test complete script conversion."""
        internal_script = to_internal_script(api_script)

        assert isinstance(internal_script, Script)
        assert internal_script.title == "Test Video"
        assert internal_script.hook == "Attention grabbing hook"
        assert len(internal_script.scenes) == 2
        assert internal_script.total_duration_seconds == 25.0
        assert internal_script.hashtags == ["test", "video"]
        assert internal_script.thumbnail_prompt == "Thumbnail description"
        assert internal_script.description == "Video description"

    def test_scenes_converted(self, api_script):
        """Test that nested scenes are properly converted."""
        internal_script = to_internal_script(api_script)

        assert all(isinstance(s, Scene) for s in internal_script.scenes)
        assert internal_script.scenes[0].scene_number == 1
        assert internal_script.scenes[1].scene_number == 2


class TestToInternalResearch:
    """Tests for to_internal_research conversion function."""

    def test_basic_conversion(self):
        """Test basic research conversion."""
        api_research = ResearchAPIModel(
            key_points=["Point 1", "Point 2"],
            facts=[
                ResearchFactAPIModel(fact="Fact 1", source="Source 1"),
                ResearchFactAPIModel(fact="Fact 2", source="Source 2"),
            ],
            examples=["Example 1"],
            analogies=["Analogy 1"],
            related_topics=["Topic 1"],
        )

        internal_research = to_internal_research(api_research, topic="Test Topic")

        assert isinstance(internal_research, ResearchResult)
        assert internal_research.topic == "Test Topic"
        assert internal_research.key_points == ["Point 1", "Point 2"]
        assert len(internal_research.facts) == 2
        assert internal_research.facts[0]["fact"] == "Fact 1"
        assert internal_research.facts[0]["source"] == "Source 1"

    def test_empty_research(self):
        """Test conversion of empty research."""
        api_research = ResearchAPIModel()

        internal_research = to_internal_research(api_research, topic="Empty Topic")

        assert internal_research.topic == "Empty Topic"
        assert internal_research.key_points == []
        assert internal_research.facts == []


class TestToApiScene:
    """Tests for to_api_scene conversion function."""

    def test_basic_conversion(self):
        """Test internal to API conversion."""
        internal_scene = Scene(
            scene_number=1,
            narration="Test narration",
            visual_description="Test visual",
            duration_seconds=10.0,
            mood="informative",
            key_visual_elements=["element1"],
        )

        api_scene = to_api_scene(internal_scene)

        assert isinstance(api_scene, SceneAPIModel)
        assert api_scene.scene_number == 1
        assert api_scene.narration == "Test narration"
        assert api_scene.key_visual_elements == ["element1"]


class TestToApiScript:
    """Tests for to_api_script conversion function."""

    def test_full_conversion(self):
        """Test complete script conversion to API model."""
        internal_script = Script(
            title="Internal Script",
            hook="Internal hook",
            scenes=[
                Scene(
                    scene_number=1,
                    narration="Narration 1",
                    visual_description="Visual 1",
                ),
            ],
            total_duration_seconds=30.0,
            hashtags=["tag1"],
            thumbnail_prompt="Thumbnail",
            description="Description",
        )

        api_script = to_api_script(internal_script)

        assert isinstance(api_script, ScriptAPIModel)
        assert api_script.title == "Internal Script"
        assert len(api_script.scenes) == 1
        assert isinstance(api_script.scenes[0], SceneAPIModel)


class TestToApiResearch:
    """Tests for to_api_research conversion function."""

    def test_basic_conversion(self):
        """Test internal to API research conversion."""
        internal_research = ResearchResult(
            topic="Test Topic",
            key_points=["Point 1", "Point 2"],
            facts=[
                {"fact": "Fact 1", "source": "Source 1"},
                {"fact": "Fact 2", "source": "Source 2"},
            ],
            examples=["Example 1"],
            analogies=["Analogy 1"],
            related_topics=["Related 1"],
        )

        api_research = to_api_research(internal_research)

        assert isinstance(api_research, ResearchAPIModel)
        assert api_research.key_points == ["Point 1", "Point 2"]
        assert len(api_research.facts) == 2
        assert api_research.facts[0].fact == "Fact 1"
        assert api_research.facts[0].source == "Source 1"


class TestRoundTripConversions:
    """Tests for round-trip conversions (API -> Internal -> API)."""

    def test_scene_round_trip(self):
        """Test scene round-trip preserves data."""
        original = SceneAPIModel(
            scene_number=5,
            narration="Original narration text",
            visual_description="Original visual description",
            duration_seconds=12.5,
            mood="dramatic",
            key_visual_elements=["fire", "water", "earth"],
        )

        # Convert to internal and back
        internal = to_internal_scene(original)
        restored = to_api_scene(internal)

        assert restored.scene_number == original.scene_number
        assert restored.narration == original.narration
        assert restored.visual_description == original.visual_description
        assert restored.duration_seconds == original.duration_seconds
        assert restored.mood == original.mood
        assert restored.key_visual_elements == original.key_visual_elements

    def test_script_round_trip(self):
        """Test script round-trip preserves data."""
        original = ScriptAPIModel(
            title="Round Trip Test",
            hook="Test hook text",
            scenes=[
                SceneAPIModel(
                    scene_number=1,
                    narration="Scene narration",
                    visual_description="Scene visual",
                    duration_seconds=10.0,
                ),
            ],
            total_duration_seconds=10.0,
            hashtags=["test", "roundtrip"],
            thumbnail_prompt="Test thumbnail",
            description="Test description",
        )

        internal = to_internal_script(original)
        restored = to_api_script(internal)

        assert restored.title == original.title
        assert restored.hook == original.hook
        assert len(restored.scenes) == len(original.scenes)
        assert restored.total_duration_seconds == original.total_duration_seconds
        assert restored.hashtags == original.hashtags

    def test_research_round_trip(self):
        """Test research round-trip preserves data."""
        original = ResearchAPIModel(
            key_points=["Key 1", "Key 2"],
            facts=[ResearchFactAPIModel(fact="Fact 1", source="Source 1")],
            examples=["Example 1"],
            analogies=["Analogy 1"],
            related_topics=["Topic 1"],
        )

        internal = to_internal_research(original, topic="Test")
        restored = to_api_research(internal)

        assert restored.key_points == original.key_points
        assert len(restored.facts) == len(original.facts)
        assert restored.facts[0].fact == original.facts[0].fact
        assert restored.examples == original.examples
        assert restored.analogies == original.analogies
        assert restored.related_topics == original.related_topics
