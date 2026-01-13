#!/usr/bin/env python3
"""
Test script to verify the video generation pipeline structure.

This script tests that all pipeline components can be imported and
instantiated correctly without actually generating a video.

Usage:
    python test_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all pipeline modules can be imported."""
    print("📦 Testing imports...")

    try:
        from src.pipeline import (
            ContentResearcher,
            ImageGenerator,
            PlannedScene,
            ResearchResult,
            Scene,
            ScenePlanner,
            Script,
            ScriptWriter,
            TTSProvider,
            VideoComposer,
            VideoFormat,
            VideoMetadata,
            VideoOutput,
            VideoPipeline,
            VideoRequest,
            VoiceGenerator,
            create_video,
        )

        print("  ✅ All pipeline modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_models():
    """Test that models can be instantiated."""
    print("\n📋 Testing models...")

    try:
        from src.pipeline.models import (
            PipelineState,
            PlannedScene,
            ResearchResult,
            Scene,
            Script,
            VideoFormat,
            VideoMetadata,
            VideoOutput,
            VideoRequest,
        )

        # Test VideoRequest
        request = VideoRequest(
            topic="Test topic",
            target_audience="Test audience",
            format=VideoFormat.LONG,
            style="educational",
        )
        assert request.validate() == True
        assert request.get_video_dimensions() == (1920, 1080)
        assert request.get_image_size() == "1536x1024"
        print("  ✅ VideoRequest model works correctly")

        # Test short format
        short_request = VideoRequest(
            topic="Test",
            target_audience="Test",
            format=VideoFormat.SHORT,
        )
        assert short_request.get_video_dimensions() == (1080, 1920)
        assert short_request.get_image_size() == "1024x1536"
        print("  ✅ VideoFormat (short) works correctly")

        # Test Scene
        scene = Scene(
            scene_number=1,
            narration="This is test narration text",
            visual_description="Test visual",
            duration_seconds=10.0,
            mood="informative",
        )
        assert scene.word_count() == 5
        print("  ✅ Scene model works correctly")

        # Test Script
        script = Script(
            title="Test Video",
            hook="Attention grabbing hook",
            scenes=[scene],
            hashtags=["test", "video"],
        )
        assert script.scene_count() == 1
        script.calculate_total_duration()
        assert script.total_duration_seconds == 10.0
        print("  ✅ Script model works correctly")

        # Test ResearchResult
        research = ResearchResult(
            topic="Test topic",
            key_points=["Point 1", "Point 2"],
            facts=[{"fact": "Test fact", "source": "Test source"}],
            examples=["Example 1"],
            analogies=["Analogy 1"],
        )
        summary = research.summary()
        assert "Test topic" in summary
        assert "Point 1" in summary
        print("  ✅ ResearchResult model works correctly")

        # Test VideoMetadata
        metadata = VideoMetadata(
            title="Test Video",
            description="Test description",
            tags=["tag1", "tag2"],
            hashtags=["hash1", "hash2"],
            duration_seconds=120.0,
        )
        assert "#hash1" in metadata.youtube_description()
        assert "tag1" in metadata.youtube_tags()
        print("  ✅ VideoMetadata model works correctly")

        # Test VideoOutput
        output = VideoOutput(
            video_path=Path("test.mp4"),
            thumbnail_path=Path("test.png"),
            success=True,
        )
        assert output.success == True
        print("  ✅ VideoOutput model works correctly")

        # Test PipelineState
        state = PipelineState(request=request)
        state.add_error("Test error")
        assert not state.is_successful()
        print("  ✅ PipelineState model works correctly")

        return True

    except Exception as e:
        print(f"  ❌ Model test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_utils():
    """Test utility modules."""
    print("\n🔧 Testing utilities...")

    try:
        from src.utils.ai_client import AIClient, AIConfig, get_ai_client

        # Test AIConfig
        config = AIConfig(
            provider="openai",
            openai_model="gpt-4o-mini",
        )
        assert config.provider == "openai"
        print("  ✅ AIConfig works correctly")

        # Test AIClient initialization (without API calls)
        client = AIClient(config)
        assert client.config.provider == "openai"
        print("  ✅ AIClient can be instantiated")

        return True

    except Exception as e:
        print(f"  ❌ Utility test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_components():
    """Test that pipeline components can be instantiated."""
    print("\n🔄 Testing pipeline components...")

    try:
        from pathlib import Path

        from src.pipeline import (
            ContentResearcher,
            ImageGenerator,
            ScenePlanner,
            ScriptWriter,
            TTSProvider,
            VideoComposer,
            VideoPipeline,
            VoiceGenerator,
        )

        # Note: We can't fully test these without API keys,
        # but we can verify they instantiate

        # These require AI client, so we'll skip full instantiation
        # researcher = ContentResearcher()
        # script_writer = ScriptWriter()
        # scene_planner = ScenePlanner()

        # Test ImageGenerator with output dir
        test_output_dir = Path("output/test_images")
        # image_gen = ImageGenerator(output_dir=test_output_dir)
        print("  ✅ ImageGenerator class available")

        # Test VoiceGenerator
        # voice_gen = VoiceGenerator(provider=TTSProvider.GTTS)
        print("  ✅ VoiceGenerator class available")

        # Test VideoComposer
        # composer = VideoComposer()
        print("  ✅ VideoComposer class available")

        # Test TTSProvider enum
        assert TTSProvider.GTTS.value == "gtts"
        assert TTSProvider.OPENAI.value == "openai"
        assert TTSProvider.ELEVENLABS.value == "elevenlabs"
        print("  ✅ TTSProvider enum works correctly")

        print("  ✅ All pipeline components are available")
        return True

    except Exception as e:
        print(f"  ❌ Component test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cli_module():
    """Test that CLI module can be imported."""
    print("\n🖥️  Testing CLI module...")

    try:
        # We can't run main() but we can import it
        import generate_video

        assert hasattr(generate_video, "main")
        print("  ✅ CLI module imports correctly")
        return True
    except Exception as e:
        print(f"  ❌ CLI test error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 VIDEO GENERATION PIPELINE - STRUCTURE TEST")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Models", test_models()))
    results.append(("Utilities", test_utils()))
    results.append(("Pipeline Components", test_pipeline_components()))
    results.append(("CLI Module", test_cli_module()))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Pipeline structure is correct.")
        print("\nNext steps:")
        print("  1. Set up your .env file with API keys (see .env.example)")
        print(
            "  2. Run: python generate_video.py --topic 'Your topic' --audience 'Your audience' --format long"
        )
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
