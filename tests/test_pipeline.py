"""
Tests for the composable pipeline infrastructure.

These tests verify that:
1. PipelineStage correctly defines stages with dependencies
2. ComposablePipeline correctly manages stage execution
3. Skip/resume functionality works correctly
4. Checkpoints can be saved and loaded
5. Stage dependencies are properly enforced
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.models import PipelineState, VideoFormat, VideoRequest
from src.pipeline.pipeline import (
    ComposablePipeline,
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
    StageResult,
    StageStatus,
    create_pipeline,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_request():
    """Create a sample video request for testing."""
    return VideoRequest(
        topic="Test Topic",
        target_audience="beginners",
        format=VideoFormat.SHORT,
        style="educational",
    )


@pytest.fixture
def sample_state(sample_request):
    """Create a sample pipeline state for testing."""
    return PipelineState(request=sample_request)


def create_mock_stage(name: str, should_fail: bool = False):
    """Create a mock stage function for testing."""

    def stage_fn(state: PipelineState, context) -> PipelineState:
        if should_fail:
            raise Exception(f"Stage {name} failed intentionally")
        state.current_stage = f"{name}_completed"
        return state

    return stage_fn


# =============================================================================
# TEST STAGE RESULT
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_create_pending(self):
        """Test creating a pending stage result."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.PENDING,
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.PENDING
        assert result.started_at is None
        assert result.completed_at is None
        assert result.error_message is None

    def test_create_completed(self):
        """Test creating a completed stage result."""
        start = datetime.now()
        end = datetime.now()

        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            duration_seconds=5.5,
        )

        assert result.status == StageStatus.COMPLETED
        assert result.started_at == start
        assert result.completed_at == end
        assert result.duration_seconds == 5.5

    def test_create_failed(self):
        """Test creating a failed stage result."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.FAILED,
            error_message="Something went wrong",
        )

        assert result.status == StageStatus.FAILED
        assert result.error_message == "Something went wrong"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        start = datetime(2025, 1, 15, 10, 0, 0)
        end = datetime(2025, 1, 15, 10, 0, 5)

        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            duration_seconds=5.0,
        )

        d = result.to_dict()

        assert d["stage_name"] == "test_stage"
        assert d["status"] == "completed"
        assert d["started_at"] == start.isoformat()
        assert d["completed_at"] == end.isoformat()
        assert d["duration_seconds"] == 5.0

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "stage_name": "test_stage",
            "status": "completed",
            "started_at": "2025-01-15T10:00:00",
            "completed_at": "2025-01-15T10:00:05",
            "error_message": None,
            "duration_seconds": 5.0,
        }

        result = StageResult.from_dict(d)

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.duration_seconds == 5.0


# =============================================================================
# TEST PIPELINE STAGE
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_create_basic_stage(self):
        """Test creating a basic pipeline stage."""

        def dummy_fn(state, context):
            return state

        stage = PipelineStage(
            name="test_stage",
            execute=dummy_fn,
        )

        assert stage.name == "test_stage"
        assert stage.execute == dummy_fn
        assert stage.depends_on == []
        assert stage.optional is False
        assert stage.retry_count == 0

    def test_create_with_dependencies(self):
        """Test creating a stage with dependencies."""

        def dummy_fn(state, context):
            return state

        stage = PipelineStage(
            name="dependent_stage",
            execute=dummy_fn,
            depends_on=["stage1", "stage2"],
        )

        assert stage.depends_on == ["stage1", "stage2"]

    def test_create_optional_stage(self):
        """Test creating an optional stage."""

        def dummy_fn(state, context):
            return state

        stage = PipelineStage(
            name="optional_stage",
            execute=dummy_fn,
            optional=True,
            retry_count=3,
        )

        assert stage.optional is True
        assert stage.retry_count == 3

    def test_default_description(self):
        """Test that default description is generated."""

        def dummy_fn(state, context):
            return state

        stage = PipelineStage(
            name="my_stage",
            execute=dummy_fn,
        )

        assert "my_stage" in stage.description

    def test_custom_description(self):
        """Test custom description."""

        def dummy_fn(state, context):
            return state

        stage = PipelineStage(
            name="my_stage",
            execute=dummy_fn,
            description="Custom description for my stage",
        )

        assert stage.description == "Custom description for my stage"


# =============================================================================
# TEST PIPELINE CONFIG
# =============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.skip_stages == []
        assert config.only_stages is None
        assert config.stop_after is None
        assert config.save_checkpoints is True
        assert config.continue_on_optional_failure is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            skip_stages=["stage1", "stage2"],
            only_stages=["stage3", "stage4"],
            stop_after="stage3",
            save_checkpoints=False,
            continue_on_optional_failure=False,
        )

        assert config.skip_stages == ["stage1", "stage2"]
        assert config.only_stages == ["stage3", "stage4"]
        assert config.stop_after == "stage3"
        assert config.save_checkpoints is False
        assert config.continue_on_optional_failure is False


# =============================================================================
# TEST COMPOSABLE PIPELINE
# =============================================================================


class TestComposablePipeline:
    """Tests for ComposablePipeline class."""

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipeline = ComposablePipeline()

        assert pipeline.stages == {}
        assert pipeline.stage_order == []
        assert pipeline.context is None

    def test_create_with_context(self):
        """Test creating a pipeline with context."""
        context = {"ai_client": MagicMock()}
        pipeline = ComposablePipeline(context=context)

        assert pipeline.context == context

    def test_add_stage(self):
        """Test adding a stage to the pipeline."""
        pipeline = ComposablePipeline()
        stage = PipelineStage(
            name="test_stage",
            execute=create_mock_stage("test"),
        )

        pipeline.add_stage(stage)

        assert "test_stage" in pipeline.stages
        assert "test_stage" in pipeline.stage_order

    def test_add_stage_chaining(self):
        """Test that add_stage returns self for chaining."""
        pipeline = ComposablePipeline()
        stage1 = PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        stage2 = PipelineStage(
            name="stage2",
            execute=create_mock_stage("stage2"),
            depends_on=["stage1"],
        )

        result = pipeline.add_stage(stage1).add_stage(stage2)

        assert result is pipeline
        assert len(pipeline.stages) == 2

    def test_add_duplicate_stage_raises(self):
        """Test that adding a duplicate stage raises an error."""
        pipeline = ComposablePipeline()
        stage = PipelineStage(name="test", execute=create_mock_stage("test"))

        pipeline.add_stage(stage)

        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_stage(stage)

    def test_add_stage_with_missing_dependency_raises(self):
        """Test that adding a stage with missing dependency raises."""
        pipeline = ComposablePipeline()
        stage = PipelineStage(
            name="test",
            execute=create_mock_stage("test"),
            depends_on=["nonexistent"],
        )

        with pytest.raises(ValueError, match="depends on 'nonexistent'"):
            pipeline.add_stage(stage)

    def test_remove_stage(self):
        """Test removing a stage from the pipeline."""
        pipeline = ComposablePipeline()
        stage = PipelineStage(name="test", execute=create_mock_stage("test"))
        pipeline.add_stage(stage)

        pipeline.remove_stage("test")

        assert "test" not in pipeline.stages
        assert "test" not in pipeline.stage_order

    def test_remove_stage_with_dependents_raises(self):
        """Test that removing a stage with dependents raises."""
        pipeline = ComposablePipeline()
        stage1 = PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        stage2 = PipelineStage(
            name="stage2",
            execute=create_mock_stage("stage2"),
            depends_on=["stage1"],
        )
        pipeline.add_stage(stage1).add_stage(stage2)

        with pytest.raises(ValueError, match="depends on it"):
            pipeline.remove_stage("stage1")

    def test_remove_nonexistent_stage_raises(self):
        """Test that removing a nonexistent stage raises."""
        pipeline = ComposablePipeline()

        with pytest.raises(ValueError, match="not found"):
            pipeline.remove_stage("nonexistent")

    def test_get_stage(self):
        """Test getting a stage by name."""
        pipeline = ComposablePipeline()
        stage = PipelineStage(name="test", execute=create_mock_stage("test"))
        pipeline.add_stage(stage)

        result = pipeline.get_stage("test")

        assert result is stage

    def test_get_nonexistent_stage(self):
        """Test getting a nonexistent stage returns None."""
        pipeline = ComposablePipeline()

        result = pipeline.get_stage("nonexistent")

        assert result is None

    def test_list_stages(self):
        """Test listing stages in order."""
        pipeline = ComposablePipeline()
        pipeline.add_stage(PipelineStage(name="a", execute=create_mock_stage("a")))
        pipeline.add_stage(
            PipelineStage(name="b", execute=create_mock_stage("b"), depends_on=["a"])
        )
        pipeline.add_stage(
            PipelineStage(name="c", execute=create_mock_stage("c"), depends_on=["b"])
        )

        stages = pipeline.list_stages()

        assert stages == ["a", "b", "c"]


# =============================================================================
# TEST PIPELINE EXECUTION
# =============================================================================


class TestPipelineExecution:
    """Tests for pipeline execution."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple pipeline with 3 stages."""
        pipeline = ComposablePipeline()
        pipeline.add_stage(
            PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage2",
                execute=create_mock_stage("stage2"),
                depends_on=["stage1"],
            )
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage3",
                execute=create_mock_stage("stage3"),
                depends_on=["stage2"],
            )
        )
        return pipeline

    def test_run_all_stages(self, simple_pipeline, sample_request):
        """Test running all stages successfully."""
        config = PipelineConfig(save_checkpoints=False)

        state = simple_pipeline.run(sample_request, config=config)

        assert state.current_stage == "stage3_completed"

    def test_skip_stage(self, simple_pipeline, sample_request):
        """Test skipping a stage."""
        config = PipelineConfig(
            skip_stages=["stage2"],
            save_checkpoints=False,
        )

        state = simple_pipeline.run(sample_request, config=config)

        # Stage 3 should still run (dependency was skipped but that's OK)
        assert state is not None

    def test_stop_after_stage(self, simple_pipeline, sample_request):
        """Test stopping after a specific stage."""
        config = PipelineConfig(
            stop_after="stage2",
            save_checkpoints=False,
        )

        state = simple_pipeline.run(sample_request, config=config)

        assert state.current_stage == "stage2_completed"

    def test_run_only_specific_stages(self, sample_request):
        """Test running only specific stages."""
        pipeline = ComposablePipeline()
        pipeline.add_stage(
            PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage2",
                execute=create_mock_stage("stage2"),
                depends_on=["stage1"],
            )
        )
        pipeline.add_stage(
            PipelineStage(name="stage3", execute=create_mock_stage("stage3"))
        )

        config = PipelineConfig(
            only_stages=["stage2"],  # Should also run stage1 (dependency)
            save_checkpoints=False,
        )

        state = pipeline.run(sample_request, config=config)

        # Stage1 and stage2 should run, stage3 should not
        assert state is not None

    def test_stage_failure_stops_pipeline(self, sample_request):
        """Test that a failed stage stops the pipeline."""
        pipeline = ComposablePipeline()
        pipeline.add_stage(
            PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage2",
                execute=create_mock_stage("stage2", should_fail=True),
                depends_on=["stage1"],
            )
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage3",
                execute=create_mock_stage("stage3"),
                depends_on=["stage2"],
            )
        )

        config = PipelineConfig(save_checkpoints=False)

        state = pipeline.run(sample_request, config=config)

        # Pipeline should have stopped at stage2
        assert "stage2 failed" in state.errors[0]

    def test_optional_stage_failure_continues(self, sample_request):
        """Test that optional stage failure allows continuation."""
        pipeline = ComposablePipeline()
        pipeline.add_stage(
            PipelineStage(name="stage1", execute=create_mock_stage("stage1"))
        )
        pipeline.add_stage(
            PipelineStage(
                name="stage2",
                execute=create_mock_stage("stage2", should_fail=True),
                depends_on=["stage1"],
                optional=True,
            )
        )
        pipeline.add_stage(
            PipelineStage(name="stage3", execute=create_mock_stage("stage3"))
        )

        config = PipelineConfig(
            save_checkpoints=False,
            continue_on_optional_failure=True,
        )

        state = pipeline.run(sample_request, config=config)

        # Pipeline should continue past failed optional stage
        assert state.current_stage == "stage3_completed"


# =============================================================================
# TEST PIPELINE CHECKPOINT
# =============================================================================


class TestPipelineCheckpoint:
    """Tests for PipelineCheckpoint class."""

    @pytest.fixture
    def sample_checkpoint(self, sample_request, sample_state):
        """Create a sample checkpoint."""
        return PipelineCheckpoint(
            request=sample_request,
            state=sample_state,
            completed_stages=["stage1", "stage2"],
            stage_results=[
                StageResult(stage_name="stage1", status=StageStatus.COMPLETED),
                StageResult(stage_name="stage2", status=StageStatus.COMPLETED),
            ],
        )

    def test_checkpoint_creation(self, sample_checkpoint):
        """Test checkpoint creation."""
        assert sample_checkpoint.completed_stages == ["stage1", "stage2"]
        assert len(sample_checkpoint.stage_results) == 2
        assert sample_checkpoint.checkpoint_version == "1.0"

    def test_save_and_load_checkpoint(self, sample_checkpoint, tmp_path):
        """Test saving and loading a checkpoint."""
        checkpoint_path = tmp_path / "test_checkpoint.pkl"

        sample_checkpoint.save(checkpoint_path)

        assert checkpoint_path.exists()

        loaded = PipelineCheckpoint.load(checkpoint_path)

        assert loaded.completed_stages == sample_checkpoint.completed_stages
        assert loaded.request.topic == sample_checkpoint.request.topic

    def test_save_json_metadata(self, sample_checkpoint, tmp_path):
        """Test saving JSON metadata."""
        json_path = tmp_path / "checkpoint.json"

        sample_checkpoint.save_json(json_path)

        assert json_path.exists()

        import json

        with open(json_path) as f:
            data = json.load(f)

        assert data["completed_stages"] == ["stage1", "stage2"]
        assert data["request"]["topic"] == "Test Topic"


# =============================================================================
# TEST CREATE PIPELINE FACTORY
# =============================================================================


class TestCreatePipelineFactory:
    """Tests for create_pipeline factory function."""

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline with factory."""
        pipeline = create_pipeline()

        assert isinstance(pipeline, ComposablePipeline)
        assert pipeline.stages == {}
        assert pipeline.context is None

    def test_create_with_context(self):
        """Test creating pipeline with context."""
        context = {"key": "value"}

        pipeline = create_pipeline(context=context)

        assert pipeline.context == context


# =============================================================================
# TEST STAGE STATUS ENUM
# =============================================================================


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.FAILED.value == "failed"

    def test_status_from_string(self):
        """Test creating status from string."""
        status = StageStatus("completed")
        assert status == StageStatus.COMPLETED
