"""
Composable Pipeline Infrastructure

This module provides a flexible pipeline system that supports:
- Skip/resume functionality
- Checkpoint saving and loading
- Configurable stage execution
- Stage dependencies

Usage:
    from src.pipeline import ComposablePipeline, PipelineStage

    # Create pipeline with stages
    pipeline = ComposablePipeline()
    pipeline.add_stage(PipelineStage("research", research_fn))
    pipeline.add_stage(PipelineStage("script", script_fn, depends_on=["research"]))

    # Run with options
    output = pipeline.run(request, skip_stages=["research"])

    # Or resume from checkpoint
    output = pipeline.run(request, resume_from="script")
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .models import PipelineState, VideoRequest

# Type for stage execution functions
StageFunction = Callable[[PipelineState, Any], PipelineState]


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""

    stage_name: str
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageResult":
        """Create from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            status=StageStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data["started_at"]
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data["completed_at"]
            else None,
            error_message=data.get("error_message"),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class PipelineStage:
    """
    Definition of a single pipeline stage.

    Attributes:
        name: Unique identifier for the stage
        execute: Function to execute the stage
        description: Human-readable description
        depends_on: List of stage names this stage depends on
        optional: If True, failure won't stop the pipeline
        retry_count: Number of times to retry on failure
    """

    name: str
    execute: StageFunction
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False
    retry_count: int = 0

    def __post_init__(self):
        if not self.description:
            self.description = f"Stage: {self.name}"


@dataclass
class PipelineCheckpoint:
    """
    Checkpoint for saving/resuming pipeline state.

    Contains all information needed to resume a pipeline from a specific point.
    """

    request: VideoRequest
    state: PipelineState
    completed_stages: list[str]
    stage_results: list[StageResult]
    created_at: datetime = field(default_factory=datetime.now)
    checkpoint_version: str = "1.0"

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  💾 Checkpoint saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "PipelineCheckpoint":
        """Load checkpoint from file."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"  📂 Checkpoint loaded: {path}")
        return checkpoint

    def save_json(self, path: Path) -> None:
        """Save checkpoint metadata as JSON (human-readable)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "checkpoint_version": self.checkpoint_version,
            "created_at": self.created_at.isoformat(),
            "completed_stages": self.completed_stages,
            "current_stage": self.state.current_stage,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "request": {
                "topic": self.request.topic,
                "target_audience": self.request.target_audience,
                "format": self.request.format.value,
                "style": self.request.style,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class PipelineConfig:
    """
    Configuration for pipeline execution.

    Attributes:
        skip_stages: List of stage names to skip
        only_stages: If set, only run these stages (and dependencies)
        stop_after: Stop after this stage completes
        save_checkpoints: Whether to save checkpoints after each stage
        checkpoint_dir: Directory for checkpoint files
        continue_on_optional_failure: Continue if optional stage fails
    """

    skip_stages: list[str] = field(default_factory=list)
    only_stages: Optional[list[str]] = None
    stop_after: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("output/checkpoints"))
    continue_on_optional_failure: bool = True


class ComposablePipeline:
    """
    A composable, resumable pipeline for video generation.

    Supports:
    - Adding/removing stages dynamically
    - Skipping specific stages
    - Resuming from checkpoints
    - Stage dependencies
    - Parallel stage execution (where dependencies allow)

    Usage:
        pipeline = ComposablePipeline()

        # Add stages
        pipeline.add_stage(PipelineStage("research", research_topic))
        pipeline.add_stage(PipelineStage("script", generate_script, depends_on=["research"]))

        # Run pipeline
        output = pipeline.run(request)

        # Or run with options
        output = pipeline.run(request, config=PipelineConfig(skip_stages=["research"]))

        # Or resume from checkpoint
        output = pipeline.resume(checkpoint_path)
    """

    def __init__(self, context: Optional[Any] = None):
        """
        Initialize the pipeline.

        Args:
            context: Shared context object passed to all stages (e.g., AI client, config)
        """
        self.stages: dict[str, PipelineStage] = {}
        self.stage_order: list[str] = []
        self.context = context
        self._stage_results: list[StageResult] = []

    def add_stage(self, stage: PipelineStage) -> "ComposablePipeline":
        """
        Add a stage to the pipeline.

        Args:
            stage: The pipeline stage to add

        Returns:
            Self for chaining
        """
        if stage.name in self.stages:
            raise ValueError(f"Stage '{stage.name}' already exists")

        # Validate dependencies exist
        for dep in stage.depends_on:
            if dep not in self.stages:
                raise ValueError(
                    f"Stage '{stage.name}' depends on '{dep}' which hasn't been added yet"
                )

        self.stages[stage.name] = stage
        self.stage_order.append(stage.name)
        return self

    def remove_stage(self, name: str) -> "ComposablePipeline":
        """
        Remove a stage from the pipeline.

        Args:
            name: Name of the stage to remove

        Returns:
            Self for chaining
        """
        if name not in self.stages:
            raise ValueError(f"Stage '{name}' not found")

        # Check if other stages depend on this one
        for stage in self.stages.values():
            if name in stage.depends_on:
                raise ValueError(
                    f"Cannot remove '{name}': stage '{stage.name}' depends on it"
                )

        del self.stages[name]
        self.stage_order.remove(name)
        return self

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        return self.stages.get(name)

    def list_stages(self) -> list[str]:
        """Get ordered list of stage names."""
        return list(self.stage_order)

    def run(
        self,
        request: VideoRequest,
        config: Optional[PipelineConfig] = None,
        initial_state: Optional[PipelineState] = None,
    ) -> PipelineState:
        """
        Execute the pipeline.

        Args:
            request: The video request to process
            config: Pipeline execution configuration
            initial_state: Optional initial state (for resuming)

        Returns:
            Final pipeline state
        """
        config = config or PipelineConfig()
        state = initial_state or PipelineState(request=request)
        self._stage_results = []

        # Determine which stages to run
        stages_to_run = self._resolve_stages_to_run(config, state)

        print(f"\n🚀 Pipeline starting with {len(stages_to_run)} stages")
        print(f"   Stages: {' → '.join(stages_to_run)}")
        if config.skip_stages:
            print(f"   Skipping: {', '.join(config.skip_stages)}")
        print()

        # Execute stages in order
        for stage_name in stages_to_run:
            stage = self.stages[stage_name]

            # Check if we should skip this stage
            if stage_name in config.skip_stages:
                print(f"⏭️  Skipping stage: {stage_name}")
                self._stage_results.append(
                    StageResult(stage_name=stage_name, status=StageStatus.SKIPPED)
                )
                continue

            # Check dependencies are met
            if not self._check_dependencies(stage, state):
                if stage.optional:
                    print(
                        f"⏭️  Skipping optional stage '{stage_name}': dependencies not met"
                    )
                    self._stage_results.append(
                        StageResult(stage_name=stage_name, status=StageStatus.SKIPPED)
                    )
                    continue
                else:
                    error_msg = f"Stage '{stage_name}' dependencies not met"
                    state.add_error(error_msg)
                    self._stage_results.append(
                        StageResult(
                            stage_name=stage_name,
                            status=StageStatus.FAILED,
                            error_message=error_msg,
                        )
                    )
                    break

            # Execute the stage
            result = self._execute_stage(stage, state)
            self._stage_results.append(result)

            if result.status == StageStatus.FAILED:
                if stage.optional and config.continue_on_optional_failure:
                    print(f"⚠️  Optional stage '{stage_name}' failed, continuing...")
                    continue
                else:
                    print(f"❌ Pipeline stopped at stage '{stage_name}'")
                    break

            # Save checkpoint if enabled
            if config.save_checkpoints:
                self._save_checkpoint(request, state, config.checkpoint_dir)

            # Check if we should stop after this stage
            if config.stop_after and stage_name == config.stop_after:
                print(f"⏹️  Stopping after stage '{stage_name}' as configured")
                break

        # Print summary
        self._print_summary()

        return state

    def resume(
        self,
        checkpoint_path: Path,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineState:
        """
        Resume pipeline from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            config: Pipeline configuration (overrides checkpoint config)

        Returns:
            Final pipeline state
        """
        checkpoint = PipelineCheckpoint.load(checkpoint_path)

        print(f"\n🔄 Resuming pipeline from checkpoint")
        print(f"   Created: {checkpoint.created_at}")
        print(f"   Completed stages: {', '.join(checkpoint.completed_stages)}")

        # Create config that skips already-completed stages
        resume_config = config or PipelineConfig()
        resume_config.skip_stages = list(
            set(resume_config.skip_stages) | set(checkpoint.completed_stages)
        )

        return self.run(
            checkpoint.request,
            config=resume_config,
            initial_state=checkpoint.state,
        )

    def _resolve_stages_to_run(
        self, config: PipelineConfig, state: PipelineState
    ) -> list[str]:
        """Determine which stages to run based on config."""
        if config.only_stages:
            # Run only specified stages (and their dependencies)
            stages = set()
            for stage_name in config.only_stages:
                stages.add(stage_name)
                stages.update(self._get_all_dependencies(stage_name))
            # Maintain original order
            return [s for s in self.stage_order if s in stages]
        else:
            return list(self.stage_order)

    def _get_all_dependencies(self, stage_name: str) -> set[str]:
        """Get all dependencies for a stage (recursive)."""
        if stage_name not in self.stages:
            return set()

        deps = set()
        stage = self.stages[stage_name]
        for dep in stage.depends_on:
            deps.add(dep)
            deps.update(self._get_all_dependencies(dep))
        return deps

    def _check_dependencies(self, stage: PipelineStage, state: PipelineState) -> bool:
        """Check if all dependencies for a stage are met."""
        completed = {
            r.stage_name
            for r in self._stage_results
            if r.status == StageStatus.COMPLETED
        }
        skipped_ok = {
            r.stage_name for r in self._stage_results if r.status == StageStatus.SKIPPED
        }

        for dep in stage.depends_on:
            if dep not in completed and dep not in skipped_ok:
                return False
        return True

    def _execute_stage(self, stage: PipelineStage, state: PipelineState) -> StageResult:
        """Execute a single stage with retry logic."""
        result = StageResult(
            stage_name=stage.name,
            status=StageStatus.RUNNING,
            started_at=datetime.now(),
        )

        attempts = 0
        max_attempts = stage.retry_count + 1

        while attempts < max_attempts:
            attempts += 1
            try:
                state.current_stage = stage.name
                print(
                    f"▶️  Stage {self.stage_order.index(stage.name) + 1}/{len(self.stage_order)}: {stage.name}"
                )

                # Execute the stage function
                state = stage.execute(state, self.context)

                result.status = StageStatus.COMPLETED
                result.completed_at = datetime.now()
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

                print(f"   ✅ Completed in {result.duration_seconds:.1f}s\n")
                return result

            except Exception as e:
                error_msg = f"{stage.name} failed: {str(e)}"
                if attempts < max_attempts:
                    print(
                        f"   ⚠️  Attempt {attempts}/{max_attempts} failed, retrying..."
                    )
                else:
                    result.status = StageStatus.FAILED
                    result.error_message = error_msg
                    result.completed_at = datetime.now()
                    result.duration_seconds = (
                        result.completed_at - result.started_at
                    ).total_seconds()
                    state.add_error(error_msg)
                    print(f"   ❌ {error_msg}\n")

        return result

    def _save_checkpoint(
        self, request: VideoRequest, state: PipelineState, checkpoint_dir: Path
    ) -> None:
        """Save a checkpoint after successful stage completion."""
        completed_stages = [
            r.stage_name
            for r in self._stage_results
            if r.status == StageStatus.COMPLETED
        ]

        checkpoint = PipelineCheckpoint(
            request=request,
            state=state,
            completed_stages=completed_stages,
            stage_results=list(self._stage_results),
        )

        # Save binary checkpoint (for resuming)
        checkpoint_path = checkpoint_dir / f"checkpoint_{state.current_stage}.pkl"
        checkpoint.save(checkpoint_path)

        # Save JSON metadata (for inspection)
        json_path = checkpoint_dir / f"checkpoint_{state.current_stage}.json"
        checkpoint.save_json(json_path)

    def _print_summary(self) -> None:
        """Print execution summary."""
        completed = sum(
            1 for r in self._stage_results if r.status == StageStatus.COMPLETED
        )
        skipped = sum(1 for r in self._stage_results if r.status == StageStatus.SKIPPED)
        failed = sum(1 for r in self._stage_results if r.status == StageStatus.FAILED)
        total_time = sum(r.duration_seconds for r in self._stage_results)

        print(f"\n{'=' * 50}")
        print("📊 Pipeline Summary")
        print(f"{'=' * 50}")
        print(f"   ✅ Completed: {completed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"{'=' * 50}\n")


# Convenience function for creating a basic pipeline
def create_pipeline(context: Optional[Any] = None) -> ComposablePipeline:
    """
    Create an empty composable pipeline.

    Args:
        context: Shared context for all stages

    Returns:
        New ComposablePipeline instance
    """
    return ComposablePipeline(context=context)
