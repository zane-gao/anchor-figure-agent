"""Figure Agent research prototype."""

from .agents import FigureEditPipeline, HierarchicalCoordinator, HierarchicalDecomposer, build_scene_from_context
from .benchmark import BenchmarkBuildConfig, BenchmarkBuilder
from .drawio_adapter import DrawIOAdapter
from .experiments import ExperimentConfig, ExperimentRunner

__all__ = [
    "FigureEditPipeline",
    "HierarchicalDecomposer",
    "HierarchicalCoordinator",
    "BenchmarkBuilder",
    "BenchmarkBuildConfig",
    "ExperimentConfig",
    "ExperimentRunner",
    "DrawIOAdapter",
    "build_scene_from_context",
]
