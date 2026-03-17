from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json

from .agents import FigureEditPipeline, build_scene_from_context
from .evaluation import EvaluationResult, evaluate_case, save_evaluation
from .models import FigureSceneGraph, PipelineConfig, PipelineRequest
from .utils import ensure_dir, load_json, save_json


@dataclass
class ExperimentMethodSpec:
    method_id: str
    display_name: str
    kind: str = "pipeline"
    description: str = ""
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)


@dataclass
class ExperimentConfig:
    benchmark_root: str
    results_root: str
    method_ids: list[str] = field(default_factory=list)
    include_ablations: bool = False
    limit_cases: int | None = None
    synthetic_only: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**load_json(path))


def main_method_specs() -> list[ExperimentMethodSpec]:
    return [
        ExperimentMethodSpec(
            method_id="draft_only",
            display_name="Draft-Only",
            description="直接使用粗糙初稿，不做结构修复。",
            pipeline_config=PipelineConfig(
                method_name="Draft-Only",
                use_verifier=False,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=False,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=False,
                use_retoucher=False,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="one_shot_edit_closed",
            display_name="One-Shot Edit (Closed Proxy)",
            description="闭源单次编辑代理基线：单轮、带验证、无多轮 critic。",
            pipeline_config=PipelineConfig(
                method_name="One-Shot Edit (Closed)",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=False,
                use_layout_anchors=True,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="one_shot_edit_open",
            display_name="One-Shot Edit (Open Proxy)",
            description="开源单次编辑代理基线：单轮、弱验证、弱结构约束。",
            pipeline_config=PipelineConfig(
                method_name="One-Shot Edit (Open)",
                use_verifier=False,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=False,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="regenerate_from_context",
            display_name="Regenerate-from-Context",
            kind="context_regenerate",
            description="从 paper context 直接重建场景图，不走后编辑。",
            pipeline_config=PipelineConfig(
                method_name="Regenerate-from-Context",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="structured_wo_anchors",
            display_name="Structured w/o Anchors",
            description="保留结构恢复和精修，但去掉双层锚点。",
            pipeline_config=PipelineConfig(
                method_name="Structured w/o Anchors",
                use_verifier=True,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=False,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_no_hierarchy",
            display_name="Ablation: No Hierarchy",
            description="启用双锚点但不使用层级分解与层级 team。",
            pipeline_config=PipelineConfig(
                method_name="Ablation-No-Hierarchy",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
                max_iterations_override=4,
            ),
        ),
        ExperimentMethodSpec(
            method_id="anchorfigure_full",
            display_name="AnchorFigure (Full)",
            description="完整双层锚点 + 分治式层级结构编辑方法。",
            pipeline_config=PipelineConfig(
                method_name="AnchorFigure (Full)",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
                max_iterations_override=4,
            ),
        ),
    ]


def ablation_method_specs() -> list[ExperimentMethodSpec]:
    return [
        ExperimentMethodSpec(
            method_id="ablation_no_semantic_anchors",
            display_name="Ablation: No Semantic Anchors",
            pipeline_config=PipelineConfig(
                method_name="Ablation-No-Semantic-Anchors",
                use_verifier=True,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_no_layout_anchors",
            display_name="Ablation: No Layout Anchors",
            pipeline_config=PipelineConfig(
                method_name="Ablation-No-Layout-Anchors",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=False,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_hierarchy_no_semantic",
            display_name="Ablation: Hierarchy No Semantic",
            pipeline_config=PipelineConfig(
                method_name="Ablation-Hierarchy-No-Semantic",
                use_verifier=True,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_hierarchy_no_layout",
            display_name="Ablation: Hierarchy No Layout",
            pipeline_config=PipelineConfig(
                method_name="Ablation-Hierarchy-No-Layout",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=False,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_no_verifier",
            display_name="Ablation: No Tool Verification",
            pipeline_config=PipelineConfig(
                method_name="Ablation-No-Tool-Verification",
                use_verifier=False,
                use_context_matching=False,
                use_semantic_anchors=False,
                use_layout_anchors=False,
                use_hierarchy_decomposition=False,
                use_hierarchical_team=False,
                use_hierarchy_metrics=False,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_no_critic",
            display_name="Ablation: No Critic Loop",
            pipeline_config=PipelineConfig(
                method_name="Ablation-No-Critic-Loop",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=False,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_no_retoucher",
            display_name="Ablation: Layout Only",
            pipeline_config=PipelineConfig(
                method_name="Ablation-Layout-Only",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=False,
                use_critic_loop=True,
            ),
        ),
        ExperimentMethodSpec(
            method_id="ablation_png_only",
            display_name="Ablation: PNG Only",
            pipeline_config=PipelineConfig(
                method_name="Ablation-PNG-Only",
                use_verifier=True,
                use_context_matching=True,
                use_semantic_anchors=True,
                use_layout_anchors=True,
                use_hierarchy_decomposition=True,
                use_hierarchical_team=True,
                use_hierarchy_metrics=True,
                use_layout_planner=True,
                use_retoucher=True,
                use_critic_loop=True,
                enforce_editable_export=False,
                export_svg=False,
                export_pptx=False,
            ),
        ),
    ]


def all_method_specs(include_ablations: bool = False) -> dict[str, ExperimentMethodSpec]:
    methods = {spec.method_id: spec for spec in main_method_specs()}
    if include_ablations:
        methods.update({spec.method_id: spec for spec in ablation_method_specs()})
    return methods


def _load_case_dirs(benchmark_root: Path, limit_cases: int | None = None) -> list[Path]:
    cases_root = benchmark_root / "cases" if (benchmark_root / "cases").exists() else benchmark_root
    case_dirs = [path for path in sorted(cases_root.iterdir()) if path.is_dir() and (path / "case_meta.json").exists()]
    return case_dirs[:limit_cases] if limit_cases else case_dirs


def _context_manifest(case_dir: Path, output_dir: Path) -> Path:
    paper_context = (case_dir / "paper_context.txt").read_text(encoding="utf-8")
    target_scene = FigureSceneGraph.load(case_dir / "target_scene.json")
    scene = build_scene_from_context(paper_context, width=target_scene.width, height=target_scene.height)
    manifest_path = output_dir / "context_regenerate_manifest.json"
    save_json(
        manifest_path,
        {
            "paper_concepts": [node.label for node in target_scene.nodes if node.kind != "background"],
            "ocr_candidates": {},
            "draft_scene_graph": scene.to_dict(),
        },
    )
    return manifest_path


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.methods = all_method_specs(include_ablations=config.include_ablations)

    def _selected_methods(self) -> list[ExperimentMethodSpec]:
        if self.config.method_ids:
            return [self.methods[method_id] for method_id in self.config.method_ids]
        return list(self.methods.values())

    def _run_method_case(self, spec: ExperimentMethodSpec, case_dir: Path, output_dir: Path) -> EvaluationResult:
        draft_manifest = case_dir / "draft_manifest.json"
        if spec.kind == "context_regenerate":
            draft_manifest = _context_manifest(case_dir, output_dir)
        request = PipelineRequest(
            draft_image=str(case_dir / "draft.png"),
            draft_manifest=str(draft_manifest),
            paper_context=(case_dir / "paper_context.txt").read_text(encoding="utf-8"),
            caption=(case_dir / "caption.txt").read_text(encoding="utf-8"),
            edit_goal=(case_dir / "edit_goal.txt").read_text(encoding="utf-8"),
            output_dir=str(output_dir),
            pipeline_config=spec.pipeline_config,
        )
        pipeline = FigureEditPipeline(config=spec.pipeline_config)
        pipeline.run(request)
        evaluation = evaluate_case(
            case_id=case_dir.name,
            method_name=spec.display_name,
            result_dir=output_dir,
            target_scene_path=case_dir / "target_scene.json",
            target_hierarchy_path=case_dir / "target_hierarchy.json",
        )
        save_evaluation(evaluation, output_dir / "metrics.json")
        return evaluation

    def run(self) -> dict[str, Any]:
        benchmark_root = Path(self.config.benchmark_root)
        results_root = ensure_dir(self.config.results_root)
        case_dirs = _load_case_dirs(benchmark_root, self.config.limit_cases)
        method_summaries: dict[str, Any] = {}
        failure_rows: list[dict[str, Any]] = []

        for spec in self._selected_methods():
            per_case_results: list[EvaluationResult] = []
            method_root = ensure_dir(results_root / spec.method_id)
            for case_dir in case_dirs:
                case_output = ensure_dir(method_root / case_dir.name)
                evaluation = self._run_method_case(spec, case_dir, case_output)
                case_meta = load_json(case_dir / "case_meta.json")
                for tag in evaluation.failure_tags:
                    failure_rows.append(
                        {
                            "method_id": spec.method_id,
                            "case_id": case_dir.name,
                            "family": case_meta["family"],
                            "difficulty": case_meta["difficulty"],
                            "failure_tag": tag,
                        }
                    )
                per_case_results.append(evaluation)
            method_summaries[spec.method_id] = self._summarize_method(spec, per_case_results, case_dirs)

        summary_payload = {
            "benchmark_root": str(benchmark_root),
            "results_root": str(results_root),
            "methods": method_summaries,
            "case_count": len(case_dirs),
        }
        save_json(results_root / "summary.json", summary_payload)
        self._write_summary_csv(results_root / "summary.csv", method_summaries)
        self._write_failure_csv(results_root / "analysis" / "failure_taxonomy.csv", failure_rows)
        return summary_payload

    def _summarize_method(
        self,
        spec: ExperimentMethodSpec,
        results: list[EvaluationResult],
        case_dirs: list[Path],
    ) -> dict[str, Any]:
        if not results:
            return {"display_name": spec.display_name, "overall": {}, "by_family": {}, "by_difficulty": {}}
        metric_keys = list(results[0].metrics.keys())
        overall = {
            key: round(sum(result.metrics[key] for result in results) / len(results), 4)
            for key in metric_keys
        }
        by_family: dict[str, dict[str, float]] = {}
        by_difficulty: dict[str, dict[str, float]] = {}
        case_meta = {case_dir.name: load_json(case_dir / "case_meta.json") for case_dir in case_dirs}
        for bucket_name, target in (("by_family", by_family), ("by_difficulty", by_difficulty)):
            groups: dict[str, list[EvaluationResult]] = {}
            for result in results:
                key = case_meta[result.case_id]["family"] if bucket_name == "by_family" else case_meta[result.case_id]["difficulty"]
                groups.setdefault(key, []).append(result)
            for key, group_results in groups.items():
                target[key] = {
                    metric: round(sum(item.metrics[metric] for item in group_results) / len(group_results), 4)
                    for metric in metric_keys
                }
        return {
            "display_name": spec.display_name,
            "description": spec.description,
            "pipeline_config": asdict(spec.pipeline_config),
            "overall": overall,
            "by_family": by_family,
            "by_difficulty": by_difficulty,
        }

    def _write_summary_csv(self, path: Path, summaries: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "method_id",
                "display_name",
                "overall_auto_score",
                "structure_fidelity",
                "text_accuracy_score",
                "layout_readability_score",
                "editability_score",
                "module_assignment_accuracy",
                "region_boundary_accuracy",
                "block_grouping_accuracy",
                "hierarchy_consistency_score",
                "publication_readiness",
            ])
            for method_id, payload in summaries.items():
                overall = payload["overall"]
                writer.writerow(
                    [
                        method_id,
                        payload["display_name"],
                        overall.get("overall_auto_score", 0.0),
                        overall.get("structure_fidelity", 0.0),
                        overall.get("text_accuracy_score", 0.0),
                        overall.get("layout_readability_score", 0.0),
                        overall.get("editability_score", 0.0),
                        overall.get("module_assignment_accuracy", 0.0),
                        overall.get("region_boundary_accuracy", 0.0),
                        overall.get("block_grouping_accuracy", 0.0),
                        overall.get("hierarchy_consistency_score", 0.0),
                        overall.get("publication_readiness", 0.0),
                    ]
                )

    def _write_failure_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method_id", "case_id", "family", "difficulty", "failure_tag"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def strongest_non_anchorfigure(summary_path: str | Path) -> str | None:
    summary = load_json(summary_path)
    best_method = None
    best_score = -1.0
    for method_id, payload in summary["methods"].items():
        if method_id == "anchorfigure_full":
            continue
        score = payload["overall"].get("overall_auto_score", 0.0)
        if score > best_score:
            best_score = score
            best_method = method_id
    return best_method
