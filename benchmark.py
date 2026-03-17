from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .agents import HierarchicalDecomposer
from .exporters import render_scene_to_png
from .models import BBox, FigureSceneGraph, PipelineConfig, SceneEdge, SceneGroup, SceneNode, make_id
from .utils import deterministic_rng, ensure_dir, save_json


BENCHMARK_NAME = "AnchorFigureBench-v1"
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


@dataclass
class BenchmarkTemplate:
    case_id: str
    family: str
    scene: FigureSceneGraph
    paper_context: str
    caption: str
    edit_goal: str


@dataclass
class BenchmarkBuildConfig:
    benchmark_name: str = BENCHMARK_NAME
    case_count_per_family: int = 120
    difficulty_levels: tuple[str, ...] = DIFFICULTY_LEVELS
    random_seed: int = 7
    synthetic_split_name: str = "cases"
    real_split_name: str = "real_validation"


@dataclass
class TemplateFamily:
    family: str
    title: str
    generator: Callable[[int], BenchmarkTemplate]


def _node(
    node_id: str,
    label: str,
    x: int,
    y: int,
    w: int = 220,
    h: int = 90,
    kind: str = "container",
) -> SceneNode:
    return SceneNode(id=node_id, kind=kind, label=label, bbox=BBox(x, y, w, h))


def _edge(source_id: str, target_id: str) -> SceneEdge:
    return SceneEdge(id=make_id("edge"), source_id=source_id, target_id=target_id)


def parse_context_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            concepts.append(line[2:].strip())
    return concepts


def _context_block(concepts: list[str], flows: list[str], groups: dict[str, list[str]]) -> str:
    lines = ["Concept inventory:"]
    lines.extend(f"- {concept}" for concept in concepts)
    lines.append("Expected flow:")
    lines.extend(flows)
    lines.append("Group summary:")
    for name, members in groups.items():
        lines.append(f"{name}: {', '.join(members)}")
    return "\n".join(lines)


def _palette_variant(seed: str) -> dict[str, str]:
    rng = deterministic_rng(seed)
    palette = [
        ("#DBEAFE", "#E0F2FE", "#ECFCCB"),
        ("#FCE7F3", "#EDE9FE", "#FEF3C7"),
        ("#DCFCE7", "#DBEAFE", "#FCE7F3"),
    ]
    fill_a, fill_b, fill_c = palette[rng.randrange(len(palette))]
    return {"group_a": fill_a, "group_b": fill_b, "group_c": fill_c}


def _single_path_pipeline(idx: int) -> BenchmarkTemplate:
    rng = deterministic_rng(f"single_path:{idx}")
    stage_prefix = ["Input", "Parse", "Align", "Route", "Polish", "Export"]
    nouns = ["Graph", "Anchor", "Layout", "Critic", "Panel", "Figure"]
    labels = [
        f"{stage_prefix[0]} {nouns[rng.randrange(len(nouns))]}",
        "Scene Proposal",
        "Constraint Merge",
        "Anchor Layout",
        "Visual Polish",
        "Editable Export",
    ]
    width = 1580 + rng.randrange(0, 80)
    height = 860 + rng.randrange(0, 40)
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100)]
    x = 100
    y = 360
    for i, label in enumerate(labels, start=1):
        scene.nodes.append(_node(f"n{i}", label, x, y + rng.randrange(-20, 20), 210 + rng.randrange(0, 40), 90))
        x += 250
    scene.edges = [_edge(f"n{i}", f"n{i+1}") for i in range(1, len(labels))]
    scene.groups = [
        SceneGroup(id="g1", label="Pipeline", bbox=BBox(60, 280, width - 120, 240), node_ids=[f"n{i}" for i in range(1, len(labels) + 1)]),
    ]
    flows = [" -> ".join(labels)]
    groups = {"Pipeline": labels}
    context = _context_block(labels, flows, groups)
    caption = "A single-path scientific figure editing pipeline from draft parsing to editable export."
    edit_goal = "修复单路径流程图中的布局错乱、连线错误和风格不一致。"
    return BenchmarkTemplate(f"single_path_pipeline_{idx:03d}", "single_path_pipeline", scene, context, caption, edit_goal)


def _multibranch_method(idx: int) -> BenchmarkTemplate:
    rng = deterministic_rng(f"multibranch:{idx}")
    branch_tokens = ["Verification", "Anchor Builder", "Layout Planner", "Retouch Executor", "Critic Stopper", "Export Heads"]
    width = 1600
    height = 900
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100),
        _node("n1", "Draft Figure", 120, 360),
        _node("n2", "Raster to Scene", 400, 360),
        _node("n3", branch_tokens[0], 700, 220),
        _node("n4", branch_tokens[1], 700, 500),
        _node("n5", branch_tokens[2], 1000, 220),
        _node("n6", branch_tokens[3], 1000, 500),
        _node("n7", branch_tokens[4], 1280, 360),
        _node("n8", branch_tokens[5], 1280, 650),
    ]
    scene.edges = [
        _edge("n1", "n2"),
        _edge("n2", "n3"),
        _edge("n2", "n4"),
        _edge("n3", "n5"),
        _edge("n4", "n6"),
        _edge("n5", "n7"),
        _edge("n6", "n7"),
        _edge("n7", "n8"),
    ]
    scene.groups = [
        SceneGroup(id="g1", label="Perception", bbox=BBox(80, 290, 620, 250), node_ids=["n1", "n2"]),
        SceneGroup(id="g2", label="Structural Editing", bbox=BBox(650, 140, 640, 540), node_ids=["n3", "n4", "n5", "n6"]),
        SceneGroup(id="g3", label="Output", bbox=BBox(1230, 300, 300, 480), node_ids=["n7", "n8"]),
    ]
    if rng.random() > 0.5:
        scene.nodes.append(_node("n9", "Icon Pool", 980, 720, 180, 80, kind="icon"))
        scene.groups[1].node_ids.append("n9")
        scene.edges.append(_edge("n6", "n9"))
    concepts = [node.label for node in scene.nodes if node.kind != "background"]
    flows = [
        "Draft Figure -> Raster to Scene -> Verification -> Layout Planner -> Critic Stopper -> Export Heads",
        "Raster to Scene -> Anchor Builder -> Retouch Executor -> Critic Stopper",
    ]
    groups = {
        "Perception": ["Draft Figure", "Raster to Scene"],
        "Structural Editing": [branch_tokens[0], branch_tokens[1], branch_tokens[2], branch_tokens[3]],
        "Output": [branch_tokens[4], branch_tokens[5]],
    }
    context = _context_block(concepts, flows, groups)
    caption = "A dual-branch scientific figure editing system that separates verification and retouching before converging at a critic."
    edit_goal = "修复多分支方法图的流程关系、分组边界和导出区域。"
    return BenchmarkTemplate(f"multibranch_method_{idx:03d}", "multibranch_method", scene, context, caption, edit_goal)


def _grouped_system(idx: int) -> BenchmarkTemplate:
    width = 1600
    height = 920
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100),
        _node("n1", "Paper Context", 100, 180),
        _node("n2", "Draft PNG", 100, 440),
        _node("n3", "Semantic Anchors", 470, 180),
        _node("n4", "Layout Anchors", 470, 440),
        _node("n5", "Constraint Solver", 840, 180),
        _node("n6", "Style Normalizer", 840, 440),
        _node("n7", "SVG Export", 1210, 140),
        _node("n8", "PPTX Export", 1210, 340),
        _node("n9", "PNG Preview", 1210, 540),
    ]
    scene.edges = [
        _edge("n1", "n3"),
        _edge("n2", "n4"),
        _edge("n3", "n5"),
        _edge("n4", "n5"),
        _edge("n5", "n6"),
        _edge("n6", "n7"),
        _edge("n6", "n8"),
        _edge("n6", "n9"),
    ]
    scene.groups = [
        SceneGroup(id="g1", label="Inputs", bbox=BBox(60, 120, 320, 510), node_ids=["n1", "n2"]),
        SceneGroup(id="g2", label="Scene Graph Core", bbox=BBox(420, 120, 700, 510), node_ids=["n3", "n4", "n5", "n6"]),
        SceneGroup(id="g3", label="Editable Outputs", bbox=BBox(1170, 90, 300, 620), node_ids=["n7", "n8", "n9"]),
    ]
    concepts = [node.label for node in scene.nodes if node.kind != "background"]
    flows = [
        "Paper Context -> Semantic Anchors -> Constraint Solver -> Style Normalizer -> SVG Export",
        "Draft PNG -> Layout Anchors -> Constraint Solver -> Style Normalizer -> PPTX Export",
        "Style Normalizer -> PNG Preview",
    ]
    groups = {
        "Inputs": ["Paper Context", "Draft PNG"],
        "Scene Graph Core": ["Semantic Anchors", "Layout Anchors", "Constraint Solver", "Style Normalizer"],
        "Editable Outputs": ["SVG Export", "PPTX Export", "PNG Preview"],
    }
    context = _context_block(concepts, flows, groups)
    caption = "Scene graph reconstruction and multi-format export for scientific figure post-editing."
    edit_goal = "修复分组系统图中的对齐问题、文本问题和导出分支排版。"
    return BenchmarkTemplate(f"grouped_system_{idx:03d}", "grouped_system", scene, context, caption, edit_goal)


def _cross_panel_pipeline(idx: int) -> BenchmarkTemplate:
    width = 1700
    height = 960
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100),
        _node("n1", "Panel A Draft", 120, 220),
        _node("n2", "Panel A Scene", 420, 220),
        _node("n3", "Shared Anchor Pool", 760, 390, 240, 110),
        _node("n4", "Panel B Draft", 120, 620),
        _node("n5", "Panel B Scene", 420, 620),
        _node("n6", "Cross Panel Planner", 1080, 250),
        _node("n7", "Consistency Critic", 1080, 550),
        _node("n8", "Submission Figure", 1390, 390),
    ]
    scene.edges = [
        _edge("n1", "n2"),
        _edge("n4", "n5"),
        _edge("n2", "n3"),
        _edge("n5", "n3"),
        _edge("n3", "n6"),
        _edge("n3", "n7"),
        _edge("n6", "n8"),
        _edge("n7", "n8"),
    ]
    scene.groups = [
        SceneGroup(id="g1", label="Panel A", bbox=BBox(70, 150, 640, 250), node_ids=["n1", "n2"]),
        SceneGroup(id="g2", label="Panel B", bbox=BBox(70, 550, 640, 250), node_ids=["n4", "n5"]),
        SceneGroup(id="g3", label="Fusion", bbox=BBox(710, 180, 900, 520), node_ids=["n3", "n6", "n7", "n8"]),
    ]
    concepts = [node.label for node in scene.nodes if node.kind != "background"]
    flows = [
        "Panel A Draft -> Panel A Scene -> Shared Anchor Pool -> Cross Panel Planner -> Submission Figure",
        "Panel B Draft -> Panel B Scene -> Shared Anchor Pool -> Consistency Critic -> Submission Figure",
    ]
    groups = {
        "Panel A": ["Panel A Draft", "Panel A Scene"],
        "Panel B": ["Panel B Draft", "Panel B Scene"],
        "Fusion": ["Shared Anchor Pool", "Cross Panel Planner", "Consistency Critic", "Submission Figure"],
    }
    context = _context_block(concepts, flows, groups)
    caption = "Cross-panel layout planning and consistency correction for multi-panel scientific figures."
    edit_goal = "修复跨 panel 方法图中的共享锚点布局和汇合连线。"
    return BenchmarkTemplate(f"cross_panel_pipeline_{idx:03d}", "cross_panel_pipeline", scene, context, caption, edit_goal)


def _shared_backbone_multihead(idx: int) -> BenchmarkTemplate:
    width = 1700
    height = 900
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100),
        _node("n1", "Draft Inputs", 90, 380),
        _node("n2", "Shared Backbone", 400, 380, 260, 100),
        _node("n3", "Semantic Head", 820, 180),
        _node("n4", "Layout Head", 820, 380),
        _node("n5", "Editability Head", 820, 580),
        _node("n6", "Structure Decoder", 1160, 180),
        _node("n7", "Retouch Decoder", 1160, 380),
        _node("n8", "Export Decoder", 1160, 580),
        _node("n9", "Final Figure", 1450, 380, 180, 120),
    ]
    scene.edges = [
        _edge("n1", "n2"),
        _edge("n2", "n3"),
        _edge("n2", "n4"),
        _edge("n2", "n5"),
        _edge("n3", "n6"),
        _edge("n4", "n7"),
        _edge("n5", "n8"),
        _edge("n6", "n9"),
        _edge("n7", "n9"),
        _edge("n8", "n9"),
    ]
    scene.groups = [
        SceneGroup(id="g1", label="Encoders", bbox=BBox(60, 320, 650, 220), node_ids=["n1", "n2"]),
        SceneGroup(id="g2", label="Prediction Heads", bbox=BBox(760, 120, 430, 620), node_ids=["n3", "n4", "n5", "n6", "n7", "n8"]),
        SceneGroup(id="g3", label="Merged Output", bbox=BBox(1400, 300, 240, 260), node_ids=["n9"]),
    ]
    concepts = [node.label for node in scene.nodes if node.kind != "background"]
    flows = [
        "Draft Inputs -> Shared Backbone -> Semantic Head -> Structure Decoder -> Final Figure",
        "Shared Backbone -> Layout Head -> Retouch Decoder -> Final Figure",
        "Shared Backbone -> Editability Head -> Export Decoder -> Final Figure",
    ]
    groups = {
        "Encoders": ["Draft Inputs", "Shared Backbone"],
        "Prediction Heads": ["Semantic Head", "Layout Head", "Editability Head", "Structure Decoder", "Retouch Decoder", "Export Decoder"],
        "Merged Output": ["Final Figure"],
    }
    context = _context_block(concepts, flows, groups)
    caption = "A shared-backbone multi-head figure editing architecture with structure, retouching, and export heads."
    edit_goal = "修复共享 backbone 多头结构图的扇出、扇入和整体留白。"
    return BenchmarkTemplate(f"shared_backbone_multihead_{idx:03d}", "shared_backbone_multihead", scene, context, caption, edit_goal)


def _ablation_comparison_pipeline(idx: int) -> BenchmarkTemplate:
    width = 1720
    height = 980
    scene = FigureSceneGraph(width=width, height=height)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, width, height), z_index=-100),
        _node("n1", "Draft Figure", 100, 410),
        _node("n2", "w/o Semantic Anchors", 450, 170),
        _node("n3", "w/o Layout Anchors", 450, 410),
        _node("n4", "w/o Critic Loop", 450, 650),
        _node("n5", "Full AnchorFigure", 850, 410, 270, 110),
        _node("n6", "Metric Table", 1260, 220),
        _node("n7", "Failure Gallery", 1260, 600),
    ]
    scene.edges = [
        _edge("n1", "n2"),
        _edge("n1", "n3"),
        _edge("n1", "n4"),
        _edge("n2", "n5"),
        _edge("n3", "n5"),
        _edge("n4", "n5"),
        _edge("n5", "n6"),
        _edge("n5", "n7"),
    ]
    scene.groups = [
        SceneGroup(id="g1", label="Ablations", bbox=BBox(390, 100, 780, 690), node_ids=["n2", "n3", "n4", "n5"]),
        SceneGroup(id="g2", label="Evaluation", bbox=BBox(1210, 150, 370, 620), node_ids=["n6", "n7"]),
    ]
    concepts = [node.label for node in scene.nodes if node.kind != "background"]
    flows = [
        "Draft Figure -> w/o Semantic Anchors -> Full AnchorFigure -> Metric Table",
        "Draft Figure -> w/o Layout Anchors -> Full AnchorFigure -> Failure Gallery",
        "Draft Figure -> w/o Critic Loop -> Full AnchorFigure",
    ]
    groups = {
        "Ablations": ["w/o Semantic Anchors", "w/o Layout Anchors", "w/o Critic Loop", "Full AnchorFigure"],
        "Evaluation": ["Metric Table", "Failure Gallery"],
    }
    context = _context_block(concepts, flows, groups)
    caption = "An ablation-driven comparison figure showing how full AnchorFigure aggregates improvements over disabled variants."
    edit_goal = "修复消融对比图的分支并列、汇合连接和评测区域布局。"
    return BenchmarkTemplate(f"ablation_comparison_pipeline_{idx:03d}", "ablation_comparison_pipeline", scene, context, caption, edit_goal)


def _families() -> list[TemplateFamily]:
    return [
        TemplateFamily("single_path_pipeline", "Single-Path Pipeline", _single_path_pipeline),
        TemplateFamily("multibranch_method", "Multi-Branch Method", _multibranch_method),
        TemplateFamily("grouped_system", "Grouped System", _grouped_system),
        TemplateFamily("cross_panel_pipeline", "Cross-Panel Pipeline", _cross_panel_pipeline),
        TemplateFamily("shared_backbone_multihead", "Shared Backbone Multihead", _shared_backbone_multihead),
        TemplateFamily("ablation_comparison_pipeline", "Ablation Comparison Pipeline", _ablation_comparison_pipeline),
    ]


def _corrupt_label(label: str, difficulty: str, rng) -> str:
    severity = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    words = label.split()
    if not words:
        return label
    for _ in range(severity):
        idx = rng.randrange(len(words))
        if len(words[idx]) > 4:
            words[idx] = words[idx][:-1]
    return " ".join(words)


def _degrade_scene(template: BenchmarkTemplate, difficulty: str) -> tuple[FigureSceneGraph, dict]:
    rng = deterministic_rng(f"{template.case_id}:{difficulty}")
    degraded = template.scene.clone()
    severity = {"easy": 0.35, "medium": 0.65, "hard": 1.0}[difficulty]
    ocr_candidates: dict[str, str] = {}
    degradation_tags: list[str] = []

    for node in degraded.nodes:
        if node.kind == "background":
            continue
        original_label = node.label
        dx = int(rng.randint(-90, 90) * severity)
        dy = int(rng.randint(-70, 70) * severity)
        dw = int(rng.randint(-50, 20) * severity)
        dh = int(rng.randint(-20, 10) * severity)
        node.bbox = BBox(
            x=max(20, node.bbox.x + dx),
            y=max(20, node.bbox.y + dy),
            width=max(150, node.bbox.width + dw),
            height=max(70, node.bbox.height + dh),
        )
        corrupted = _corrupt_label(node.label, difficulty, rng)
        node.metadata["ocr_text"] = corrupted
        ocr_candidates[node.id] = corrupted
        node.label = corrupted
        node.style["fill"] = "#F1F5F9" if rng.random() > 0.5 else "#E2E8F0"
        node.style["stroke"] = "#64748B"
        if corrupted != original_label:
            degradation_tags.append("text_blur")
        if dx or dy or dw or dh:
            degradation_tags.append("layout_perturbation")

    edge_error_count = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    valid_targets = [edge.target_id for edge in degraded.edges]
    for idx in range(min(edge_error_count, len(degraded.edges))):
        replacement = valid_targets[(idx + 1) % len(valid_targets)]
        degraded.edges[idx].target_id = replacement
    if degraded.edges:
        degradation_tags.append("arrow_miswire")

    for group in degraded.groups:
        group.bbox = BBox(
            group.bbox.x + int(20 * severity),
            group.bbox.y + int(20 * severity),
            max(160, group.bbox.width - int(60 * severity)),
            max(160, group.bbox.height - int(50 * severity)),
        )
        group.style["fill"] = "#F8FAFC"
        group.style["stroke"] = "#CBD5E1"
    if degraded.groups:
        degradation_tags.append("group_boundary_break")

    if difficulty in {"medium", "hard"}:
        degradation_tags.append("background_pollution")
        degraded.metadata["background_noise"] = difficulty
    if difficulty == "hard":
        degradation_tags.extend(["layout_perturbation", "color_imbalance"])

    manifest = {
        "paper_concepts": parse_context_concepts(template.paper_context),
        "ocr_candidates": ocr_candidates,
        "draft_scene_graph": degraded.to_dict(),
    }
    degradation_meta = {
        "family": template.family,
        "difficulty": difficulty,
        "severity": severity,
        "degradation_tags": sorted(set(degradation_tags)),
        "target_case_id": template.case_id,
    }
    return degraded, {"manifest": manifest, "degradation_meta": degradation_meta}


class BenchmarkBuilder:
    def __init__(self, config: BenchmarkBuildConfig | None = None) -> None:
        self.config = config or BenchmarkBuildConfig()
        self._families = _families()
        self._decomposer = HierarchicalDecomposer()
        self._hierarchy_config = PipelineConfig(
            use_hierarchy_decomposition=True,
            use_hierarchical_team=True,
            use_hierarchy_metrics=True,
        )

    def templates(self) -> list[TemplateFamily]:
        return self._families

    def build(self, output_dir: str | Path) -> list[Path]:
        output_dir = ensure_dir(output_dir)
        cases_dir = ensure_dir(output_dir / self.config.synthetic_split_name)
        case_dirs: list[Path] = []
        summary_cases: list[dict] = []

        for family in self._families:
            for idx in range(self.config.case_count_per_family):
                template = family.generator(idx)
                for difficulty in self.config.difficulty_levels:
                    case_id = f"{template.case_id}_{difficulty}"
                    case_dir = ensure_dir(cases_dir / case_id)
                    case_dirs.append(case_dir)

                    degraded_scene, payload = _degrade_scene(template, difficulty)
                    target_scene = self._decomposer.decompose(template.scene.clone(), self._hierarchy_config)
                    draft_hierarchy_seed = self._decomposer.decompose(degraded_scene.clone(), self._hierarchy_config)
                    target_scene_path = target_scene.save(case_dir / "target_scene.json")
                    render_scene_to_png(target_scene.clone(), case_dir / "target.png")
                    render_scene_to_png(degraded_scene.clone(), case_dir / "draft.png")

                    payload["manifest"]["draft_scene_graph"] = draft_hierarchy_seed.to_dict()
                    save_json(case_dir / "draft_manifest.json", payload["manifest"])
                    degradation_meta = payload["degradation_meta"]
                    severity_tags = set(degradation_meta["degradation_tags"])
                    degradation_meta["global_level_tags"] = sorted(tag for tag in severity_tags if tag in {"layout_perturbation", "color_imbalance"})
                    degradation_meta["module_level_tags"] = sorted(tag for tag in severity_tags if tag in {"arrow_miswire"})
                    degradation_meta["region_level_tags"] = sorted(tag for tag in severity_tags if tag in {"group_boundary_break", "background_pollution"})
                    degradation_meta["block_level_tags"] = sorted(tag for tag in severity_tags if tag in {"arrow_miswire", "layout_perturbation"})
                    degradation_meta["element_level_tags"] = sorted(tag for tag in severity_tags if tag in {"text_blur", "arrow_miswire"})
                    save_json(case_dir / "degradation_meta.json", degradation_meta)
                    save_json(case_dir / "target_hierarchy.json", {
                        "hierarchy_root_id": target_scene.hierarchy_root_id,
                        "hierarchy_units": [unit.to_dict() for unit in target_scene.hierarchy_units],
                        "hierarchy_relations": [relation.__dict__ for relation in target_scene.hierarchy_relations],
                        "hierarchy_state": target_scene.hierarchy_state.__dict__ if target_scene.hierarchy_state else None,
                        "decomposition_report": target_scene.decomposition_report,
                    })
                    save_json(case_dir / "draft_hierarchy_seed.json", {
                        "hierarchy_root_id": draft_hierarchy_seed.hierarchy_root_id,
                        "hierarchy_units": [unit.to_dict() for unit in draft_hierarchy_seed.hierarchy_units],
                        "hierarchy_relations": [relation.__dict__ for relation in draft_hierarchy_seed.hierarchy_relations],
                        "hierarchy_state": draft_hierarchy_seed.hierarchy_state.__dict__ if draft_hierarchy_seed.hierarchy_state else None,
                        "decomposition_report": draft_hierarchy_seed.decomposition_report,
                    })
                    (case_dir / "paper_context.txt").write_text(template.paper_context, encoding="utf-8")
                    (case_dir / "caption.txt").write_text(template.caption, encoding="utf-8")
                    (case_dir / "edit_goal.txt").write_text(template.edit_goal, encoding="utf-8")
                    case_meta = {
                        "benchmark_name": self.config.benchmark_name,
                        "case_id": case_id,
                        "family": family.family,
                        "difficulty": difficulty,
                        "target_case_id": template.case_id,
                        "target_scene_json": str(target_scene_path),
                        "draft_image": str(case_dir / "draft.png"),
                        "draft_manifest": str(case_dir / "draft_manifest.json"),
                    }
                    save_json(case_dir / "case_meta.json", case_meta)
                    summary_cases.append(case_meta)

        save_json(
            output_dir / "benchmark_manifest.json",
            {
                "benchmark_name": self.config.benchmark_name,
                "case_count_per_family": self.config.case_count_per_family,
                "families": [family.family for family in self._families],
                "difficulty_levels": list(self.config.difficulty_levels),
                "total_cases": len(summary_cases),
                "synthetic_split": self.config.synthetic_split_name,
                "real_split": self.config.real_split_name,
                "cases": summary_cases,
            },
        )
        self._build_real_data_scaffold(output_dir / self.config.real_split_name)
        return case_dirs

    def _build_real_data_scaffold(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        readme = """# Real Validation Scaffold

这个目录用于存放真实科研图 draft-target pair。

推荐结构：

- `cases/<case_id>/draft.png`
- `cases/<case_id>/target.png`
- `cases/<case_id>/target_scene.json`
- `cases/<case_id>/paper_context.txt`
- `cases/<case_id>/caption.txt`
- `cases/<case_id>/edit_goal.txt`
- `cases/<case_id>/case_meta.json`

当前仓库只提供脚手架，不包含真实数据本体。
"""
        (Path(output_dir) / "README.md").write_text(readme, encoding="utf-8")
        csv_template = "\n".join(
            [
                "case_id,paper_id,permission_status,draft_path,target_path,notes",
                "real_case_001,paper_x,pending,,,",
                "real_case_002,paper_y,pending,,,",
            ]
        )
        (Path(output_dir) / "real_case_manifest_template.csv").write_text(csv_template, encoding="utf-8")
