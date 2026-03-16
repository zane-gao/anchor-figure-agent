from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exporters import render_scene_to_png
from .models import BBox, FigureSceneGraph, SceneEdge, SceneGroup, SceneNode, make_id
from .utils import deterministic_rng, ensure_dir, save_json


@dataclass
class BenchmarkTemplate:
    case_id: str
    scene: FigureSceneGraph
    paper_context: str
    caption: str
    edit_goal: str


def _node(node_id: str, label: str, x: int, y: int, w: int = 220, h: int = 90, kind: str = "container") -> SceneNode:
    return SceneNode(id=node_id, kind=kind, label=label, bbox=BBox(x, y, w, h))


def _edge(source_id: str, target_id: str) -> SceneEdge:
    return SceneEdge(id=make_id("edge"), source_id=source_id, target_id=target_id)


def _build_multibranch_method() -> BenchmarkTemplate:
    scene = FigureSceneGraph(width=1600, height=900)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, 1600, 900), z_index=-100),
        _node("n1", "Draft Figure", 120, 360),
        _node("n2", "Raster to Scene", 410, 360),
        _node("n3", "Verification Layer", 700, 220),
        _node("n4", "Anchor Builder", 700, 500),
        _node("n5", "Layout Planner", 990, 220),
        _node("n6", "Retouch Executor", 990, 500),
        _node("n7", "Critic Stopper", 1280, 360),
        _node("n8", "Export Heads", 1280, 650),
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
    context = "\n".join(
        [
            "Concept inventory:",
            "- Draft Figure",
            "- Raster to Scene",
            "- Verification Layer",
            "- Anchor Builder",
            "- Layout Planner",
            "- Retouch Executor",
            "- Critic Stopper",
            "- Export Heads",
            "Expected flow:",
            "Draft Figure -> Raster to Scene -> Verification Layer -> Layout Planner -> Critic Stopper -> Export Heads",
            "Raster to Scene -> Anchor Builder -> Retouch Executor -> Critic Stopper",
            "Group summary:",
            "Perception: Draft Figure, Raster to Scene",
            "Structural Editing: Verification Layer, Anchor Builder, Layout Planner, Retouch Executor",
            "Output: Critic Stopper, Export Heads",
        ]
    )
    caption = "A dual-anchor scientific figure editing loop that transforms a rough draft into editable exports."
    edit_goal = "修复布局错乱、箭头错误、分组边界和文本可读性，输出可编辑终稿。"
    return BenchmarkTemplate("multibranch_method", scene, context, caption, edit_goal)


def _build_grouped_system() -> BenchmarkTemplate:
    scene = FigureSceneGraph(width=1600, height=920)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, 1600, 920), z_index=-100),
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
    context = "\n".join(
        [
            "Concept inventory:",
            "- Paper Context",
            "- Draft PNG",
            "- Semantic Anchors",
            "- Layout Anchors",
            "- Constraint Solver",
            "- Style Normalizer",
            "- SVG Export",
            "- PPTX Export",
            "- PNG Preview",
            "Expected flow:",
            "Paper Context -> Semantic Anchors -> Constraint Solver -> Style Normalizer -> SVG Export",
            "Draft PNG -> Layout Anchors -> Constraint Solver -> Style Normalizer -> PPTX Export",
            "Style Normalizer -> PNG Preview",
            "Group summary:",
            "Inputs: Paper Context, Draft PNG",
            "Scene Graph Core: Semantic Anchors, Layout Anchors, Constraint Solver, Style Normalizer",
            "Editable Outputs: SVG Export, PPTX Export, PNG Preview",
        ]
    )
    caption = "Scene graph reconstruction and multi-format export for scientific figure post-editing."
    edit_goal = "让分组更清晰、保证多输出头对齐并修复文字与箭头。"
    return BenchmarkTemplate("grouped_system", scene, context, caption, edit_goal)


def _build_cross_panel_pipeline() -> BenchmarkTemplate:
    scene = FigureSceneGraph(width=1700, height=960)
    scene.nodes = [
        SceneNode(id="background", kind="background", label="", bbox=BBox(0, 0, 1700, 960), z_index=-100),
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
    context = "\n".join(
        [
            "Concept inventory:",
            "- Panel A Draft",
            "- Panel A Scene",
            "- Shared Anchor Pool",
            "- Panel B Draft",
            "- Panel B Scene",
            "- Cross Panel Planner",
            "- Consistency Critic",
            "- Submission Figure",
            "Expected flow:",
            "Panel A Draft -> Panel A Scene -> Shared Anchor Pool -> Cross Panel Planner -> Submission Figure",
            "Panel B Draft -> Panel B Scene -> Shared Anchor Pool -> Consistency Critic -> Submission Figure",
            "Group summary:",
            "Panel A: Panel A Draft, Panel A Scene",
            "Panel B: Panel B Draft, Panel B Scene",
            "Fusion: Shared Anchor Pool, Cross Panel Planner, Consistency Critic, Submission Figure",
        ]
    )
    caption = "Cross-panel layout planning and consistency correction for multi-panel scientific figures."
    edit_goal = "修复跨 panel 布局、共享锚点位置和终稿导出流程。"
    return BenchmarkTemplate("cross_panel_pipeline", scene, context, caption, edit_goal)


def _corrupt_label(label: str, rng) -> str:
    if len(label) <= 4:
        return label
    if " " in label:
        words = label.split()
        idx = rng.randrange(len(words))
        word = words[idx]
        if len(word) > 4:
            words[idx] = word[:-1]
        return " ".join(words)
    return label[:-1]


class BenchmarkBuilder:
    def templates(self) -> list[BenchmarkTemplate]:
        return [_build_multibranch_method(), _build_grouped_system(), _build_cross_panel_pipeline()]

    def _degrade_scene(self, template: BenchmarkTemplate) -> tuple[FigureSceneGraph, dict]:
        rng = deterministic_rng(template.case_id)
        degraded = template.scene.clone()
        ocr_candidates = {}
        for node in degraded.nodes:
            if node.kind == "background":
                continue
            dx = rng.randint(-90, 90)
            dy = rng.randint(-70, 70)
            dw = rng.randint(-50, 20)
            dh = rng.randint(-20, 10)
            node.bbox = BBox(
                x=max(20, node.bbox.x + dx),
                y=max(20, node.bbox.y + dy),
                width=max(150, node.bbox.width + dw),
                height=max(70, node.bbox.height + dh),
            )
            corrupted = _corrupt_label(node.label, rng)
            node.metadata["ocr_text"] = corrupted
            ocr_candidates[node.id] = corrupted
            node.label = corrupted
            node.style["fill"] = "#F1F5F9" if rng.random() > 0.5 else "#E2E8F0"
            node.style["stroke"] = "#64748B"

        if len(degraded.edges) >= 2:
            degraded.edges[0].target_id = degraded.edges[1].target_id
        for group in degraded.groups:
            group.bbox = BBox(group.bbox.x + 20, group.bbox.y + 20, max(160, group.bbox.width - 60), max(160, group.bbox.height - 50))
            group.style["fill"] = "#F8FAFC"
            group.style["stroke"] = "#CBD5E1"

        manifest = {
            "paper_concepts": parse_context_concepts(template.paper_context),
            "ocr_candidates": ocr_candidates,
            "draft_scene_graph": degraded.to_dict(),
        }
        return degraded, manifest

    def build(self, output_dir: str | Path) -> list[Path]:
        output_dir = ensure_dir(output_dir)
        case_dirs: list[Path] = []
        for template in self.templates():
            case_dir = ensure_dir(output_dir / template.case_id)
            case_dirs.append(case_dir)
            target_scene_path = template.scene.save(case_dir / "target_scene.json")
            render_scene_to_png(template.scene.clone(), case_dir / "target.png")
            degraded_scene, manifest = self._degrade_scene(template)
            render_scene_to_png(degraded_scene.clone(), case_dir / "draft.png")
            save_json(case_dir / "draft_manifest.json", manifest)
            (case_dir / "paper_context.txt").write_text(template.paper_context, encoding="utf-8")
            (case_dir / "caption.txt").write_text(template.caption, encoding="utf-8")
            (case_dir / "edit_goal.txt").write_text(template.edit_goal, encoding="utf-8")
            save_json(
                case_dir / "case_meta.json",
                {
                    "case_id": template.case_id,
                    "target_scene_json": str(target_scene_path),
                    "draft_image": str(case_dir / "draft.png"),
                    "draft_manifest": str(case_dir / "draft_manifest.json"),
                },
            )
        return case_dirs


def parse_context_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            concepts.append(line[2:].strip())
    return concepts
