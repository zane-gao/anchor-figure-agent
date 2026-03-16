from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from PIL import Image

from .exporters import render_scene_to_png, export_scene_to_pptx, export_scene_to_svg, reroute_edges
from .models import (
    AssetState,
    AtomicOperation,
    BBox,
    FigureSceneGraph,
    Issue,
    LayoutAnchor,
    LayoutConstraint,
    MemoryState,
    PipelineArtifacts,
    PipelineRequest,
    PlanningState,
    SceneEdge,
    SceneGroup,
    SceneNode,
    SemanticAnchor,
    VersionDag,
    VersionNode,
    make_id,
)
from .utils import bbox_union, clamp, ensure_dir, load_json, normalize_text, save_json, similarity


EDITABLE_KINDS = {"text", "container", "icon", "group", "background"}


@dataclass
class ContextKnowledge:
    concepts: list[str]
    flows: list[tuple[str, str]]
    groups: dict[str, list[str]]


@dataclass
class CritiqueResult:
    overall: float
    metrics: dict[str, float]
    issues: list[Issue]


def parse_context_knowledge(text: str) -> ContextKnowledge:
    concepts: list[str] = []
    flows: list[tuple[str, str]] = []
    groups: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            concepts.append(line[2:].strip())
            continue
        if "->" in line:
            parts = [segment.strip() for segment in line.split("->") if segment.strip()]
            for idx in range(1, len(parts)):
                flows.append((parts[idx - 1], parts[idx]))
            continue
        if ":" in line and "," in line:
            name, members = line.split(":", 1)
            name = name.strip()
            if name.lower() in {"concept inventory", "expected flow", "group summary"}:
                continue
            groups[name] = [member.strip() for member in members.split(",") if member.strip()]
    for left, right in flows:
        if left not in concepts:
            concepts.append(left)
        if right not in concepts:
            concepts.append(right)
    for members in groups.values():
        for member in members:
            if member not in concepts:
                concepts.append(member)
    return ContextKnowledge(concepts=concepts, flows=flows, groups=groups)


def _find_best_concept(label: str, concepts: list[str]) -> tuple[str, float]:
    current = label.strip()
    if not concepts:
        return current, 0.0
    scores = [(concept, similarity(current, concept)) for concept in concepts]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[0]


def _non_background_nodes(scene: FigureSceneGraph) -> list[SceneNode]:
    return [node for node in scene.nodes if node.kind != "background"]


def _ensure_background(scene: FigureSceneGraph) -> None:
    if any(node.kind == "background" for node in scene.nodes):
        return
    scene.nodes.insert(
        0,
        SceneNode(
            id="background",
            kind="background",
            label="",
            bbox=BBox(0, 0, scene.width, scene.height),
            z_index=-100,
        ),
    )


def _connected_components_scene(image_path: str) -> FigureSceneGraph:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image)
    mask = np.any(array < 245, axis=2)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[BBox] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if area >= 600:
                components.append(BBox(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    components.sort(key=lambda box: (box.y, box.x))
    scene = FigureSceneGraph(width=width, height=height)
    _ensure_background(scene)
    for idx, box in enumerate(components, start=1):
        scene.nodes.append(
            SceneNode(
                id=f"component_{idx}",
                kind="container",
                label=f"Component {idx}",
                bbox=box.inflate(4, 4),
                confidence=0.35,
                metadata={"ocr_text": "", "proposal_source": "connected_component"},
            )
        )
    return scene


def _build_semantic_anchors(scene: FigureSceneGraph, knowledge: ContextKnowledge) -> list[SemanticAnchor]:
    anchors: list[SemanticAnchor] = []
    node_by_label = defaultdict(list)
    for node in _non_background_nodes(scene):
        node_by_label[normalize_text(node.label)].append(node.id)
    for concept in knowledge.concepts:
        matched = node_by_label.get(normalize_text(concept), [])
        if not matched:
            matched = [
                node.id
                for node in _non_background_nodes(scene)
                if similarity(node.label, concept) >= 0.55
            ]
        if not matched:
            continue
        anchor = SemanticAnchor(
            id=make_id("sem"),
            name=concept,
            concept_key=normalize_text(concept),
            node_ids=matched,
            importance=1.0,
        )
        anchors.append(anchor)
        for node_id in matched:
            node = scene.node_map()[node_id]
            if anchor.id not in node.semantic_anchor_ids:
                node.semantic_anchor_ids.append(anchor.id)
    return anchors


def _topological_layers(scene: FigureSceneGraph) -> list[list[str]]:
    nodes = [node.id for node in _non_background_nodes(scene)]
    indegree = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in scene.edges:
        if edge.source_id in indegree and edge.target_id in indegree:
            indegree[edge.target_id] += 1
            outgoing[edge.source_id].append(edge.target_id)
    queue = deque([node_id for node_id, value in indegree.items() if value == 0])
    layers: list[list[str]] = []
    assigned: set[str] = set()
    while queue:
        current_layer: list[str] = []
        for _ in range(len(queue)):
            node_id = queue.popleft()
            if node_id in assigned:
                continue
            assigned.add(node_id)
            current_layer.append(node_id)
        if not current_layer:
            continue
        layers.append(current_layer)
        for node_id in current_layer:
            for child in outgoing[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
    remaining = [node_id for node_id in nodes if node_id not in assigned]
    for node_id in sorted(remaining):
        layers.append([node_id])
    return layers or [nodes]


def _build_layout_anchors_and_constraints(scene: FigureSceneGraph) -> tuple[list[LayoutAnchor], list[LayoutConstraint]]:
    anchors: list[LayoutAnchor] = []
    constraints: list[LayoutConstraint] = []
    for layer_idx, layer in enumerate(_topological_layers(scene)):
        if not layer:
            continue
        sample_node = scene.node_map()[layer[0]]
        anchors.append(
            LayoutAnchor(
                id=make_id("layout"),
                kind="column",
                node_ids=layer,
                direction="right",
                target={"x": sample_node.bbox.x, "layer": layer_idx},
            )
        )
        constraints.append(
            LayoutConstraint(
                id=make_id("constraint"),
                kind="align_vertical",
                subject_ids=layer,
                value={"axis": "x", "value": sample_node.bbox.x},
                priority=0.8,
            )
        )
    for edge in scene.edges:
        anchors.append(
            LayoutAnchor(
                id=make_id("layout"),
                kind="flow",
                node_ids=[edge.source_id, edge.target_id],
                direction="right",
                target={"edge_id": edge.id},
            )
        )
        constraints.append(
            LayoutConstraint(
                id=make_id("constraint"),
                kind="flow_order",
                subject_ids=[edge.source_id, edge.target_id],
                value={"direction": "right"},
                priority=1.0,
            )
        )
    return anchors, constraints


class RasterToSceneAgent:
    def propose(self, request: PipelineRequest) -> tuple[FigureSceneGraph, dict]:
        manifest_path = request.draft_manifest
        sidecar_path = Path(request.draft_image).with_suffix(".draft_manifest.json")
        payload = {}
        if manifest_path and Path(manifest_path).exists():
            payload = load_json(manifest_path)
        elif sidecar_path.exists():
            payload = load_json(sidecar_path)
        if payload.get("draft_scene_graph"):
            scene = FigureSceneGraph.from_dict(payload["draft_scene_graph"])
            scene.metadata["proposal_source"] = "manifest"
            return scene, payload
        scene = _connected_components_scene(request.draft_image)
        scene.metadata["proposal_source"] = "connected_component"
        return scene, payload


class ToolVerificationLayer:
    def verify(
        self,
        scene: FigureSceneGraph,
        request: PipelineRequest,
        manifest: dict,
    ) -> FigureSceneGraph:
        _ensure_background(scene)
        knowledge = parse_context_knowledge(request.paper_context)
        node_map = scene.node_map()
        concept_inventory = knowledge.concepts or manifest.get("paper_concepts", [])
        ocr_candidates = manifest.get("ocr_candidates", {})
        duplicate_counter: dict[str, int] = defaultdict(int)

        for node in _non_background_nodes(scene):
            candidate = ocr_candidates.get(node.id) or node.metadata.get("ocr_text") or node.label
            best_concept, score = _find_best_concept(candidate or node.label, concept_inventory)
            if score >= 0.34:
                node.label = best_concept
            else:
                node.label = candidate or node.label or "Untitled Stage"
            normalized = normalize_text(node.label)
            duplicate_counter[normalized] += 1
            if duplicate_counter[normalized] > 1:
                node.label = f"{node.label} {duplicate_counter[normalized]}"
            node.confidence = clamp(max(node.confidence, score), 0.2, 0.98)
            node.metadata["verified_text"] = node.label

        scene.semantic_anchors = _build_semantic_anchors(scene, knowledge)
        label_to_nodes = defaultdict(list)
        for node in _non_background_nodes(scene):
            label_to_nodes[normalize_text(node.label)].append(node.id)

        rebuilt_edges: list[SceneEdge] = []
        if knowledge.flows:
            for left, right in knowledge.flows:
                left_candidates = label_to_nodes.get(normalize_text(left), [])
                right_candidates = label_to_nodes.get(normalize_text(right), [])
                if not left_candidates or not right_candidates:
                    continue
                rebuilt_edges.append(
                    SceneEdge(
                        id=make_id("edge"),
                        source_id=left_candidates[0],
                        target_id=right_candidates[0],
                        metadata={"verified_from_context": True},
                        confidence=0.9,
                    )
                )
        else:
            rebuilt_edges = scene.edges
        scene.edges = rebuilt_edges

        groups: list[SceneGroup] = []
        if knowledge.groups:
            for name, members in knowledge.groups.items():
                member_ids: list[str] = []
                for member in members:
                    member_ids.extend(label_to_nodes.get(normalize_text(member), []))
                member_ids = list(dict.fromkeys(member_ids))
                if not member_ids:
                    continue
                union_box = bbox_union([node_map[node_id].bbox for node_id in member_ids]).inflate(32, 36)
                groups.append(
                    SceneGroup(
                        id=make_id("group"),
                        label=name,
                        bbox=union_box,
                        node_ids=member_ids,
                        confidence=0.9,
                    )
                )
        else:
            for group in scene.groups:
                valid_ids = [node_id for node_id in group.node_ids if node_id in node_map]
                if not valid_ids:
                    continue
                group.node_ids = valid_ids
                group.bbox = bbox_union([node_map[node_id].bbox for node_id in valid_ids]).inflate(28, 32)
                groups.append(group)
        scene.groups = groups
        scene.layout_anchors, scene.constraints = _build_layout_anchors_and_constraints(scene)
        reroute_edges(scene)
        return scene


class LayoutPlanner:
    def apply(self, scene: FigureSceneGraph, issues: list[Issue]) -> tuple[FigureSceneGraph, list[AtomicOperation]]:
        scene = scene.clone()
        node_map = scene.node_map()
        layers = _topological_layers(scene)
        movable_nodes = [node for node in _non_background_nodes(scene) if node.kind in {"container", "text", "icon"}]
        if not movable_nodes:
            return scene, []

        column_count = max(1, len(layers))
        left_margin = 140
        right_margin = 120
        top_margin = 130
        bottom_margin = 100
        usable_width = max(200, scene.width - left_margin - right_margin)
        usable_height = max(200, scene.height - top_margin - bottom_margin)
        column_gap = usable_width / max(1, column_count)
        operations: list[AtomicOperation] = []

        for layer_idx, layer in enumerate(layers):
            layer_nodes = [node_map[node_id] for node_id in layer if node_id in node_map]
            layer_nodes = [node for node in layer_nodes if node.kind != "background"]
            if not layer_nodes:
                continue
            layer_nodes.sort(key=lambda node: (node.metadata.get("group_priority", 0), node.bbox.y, node.bbox.x))
            row_gap = usable_height / max(1, len(layer_nodes))
            for row_idx, node in enumerate(layer_nodes):
                width = clamp(max(node.bbox.width, 210 + len(node.label) * 5), 190, 320)
                height = 92 if node.kind != "icon" else max(node.bbox.height, 84)
                x = left_margin + layer_idx * column_gap + (column_gap - width) / 2
                y = top_margin + row_idx * row_gap + max(0, (row_gap - height) / 2)
                if abs(node.bbox.x - x) > 2 or abs(node.bbox.y - y) > 2:
                    operations.append(
                        AtomicOperation(
                            id=make_id("op"),
                            op_type="move",
                            target_ids=[node.id],
                            params={"x": round(x, 2), "y": round(y, 2)},
                            rationale="按流程层级重排节点位置。",
                            status="done",
                        )
                    )
                if abs(node.bbox.width - width) > 2 or abs(node.bbox.height - height) > 2:
                    operations.append(
                        AtomicOperation(
                            id=make_id("op"),
                            op_type="resize",
                            target_ids=[node.id],
                            params={"width": round(width, 2), "height": round(height, 2)},
                            rationale="规范节点尺寸以提升可读性。",
                            status="done",
                        )
                    )
                node.bbox = BBox(x=x, y=y, width=width, height=height)

        for group in scene.groups:
            member_boxes = [node_map[node_id].bbox for node_id in group.node_ids if node_id in node_map]
            if not member_boxes:
                continue
            new_bbox = bbox_union(member_boxes).inflate(34, 40)
            if (
                abs(group.bbox.x - new_bbox.x) > 2
                or abs(group.bbox.y - new_bbox.y) > 2
                or abs(group.bbox.width - new_bbox.width) > 2
                or abs(group.bbox.height - new_bbox.height) > 2
            ):
                operations.append(
                    AtomicOperation(
                        id=make_id("op"),
                        op_type="regroup",
                        target_ids=[group.id],
                        params={"bbox": new_bbox.to_dict()},
                        rationale="扩大分组区域以包裹全部节点。",
                        status="done",
                    )
                )
            group.bbox = new_bbox

        reroute_edges(scene)
        for edge in scene.edges:
            operations.append(
                AtomicOperation(
                    id=make_id("op"),
                    op_type="reroute_arrow",
                    target_ids=[edge.id],
                    params={"points": edge.points},
                    rationale="根据重排后的节点位置重新布线。",
                    status="done",
                )
            )
        scene.layout_anchors, scene.constraints = _build_layout_anchors_and_constraints(scene)
        return scene, operations


class RetouchExecutor:
    def apply(
        self,
        scene: FigureSceneGraph,
        issues: list[Issue],
        request: PipelineRequest,
    ) -> tuple[FigureSceneGraph, list[AtomicOperation]]:
        scene = scene.clone()
        operations: list[AtomicOperation] = []
        group_palette = ["#DBEAFE", "#E0F2FE", "#ECFCCB", "#FCE7F3"]
        group_styles = {}
        for idx, group in enumerate(scene.groups):
            fill = group_palette[idx % len(group_palette)]
            group.style.update({"fill": fill, "stroke": "#94A3B8", "text": "#334155"})
            group_styles[group.id] = fill
            operations.append(
                AtomicOperation(
                    id=make_id("op"),
                    op_type="normalize_style",
                    target_ids=[group.id],
                    params={"fill": fill},
                    rationale="统一分组底色与描边。",
                    status="done",
                )
            )

        membership = {}
        for group in scene.groups:
            for node_id in group.node_ids:
                membership[node_id] = group.id

        indegree = defaultdict(int)
        outgoing = defaultdict(int)
        for edge in scene.edges:
            indegree[edge.target_id] += 1
            outgoing[edge.source_id] += 1

        for node in _non_background_nodes(scene):
            fill = "#FFFFFF"
            if node.id in membership:
                fill = group_styles.get(membership[node.id], "#FFFFFF")
            if indegree[node.id] == 0:
                fill = "#E0F2FE"
            if outgoing[node.id] == 0:
                fill = "#DCFCE7"
            if node.kind == "icon":
                fill = "#FDE68A"
            node.style.update({"fill": fill, "stroke": "#0F172A", "text": "#0F172A"})
            operations.append(
                AtomicOperation(
                    id=make_id("op"),
                    op_type="normalize_style",
                    target_ids=[node.id],
                    params=node.style,
                    rationale="统一节点颜色、描边和文本样式。",
                    status="done",
                )
            )
            verified_text = node.metadata.get("verified_text")
            if verified_text and node.label != verified_text:
                node.label = verified_text
                operations.append(
                    AtomicOperation(
                        id=make_id("op"),
                        op_type="replace_text",
                        target_ids=[node.id],
                        params={"text": verified_text},
                        rationale="使用验证后的文本替换模糊内容。",
                        status="done",
                    )
                )
            node.metadata["clean_background"] = True

        for issue in issues:
            if issue.category == "layout_overlap":
                operations.append(
                    AtomicOperation(
                        id=make_id("op"),
                        op_type="local_cleanup",
                        target_ids=issue.target_ids,
                        params={"strategy": "spacing_boost"},
                        rationale="在布局稳定后进行局部留白清理。",
                        status="done",
                    )
                )
            if issue.category == "icon_missing":
                operations.append(
                    AtomicOperation(
                        id=make_id("op"),
                        op_type="swap_icon",
                        target_ids=issue.target_ids,
                        params={"mode": "accent_placeholder"},
                        rationale="使用统一示意图标风格修复缺失元素。",
                        status="done",
                    )
                )
        return scene, operations


class CriticStopper:
    def assess(self, scene: FigureSceneGraph) -> CritiqueResult:
        nodes = _non_background_nodes(scene)
        overlaps = 0
        pairs = 0
        for idx, node in enumerate(nodes):
            for other in nodes[idx + 1 :]:
                pairs += 1
                if node.bbox.overlaps(other.bbox):
                    overlaps += 1
        overlap_score = 1.0 if pairs == 0 else 1.0 - overlaps / pairs
        text_score = 0.0 if not nodes else sum(1 for node in nodes if node.label.strip()) / len(nodes)
        editable_score = 0.0 if not nodes else sum(1 for node in nodes if node.kind in EDITABLE_KINDS) / len(nodes)

        flow_checks = 0
        flow_hits = 0
        node_map = scene.node_map()
        for edge in scene.edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            if not source or not target:
                continue
            flow_checks += 1
            if target.bbox.center_x >= source.bbox.center_x - 5 or target.bbox.center_y >= source.bbox.center_y - 5:
                flow_hits += 1
        flow_score = 0.0 if flow_checks == 0 else flow_hits / flow_checks

        group_checks = 0
        group_hits = 0
        for group in scene.groups:
            for node_id in group.node_ids:
                if node_id not in node_map:
                    continue
                group_checks += 1
                if group.bbox.contains(node_map[node_id].bbox):
                    group_hits += 1
        group_score = 1.0 if group_checks == 0 else group_hits / group_checks

        routing_checks = 0
        routing_hits = 0
        for edge in scene.edges:
            if len(edge.points) >= 2:
                routing_checks += 1
                if edge.source_port and edge.target_port:
                    routing_hits += 1
        routing_score = 1.0 if routing_checks == 0 else routing_hits / routing_checks

        overall = (
            overlap_score * 0.22
            + text_score * 0.2
            + editable_score * 0.18
            + flow_score * 0.22
            + group_score * 0.1
            + routing_score * 0.08
        )

        issues: list[Issue] = []
        if overlaps:
            issues.append(
                Issue(
                    id=make_id("issue"),
                    category="layout_overlap",
                    severity=min(1.0, overlaps / max(1, pairs)),
                    message="存在节点重叠，需要重排。",
                    target_ids=[node.id for node in nodes],
                    suggested_operations=["move", "resize"],
                )
            )
        missing_text_nodes = [node.id for node in nodes if not node.label.strip()]
        if missing_text_nodes:
            issues.append(
                Issue(
                    id=make_id("issue"),
                    category="text_missing",
                    severity=0.7,
                    message="有节点缺少清晰文本。",
                    target_ids=missing_text_nodes,
                    suggested_operations=["replace_text"],
                )
            )
        broken_groups = [
            group.id
            for group in scene.groups
            if any(node_id in node_map and not group.bbox.contains(node_map[node_id].bbox) for node_id in group.node_ids)
        ]
        if broken_groups:
            issues.append(
                Issue(
                    id=make_id("issue"),
                    category="group_boundary",
                    severity=0.6,
                    message="分组边界未覆盖全部成员。",
                    target_ids=broken_groups,
                    suggested_operations=["regroup"],
                )
            )
        routing_failures = [edge.id for edge in scene.edges if len(edge.points) < 2]
        if routing_failures:
            issues.append(
                Issue(
                    id=make_id("issue"),
                    category="edge_routing",
                    severity=0.65,
                    message="存在未正确布线的箭头。",
                    target_ids=routing_failures,
                    suggested_operations=["reroute_arrow"],
                )
            )
        return CritiqueResult(
            overall=round(overall, 4),
            metrics={
                "overlap_score": round(overlap_score, 4),
                "text_score": round(text_score, 4),
                "editable_score": round(editable_score, 4),
                "flow_score": round(flow_score, 4),
                "group_score": round(group_score, 4),
                "routing_score": round(routing_score, 4),
            },
            issues=issues,
        )

    def should_stop(
        self,
        critique: CritiqueResult,
        best_score: float,
        iteration: int,
        max_iterations: int,
        stagnation_rounds: int,
    ) -> tuple[bool, str | None]:
        if critique.overall >= 0.9 and not critique.issues:
            return True, "达到目标质量且无剩余问题。"
        if iteration >= max_iterations:
            return True, "达到最大迭代次数。"
        if stagnation_rounds >= 2:
            return True, "连续两轮无显著提升。"
        if best_score >= 0.88 and critique.overall >= best_score - 0.005:
            return True, "已达到可接受分数并进入平台期。"
        return False, None


class FigureEditPipeline:
    def __init__(self) -> None:
        self.proposer = RasterToSceneAgent()
        self.verifier = ToolVerificationLayer()
        self.layout_planner = LayoutPlanner()
        self.retoucher = RetouchExecutor()
        self.critic = CriticStopper()

    def _record_version(
        self,
        dag: VersionDag,
        scene: FigureSceneGraph,
        score: float,
        note: str,
        issues: list[Issue],
        operations: list[AtomicOperation],
        versions_dir: Path,
        parent_id: str | None,
    ) -> VersionNode:
        version_id = make_id("version")
        snapshot_path = versions_dir / f"{version_id}.png"
        render_scene_to_png(scene.clone(), snapshot_path)
        version = VersionNode(
            id=version_id,
            parent_id=parent_id,
            scene_graph=scene.clone(),
            score=score,
            note=note,
            issues=issues,
            operations=operations,
            snapshot_png=str(snapshot_path),
        )
        dag.add_version(version)
        return version

    def run(self, request: PipelineRequest) -> PipelineArtifacts:
        output_dir = ensure_dir(request.output_dir)
        versions_dir = ensure_dir(output_dir / "versions")
        memory = MemoryState(
            asset_state=AssetState(draft_image_path=request.draft_image),
            planning_state=PlanningState(
                edit_goal=request.edit_goal,
                paper_context=request.paper_context,
                caption=request.caption,
                target_score=0.88,
            ),
        )

        initial_scene, manifest = self.proposer.propose(request)
        verified_scene = self.verifier.verify(initial_scene, request, manifest)
        initial_critique = self.critic.assess(verified_scene)
        memory.planning_state.issue_backlog = initial_critique.issues
        memory.planning_state.best_score = initial_critique.overall
        current_version = self._record_version(
            dag=memory.asset_state.version_dag,
            scene=verified_scene,
            score=initial_critique.overall,
            note="verified_initial_scene",
            issues=initial_critique.issues,
            operations=[],
            versions_dir=versions_dir,
            parent_id=None,
        )

        current_critique = initial_critique
        stagnation_rounds = 0
        for iteration in range(1, request.max_iterations + 1):
            memory.execution_state.iteration = iteration
            stop, reason = self.critic.should_stop(
                critique=current_critique,
                best_score=memory.planning_state.best_score,
                iteration=iteration - 1,
                max_iterations=request.max_iterations,
                stagnation_rounds=stagnation_rounds,
            )
            if stop:
                memory.planning_state.stop_reason = reason
                break

            base_scene = memory.asset_state.version_dag.best().scene_graph.clone()
            planned_scene, plan_ops = self.layout_planner.apply(base_scene, current_critique.issues)
            refined_scene, exec_ops = self.retoucher.apply(planned_scene, current_critique.issues, request)
            critique = self.critic.assess(refined_scene)
            all_ops = plan_ops + exec_ops
            current_version = self._record_version(
                dag=memory.asset_state.version_dag,
                scene=refined_scene,
                score=critique.overall,
                note=f"iteration_{iteration}",
                issues=critique.issues,
                operations=all_ops,
                versions_dir=versions_dir,
                parent_id=current_version.id,
            )
            memory.execution_state.operations = all_ops
            memory.execution_state.notes.append(
                f"Iteration {iteration}: score={critique.overall:.4f}, issues={len(critique.issues)}"
            )
            improvement = critique.overall - memory.planning_state.best_score
            if improvement > 0.004:
                memory.planning_state.best_score = critique.overall
                current_critique = critique
                stagnation_rounds = 0
            else:
                current_critique = critique
                stagnation_rounds += 1

        final_version = memory.asset_state.version_dag.best()
        final_scene = final_version.scene_graph.clone()
        scene_graph_path = final_scene.save(output_dir / "scene_graph.json")
        revised_png_path = render_scene_to_png(final_scene.clone(), output_dir / "revised.png")
        editable_svg_path = export_scene_to_svg(final_scene.clone(), output_dir / "editable.svg")
        editable_pptx_path = export_scene_to_pptx(final_scene.clone(), output_dir / "editable.pptx")

        report_payload = {
            "edit_goal": request.edit_goal,
            "draft_image": request.draft_image,
            "caption": request.caption,
            "stop_reason": memory.planning_state.stop_reason or "达到最佳版本。",
            "best_score": final_version.score,
            "best_version_id": final_version.id,
            "final_metrics": self.critic.assess(final_scene).metrics,
            "version_history": [
                {
                    "id": version.id,
                    "parent_id": version.parent_id,
                    "score": version.score,
                    "note": version.note,
                    "snapshot_png": version.snapshot_png,
                    "issues": [asdict(issue) for issue in version.issues],
                    "operations": [asdict(operation) for operation in version.operations],
                }
                for version in memory.asset_state.version_dag.versions.values()
            ],
        }
        edit_report_path = save_json(output_dir / "edit_report.json", report_payload)
        memory.asset_state.scene_graph_path = str(scene_graph_path)
        memory_json_path = save_json(output_dir / "memory.json", asdict(memory))

        return PipelineArtifacts(
            revised_png=str(revised_png_path),
            scene_graph_json=str(scene_graph_path),
            editable_pptx=str(editable_pptx_path),
            editable_svg=str(editable_svg_path),
            edit_report_json=str(edit_report_path),
            memory_json=str(memory_json_path),
        )
