from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict, deque
import math

import numpy as np
from PIL import Image

from .exporters import render_scene_to_png, export_scene_to_pptx, export_scene_to_svg, reroute_edges
from .models import (
    AssetState,
    AtomicOperation,
    BBox,
    FigureSceneGraph,
    HierarchyRelation,
    HierarchyState,
    HierarchyUnit,
    Issue,
    LayoutAnchor,
    LayoutConstraint,
    MemoryState,
    PipelineArtifacts,
    PipelineConfig,
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
    node_map = scene.node_map()
    group_map = scene.group_map()
    for layer_idx, layer in enumerate(_topological_layers(scene)):
        if not layer:
            continue
        sample_node = scene.node_map()[layer[0]]
        anchor = LayoutAnchor(
            id=make_id("layout"),
            kind="column",
            node_ids=layer,
            direction="right",
            target={"x": sample_node.bbox.x, "layer": layer_idx},
        )
        anchors.append(anchor)
        for node_id in layer:
            if node_id in node_map and anchor.id not in node_map[node_id].layout_anchor_ids:
                node_map[node_id].layout_anchor_ids.append(anchor.id)
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
        anchor = LayoutAnchor(
            id=make_id("layout"),
            kind="flow",
            node_ids=[edge.source_id, edge.target_id],
            direction="right",
            target={"edge_id": edge.id},
        )
        anchors.append(anchor)
        for node_id in [edge.source_id, edge.target_id]:
            if node_id in node_map and anchor.id not in node_map[node_id].layout_anchor_ids:
                node_map[node_id].layout_anchor_ids.append(anchor.id)
        constraints.append(
            LayoutConstraint(
                id=make_id("constraint"),
                kind="flow_order",
                subject_ids=[edge.source_id, edge.target_id],
                value={"direction": "right"},
                priority=1.0,
            )
        )
    for group in scene.groups:
        for anchor in anchors:
            if any(node_id in group.node_ids for node_id in anchor.node_ids):
                if anchor.id not in group.layout_anchor_ids:
                    group.layout_anchor_ids.append(anchor.id)
    return anchors, constraints


def _bbox_from_points(points: list[tuple[float, float]]) -> BBox | None:
    if not points:
        return None
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    return BBox(min_x, min_y, max_x - min_x, max_y - min_y)


def _collect_unit_bbox(
    scene: FigureSceneGraph,
    node_ids: list[str],
    edge_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
) -> BBox | None:
    boxes: list[BBox] = []
    node_map = scene.node_map()
    edge_map = scene.edge_map()
    group_map = scene.group_map()
    for node_id in node_ids:
        if node_id in node_map:
            boxes.append(node_map[node_id].bbox)
    for group_id in group_ids or []:
        if group_id in group_map:
            boxes.append(group_map[group_id].bbox)
    for edge_id in edge_ids or []:
        if edge_id in edge_map:
            bbox = _bbox_from_points(edge_map[edge_id].points)
            if bbox is not None:
                boxes.append(bbox)
    return bbox_union(boxes).inflate(12, 12) if boxes else None


def _same_level_relation_type(left: HierarchyUnit, right: HierarchyUnit, edge_lookup: set[tuple[str, str]]) -> str | None:
    if left.id == right.id or left.bbox is None or right.bbox is None:
        return None
    if left.node_ids and right.node_ids:
        for source_id in left.node_ids:
            for target_id in right.node_ids:
                if (source_id, target_id) in edge_lookup or (target_id, source_id) in edge_lookup:
                    return "flows_to"
    if abs(left.bbox.center_x - right.bbox.center_x) < 28 or abs(left.bbox.center_y - right.bbox.center_y) < 28:
        return "aligned_with"
    horizontal_gap = max(0.0, max(left.bbox.x - right.bbox.right, right.bbox.x - left.bbox.right))
    vertical_gap = max(0.0, max(left.bbox.y - right.bbox.bottom, right.bbox.y - left.bbox.bottom))
    if horizontal_gap < 36 or vertical_gap < 36:
        return "adjacent_to"
    return None


def _connected_components(node_ids: list[str], edges: list[SceneEdge]) -> list[list[str]]:
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    node_set = set(node_ids)
    for edge in edges:
        if edge.source_id in node_set and edge.target_id in node_set:
            outgoing[edge.source_id].add(edge.target_id)
            outgoing[edge.target_id].add(edge.source_id)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        component: list[str] = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for nxt in outgoing[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(component))
    return components


def _partition_nodes_by_axis(scene: FigureSceneGraph, node_ids: list[str], parts: int = 2) -> list[list[str]]:
    node_map = scene.node_map()
    nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map]
    if len(nodes) <= 2 or parts <= 1:
        return [sorted(node_ids)]
    bbox = bbox_union([node.bbox for node in nodes])
    split_axis = "x" if bbox.width >= bbox.height else "y"
    ordered = sorted(nodes, key=lambda node: node.bbox.center_x if split_axis == "x" else node.bbox.center_y)
    bucket_size = max(1, math.ceil(len(ordered) / parts))
    buckets: list[list[str]] = []
    for start in range(0, len(ordered), bucket_size):
        buckets.append([node.id for node in ordered[start : start + bucket_size]])
    return buckets


class HierarchicalDecomposer:
    def decompose(self, scene: FigureSceneGraph, config: PipelineConfig) -> FigureSceneGraph:
        scene = scene.clone()
        if not config.use_hierarchy_decomposition:
            scene.hierarchy_units = []
            scene.hierarchy_relations = []
            scene.hierarchy_root_id = None
            scene.hierarchy_state = None
            scene.decomposition_report = {
                "enabled": False,
                "reason": "use_hierarchy_decomposition disabled",
            }
            return scene

        node_ids = [node.id for node in _non_background_nodes(scene)]
        edge_ids = [edge.id for edge in scene.edges]
        group_ids = [group.id for group in scene.groups]
        units: list[HierarchyUnit] = []
        relations: list[HierarchyRelation] = []
        level_index: dict[str, list[str]] = {level: [] for level in config.hierarchy_levels}

        root_id = make_id("hier")
        root_unit = HierarchyUnit(
            id=root_id,
            level="global",
            label="Global Figure",
            node_ids=node_ids,
            edge_ids=edge_ids,
            group_ids=group_ids,
            bbox=_collect_unit_bbox(scene, node_ids, edge_ids=edge_ids, group_ids=group_ids),
            metadata={"source": "scene_extent"},
        )
        units.append(root_unit)
        level_index["global"].append(root_id)

        module_units: list[HierarchyUnit] = []
        if scene.groups and config.hierarchy_from_groups_first:
            for group in scene.groups:
                unit = HierarchyUnit(
                    id=make_id("hier"),
                    level="module",
                    label=group.label or f"Module {len(module_units)+1}",
                    parent_id=root_id,
                    node_ids=list(group.node_ids),
                    group_ids=[group.id],
                    bbox=group.bbox.inflate(8, 8),
                    metadata={"source": "group"},
                )
                module_units.append(unit)
        else:
            layers = _topological_layers(scene) if scene.edges else _fallback_layers_without_anchors(scene)
            for idx, layer in enumerate(layers):
                unit = HierarchyUnit(
                    id=make_id("hier"),
                    level="module",
                    label=f"Module {idx+1}",
                    parent_id=root_id,
                    node_ids=list(layer),
                    bbox=_collect_unit_bbox(scene, layer),
                    metadata={"source": "topology_layer"},
                )
                module_units.append(unit)
        root_unit.child_ids = [unit.id for unit in module_units]
        units.extend(module_units)
        level_index["module"] = [unit.id for unit in module_units]

        region_units: list[HierarchyUnit] = []
        block_units: list[HierarchyUnit] = []
        element_units: list[HierarchyUnit] = []
        edge_lookup = {(edge.source_id, edge.target_id) for edge in scene.edges}
        node_map = scene.node_map()

        for module_idx, module_unit in enumerate(module_units):
            source_node_ids = module_unit.node_ids or node_ids
            partitions = _partition_nodes_by_axis(scene, source_node_ids, parts=2 if len(source_node_ids) > 3 else 1)
            module_unit.child_ids = []
            for region_idx, region_node_ids in enumerate(partitions):
                region_unit = HierarchyUnit(
                    id=make_id("hier"),
                    level="region",
                    label=f"{module_unit.label}-Region {region_idx+1}",
                    parent_id=module_unit.id,
                    node_ids=list(region_node_ids),
                    bbox=_collect_unit_bbox(scene, region_node_ids),
                    metadata={"source": "axis_partition"},
                )
                region_units.append(region_unit)
                module_unit.child_ids.append(region_unit.id)

                components = _connected_components(region_node_ids, scene.edges)
                region_unit.child_ids = []
                for block_idx, component in enumerate(components):
                    block_edge_ids = [
                        edge.id
                        for edge in scene.edges
                        if edge.source_id in component and edge.target_id in component
                    ]
                    block_unit = HierarchyUnit(
                        id=make_id("hier"),
                        level="block",
                        label=f"{region_unit.label}-Block {block_idx+1}",
                        parent_id=region_unit.id,
                        node_ids=list(component),
                        edge_ids=block_edge_ids,
                        bbox=_collect_unit_bbox(scene, component, edge_ids=block_edge_ids),
                        metadata={"source": "connected_component"},
                    )
                    block_units.append(block_unit)
                    region_unit.child_ids.append(block_unit.id)

                    block_unit.child_ids = []
                    for item_id in component:
                        node = node_map[item_id]
                        element_unit = HierarchyUnit(
                            id=make_id("hier"),
                            level="element",
                            label=node.label,
                            parent_id=block_unit.id,
                            node_ids=[node.id],
                            bbox=node.bbox,
                            metadata={"kind": node.kind, "source": "node"},
                        )
                        element_units.append(element_unit)
                        block_unit.child_ids.append(element_unit.id)
                    for block_edge_id in block_edge_ids:
                        edge = scene.edge_map()[block_edge_id]
                        edge_bbox = _bbox_from_points(edge.points) or _collect_unit_bbox(scene, [edge.source_id, edge.target_id])
                        element_unit = HierarchyUnit(
                            id=make_id("hier"),
                            level="element",
                            label=edge.label or edge.kind,
                            parent_id=block_unit.id,
                            edge_ids=[edge.id],
                            bbox=edge_bbox,
                            metadata={"kind": edge.kind, "source": "edge"},
                        )
                        element_units.append(element_unit)
                        block_unit.child_ids.append(element_unit.id)

        units.extend(region_units)
        units.extend(block_units)
        units.extend(element_units)
        level_index["region"] = [unit.id for unit in region_units]
        level_index["block"] = [unit.id for unit in block_units]
        level_index["element"] = [unit.id for unit in element_units]

        for unit in units:
            if unit.parent_id:
                relations.append(
                    HierarchyRelation(
                        id=make_id("hrel"),
                        relation_type="contains",
                        source_unit_id=unit.parent_id,
                        target_unit_id=unit.id,
                        level=unit.level,
                        metadata={"source": "parent_child"},
                    )
                )

        units_by_level: dict[str, list[HierarchyUnit]] = defaultdict(list)
        for unit in units:
            units_by_level[unit.level].append(unit)
        for level, siblings in units_by_level.items():
            for idx, left in enumerate(siblings):
                for right in siblings[idx + 1 :]:
                    if left.parent_id != right.parent_id:
                        continue
                    relation_type = _same_level_relation_type(left, right, edge_lookup)
                    if relation_type:
                        relations.append(
                            HierarchyRelation(
                                id=make_id("hrel"),
                                relation_type=relation_type,
                                source_unit_id=left.id,
                                target_unit_id=right.id,
                                level=level,
                                metadata={"source": "same_level_heuristic"},
                            )
                        )

        module_lookup = {
            node_id: module_unit.id
            for module_unit in module_units
            for node_id in module_unit.node_ids
        }
        dependency_pairs = set()
        for edge in scene.edges:
            source_module = module_lookup.get(edge.source_id)
            target_module = module_lookup.get(edge.target_id)
            if source_module and target_module and source_module != target_module:
                dependency_pairs.add((source_module, target_module))
        for source_unit_id, target_unit_id in sorted(dependency_pairs):
            relations.append(
                HierarchyRelation(
                    id=make_id("hrel"),
                    relation_type="depends_on",
                    source_unit_id=source_unit_id,
                    target_unit_id=target_unit_id,
                    level="module",
                    metadata={"source": "cross_module_flow"},
                )
            )

        scene.hierarchy_units = units
        scene.hierarchy_relations = relations
        scene.hierarchy_root_id = root_id
        scene.hierarchy_state = HierarchyState(
            root_id=root_id,
            level_index=level_index,
            source="heuristic",
            confidence=0.78,
        )
        scene.decomposition_report = {
            "enabled": True,
            "levels": {level: len(level_index.get(level, [])) for level in config.hierarchy_levels},
            "module_source": "groups" if scene.groups and config.hierarchy_from_groups_first else "topology_layer",
            "notes": [],
        }
        return scene


def _unit_map_by_member(scene: FigureSceneGraph) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    node_to_unit: dict[str, str] = {}
    edge_to_unit: dict[str, str] = {}
    group_to_unit: dict[str, str] = {}
    for level in ["element", "block", "region", "module", "global"]:
        for unit in scene.hierarchy_units_by_level(level):
            for node_id in unit.node_ids:
                node_to_unit.setdefault(node_id, unit.id)
            for edge_id in unit.edge_ids:
                edge_to_unit.setdefault(edge_id, unit.id)
            for group_id in unit.group_ids:
                group_to_unit.setdefault(group_id, unit.id)
    return node_to_unit, edge_to_unit, group_to_unit


def _issue_hierarchy_level(scene: FigureSceneGraph, issue: Issue) -> tuple[str, list[str]]:
    if not scene.hierarchy_units:
        return "global", []
    unit_map = scene.hierarchy_unit_map()
    node_to_unit, edge_to_unit, group_to_unit = _unit_map_by_member(scene)
    matched_unit_ids: list[str] = []
    for target_id in issue.target_ids:
        if target_id in node_to_unit:
            matched_unit_ids.append(node_to_unit[target_id])
        elif target_id in edge_to_unit:
            matched_unit_ids.append(edge_to_unit[target_id])
        elif target_id in group_to_unit:
            matched_unit_ids.append(group_to_unit[target_id])
        elif target_id in unit_map:
            matched_unit_ids.append(target_id)
    matched_levels = [unit_map[unit_id].level for unit_id in matched_unit_ids if unit_id in unit_map]
    if issue.category in {"text_missing", "icon_missing"}:
        return "element", matched_unit_ids
    if issue.category in {"edge_routing"}:
        return "block", matched_unit_ids
    if issue.category in {"group_boundary"}:
        return "region", matched_unit_ids
    if issue.category in {"layout_overlap"}:
        if "module" in matched_levels or len(issue.target_ids) > 4:
            return "module", matched_unit_ids
        if "block" in matched_levels:
            return "block", matched_unit_ids
    if "global" in matched_levels:
        return "global", matched_unit_ids
    return (matched_levels[0], matched_unit_ids) if matched_levels else ("global", [])


def _issue_stats_by_level(issues: list[Issue]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for issue in issues:
        level = issue.metadata.get("hierarchy_level", "unknown")
        stats[level] += 1
    return dict(stats)


def _operation_stats_by_level(operations: list[AtomicOperation]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for operation in operations:
        level = str(operation.params.get("level", "unknown"))
        stats[level] += 1
    return dict(stats)


def _hierarchy_level_order(level: str) -> int:
    levels = {"global": 0, "module": 1, "region": 2, "block": 3, "element": 4}
    return levels.get(level, 99)


def _unit_bbox_or_collect(scene: FigureSceneGraph, unit: HierarchyUnit) -> BBox | None:
    return unit.bbox or _collect_unit_bbox(scene, unit.node_ids, edge_ids=unit.edge_ids, group_ids=unit.group_ids)


def _shift_nodes(scene: FigureSceneGraph, node_ids: list[str], dx: float, dy: float) -> None:
    node_map = scene.node_map()
    for node_id in node_ids:
        if node_id not in node_map:
            continue
        node = node_map[node_id]
        node.bbox = node.bbox.move_to(node.bbox.x + dx, node.bbox.y + dy)


def _refresh_group_bboxes(scene: FigureSceneGraph) -> None:
    node_map = scene.node_map()
    for group in scene.groups:
        member_boxes = [node_map[node_id].bbox for node_id in group.node_ids if node_id in node_map]
        if member_boxes:
            group.bbox = bbox_union(member_boxes).inflate(28, 32)


def _arrange_units(scene: FigureSceneGraph, units: list[HierarchyUnit], container_bbox: BBox, axis: str) -> list[AtomicOperation]:
    if not units:
        return []
    operations: list[AtomicOperation] = []
    units = [unit for unit in units if _unit_bbox_or_collect(scene, unit) is not None]
    if not units:
        return operations
    units.sort(key=lambda unit: (_unit_bbox_or_collect(scene, unit).x, _unit_bbox_or_collect(scene, unit).y))
    gap = 24.0
    if axis == "x":
        total_width = sum((_unit_bbox_or_collect(scene, unit).width for unit in units)) + gap * max(0, len(units) - 1)
        cursor = container_bbox.x + max(0.0, (container_bbox.width - total_width) / 2.0)
        center_y = container_bbox.center_y
        for unit in units:
            bbox = _unit_bbox_or_collect(scene, unit)
            target_x = cursor
            target_y = center_y - bbox.height / 2.0
            dx = target_x - bbox.x
            dy = target_y - bbox.y
            if abs(dx) > 1 or abs(dy) > 1:
                _shift_nodes(scene, unit.node_ids, dx, dy)
                operations.append(
                    AtomicOperation(
                        id=make_id("hop"),
                        op_type="move",
                        target_ids=[unit.id],
                        params={"dx": round(dx, 2), "dy": round(dy, 2), "level": unit.level},
                        rationale=f"在 {unit.level} 层级上按 {axis} 方向重排。",
                        status="done",
                    )
                )
            cursor += bbox.width + gap
    else:
        total_height = sum((_unit_bbox_or_collect(scene, unit).height for unit in units)) + gap * max(0, len(units) - 1)
        cursor = container_bbox.y + max(0.0, (container_bbox.height - total_height) / 2.0)
        center_x = container_bbox.center_x
        for unit in units:
            bbox = _unit_bbox_or_collect(scene, unit)
            target_x = center_x - bbox.width / 2.0
            target_y = cursor
            dx = target_x - bbox.x
            dy = target_y - bbox.y
            if abs(dx) > 1 or abs(dy) > 1:
                _shift_nodes(scene, unit.node_ids, dx, dy)
                operations.append(
                    AtomicOperation(
                        id=make_id("hop"),
                        op_type="move",
                        target_ids=[unit.id],
                        params={"dx": round(dx, 2), "dy": round(dy, 2), "level": unit.level},
                        rationale=f"在 {unit.level} 层级上按 {axis} 方向重排。",
                        status="done",
                    )
                )
            cursor += bbox.height + gap
    _refresh_group_bboxes(scene)
    return operations


def _spread_block(scene: FigureSceneGraph, unit: HierarchyUnit) -> list[AtomicOperation]:
    bbox = _unit_bbox_or_collect(scene, unit)
    if bbox is None or len(unit.node_ids) <= 1:
        return []
    node_map = scene.node_map()
    nodes = [node_map[node_id] for node_id in unit.node_ids if node_id in node_map]
    if not nodes:
        return []
    columns = 2 if len(nodes) > 2 else 1
    rows = math.ceil(len(nodes) / columns)
    slot_w = max(160.0, bbox.width / max(1, columns))
    slot_h = max(84.0, bbox.height / max(1, rows))
    operations: list[AtomicOperation] = []
    for idx, node in enumerate(nodes):
        row = idx // columns
        col = idx % columns
        target_x = bbox.x + col * slot_w + (slot_w - node.bbox.width) / 2.0
        target_y = bbox.y + row * slot_h + (slot_h - node.bbox.height) / 2.0
        dx = target_x - node.bbox.x
        dy = target_y - node.bbox.y
        if abs(dx) > 1 or abs(dy) > 1:
            node.bbox = node.bbox.move_to(target_x, target_y)
            operations.append(
                AtomicOperation(
                    id=make_id("hop"),
                    op_type="move",
                    target_ids=[node.id],
                    params={"x": round(target_x, 2), "y": round(target_y, 2), "level": "block"},
                    rationale="在 block 层级内拉开局部节点以避免重叠。",
                    status="done",
                )
            )
    _refresh_group_bboxes(scene)
    return operations


class HierarchicalCoordinator:
    def __init__(self, decomposer: HierarchicalDecomposer) -> None:
        self.decomposer = decomposer

    def coordinate(
        self,
        scene: FigureSceneGraph,
        issues: list[Issue],
        request: PipelineRequest,
        config: PipelineConfig,
        layout_planner: "LayoutPlanner",
        retoucher: "RetouchExecutor",
    ) -> tuple[FigureSceneGraph, list[AtomicOperation], dict[str, Any]]:
        scene = scene.clone()
        if not config.use_hierarchical_team or not scene.hierarchy_units:
            planned_scene, plan_ops = layout_planner.apply(scene, issues, config)
            refined_scene, retouch_ops = retoucher.apply(planned_scene, issues, request, config)
            return refined_scene, plan_ops + retouch_ops, {
                "hierarchy_issue_stats": _issue_stats_by_level(issues),
                "hierarchy_operation_stats": {"global": len(plan_ops), "element": len(retouch_ops)},
            }

        routed: dict[str, list[Issue]] = defaultdict(list)
        scene_unit_map = scene.hierarchy_unit_map()
        for issue in issues:
            level, unit_ids = _issue_hierarchy_level(scene, issue)
            issue.metadata["hierarchy_level"] = level
            issue.metadata["hierarchy_unit_ids"] = unit_ids
            routed[level].append(issue)

        all_ops: list[AtomicOperation] = []
        operation_stats: dict[str, int] = defaultdict(int)

        global_scene, global_ops = layout_planner.apply(scene, routed.get("global", []) + routed.get("module", []), config)
        all_ops.extend(global_ops)
        operation_stats["global"] += len(global_ops)
        scene = self.decomposer.decompose(global_scene, config)

        unit_map = scene.hierarchy_unit_map()
        root_unit = unit_map.get(scene.hierarchy_root_id) if scene.hierarchy_root_id else None
        module_units = [unit_map[unit_id] for unit_id in (root_unit.child_ids if root_unit else []) if unit_id in unit_map]
        if root_unit and root_unit.bbox:
            module_ops = _arrange_units(scene, module_units, root_unit.bbox.inflate(-24, -24), axis="x")
            all_ops.extend(module_ops)
            operation_stats["module"] += len(module_ops)
            scene = self.decomposer.decompose(scene, config)
            unit_map = scene.hierarchy_unit_map()

        for module_unit in scene.hierarchy_units_by_level("module"):
            region_units = [unit_map[child_id] for child_id in module_unit.child_ids if child_id in unit_map and unit_map[child_id].level == "region"]
            module_bbox = _unit_bbox_or_collect(scene, module_unit)
            if module_bbox and region_units:
                region_ops = _arrange_units(scene, region_units, module_bbox.inflate(-18, -18), axis="y")
                all_ops.extend(region_ops)
                operation_stats["region"] += len(region_ops)
        scene = self.decomposer.decompose(scene, config)
        unit_map = scene.hierarchy_unit_map()

        for region_unit in scene.hierarchy_units_by_level("region"):
            block_units = [unit_map[child_id] for child_id in region_unit.child_ids if child_id in unit_map and unit_map[child_id].level == "block"]
            region_bbox = _unit_bbox_or_collect(scene, region_unit)
            if region_bbox and block_units:
                block_frame_ops = _arrange_units(scene, block_units, region_bbox.inflate(-16, -16), axis="y")
                all_ops.extend(block_frame_ops)
                operation_stats["block"] += len(block_frame_ops)
                for block_unit in block_units:
                    spread_ops = _spread_block(scene, block_unit)
                    all_ops.extend(spread_ops)
                    operation_stats["block"] += len(spread_ops)
        scene = self.decomposer.decompose(scene, config)

        element_targets = {
            target_id
            for issue in routed.get("element", [])
            for target_id in issue.target_ids
        }
        refined_scene, element_ops = retoucher.apply(scene, routed.get("element", []), request, config, target_scope=element_targets or None)
        all_ops.extend(element_ops)
        operation_stats["element"] += len(element_ops)
        reroute_edges(refined_scene)
        refined_scene = self.decomposer.decompose(refined_scene, config)

        return refined_scene, all_ops, {
            "hierarchy_issue_stats": _issue_stats_by_level(issues),
            "hierarchy_operation_stats": dict(operation_stats),
        }

def _fallback_layers_without_anchors(scene: FigureSceneGraph) -> list[list[str]]:
    nodes = sorted(_non_background_nodes(scene), key=lambda node: (node.bbox.x, node.bbox.y))
    if not nodes:
        return []
    if len(nodes) <= 2:
        return [[node.id] for node in nodes]
    min_x = min(node.bbox.x for node in nodes)
    max_x = max(node.bbox.x for node in nodes)
    span = max(1.0, max_x - min_x)
    bins: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        bucket = int(((node.bbox.x - min_x) / span) * 2.99)
        bins[bucket].append(node.id)
    return [bins[idx] for idx in sorted(bins) if bins[idx]]


def build_scene_from_context(
    paper_context: str,
    width: int = 1600,
    height: int = 900,
) -> FigureSceneGraph:
    knowledge = parse_context_knowledge(paper_context)
    scene = FigureSceneGraph(width=width, height=height)
    _ensure_background(scene)
    concepts = knowledge.concepts or ["Draft Figure", "Layout Planner", "Final Figure"]
    indegree = defaultdict(int)
    outgoing = defaultdict(list)
    for left, right in knowledge.flows:
        indegree[right] += 1
        outgoing[left].append(right)
        if left not in concepts:
            concepts.append(left)
        if right not in concepts:
            concepts.append(right)
    queue = deque([concept for concept in concepts if indegree[concept] == 0])
    layers: list[list[str]] = []
    assigned: set[str] = set()
    while queue:
        current_layer: list[str] = []
        for _ in range(len(queue)):
            concept = queue.popleft()
            if concept in assigned:
                continue
            assigned.add(concept)
            current_layer.append(concept)
        if current_layer:
            layers.append(current_layer)
        for concept in current_layer:
            for child in outgoing.get(concept, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
    for concept in concepts:
        if concept not in assigned:
            layers.append([concept])

    left_margin = 120
    top_margin = 120
    usable_width = width - 240
    usable_height = height - 220
    column_gap = usable_width / max(1, len(layers))
    concept_to_id: dict[str, str] = {}
    for layer_idx, layer in enumerate(layers):
        row_gap = usable_height / max(1, len(layer))
        for row_idx, concept in enumerate(layer):
            node_id = make_id("ctx")
            concept_to_id[concept] = node_id
            scene.nodes.append(
                SceneNode(
                    id=node_id,
                    kind="container",
                    label=concept,
                    bbox=BBox(
                        left_margin + layer_idx * column_gap,
                        top_margin + row_idx * row_gap,
                        240,
                        90,
                    ),
                )
            )
    for left, right in knowledge.flows:
        if left in concept_to_id and right in concept_to_id:
            scene.edges.append(SceneEdge(id=make_id("edge"), source_id=concept_to_id[left], target_id=concept_to_id[right]))
    for group_name, members in knowledge.groups.items():
        member_ids = [concept_to_id[member] for member in members if member in concept_to_id]
        if not member_ids:
            continue
        member_boxes = [scene.node_map()[node_id].bbox for node_id in member_ids]
        scene.groups.append(
            SceneGroup(
                id=make_id("group"),
                label=group_name,
                bbox=bbox_union(member_boxes).inflate(28, 32),
                node_ids=member_ids,
            )
        )
    scene.semantic_anchors = _build_semantic_anchors(scene, knowledge)
    scene.layout_anchors, scene.constraints = _build_layout_anchors_and_constraints(scene)
    reroute_edges(scene)
    return scene


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
        config: PipelineConfig,
    ) -> FigureSceneGraph:
        _ensure_background(scene)
        if not config.use_verifier:
            scene.semantic_anchors = []
            scene.layout_anchors = []
            scene.constraints = []
            reroute_edges(scene)
            return scene

        knowledge = parse_context_knowledge(request.paper_context)
        node_map = scene.node_map()
        concept_inventory = knowledge.concepts or manifest.get("paper_concepts", [])
        ocr_candidates = manifest.get("ocr_candidates", {})
        duplicate_counter: dict[str, int] = defaultdict(int)

        for node in _non_background_nodes(scene):
            candidate = ocr_candidates.get(node.id) or node.metadata.get("ocr_text") or node.label
            best_concept, score = _find_best_concept(candidate or node.label, concept_inventory)
            if config.use_context_matching and score >= 0.34:
                node.label = best_concept
            else:
                node.label = candidate or node.label or "Untitled Stage"
            normalized = normalize_text(node.label)
            duplicate_counter[normalized] += 1
            if duplicate_counter[normalized] > 1:
                node.label = f"{node.label} {duplicate_counter[normalized]}"
            node.confidence = clamp(max(node.confidence, score), 0.2, 0.98)
            node.metadata["verified_text"] = node.label

        scene.semantic_anchors = _build_semantic_anchors(scene, knowledge) if config.use_semantic_anchors else []
        label_to_nodes = defaultdict(list)
        for node in _non_background_nodes(scene):
            label_to_nodes[normalize_text(node.label)].append(node.id)

        rebuilt_edges: list[SceneEdge] = []
        if config.use_layout_anchors and knowledge.flows:
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
        if config.use_layout_anchors and knowledge.groups:
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
        if config.use_layout_anchors:
            scene.layout_anchors, scene.constraints = _build_layout_anchors_and_constraints(scene)
        else:
            scene.layout_anchors = []
            scene.constraints = []
        reroute_edges(scene)
        return scene


class LayoutPlanner:
    def apply(
        self,
        scene: FigureSceneGraph,
        issues: list[Issue],
        config: PipelineConfig,
    ) -> tuple[FigureSceneGraph, list[AtomicOperation]]:
        scene = scene.clone()
        if not config.use_layout_planner:
            reroute_edges(scene)
            return scene, []
        node_map = scene.node_map()
        layers = _topological_layers(scene) if config.use_layout_anchors and scene.layout_anchors else _fallback_layers_without_anchors(scene)
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
        if config.use_layout_anchors:
            scene.layout_anchors, scene.constraints = _build_layout_anchors_and_constraints(scene)
        else:
            scene.layout_anchors = []
            scene.constraints = []
        return scene, operations


class RetouchExecutor:
    def apply(
        self,
        scene: FigureSceneGraph,
        issues: list[Issue],
        request: PipelineRequest,
        config: PipelineConfig,
        target_scope: set[str] | None = None,
    ) -> tuple[FigureSceneGraph, list[AtomicOperation]]:
        scene = scene.clone()
        if not config.use_retoucher:
            return scene, []
        operations: list[AtomicOperation] = []
        group_palette = ["#DBEAFE", "#E0F2FE", "#ECFCCB", "#FCE7F3"]
        group_styles = {}
        for idx, group in enumerate(scene.groups):
            if target_scope is not None and group.id not in target_scope and not any(node_id in target_scope for node_id in group.node_ids):
                continue
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
            if target_scope is not None and node.id not in target_scope:
                continue
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
        if scene.hierarchy_units:
            for issue in issues:
                level, unit_ids = _issue_hierarchy_level(scene, issue)
                issue.metadata["hierarchy_level"] = level
                issue.metadata["hierarchy_unit_ids"] = unit_ids
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
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.proposer = RasterToSceneAgent()
        self.verifier = ToolVerificationLayer()
        self.decomposer = HierarchicalDecomposer()
        self.coordinator = HierarchicalCoordinator(self.decomposer)
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
        config = request.pipeline_config or self.config
        output_dir = ensure_dir(request.output_dir)
        versions_dir = ensure_dir(output_dir / "versions")
        max_iterations = config.max_iterations_override or request.max_iterations
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
        verified_scene = self.verifier.verify(initial_scene, request, manifest, config)
        verified_scene = self.decomposer.decompose(verified_scene, config)
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
        hierarchy_debug = {
            "hierarchy_issue_stats": _issue_stats_by_level(initial_critique.issues),
            "hierarchy_operation_stats": {},
        }
        iteration_budget = max_iterations if config.use_critic_loop else 1
        for iteration in range(1, iteration_budget + 1):
            memory.execution_state.iteration = iteration
            stop, reason = self.critic.should_stop(
                critique=current_critique,
                best_score=memory.planning_state.best_score,
                iteration=iteration - 1,
                max_iterations=iteration_budget,
                stagnation_rounds=stagnation_rounds,
            )
            if stop:
                memory.planning_state.stop_reason = reason
                break

            base_scene = memory.asset_state.version_dag.best().scene_graph.clone()
            planned_scene, all_ops, hierarchy_debug = self.coordinator.coordinate(
                base_scene,
                current_critique.issues,
                request,
                config,
                self.layout_planner,
                self.retoucher,
            )
            refined_scene = planned_scene
            critique = self.critic.assess(refined_scene)
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
        editable_svg_path = None
        editable_pptx_path = None
        if config.enforce_editable_export and config.export_svg:
            editable_svg_path = str(export_scene_to_svg(final_scene.clone(), output_dir / "editable.svg"))
        if config.enforce_editable_export and config.export_pptx:
            editable_pptx_path = str(export_scene_to_pptx(final_scene.clone(), output_dir / "editable.pptx"))

        report_payload = {
            "method_name": config.method_name,
            "pipeline_config": asdict(config),
            "edit_goal": request.edit_goal,
            "draft_image": request.draft_image,
            "caption": request.caption,
            "stop_reason": memory.planning_state.stop_reason or "达到最佳版本。",
            "best_score": final_version.score,
            "best_version_id": final_version.id,
            "final_metrics": self.critic.assess(final_scene).metrics,
            "hierarchy_issue_stats": hierarchy_debug.get("hierarchy_issue_stats", {}),
            "hierarchy_operation_stats": hierarchy_debug.get("hierarchy_operation_stats", {}),
            "decomposition_report": final_scene.decomposition_report,
            "version_history": [
                {
                    "id": version.id,
                    "parent_id": version.parent_id,
                    "score": version.score,
                    "note": version.note,
                    "snapshot_png": version.snapshot_png,
                    "issue_level_stats": _issue_stats_by_level(version.issues),
                    "operation_level_stats": _operation_stats_by_level(version.operations),
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
            editable_pptx=editable_pptx_path,
            editable_svg=editable_svg_path,
            edit_report_json=str(edit_report_path),
            memory_json=str(memory_json_path),
        )
