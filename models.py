from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import copy
import json
import uuid


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def inflate(self, dx: float, dy: float) -> "BBox":
        return BBox(
            x=self.x - dx,
            y=self.y - dy,
            width=self.width + 2 * dx,
            height=self.height + 2 * dy,
        )

    def move_to(self, x: float, y: float) -> "BBox":
        return BBox(x=x, y=y, width=self.width, height=self.height)

    def overlaps(self, other: "BBox") -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def contains(self, other: "BBox") -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


@dataclass
class StyleTokens:
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "background": "#F8FAFC",
            "panel": "#FFFFFF",
            "group_fill": "#EEF4FF",
            "group_stroke": "#94A3B8",
            "node_fill": "#FFFFFF",
            "node_stroke": "#1E293B",
            "accent": "#2563EB",
            "text": "#0F172A",
            "muted_text": "#475569",
        }
    )
    font_family: str = "Arial"
    font_size: int = 24
    title_size: int = 28
    stroke_width: int = 3
    corner_radius: int = 18
    arrow_width: int = 4
    shadow: bool = False


@dataclass
class SceneNode:
    id: str
    kind: str
    label: str
    bbox: BBox
    style: dict[str, Any] = field(default_factory=dict)
    z_index: int = 0
    asset_ref_id: str | None = None
    semantic_anchor_ids: list[str] = field(default_factory=list)
    layout_anchor_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneNode":
        payload = dict(data)
        payload["bbox"] = BBox.from_dict(data["bbox"])
        return cls(**payload)


@dataclass
class SceneEdge:
    id: str
    source_id: str
    target_id: str
    kind: str = "arrow"
    label: str = ""
    points: list[tuple[float, float]] = field(default_factory=list)
    source_port: str | None = None
    target_port: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    semantic_anchor_ids: list[str] = field(default_factory=list)
    layout_anchor_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneEdge":
        payload = dict(data)
        payload["points"] = [tuple(point) for point in data.get("points", [])]
        return cls(**payload)


@dataclass
class SceneGroup:
    id: str
    label: str
    bbox: BBox
    node_ids: list[str] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)
    semantic_anchor_ids: list[str] = field(default_factory=list)
    layout_anchor_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneGroup":
        payload = dict(data)
        payload["bbox"] = BBox.from_dict(data["bbox"])
        return cls(**payload)


@dataclass
class SemanticAnchor:
    id: str
    name: str
    concept_key: str
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutAnchor:
    id: str
    kind: str
    node_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    direction: str = "right"
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutConstraint:
    id: str
    kind: str
    subject_ids: list[str]
    value: dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetRef:
    id: str
    asset_type: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyUnit:
    id: str
    level: str
    label: str
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    bbox: BBox | None = None
    confidence: float = 1.0
    source: str = "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict() if self.bbox is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchyUnit":
        payload = dict(data)
        payload["bbox"] = BBox.from_dict(data["bbox"]) if data.get("bbox") else None
        return cls(**payload)


@dataclass
class HierarchyRelation:
    id: str
    relation_type: str
    source_unit_id: str
    target_unit_id: str
    level: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyState:
    root_id: str | None = None
    level_index: dict[str, list[str]] = field(default_factory=dict)
    source: str = "heuristic"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureSceneGraph:
    width: int
    height: int
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    groups: list[SceneGroup] = field(default_factory=list)
    semantic_anchors: list[SemanticAnchor] = field(default_factory=list)
    layout_anchors: list[LayoutAnchor] = field(default_factory=list)
    constraints: list[LayoutConstraint] = field(default_factory=list)
    style_tokens: StyleTokens = field(default_factory=StyleTokens)
    asset_refs: list[AssetRef] = field(default_factory=list)
    hierarchy_units: list[HierarchyUnit] = field(default_factory=list)
    hierarchy_relations: list[HierarchyRelation] = field(default_factory=list)
    hierarchy_root_id: str | None = None
    hierarchy_state: HierarchyState | None = None
    decomposition_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "FigureSceneGraph":
        return copy.deepcopy(self)

    def node_map(self) -> dict[str, SceneNode]:
        return {node.id: node for node in self.nodes}

    def group_map(self) -> dict[str, SceneGroup]:
        return {group.id: group for group in self.groups}

    def edge_map(self) -> dict[str, SceneEdge]:
        return {edge.id: edge for edge in self.edges}

    def hierarchy_unit_map(self) -> dict[str, HierarchyUnit]:
        return {unit.id: unit for unit in self.hierarchy_units}

    def hierarchy_units_by_level(self, level: str) -> list[HierarchyUnit]:
        return [unit for unit in self.hierarchy_units if unit.level == level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "groups": [group.to_dict() for group in self.groups],
            "semantic_anchors": [asdict(anchor) for anchor in self.semantic_anchors],
            "layout_anchors": [asdict(anchor) for anchor in self.layout_anchors],
            "constraints": [asdict(constraint) for constraint in self.constraints],
            "style_tokens": asdict(self.style_tokens),
            "asset_refs": [asdict(asset) for asset in self.asset_refs],
            "hierarchy_units": [unit.to_dict() for unit in self.hierarchy_units],
            "hierarchy_relations": [asdict(relation) for relation in self.hierarchy_relations],
            "hierarchy_root_id": self.hierarchy_root_id,
            "hierarchy_state": asdict(self.hierarchy_state) if self.hierarchy_state else None,
            "decomposition_report": self.decomposition_report,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FigureSceneGraph":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            nodes=[SceneNode.from_dict(node) for node in data.get("nodes", [])],
            edges=[SceneEdge.from_dict(edge) for edge in data.get("edges", [])],
            groups=[SceneGroup.from_dict(group) for group in data.get("groups", [])],
            semantic_anchors=[
                SemanticAnchor(**anchor) for anchor in data.get("semantic_anchors", [])
            ],
            layout_anchors=[
                LayoutAnchor(**anchor) for anchor in data.get("layout_anchors", [])
            ],
            constraints=[
                LayoutConstraint(**constraint) for constraint in data.get("constraints", [])
            ],
            style_tokens=StyleTokens(**data.get("style_tokens", {})),
            asset_refs=[AssetRef(**asset) for asset in data.get("asset_refs", [])],
            hierarchy_units=[
                HierarchyUnit.from_dict(unit) for unit in data.get("hierarchy_units", [])
            ],
            hierarchy_relations=[
                HierarchyRelation(**relation) for relation in data.get("hierarchy_relations", [])
            ],
            hierarchy_root_id=data.get("hierarchy_root_id"),
            hierarchy_state=HierarchyState(**data["hierarchy_state"]) if data.get("hierarchy_state") else None,
            decomposition_report=data.get("decomposition_report", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FigureSceneGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass
class Issue:
    id: str
    category: str
    severity: float
    message: str
    target_ids: list[str] = field(default_factory=list)
    suggested_operations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicOperation:
    id: str
    op_type: str
    target_ids: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    status: str = "planned"


@dataclass
class VersionNode:
    id: str
    parent_id: str | None
    scene_graph: FigureSceneGraph
    score: float
    note: str
    issues: list[Issue] = field(default_factory=list)
    operations: list[AtomicOperation] = field(default_factory=list)
    snapshot_png: str | None = None


@dataclass
class VersionDag:
    versions: dict[str, VersionNode] = field(default_factory=dict)
    head_id: str | None = None
    best_id: str | None = None

    def add_version(self, version: VersionNode) -> None:
        self.versions[version.id] = version
        self.head_id = version.id
        if self.best_id is None or version.score >= self.versions[self.best_id].score:
            self.best_id = version.id

    def head(self) -> VersionNode:
        if self.head_id is None:
            raise ValueError("Version DAG is empty.")
        return self.versions[self.head_id]

    def best(self) -> VersionNode:
        if self.best_id is None:
            raise ValueError("Version DAG is empty.")
        return self.versions[self.best_id]


@dataclass
class AssetState:
    version_dag: VersionDag = field(default_factory=VersionDag)
    scene_graph_path: str | None = None
    draft_image_path: str | None = None


@dataclass
class ExecutionState:
    iteration: int = 0
    operations: list[AtomicOperation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanningState:
    edit_goal: str = ""
    paper_context: str = ""
    caption: str = ""
    issue_backlog: list[Issue] = field(default_factory=list)
    target_score: float = 0.88
    best_score: float = 0.0
    stop_reason: str | None = None


@dataclass
class MemoryState:
    asset_state: AssetState = field(default_factory=AssetState)
    execution_state: ExecutionState = field(default_factory=ExecutionState)
    planning_state: PlanningState = field(default_factory=PlanningState)


@dataclass
class PipelineRequest:
    draft_image: str
    edit_goal: str
    output_dir: str
    paper_context: str = ""
    caption: str = ""
    reference_figures: list[str] = field(default_factory=list)
    draft_manifest: str | None = None
    max_iterations: int = 4
    random_seed: int = 0
    pipeline_config: "PipelineConfig | None" = None


@dataclass
class PipelineConfig:
    method_name: str = "AnchorFigure-Full"
    use_verifier: bool = True
    use_context_matching: bool = True
    use_semantic_anchors: bool = True
    use_layout_anchors: bool = True
    use_hierarchy_decomposition: bool = True
    use_hierarchical_team: bool = True
    use_hierarchy_metrics: bool = True
    use_layout_planner: bool = True
    use_retoucher: bool = True
    use_critic_loop: bool = True
    enforce_editable_export: bool = True
    export_svg: bool = True
    export_pptx: bool = True
    hierarchy_levels: list[str] = field(default_factory=lambda: ["global", "module", "region", "block", "element"])
    hierarchy_from_groups_first: bool = True
    max_iterations_override: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineArtifacts:
    revised_png: str
    scene_graph_json: str
    editable_pptx: str | None
    editable_svg: str | None
    edit_report_json: str
    memory_json: str
