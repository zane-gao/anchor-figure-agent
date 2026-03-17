from __future__ import annotations

from dataclasses import dataclass

from .models import FigureSceneGraph


@dataclass
class DrawIOExportResult:
    xml: str
    metadata: dict


class DrawIOAdapter:
    """Placeholder adapter for future DrawIO/XML export support."""

    def export(self, scene: FigureSceneGraph) -> DrawIOExportResult:
        metadata = {
            "hierarchy_root_id": scene.hierarchy_root_id,
            "hierarchy_unit_count": len(scene.hierarchy_units),
            "note": "DrawIO/XML export is not implemented in this iteration. Use SVG/PPTX export as the primary editable output.",
        }
        xml = "<mxGraphModel><root/></mxGraphModel>"
        return DrawIOExportResult(xml=xml, metadata=metadata)
