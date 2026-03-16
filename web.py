from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import base64
import json
import mimetypes
import tempfile
import time

from .agents import FigureEditPipeline
from .utils import ensure_dir, load_json
from .models import PipelineRequest


def _safe_relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")


def _read_case(case_dir: Path) -> dict:
    meta = load_json(case_dir / "case_meta.json")
    return {
        "case_id": case_dir.name,
        "draft_image": f"/files/{_safe_relative(case_dir / 'draft.png', case_dir.parent)}",
        "target_image": f"/files/{_safe_relative(case_dir / 'target.png', case_dir.parent)}",
        "caption": (case_dir / "caption.txt").read_text(encoding="utf-8"),
        "edit_goal": (case_dir / "edit_goal.txt").read_text(encoding="utf-8"),
        "paper_context": (case_dir / "paper_context.txt").read_text(encoding="utf-8"),
        "meta": meta,
    }


def build_server(root_dir: str | Path, port: int = 8765) -> ThreadingHTTPServer:
    root_dir = Path(root_dir).resolve()
    runs_dir = ensure_dir(root_dir / ".web_runs")
    static_dir = Path(__file__).resolve().parent.parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(404, "File not found")
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_static(self, relative_path: str) -> None:
            relative = "index.html" if relative_path in {"", "/"} else relative_path.lstrip("/")
            self._send_file(static_dir / relative)

        def _list_examples(self) -> list[dict]:
            return [_read_case(case_dir) for case_dir in sorted(root_dir.iterdir()) if case_dir.is_dir() and (case_dir / "case_meta.json").exists()]

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/examples":
                self._send_json({"examples": self._list_examples()})
                return
            if parsed.path.startswith("/files/"):
                relative = parsed.path[len("/files/") :]
                target = (root_dir / relative).resolve()
                allowed = [root_dir.resolve(), runs_dir.resolve()]
                if not any(str(target).startswith(str(base)) for base in allowed):
                    self.send_error(403, "Forbidden")
                    return
                self._send_file(target)
                return
            if parsed.path in {"/", "/index.html", "/app.js", "/styles.css"}:
                relative = "" if parsed.path in {"/", "/index.html"} else parsed.path.lstrip("/")
                self._serve_static(relative)
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/run":
                self.send_error(404, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            case_id = payload.get("case_id")
            edit_goal = payload.get("edit_goal", "")
            caption = payload.get("caption", "")
            paper_context = payload.get("paper_context", "")
            draft_image = ""
            draft_manifest = None
            if case_id:
                case_dir = root_dir / case_id
                if not case_dir.exists():
                    self._send_json({"error": f"case {case_id} not found"}, status=404)
                    return
                draft_image = str(case_dir / "draft.png")
                draft_manifest = str(case_dir / "draft_manifest.json")
                if not edit_goal:
                    edit_goal = (case_dir / "edit_goal.txt").read_text(encoding="utf-8")
                if not caption:
                    caption = (case_dir / "caption.txt").read_text(encoding="utf-8")
                if not paper_context:
                    paper_context = (case_dir / "paper_context.txt").read_text(encoding="utf-8")
            else:
                upload = payload.get("draft_image_b64")
                if not upload:
                    self._send_json({"error": "missing case_id or draft_image_b64"}, status=400)
                    return
                header, encoded = upload.split(",", 1) if "," in upload else ("", upload)
                upload_dir = ensure_dir(runs_dir / f"upload_{int(time.time() * 1000)}")
                draft_path = upload_dir / "draft.png"
                draft_path.write_bytes(base64.b64decode(encoded))
                draft_image = str(draft_path)

            run_dir = ensure_dir(runs_dir / f"run_{int(time.time() * 1000)}")
            pipeline = FigureEditPipeline()
            artifacts = pipeline.run(
                PipelineRequest(
                    draft_image=draft_image,
                    edit_goal=edit_goal or "规范化科研图布局和样式。",
                    output_dir=str(run_dir),
                    paper_context=paper_context,
                    caption=caption,
                    draft_manifest=draft_manifest,
                )
            )
            report = load_json(artifacts.edit_report_json)
            scene_graph = load_json(artifacts.scene_graph_json)
            self._send_json(
                {
                    "artifacts": {
                        "revised_png": f"/files/{_safe_relative(Path(artifacts.revised_png), root_dir)}",
                        "editable_svg": f"/files/{_safe_relative(Path(artifacts.editable_svg), root_dir)}",
                        "editable_pptx": f"/files/{_safe_relative(Path(artifacts.editable_pptx), root_dir)}",
                        "scene_graph_json": f"/files/{_safe_relative(Path(artifacts.scene_graph_json), root_dir)}",
                        "edit_report_json": f"/files/{_safe_relative(Path(artifacts.edit_report_json), root_dir)}",
                    },
                    "report": report,
                    "scene_graph": scene_graph,
                }
            )

        def log_message(self, format: str, *args) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root_dir: str | Path, port: int = 8765) -> None:
    server = build_server(root_dir=root_dir, port=port)
    print(f"Serving Figure Agent demo at http://127.0.0.1:{port}")
    server.serve_forever()
