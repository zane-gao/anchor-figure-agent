from __future__ import annotations

from pathlib import Path
from typing import Iterable
import base64
import json
import math
import random
from difflib import SequenceMatcher

from .models import BBox


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def bbox_union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return BBox(0, 0, 0, 0)
    min_x = min(box.x for box in boxes)
    min_y = min(box.y for box in boxes)
    max_x = max(box.right for box in boxes)
    max_y = max(box.bottom for box in boxes)
    return BBox(min_x, min_y, max_x - min_x, max_y - min_y)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    set_a = set(a_norm.split())
    set_b = set(b_norm.split())
    token_score = 0.0
    if set_a and set_b:
        token_score = len(set_a & set_b) / len(set_a | set_b)
    char_score = SequenceMatcher(a=a_norm, b=b_norm).ratio()
    return max(token_score, char_score)


def save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deterministic_rng(seed: str | int) -> random.Random:
    if isinstance(seed, str):
        seed_value = sum(ord(ch) * (idx + 1) for idx, ch in enumerate(seed))
    else:
        seed_value = seed
    return random.Random(seed_value)


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(points)):
        x1, y1 = points[idx - 1]
        x2, y2 = points[idx]
        total += math.dist((x1, y1), (x2, y2))
    return total


def to_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"
