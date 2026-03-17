from __future__ import annotations

from pathlib import Path
import csv

from .experiments import strongest_non_anchorfigure
from .utils import deterministic_rng, ensure_dir, load_json


def prepare_human_eval(
    benchmark_root: str | Path,
    results_root: str | Path,
    output_dir: str | Path,
    synthetic_count: int = 30,
    real_count: int = 20,
) -> Path:
    benchmark_root = Path(benchmark_root)
    results_root = Path(results_root)
    output_dir = ensure_dir(output_dir)
    summary_path = results_root / "summary.json"
    baseline_method = strongest_non_anchorfigure(summary_path)
    if baseline_method is None:
        raise ValueError("无法从 summary.json 中找到 strongest non-AnchorFigure baseline。")

    rng = deterministic_rng("human-eval")
    synthetic_root = benchmark_root / "cases" if (benchmark_root / "cases").exists() else benchmark_root
    real_root = benchmark_root / "real_validation" / "cases"
    synthetic_cases = [path for path in sorted(synthetic_root.iterdir()) if path.is_dir() and (path / "case_meta.json").exists()]
    real_cases = []
    if real_root.exists():
        real_cases = [path for path in sorted(real_root.iterdir()) if path.is_dir() and (path / "case_meta.json").exists()]
    rng.shuffle(synthetic_cases)
    rng.shuffle(real_cases)
    selected_cases = [("synthetic", path) for path in synthetic_cases[:synthetic_count]]
    selected_cases.extend(("real", path) for path in real_cases[:real_count])

    csv_path = output_dir / "pairwise_annotations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "split",
                "reviewer_id",
                "method_a",
                "method_b",
                "image_a",
                "image_b",
                "publication_readiness",
                "faithfulness",
                "readability",
                "editability_usefulness",
                "winner",
                "notes",
            ]
        )
        for split_name, case_dir in selected_cases:
            anchor_path = results_root / "anchorfigure_full" / case_dir.name / "revised.png"
            baseline_path = results_root / baseline_method / case_dir.name / "revised.png"
            if not anchor_path.exists() or not baseline_path.exists():
                continue
            if rng.random() > 0.5:
                method_a, image_a = "anchorfigure_full", anchor_path
                method_b, image_b = baseline_method, baseline_path
            else:
                method_a, image_a = baseline_method, baseline_path
                method_b, image_b = "anchorfigure_full", anchor_path
            writer.writerow(
                [
                    case_dir.name,
                    split_name,
                    "",
                    method_a,
                    method_b,
                    str(image_a),
                    str(image_b),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    protocol = """# Human Evaluation Protocol

## 任务

每个 case 展示两个匿名结果，请从以下四个维度进行比较：

1. publication readiness
2. faithfulness
3. readability
4. editability usefulness

## 规则

- 不允许根据文件名推断方法来源
- 先完成 5 个 warm-up case，再进入正式标注
- 若两者几乎相同，可在 notes 中记录，但 winner 仍需填更优者
"""
    (output_dir / "protocol.md").write_text(protocol, encoding="utf-8")
    return csv_path
