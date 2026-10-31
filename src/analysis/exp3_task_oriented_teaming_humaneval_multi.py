from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from exp3_task_oriented_teaming_humaneval import _extract_task_rows


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "datas" / "mbpp_test.jsonl"
DEFAULT_DB = REPO_ROOT / "demo_registry.sqlite"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp3_teaming_multi"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp3 multi-report: aggregate task-oriented teaming analysis across multiple MBPP reports."
    )
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--registry_db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--report_paths",
        type=str,
        default="",
        help="Comma-separated absolute/relative report paths.",
    )
    parser.add_argument(
        "--report_glob",
        type=str,
        default="",
        help="Glob pattern for report paths, e.g. 'data/MBPP/mbpp_report*.json'",
    )
    parser.add_argument(
        "--report_labels",
        type=str,
        default="",
        help="Comma-separated labels matching report order. Optional.",
    )
    return parser.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_report_paths(args: argparse.Namespace) -> List[Path]:
    report_paths: List[Path] = []
    for item in _split_csv(args.report_paths):
        report_paths.append(Path(item).resolve())
    if args.report_glob:
        matches = sorted(glob.glob(args.report_glob))
        report_paths.extend(Path(item).resolve() for item in matches)
    unique: List[Path] = []
    seen = set()
    for path in report_paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        unique.append(path)
    return unique


def _resolve_labels(paths: List[Path], raw_labels: str) -> List[str]:
    labels = _split_csv(raw_labels)
    if labels and len(labels) != len(paths):
        raise ValueError(
            f"report_labels count ({len(labels)}) must match report count ({len(paths)})."
        )
    if not labels:
        labels = [path.stem for path in paths]
    return labels


def _save_task_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "report_label",
        "report_path",
        "name",
        "task_id",
        "family",
        "ok",
        "team_size",
        "declared_team_size",
        "selected_domains",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dumped = dict(row)
            dumped["selected_domains"] = json.dumps(row.get("selected_domains") or [], ensure_ascii=True)
            writer.writerow(dumped)


def _summarize_by_family(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["report_label"]), str(row["family"]))
        buckets.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (report_label, family), group in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        team_sizes = [int(item["team_size"]) for item in group]
        declared_sizes = [int(item["declared_team_size"]) for item in group]
        oks = [1 if bool(item["ok"]) else 0 for item in group]
        summary_rows.append(
            {
                "report_label": report_label,
                "family": family,
                "count": len(group),
                "avg_team_size": round(mean(team_sizes), 6) if team_sizes else 0.0,
                "avg_declared_team_size": round(mean(declared_sizes), 6) if declared_sizes else 0.0,
                "pass_rate": round(mean(oks), 6) if oks else 0.0,
            }
        )
    return summary_rows


def _summarize_overall(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["report_label"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for report_label, group in sorted(buckets.items(), key=lambda x: x[0]):
        team_sizes = [int(item["team_size"]) for item in group]
        oks = [1 if bool(item["ok"]) else 0 for item in group]
        summary_rows.append(
            {
                "report_label": report_label,
                "num_tasks": len(group),
                "pass_rate": round(mean(oks), 6) if oks else 0.0,
                "avg_team_size": round(mean(team_sizes), 6) if team_sizes else 0.0,
            }
        )
    return summary_rows


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_import_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import matplotlib.pyplot as plt  # type: ignore

            return plt
        except Exception:
            return None


def _plot_family_compare(
    family_rows: List[Dict[str, object]],
    output_dir: Path,
) -> Path | None:
    plt = _maybe_import_plt()
    if plt is None:
        return None

    families = sorted({str(row["family"]) for row in family_rows})
    reports = sorted({str(row["report_label"]) for row in family_rows})
    if not families or not reports:
        return None

    metric_map_team: Dict[Tuple[str, str], float] = {}
    metric_map_pass: Dict[Tuple[str, str], float] = {}
    for row in family_rows:
        key = (str(row["report_label"]), str(row["family"]))
        metric_map_team[key] = float(row["avg_team_size"])
        metric_map_pass[key] = float(row["pass_rate"]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    width = 0.8 / max(1, len(reports))
    x = list(range(len(families)))

    ax1 = axes[0]
    for idx, report in enumerate(reports):
        vals = [metric_map_team.get((report, fam), 0.0) for fam in families]
        offsets = [v + (idx - (len(reports) - 1) / 2.0) * width for v in x]
        ax1.bar(offsets, vals, width=width, label=report)
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=20)
    ax1.set_title("Average Team Size by Family (Multi-report)")
    ax1.set_ylabel("Avg Team Size")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    for idx, report in enumerate(reports):
        vals = [metric_map_pass.get((report, fam), 0.0) for fam in families]
        offsets = [v + (idx - (len(reports) - 1) / 2.0) * width for v in x]
        ax2.bar(offsets, vals, width=width, label=report)
    ax2.set_xticks(x)
    ax2.set_xticklabels(families, rotation=20)
    ax2.set_title("Pass Rate by Family (Multi-report)")
    ax2.set_ylabel("Pass Rate (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "multi_family_compare.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def _plot_team_size_overlay(
    task_rows: List[Dict[str, object]],
    output_dir: Path,
) -> Path | None:
    plt = _maybe_import_plt()
    if plt is None:
        return None

    reports = sorted({str(row["report_label"]) for row in task_rows})
    if not reports:
        return None

    max_size = max(int(row["team_size"]) for row in task_rows) if task_rows else 1
    bins = list(range(0, max_size + 2))

    fig, ax = plt.subplots(figsize=(9, 5))
    for report in reports:
        sizes = [int(row["team_size"]) for row in task_rows if str(row["report_label"]) == report]
        ax.hist(sizes, bins=bins, alpha=0.35, label=report, density=False)
    ax.set_title("Team Size Distribution Overlay (Multi-report)")
    ax.set_xlabel("Team Size")
    ax.set_ylabel("Task Count")
    ax.set_xticks(list(range(0, max_size + 1)))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "multi_team_size_hist_overlay.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def main() -> None:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    db_path = Path(args.registry_db).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    report_paths = _resolve_report_paths(args)
    if not report_paths:
        raise SystemExit("Provide --report_paths or --report_glob")
    for path in report_paths:
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")
    labels = _resolve_labels(report_paths, args.report_labels)

    merged_rows: List[Dict[str, object]] = []
    for label, report_path in zip(labels, report_paths):
        rows = _extract_task_rows(report_path=report_path, tasks_path=tasks_path, db_path=db_path)
        for row in rows:
            enriched = dict(row)
            enriched["report_label"] = label
            enriched["report_path"] = str(report_path)
            merged_rows.append(enriched)

    if not merged_rows:
        raise RuntimeError("No rows extracted from the provided reports.")

    task_rows_csv = output_dir / "multi_task_rows.csv"
    family_summary_csv = output_dir / "multi_family_summary.csv"
    report_overall_csv = output_dir / "multi_report_overall.csv"

    _save_task_rows_csv(merged_rows, task_rows_csv)
    family_rows = _summarize_by_family(merged_rows)
    overall_rows = _summarize_overall(merged_rows)
    _write_csv(family_rows, family_summary_csv)
    _write_csv(overall_rows, report_overall_csv)

    family_plot = _plot_family_compare(family_rows, output_dir=output_dir)
    hist_plot = _plot_team_size_overlay(merged_rows, output_dir=output_dir)

    summary = {
        "tasks_path": str(tasks_path),
        "registry_db": str(db_path),
        "reports": [{"label": label, "path": str(path)} for label, path in zip(labels, report_paths)],
        "num_merged_rows": len(merged_rows),
        "task_rows_csv": str(task_rows_csv),
        "family_summary_csv": str(family_summary_csv),
        "report_overall_csv": str(report_overall_csv),
        "family_compare_plot": str(family_plot) if family_plot else "",
        "team_size_overlay_plot": str(hist_plot) if hist_plot else "",
    }
    summary_path = output_dir / "multi_teaming_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "task_rows_csv": summary["task_rows_csv"],
                "family_summary_csv": summary["family_summary_csv"],
                "report_overall_csv": summary["report_overall_csv"],
                "family_compare_plot": summary["family_compare_plot"],
                "team_size_overlay_plot": summary["team_size_overlay_plot"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
