from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

from paper_plot_style import BLUE, ORANGE, apply_publication_style, style_axes, style_legend


METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "dynamic_mcts": {
        "label": "Dynamic + MCTS",
        "color": BLUE,
        "linestyle": "-",
        "marker": "o",
    },
    "dynamic_router_local": {
        "label": "Dynamic Router",
        "color": ORANGE,
        "linestyle": (0, (5, 2.4)),
        "marker": "s",
    },
    "dynamic_router_gpt_4o": {
        "label": "Dynamic Router",
        "color": ORANGE,
        "linestyle": (0, (5, 2.4)),
        "marker": "s",
    },
}


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _binomial_se(passed: float, total: float) -> float:
    if total <= 0:
        return 0.0
    p = passed / total
    return float(math.sqrt(max(p * (1.0 - p), 0.0) / total))


def _summarize_report_uncertainty(report_path: Path) -> Dict[str, float]:
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    results = payload.get("results") or []

    workflow_times: List[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        value = _safe_float(item.get("workflow_time_s"))
        if value is not None:
            workflow_times.append(value)

    total = float(payload.get("total") or 0.0)
    passed = float(payload.get("passed") or 0.0)
    workflow_std = float(stdev(workflow_times)) if len(workflow_times) >= 2 else 0.0
    workflow_se = workflow_std / math.sqrt(len(workflow_times)) if workflow_times else 0.0
    return {
        "pass_rate_se": _binomial_se(passed, total),
        "workflow_time_std_s": workflow_std,
        "workflow_time_se_s": workflow_se,
    }


def enrich_scalability_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        total = _safe_float(item.get("total")) or 0.0
        passed = _safe_float(item.get("passed")) or 0.0
        item.setdefault("pass_rate_se", _binomial_se(passed, total))
        item.setdefault("workflow_time_std_s", 0.0)
        item.setdefault("workflow_time_se_s", 0.0)

        report_path_raw = str(item.get("report_path") or "").strip()
        if report_path_raw:
            report_path = Path(report_path_raw)
            if report_path.exists():
                item.update(_summarize_report_uncertainty(report_path))

        enriched.append(item)
    return enriched


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda item: int(item["pool_size"]))
    return grouped


def _method_style(method: str) -> Dict[str, object]:
    style = METHOD_STYLES.get(method)
    if style:
        return style
    return {
        "label": method.replace("_", " "),
        "color": BLUE,
        "linestyle": "-",
        "marker": "o",
    }


def _plot_single_metric(
    *,
    ax,
    grouped: Dict[str, List[Dict[str, Any]]],
    y_key: str,
    y_transform,
    title: str,
    ylabel: str,
) -> Tuple[List[object], List[str]]:
    handles: List[object] = []
    labels: List[str] = []

    for method, method_rows in grouped.items():
        style = _method_style(method)
        xs = [int(item["pool_size"]) for item in method_rows]
        ys = [y_transform(float(item[y_key])) for item in method_rows]
        artist = ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.4,
            markersize=6.8,
            markerfacecolor=style["color"],
            markeredgecolor="#ffffff",
            markeredgewidth=0.9,
            label=style["label"],
            zorder=3,
        )
        handles.append(artist[0])
        labels.append(str(style["label"]))

    style_axes(ax, grid_axis="y")
    ax.set_title(title, pad=10)
    ax.set_xlabel("Agent pool size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row["pool_size"]) for rows in grouped.values() for row in rows}))
    return handles, labels


def _plot_metric_only(
    rows: List[Dict[str, Any]],
    *,
    output_path: Path,
    y_key: str,
    y_transform,
    title: str,
    ylabel: str,
) -> Path:
    apply_publication_style()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - handled by caller
        raise RuntimeError("matplotlib is required for plotting") from exc

    enriched_rows = enrich_scalability_rows(rows)
    grouped = _group_rows(enriched_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    handles, labels = _plot_single_metric(
        ax=ax,
        grouped=grouped,
        y_key=y_key,
        y_transform=y_transform,
        title=title,
        ylabel=ylabel,
    )
    legend = ax.legend(handles, labels, loc="best", frameon=True)
    style_legend(legend)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_scalability_passrate(
    rows: List[Dict[str, Any]],
    *,
    output_path: Path,
) -> Path:
    return _plot_metric_only(
        rows,
        output_path=output_path,
        y_key="pass_rate",
        y_transform=lambda value: value * 100.0,
        title="Pass@1 vs Agent Pool Size",
        ylabel="Pass@1 (%)",
    )


def plot_scalability_latency(
    rows: List[Dict[str, Any]],
    *,
    output_path: Path,
) -> Path:
    return _plot_metric_only(
        rows,
        output_path=output_path,
        y_key="avg_workflow_time_s",
        y_transform=lambda value: value,
        title="Latency vs Agent Pool Size",
        ylabel="Avg Workflow Time per Task (s)",
    )
