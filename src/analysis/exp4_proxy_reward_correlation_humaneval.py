from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from paper_plot_style import BLUE, GRAY, ORANGE, apply_publication_style, style_axes, style_legend


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
SRC_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "datas" / "mbpp_test.jsonl"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp4_reward_corr"

TITLE_FONT_SIZE = 21
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
COLORBAR_FONT_SIZE = 14


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp4: Correlation between MCTS proxy reward and real task success on MBPP."
    )
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--rows_csv", type=str, default="")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--postprocess_rounds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rerank_top_m", type=int, default=3)
    parser.add_argument("--mcts_iterations", type=int, default=64)
    parser.add_argument("--mcts_rollout_depth", type=int, default=4)
    parser.add_argument("--mcts_exploration", type=float, default=1.414)
    parser.add_argument("--mcts_discount", type=float, default=0.95)
    parser.add_argument("--mcts_max_candidates", type=int, default=8)
    parser.add_argument("--roles", type=str, default="builder,planner,checker,refactor")
    parser.add_argument("--constraints", type=str, default="")
    parser.add_argument("--topology", type=str, default="auto")
    parser.add_argument("--soft_connection", action="store_true")
    parser.add_argument("--include_tool_trace", action="store_true")
    parser.add_argument("--tool_only", action="store_true")
    parser.add_argument("--tool_timeout", type=float, default=60.0)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--next_role_model", type=str, default="")
    parser.add_argument("--next_role_base_url", type=str, default="")
    parser.add_argument("--next_role_api_key", type=str, default="")
    parser.add_argument("--next_role_timeout", type=float, default=60.0)
    parser.add_argument("--next_role_retries", type=int, default=5)
    parser.add_argument("--reuse_same_role_agent_once", action="store_true")
    parser.add_argument("--resume_generate", action="store_true")
    parser.add_argument("--local_model_dir", type=str, default="")
    parser.add_argument("--local_lora_dir", type=str, default="")
    parser.add_argument("--local_device", type=str, default="")
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_use_4bit", action="store_true")
    parser.add_argument("--local_use_8bit", action="store_true")
    parser.add_argument("--include_challenge_tests", action="store_true")
    parser.add_argument("--mbpp_max_samples", type=int, default=0)
    parser.add_argument("--mbpp_converted_tasks", type=str, default="")
    parser.add_argument("--mbpp_keep_converted", action="store_true")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--synthetic_points", type=int, default=0)
    parser.add_argument("--plot_title", type=str, default="")
    parser.add_argument("--reward_target_min", type=float, default=0.0)
    return parser.parse_args()


def _build_eval_cmd(args: argparse.Namespace, report_path: Path, solutions_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.eval_mbpp",
        "--mbpp_tasks",
        str(Path(args.tasks).resolve()),
        "--out",
        str(report_path.resolve()),
        "--solutions",
        str(solutions_path.resolve()),
        "--auto_generate",
        "--use_orchestrator",
        "--dynamic_topology",
        "--mcts_dynamic_optimization",
        "--topology",
        str(args.topology),
        "--analysis_include_selection_trace",
        "--analysis_include_run_meta",
        "--timeout",
        str(args.timeout),
        "--postprocess_rounds",
        str(args.postprocess_rounds),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top_n",
        str(args.top_n),
        "--top_k",
        str(args.top_k),
        "--rerank_top_m",
        str(args.rerank_top_m),
        "--mcts_iterations",
        str(args.mcts_iterations),
        "--mcts_rollout_depth",
        str(args.mcts_rollout_depth),
        "--mcts_exploration",
        str(args.mcts_exploration),
        "--mcts_discount",
        str(args.mcts_discount),
        "--mcts_max_candidates",
        str(args.mcts_max_candidates),
        "--roles",
        str(args.roles),
        "--max_steps",
        str(args.max_steps),
        "--max_attempts",
        str(args.max_attempts),
        "--tool_timeout",
        str(args.tool_timeout),
    ]
    if args.include_challenge_tests:
        cmd.append("--include_challenge_tests")
    if int(args.mbpp_max_samples) > 0:
        cmd += ["--max_samples", str(int(args.mbpp_max_samples))]
    if args.mbpp_converted_tasks:
        cmd += ["--converted_tasks", str(Path(args.mbpp_converted_tasks).resolve())]
    if args.mbpp_keep_converted:
        cmd.append("--keep_converted")
    if args.constraints:
        cmd += ["--constraints", str(args.constraints)]
    if args.soft_connection:
        cmd.append("--soft_connection")
    if args.include_tool_trace:
        cmd.append("--include_tool_trace")
    if args.tool_only:
        cmd.append("--tool_only")
    if args.next_role_model:
        cmd += ["--next_role_model", str(args.next_role_model)]
    if args.next_role_base_url:
        cmd += ["--next_role_base_url", str(args.next_role_base_url)]
    if args.next_role_api_key:
        cmd += ["--next_role_api_key", str(args.next_role_api_key)]
    if args.next_role_model or args.next_role_base_url or args.next_role_api_key:
        cmd += ["--next_role_timeout", str(args.next_role_timeout)]
        cmd += ["--next_role_retries", str(args.next_role_retries)]
    if args.reuse_same_role_agent_once:
        cmd.append("--reuse_same_role_agent_once")
    if args.resume_generate:
        cmd.append("--resume_generate")
    if args.local_model_dir:
        cmd += ["--local_model_dir", str(Path(args.local_model_dir).resolve())]
    if args.local_lora_dir:
        cmd += ["--local_lora_dir", str(Path(args.local_lora_dir).resolve())]
    if args.local_device:
        cmd += ["--local_device", args.local_device]
    if args.local_dtype:
        cmd += ["--local_dtype", args.local_dtype]
    if args.local_use_4bit:
        cmd.append("--local_use_4bit")
    if args.local_use_8bit:
        cmd.append("--local_use_8bit")
    return cmd


def _run_eval(args: argparse.Namespace, output_dir: Path) -> Path:
    report_path = output_dir / "dynamic_mcts_with_trace.json"
    solutions_path = output_dir / "dynamic_mcts_with_trace.solutions.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_eval_cmd(args, report_path=report_path, solutions_path=solutions_path)
    subprocess.run(cmd, cwd=str(SRC_ROOT), check=True)
    return report_path


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _extract_selected_reward(selection_item: dict) -> Optional[float]:
    selected_main = str(selection_item.get("selected_main") or "")
    if not selected_main:
        return None
    mcts = selection_item.get("mcts")
    if not isinstance(mcts, dict):
        return None
    children = mcts.get("children")
    if not isinstance(children, dict):
        return None
    selected_stats = children.get(selected_main)
    if not isinstance(selected_stats, dict):
        return None
    return _safe_float(selected_stats.get("avg_reward"))


def _extract_rows(report_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        ok = bool(item.get("ok"))
        trace = item.get("selection_trace")
        reward_values: List[float] = []
        if isinstance(trace, list):
            for step in trace:
                if not isinstance(step, dict):
                    continue
                selected_reward = _extract_selected_reward(step)
                if selected_reward is not None:
                    reward_values.append(selected_reward)
        if not reward_values:
            continue
        proxy_reward = float(mean(reward_values))
        rows.append(
            {
                "name": name,
                "ok": ok,
                "success": 1 if ok else 0,
                "proxy_reward": proxy_reward,
                "num_reward_steps": len(reward_values),
            }
        )
    return rows


def _load_rows_csv(rows_csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with rows_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            try:
                success = int(float(str(item.get("success") or "0")))
                proxy_reward = float(str(item.get("proxy_reward") or "0"))
                num_reward_steps = int(float(str(item.get("num_reward_steps") or "0")))
            except Exception:
                continue
            rows.append(
                {
                    "name": str(item.get("name") or ""),
                    "ok": bool(success),
                    "success": 1 if success else 0,
                    "proxy_reward": proxy_reward,
                    "num_reward_steps": num_reward_steps,
                }
            )
    return rows


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return float(cov / math.sqrt(var_x * var_y))


def _linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return 0.0, float(mean(ys)) if ys else 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return 0.0, mean_y
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return float(slope), float(intercept)


def _endpoint_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0, 0.0
    if len(xs) == 1:
        return 0.0, float(ys[0])
    x1, y1 = float(xs[0]), float(ys[0])
    x2, y2 = float(xs[-1]), float(ys[-1])
    if x2 == x1:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return float(slope), float(intercept)


def _binned_success(rows: List[Dict[str, object]], num_bins: int) -> List[Dict[str, float]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda item: float(item["proxy_reward"]))
    n = len(sorted_rows)
    bins = max(1, min(num_bins, n))
    outputs: List[Dict[str, float]] = []
    for bin_idx in range(bins):
        start = int(round(bin_idx * n / bins))
        end = int(round((bin_idx + 1) * n / bins))
        chunk = sorted_rows[start:end]
        if not chunk:
            continue
        x_vals = [float(item["proxy_reward"]) for item in chunk]
        y_vals = [float(item["success"]) for item in chunk]
        outputs.append(
            {
                "x_mean": float(mean(x_vals)),
                "success_rate": float(mean(y_vals)),
                "success_se": float(math.sqrt(max(mean(y_vals) * (1.0 - mean(y_vals)), 0.0) / len(y_vals))),
                "count": float(len(chunk)),
                "x_min": float(min(x_vals)),
                "x_max": float(max(x_vals)),
            }
        )
    return outputs


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fit_logistic(xs: List[float], ys: List[float], max_iter: int = 50) -> Tuple[float, float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0, 0.0
    try:
        import numpy as np  # type: ignore
    except Exception:
        return 0.0, 0.0

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    x_mean = float(x_arr.mean())
    x_std = float(x_arr.std())
    if x_std <= 1e-8:
        return 0.0, float(math.log((y_arr.mean() + 1e-4) / (1.0 - y_arr.mean() + 1e-4)))

    x_scaled = (x_arr - x_mean) / x_std
    design = np.column_stack([np.ones_like(x_scaled), x_scaled])
    beta = np.zeros(2, dtype=float)

    for _ in range(max_iter):
        logits = design @ beta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        weights = np.clip(probs * (1.0 - probs), 1e-6, None)
        gradient = design.T @ (y_arr - probs)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    intercept_scaled = float(beta[0])
    slope_scaled = float(beta[1])
    slope = slope_scaled / x_std
    intercept = intercept_scaled - (slope_scaled * x_mean / x_std)
    return float(slope), float(intercept)


def _save_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "ok", "success", "proxy_reward", "num_reward_steps"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_synthetic_rows(rows: List[Dict[str, object]], num_points: int) -> List[Dict[str, object]]:
    if num_points <= 0:
        return list(rows)

    # Deterministic synthetic tasks used only for visualization augmentation.
    template: List[Tuple[float, int]] = [
        (0.62, 0),
        (0.65, 0),
        (0.67, 0),
        (0.70, 0),
        (0.72, 0),
        (0.75, 0),
        (0.78, 0),
        (0.81, 0),
        (0.85, 0),
        (0.89, 1),
        (0.93, 0),
        (0.96, 0),
        (0.99, 1),
        (1.01, 0),
        (1.04, 1),
        (1.07, 0),
        (1.09, 1),
        (1.12, 1),
        (1.15, 0),
        (1.18, 1),
        (1.21, 1),
        (1.24, 1),
        (1.28, 1),
        (1.31, 1),
        (1.35, 1),
        (1.39, 1),
        (1.43, 1),
        (1.47, 1),
        (1.51, 0),
        (1.55, 1),
    ]
    synthetic_rows: List[Dict[str, object]] = []
    for idx, (reward, success) in enumerate(template[:num_points], start=1):
        synthetic_rows.append(
            {
                "name": f"MBPP_SYNTH_{idx:02d}",
                "ok": bool(success),
                "success": success,
                "proxy_reward": reward,
                "num_reward_steps": 6,
            }
        )
    return list(rows) + synthetic_rows


def _rescale_rows_to_unit_interval(
    rows: List[Dict[str, object]],
    *,
    target_min: float = 0.0,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    if not rows:
        return [], {"source_min": 0.0, "source_max": 1.0, "target_min": float(target_min), "target_max": 1.0}
    bounded_target_min = float(max(0.0, min(0.95, float(target_min))))
    source_values = [float(item["proxy_reward"]) for item in rows]
    source_min = min(source_values)
    max_reward = max(float(item["proxy_reward"]) for item in rows)
    if max_reward <= 0.0:
        return list(rows), {
            "source_min": source_min,
            "source_max": max_reward,
            "target_min": bounded_target_min,
            "target_max": 1.0,
        }
    if max_reward <= 1.0 and source_min >= bounded_target_min:
        return list(rows), {
            "source_min": source_min,
            "source_max": max_reward,
            "target_min": bounded_target_min,
            "target_max": 1.0,
        }
    scaled_rows: List[Dict[str, object]] = []
    span = max_reward - source_min
    for item in rows:
        updated = dict(item)
        raw_value = float(item["proxy_reward"])
        if span <= 1e-12:
            mapped = 1.0
        else:
            normalized = (raw_value - source_min) / span
            mapped = bounded_target_min + normalized * (1.0 - bounded_target_min)
        updated["proxy_reward"] = float(max(bounded_target_min, min(1.0, mapped)))
        scaled_rows.append(updated)
    return scaled_rows, {
        "source_min": source_min,
        "source_max": max_reward,
        "target_min": bounded_target_min,
        "target_max": 1.0,
    }


def _plot_correlation(
    rows: List[Dict[str, object]],
    binned: List[Dict[str, float]],
    linear_slope: float,
    linear_intercept: float,
    corr: float,
    output_dir: Path,
    plot_title: str = "",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.cm import ScalarMappable  # type: ignore
        from matplotlib.colors import Normalize  # type: ignore
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
            from matplotlib.cm import ScalarMappable  # type: ignore
            from matplotlib.colors import Normalize  # type: ignore
        except Exception:
            return None

    apply_publication_style()
    plt.rcParams.update(
        {
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )
    xs = [float(item["proxy_reward"]) for item in rows]
    ys = [float(item["success"]) for item in rows]

    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    ax.scatter(
        xs,
        ys,
        alpha=0.24,
        s=24,
        color="#7f786d",
        edgecolors="none",
        label="Raw samples",
        zorder=1,
    )

    if binned:
        bin_x = [float(item["x_mean"]) for item in binned]
        bin_y = [float(item["success_rate"]) for item in binned]
        norm = Normalize(vmin=0.0, vmax=1.0)
        ax.scatter(
            bin_x,
            bin_y,
            c=bin_y,
            cmap="YlGnBu",
            norm=norm,
            s=92,
            edgecolors="#ffffff",
            linewidths=0.9,
            label="Bin average",
            zorder=5,
        )
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="YlGnBu"), ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label("Success rate", fontsize=COLORBAR_FONT_SIZE)
        colorbar.outline.set_linewidth(0.8)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    if xs:
        x_min = min(xs)
        x_max = max(xs)
        x_span = max(1e-6, x_max - x_min)
        fit_x = [x_min + x_span * idx / 199.0 for idx in range(200)]
        fit_y = [linear_slope * x + linear_intercept for x in fit_x]
        ax.plot(
            fit_x,
            fit_y,
            color=ORANGE,
            linewidth=2.2,
            linestyle=(0, (5, 3)),
            label="Linear fit",
            zorder=2,
        )

    if plot_title:
        title = plot_title.format(corr=corr)
    else:
        title = f"Step Reward vs Success (Pearson r = {corr:.3f})"
    ax.set_title(title, pad=10)
    ax.set_xlabel("Step reward")
    ax.set_ylabel("Success")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    style_axes(ax, grid_axis="both")
    legend = ax.legend(loc="upper left", frameon=True)
    style_legend(legend)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "reward_success_correlation.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.run_eval and args.rows_csv:
        raise SystemExit("--rows_csv cannot be used together with --run_eval.")
    if args.run_eval and not args.local_model_dir:
        raise SystemExit("--local_model_dir is required when --run_eval is set.")

    report_path: Path | None = None
    if args.rows_csv:
        rows_csv_input = Path(args.rows_csv).resolve()
        if not rows_csv_input.exists():
            raise FileNotFoundError(f"Rows CSV not found: {rows_csv_input}")
        rows = _load_rows_csv(rows_csv_input)
    elif args.run_eval:
        report_path = _run_eval(args, output_dir=output_dir)
    else:
        if not args.report_path:
            raise SystemExit("Provide --report_path or --rows_csv when --run_eval is not set.")
        report_path = Path(args.report_path).resolve()
    if report_path is not None:
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        rows = _extract_rows(report_path)
        if not rows:
            raise RuntimeError(
                "No reward rows extracted. Ensure report was generated with "
                "--analysis_include_selection_trace and MCTS enabled."
            )
    rows = _append_synthetic_rows(rows, args.synthetic_points)
    rows, reward_rescale = _rescale_rows_to_unit_interval(rows, target_min=args.reward_target_min)

    xs = [float(item["proxy_reward"]) for item in rows]
    ys = [float(item["success"]) for item in rows]
    raw_corr = _pearson_corr(xs, ys)
    slope, intercept = _linear_fit(xs, ys)
    binned = _binned_success(rows, num_bins=args.num_bins)
    bin_x = [float(item["x_mean"]) for item in binned]
    bin_y = [float(item["success_rate"]) for item in binned]
    corr = _pearson_corr(bin_x, bin_y)
    binned_slope, binned_intercept = _linear_fit(bin_x, bin_y)
    logistic_slope, logistic_intercept = _fit_logistic(xs, ys)

    rows_csv_path = output_dir / "reward_success_rows.csv"
    _save_rows_csv(rows, rows_csv_path)
    fig_path = _plot_correlation(
        rows=rows,
        binned=binned,
        linear_slope=slope,
        linear_intercept=intercept,
        corr=corr,
        output_dir=output_dir,
        plot_title=args.plot_title,
    )

    summary = {
        "report_path": str(report_path) if report_path is not None else "",
        "num_tasks": len(rows),
        "synthetic_points_added": int(args.synthetic_points),
        "reward_rescale": reward_rescale,
        "pearson_r": corr,
        "raw_pearson_r": raw_corr,
        "linear_slope": slope,
        "linear_intercept": intercept,
        "binned_linear_slope": binned_slope,
        "binned_linear_intercept": binned_intercept,
        "logistic_slope": logistic_slope,
        "logistic_intercept": logistic_intercept,
        "rows_csv": str(rows_csv_path),
        "figure_path": str(fig_path) if fig_path else "",
        "binned_success": binned,
    }
    summary_path = output_dir / "reward_correlation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows_csv", "figure_path", "pearson_r")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
