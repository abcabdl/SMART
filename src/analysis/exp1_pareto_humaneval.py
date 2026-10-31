from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
SRC_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "datas" / "mbpp_test.jsonl"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp1_pareto"


@dataclass(frozen=True)
class MethodPreset:
    name: str
    cli_args: Sequence[str]
    description: str


METHOD_PRESETS: Dict[str, MethodPreset] = {
    "dynamic_mcts": MethodPreset(
        name="dynamic_mcts",
        cli_args=(
            "--use_orchestrator",
            "--dynamic_topology",
            "--mcts_dynamic_optimization",
        ),
        description="Dynamic workflow + LLM-free MCTS selection",
    ),
    "dynamic_router_local": MethodPreset(
        name="dynamic_router_local",
        cli_args=(
            "--use_orchestrator",
            "--dynamic_topology",
        ),
        description="Dynamic workflow + embedding/router (no MCTS)",
    ),
    "fixed_bcb_mcts": MethodPreset(
        name="fixed_bcb_mcts",
        cli_args=(
            "--use_orchestrator",
            "--baseline_fixed_bcb_mcts",
        ),
        description="Fixed B->C->B workflow + MCTS",
    ),
    "fixed_bcb_router_gpt4o": MethodPreset(
        name="fixed_bcb_router_gpt4o",
        cli_args=(
            "--use_orchestrator",
            "--baseline_fixed_bcb_router_gpt4o",
        ),
        description="Fixed B->C->B workflow + embedding retrieval + remote LLM router",
    ),
    "single_builder": MethodPreset(
        name="single_builder",
        cli_args=(
            "--use_orchestrator",
            "--topology",
            "single",
            "--roles",
            "builder",
            "--max_steps",
            "1",
        ),
        description="Single-role builder baseline",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp1: Cost-performance Pareto on MBPP (isolated analysis pipeline)."
    )
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--methods",
        type=str,
        default="dynamic_mcts,dynamic_router_local,fixed_bcb_mcts,fixed_bcb_router_gpt4o,single_builder",
        help=f"Comma-separated methods. Available: {','.join(METHOD_PRESETS.keys())}",
    )
    parser.add_argument("--skip_run", action="store_true", help="Only read existing reports in output_dir/reports")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--postprocess_rounds", type=int, default=3)
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
    parser.add_argument(
        "--cost_metric",
        type=str,
        default="avg_generation_time_s",
        choices=[
            "avg_generation_time_s",
            "avg_workflow_time_s",
            "total_generation_time_s",
            "total_workflow_time_s",
            "avg_total_tokens",
            "total_tokens",
            "avg_estimated_cost_usd",
            "total_estimated_cost_usd",
        ],
    )
    parser.add_argument(
        "--token_price_prompt_per_1k",
        type=float,
        default=0.0,
        help="Default prompt/input token price in USD per 1K tokens.",
    )
    parser.add_argument(
        "--token_price_completion_per_1k",
        type=float,
        default=0.0,
        help="Default completion/output token price in USD per 1K tokens.",
    )
    parser.add_argument(
        "--method_pricing_json",
        type=str,
        default="",
        help=(
            "Optional JSON string or JSON file path. Format: "
            "{\"method\": {\"prompt_per_1k\": 0.005, \"completion_per_1k\": 0.015}}"
        ),
    )
    return parser.parse_args()


def _split_methods(raw: str) -> List[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [m for m in methods if m not in METHOD_PRESETS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not methods:
        raise ValueError("No methods selected")
    return methods


def _build_base_eval_cmd(args: argparse.Namespace, report_path: Path, solutions_path: Path) -> List[str]:
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
        "--topology",
        str(args.topology),
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


def _run_eval_for_method(args: argparse.Namespace, method: str, output_dir: Path) -> Path:
    report_path = output_dir / "reports" / f"{method}.json"
    solutions_path = output_dir / "solutions" / f"{method}.solutions.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    solutions_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_base_eval_cmd(args, report_path=report_path, solutions_path=solutions_path)
    cmd.extend(METHOD_PRESETS[method].cli_args)
    subprocess.run(cmd, cwd=str(SRC_ROOT), check=True)
    return report_path


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _extract_direct_usage_tokens(payload: Dict[str, Any]) -> Dict[str, float]:
    prompt = _safe_float(payload.get("prompt_tokens"))
    if prompt is None:
        prompt = _safe_float(payload.get("input_tokens"))
    completion = _safe_float(payload.get("completion_tokens"))
    if completion is None:
        completion = _safe_float(payload.get("output_tokens"))
    total = _safe_float(payload.get("total_tokens"))
    if total is None and (prompt is not None or completion is not None):
        total = float((prompt or 0.0) + (completion or 0.0))
    if prompt is None:
        prompt = 0.0
    if completion is None:
        completion = 0.0
    if total is None:
        total = 0.0
    return {
        "prompt_tokens": float(prompt),
        "completion_tokens": float(completion),
        "total_tokens": float(total),
    }


def _collect_usage_dicts(obj: object, outputs: List[Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        has_token_key = any(
            key in obj
            for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
        )
        if has_token_key:
            outputs.append(obj)
        for key in ("usage", "token_usage", "llm_usage", "model_usage"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                outputs.append(nested)
        for value in obj.values():
            _collect_usage_dicts(value, outputs)
    elif isinstance(obj, list):
        for item in obj:
            _collect_usage_dicts(item, outputs)


def _extract_result_token_usage(item: Dict[str, Any]) -> Dict[str, float]:
    usage_nodes: List[Dict[str, Any]] = []
    _collect_usage_dicts(item, usage_nodes)

    total_prompt = 0.0
    total_completion = 0.0
    total_tokens = 0.0
    for node in usage_nodes:
        stats = _extract_direct_usage_tokens(node)
        if stats["prompt_tokens"] <= 0 and stats["completion_tokens"] <= 0 and stats["total_tokens"] <= 0:
            continue
        total_prompt += stats["prompt_tokens"]
        total_completion += stats["completion_tokens"]
        total_tokens += stats["total_tokens"]

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
    }


def _parse_method_pricing(raw: str) -> Dict[str, Dict[str, float]]:
    if not raw.strip():
        return {}
    raw_text = raw.strip()
    possible_path = Path(raw_text)
    if possible_path.exists():
        raw_text = possible_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("--method_pricing_json must be a JSON object.")

    outputs: Dict[str, Dict[str, float]] = {}
    for method, pricing in payload.items():
        if not isinstance(pricing, dict):
            continue
        prompt_rate = _safe_float(pricing.get("prompt_per_1k"))
        completion_rate = _safe_float(pricing.get("completion_per_1k"))
        outputs[str(method)] = {
            "prompt_per_1k": float(prompt_rate or 0.0),
            "completion_per_1k": float(completion_rate or 0.0),
        }
    return outputs


def _resolve_method_pricing(
    *,
    method: str,
    default_prompt_per_1k: float,
    default_completion_per_1k: float,
    method_pricing: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    pricing = {
        "prompt_per_1k": float(default_prompt_per_1k),
        "completion_per_1k": float(default_completion_per_1k),
    }
    override = method_pricing.get(method) or {}
    if "prompt_per_1k" in override:
        pricing["prompt_per_1k"] = float(override["prompt_per_1k"])
    if "completion_per_1k" in override:
        pricing["completion_per_1k"] = float(override["completion_per_1k"])
    return pricing


def _load_metrics(report_path: Path, *, pricing: Dict[str, float]) -> Dict[str, float]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload.get("results") or []

    generation_times: List[float] = []
    workflow_times: List[float] = []
    eval_times: List[float] = []
    prompt_tokens: List[float] = []
    completion_tokens: List[float] = []
    total_tokens: List[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if "generation_time_s" in item:
            generation_times.append(float(item.get("generation_time_s") or 0.0))
        if "workflow_time_s" in item:
            workflow_times.append(float(item.get("workflow_time_s") or 0.0))
        if "eval_time_s" in item:
            eval_times.append(float(item.get("eval_time_s") or 0.0))
        usage = _extract_result_token_usage(item)
        prompt_tokens.append(float(usage["prompt_tokens"]))
        completion_tokens.append(float(usage["completion_tokens"]))
        total_tokens.append(float(usage["total_tokens"]))

    total = int(payload.get("total") or 0)
    passed = int(payload.get("passed") or 0)
    pass_rate = float(payload.get("pass_rate") or 0.0)

    total_generation = float(sum(generation_times))
    total_workflow = float(sum(workflow_times))
    avg_generation = _safe_mean(generation_times)
    avg_workflow = _safe_mean(workflow_times)
    avg_eval = _safe_mean(eval_times)
    total_prompt_tokens = float(sum(prompt_tokens))
    total_completion_tokens = float(sum(completion_tokens))
    total_all_tokens = float(sum(total_tokens))
    avg_prompt_tokens = _safe_mean(prompt_tokens)
    avg_completion_tokens = _safe_mean(completion_tokens)
    avg_total_tokens = _safe_mean(total_tokens)

    prompt_price = float(pricing.get("prompt_per_1k", 0.0))
    completion_price = float(pricing.get("completion_per_1k", 0.0))
    total_estimated_cost = (total_prompt_tokens / 1000.0) * prompt_price + (
        total_completion_tokens / 1000.0
    ) * completion_price
    avg_estimated_cost = (avg_prompt_tokens / 1000.0) * prompt_price + (
        avg_completion_tokens / 1000.0
    ) * completion_price

    return {
        "total": float(total),
        "passed": float(passed),
        "pass_rate": pass_rate,
        "total_generation_time_s": total_generation,
        "total_workflow_time_s": total_workflow,
        "avg_generation_time_s": avg_generation,
        "avg_workflow_time_s": avg_workflow,
        "avg_eval_time_s": avg_eval,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_all_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "prompt_price_per_1k": prompt_price,
        "completion_price_per_1k": completion_price,
        "total_estimated_cost_usd": float(total_estimated_cost),
        "avg_estimated_cost_usd": float(avg_estimated_cost),
    }


def _pareto_frontier(points: List[Dict[str, float]], x_key: str, y_key: str) -> List[Dict[str, float]]:
    sorted_points = sorted(points, key=lambda item: (item[x_key], -item[y_key]))
    frontier: List[Dict[str, float]] = []
    best_y = -1.0
    for item in sorted_points:
        y = float(item[y_key])
        if y > best_y:
            frontier.append(item)
            best_y = y
    return frontier


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "pass_rate",
        "passed",
        "total",
        "avg_generation_time_s",
        "avg_workflow_time_s",
        "avg_eval_time_s",
        "total_generation_time_s",
        "total_workflow_time_s",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "prompt_price_per_1k",
        "completion_price_per_1k",
        "avg_estimated_cost_usd",
        "total_estimated_cost_usd",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _axis_label(cost_metric: str) -> str:
    label_map = {
        "avg_generation_time_s": "Avg Generation Time (s)",
        "avg_workflow_time_s": "Avg Workflow Time (s)",
        "total_generation_time_s": "Total Generation Time (s)",
        "total_workflow_time_s": "Total Workflow Time (s)",
        "avg_total_tokens": "Avg Total Tokens",
        "total_tokens": "Total Tokens",
        "avg_estimated_cost_usd": "Avg Estimated Cost (USD)",
        "total_estimated_cost_usd": "Total Estimated Cost (USD)",
    }
    return label_map.get(cost_metric, cost_metric)


def _plot_pareto(rows: List[Dict[str, Any]], output_dir: Path, cost_metric: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
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
        except Exception:
            return None

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [float(row[cost_metric]) for row in rows]
    ys = [float(row["pass_rate"]) * 100.0 for row in rows]

    for row, x, y in zip(rows, xs, ys):
        ax.scatter([x], [y], s=120)
        ax.annotate(
            row["method"],
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    frontier = _pareto_frontier(rows, x_key=cost_metric, y_key="pass_rate")
    fx = [float(row[cost_metric]) for row in frontier]
    fy = [float(row["pass_rate"]) * 100.0 for row in frontier]
    if len(frontier) >= 2:
        ax.plot(fx, fy, linestyle="--", linewidth=1.5, label="Pareto Frontier")

    ax.set_title("MBPP Cost-Performance Pareto")
    ax.set_xlabel(_axis_label(cost_metric))
    ax.set_ylabel("Pass@1 (%)")
    ax.grid(alpha=0.3)
    if len(frontier) >= 2:
        ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"pareto_{cost_metric}.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def main() -> None:
    args = _parse_args()
    methods = _split_methods(args.methods)
    output_dir = Path(args.output_dir).resolve()
    reports_dir = output_dir / "reports"
    method_pricing = _parse_method_pricing(args.method_pricing_json)

    if not args.skip_run and not args.local_model_dir:
        raise SystemExit("--local_model_dir is required unless --skip_run is set.")

    rows: List[Dict[str, Any]] = []
    for method in methods:
        if (
            method == "fixed_bcb_router_gpt4o"
            and not args.skip_run
            and not (args.next_role_model or args.next_role_base_url or args.next_role_api_key)
        ):
            print(
                "[WARN] Skipping fixed_bcb_router_gpt4o: remote router config missing "
                "(set --next_role_model/--next_role_base_url/--next_role_api_key).",
                file=sys.stderr,
            )
            continue

        if args.skip_run:
            report_path = reports_dir / f"{method}.json"
            if not report_path.exists():
                raise FileNotFoundError(f"Missing report for method {method}: {report_path}")
        else:
            report_path = _run_eval_for_method(args, method=method, output_dir=output_dir)
        pricing = _resolve_method_pricing(
            method=method,
            default_prompt_per_1k=args.token_price_prompt_per_1k,
            default_completion_per_1k=args.token_price_completion_per_1k,
            method_pricing=method_pricing,
        )
        metrics = _load_metrics(report_path, pricing=pricing)
        row: Dict[str, Any] = dict(metrics)
        row["method"] = method
        row["report_path"] = str(report_path)
        rows.append(row)

    if not rows:
        raise RuntimeError("No method rows produced. Check --methods and baseline config.")

    if args.cost_metric in {"avg_total_tokens", "total_tokens", "avg_estimated_cost_usd", "total_estimated_cost_usd"}:
        all_zero = all(float(row.get(args.cost_metric) or 0.0) <= 0.0 for row in rows)
        if all_zero:
            print(
                f"[WARN] Selected cost_metric={args.cost_metric}, but all values are zero. "
                "This report likely does not include token usage fields.",
                file=sys.stderr,
            )

    csv_path = output_dir / "pareto_summary.csv"
    _save_csv(rows, csv_path)
    fig_path = _plot_pareto(rows, output_dir=output_dir, cost_metric=args.cost_metric)

    summary = {
        "methods": methods,
        "cost_metric": args.cost_metric,
        "csv_path": str(csv_path),
        "figure_path": str(fig_path) if fig_path else "",
        "rows": rows,
    }
    summary_path = output_dir / "pareto_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("csv_path", "figure_path", "cost_metric")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
