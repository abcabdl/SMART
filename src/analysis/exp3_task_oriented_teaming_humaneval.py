from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
SRC_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "datas" / "mbpp_test.jsonl"
DEFAULT_DB = REPO_ROOT / "demo_registry.sqlite"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp3_teaming"


@dataclass(frozen=True)
class MethodPreset:
    name: str
    cli_args: Sequence[str]


METHOD_PRESETS: Dict[str, MethodPreset] = {
    "dynamic_mcts": MethodPreset(
        name="dynamic_mcts",
        cli_args=(
            "--use_orchestrator",
            "--dynamic_topology",
            "--mcts_dynamic_optimization",
            "--topology",
            "auto",
        ),
    ),
    "dynamic_router_local": MethodPreset(
        name="dynamic_router_local",
        cli_args=(
            "--use_orchestrator",
            "--dynamic_topology",
            "--topology",
            "auto",
        ),
    ),
}


TASK_FAMILY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "string": ("string", "char", "substring", "text", "alphabet", "case", "token"),
    "math_number": ("number", "integer", "prime", "factor", "digit", "binary", "sum", "mod", "gcd"),
    "list_array": ("list", "array", "sequence", "tuple", "permutation", "sorted", "elements"),
    "dict_set": ("dict", "dictionary", "set", "mapping", "key", "value"),
    "graph_tree": ("graph", "tree", "node", "edge", "path"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp3: Task-oriented teaming analysis on MBPP."
    )
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--registry_db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--method", type=str, default="dynamic_mcts", choices=sorted(METHOD_PRESETS.keys()))
    parser.add_argument("--run_eval", action="store_true", help="Run eval_mbpp before analysis.")
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
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
    ]
    if args.include_challenge_tests:
        cmd.append("--include_challenge_tests")
    if int(args.mbpp_max_samples) > 0:
        cmd += ["--max_samples", str(int(args.mbpp_max_samples))]
    if args.mbpp_converted_tasks:
        cmd += ["--converted_tasks", str(Path(args.mbpp_converted_tasks).resolve())]
    if args.mbpp_keep_converted:
        cmd.append("--keep_converted")
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
    cmd.extend(METHOD_PRESETS[args.method].cli_args)
    return cmd


def _run_eval(args: argparse.Namespace, output_dir: Path) -> Path:
    report_path = output_dir / f"{args.method}.json"
    solutions_path = output_dir / f"{args.method}.solutions.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_eval_cmd(args, report_path=report_path, solutions_path=solutions_path)
    subprocess.run(cmd, cwd=str(SRC_ROOT), check=True)
    return report_path


def _name_to_task_id(report_name: str) -> str:
    text = str(report_name or "")
    if "/" in text:
        return text
    mbpp = re.match(r"^MBPP_(\d+)(?:_.*)?$", text)
    if mbpp:
        return str(int(mbpp.group(1)))
    m = re.match(r"^HumanEval_(\d+)(?:_.*)?$", text)
    if m:
        return f"HumanEval/{m.group(1)}"
    return text


def _build_prompt_map(tasks_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in _load_jsonl(tasks_path):
        task_id_raw = row.get("task_id")
        name_raw = row.get("name")
        task_id = str(task_id_raw or name_raw or "").strip()
        if not task_id:
            continue
        prompt = str(row.get("prompt") or row.get("text") or row.get("question") or "")
        mapping[task_id] = prompt
        if str(task_id).isdigit():
            mapping[f"MBPP_{int(task_id)}"] = prompt
        if name_raw:
            mapping[str(name_raw)] = prompt
    return mapping


def _task_family(prompt: str, report_name: str) -> str:
    text = (prompt or report_name or "").lower()
    for family, words in TASK_FAMILY_KEYWORDS.items():
        if any(word in text for word in words):
            return family
    return "other"


def _load_agent_domain_map(db_path: Path) -> Dict[str, str]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    result: Dict[str, str] = {}
    try:
        cursor = conn.execute(
            "SELECT name, domain_tags FROM cards WHERE kind = 'agent' AND name IS NOT NULL"
        )
        for row in cursor.fetchall():
            name = str(row["name"] or "").strip()
            if not name:
                continue
            raw_domain = row["domain_tags"]
            domain = "unknown"
            if isinstance(raw_domain, str) and raw_domain.strip():
                try:
                    parsed = json.loads(raw_domain)
                    if isinstance(parsed, list) and parsed:
                        domain = str(parsed[0]).strip() or "unknown"
                except Exception:
                    domain = raw_domain.strip() or "unknown"
            result[name] = domain
    finally:
        conn.close()
    return result


def _fallback_domain_from_name(name: str) -> str:
    text = (name or "").lower()
    if "string" in text:
        return "string"
    if "math" in text or "algorithm" in text or "number" in text:
        return "math_number"
    if "checker" in text or "test" in text:
        return "checker"
    if "research" in text:
        return "research"
    if "refactor" in text:
        return "refactor"
    if "builder" in text:
        return "builder"
    return "other"


def _extract_task_rows(report_path: Path, tasks_path: Path, db_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    prompt_map = _build_prompt_map(tasks_path)
    agent_domain_map = _load_agent_domain_map(db_path)
    rows: List[Dict[str, object]] = []

    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        task_id = _name_to_task_id(name)
        prompt = prompt_map.get(task_id, "")
        family = _task_family(prompt, name)
        selected_agents = item.get("selected_agents") or {}
        topology = item.get("topology") or {}
        topology_roles = topology.get("roles") or []
        team_size = sum(1 for _, v in selected_agents.items() if v)
        declared_team_size = len(topology_roles) if isinstance(topology_roles, list) else 0

        selected_domains: List[str] = []
        if isinstance(selected_agents, dict):
            for _, agent_name in selected_agents.items():
                if not agent_name:
                    continue
                agent_name_text = str(agent_name)
                domain = agent_domain_map.get(agent_name_text) or _fallback_domain_from_name(agent_name_text)
                selected_domains.append(domain)

        rows.append(
            {
                "name": name,
                "task_id": task_id,
                "family": family,
                "ok": bool(item.get("ok")),
                "team_size": team_size,
                "declared_team_size": declared_team_size,
                "selected_domains": selected_domains,
            }
        )
    return rows


def _save_task_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "task_id", "family", "ok", "team_size", "declared_team_size", "selected_domains"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dumped = dict(row)
            dumped["selected_domains"] = json.dumps(row.get("selected_domains") or [], ensure_ascii=True)
            writer.writerow(dumped)


def _save_family_summary(rows: List[Dict[str, object]], path: Path) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["family"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for family, family_rows in sorted(buckets.items(), key=lambda item: item[0]):
        sizes = [int(item["team_size"]) for item in family_rows]
        ok_flags = [1 if bool(item["ok"]) else 0 for item in family_rows]
        summary_rows.append(
            {
                "family": family,
                "count": len(family_rows),
                "avg_team_size": round(mean(sizes), 4) if sizes else 0.0,
                "pass_rate": round(mean(ok_flags), 4) if ok_flags else 0.0,
            }
        )

    fieldnames = ["family", "count", "avg_team_size", "pass_rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_rows


def _plot_team_size_hist(rows: List[Dict[str, object]], output_dir: Path) -> Path | None:
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

    team_sizes = [int(row["team_size"]) for row in rows]
    max_size = max(team_sizes) if team_sizes else 1
    bins = list(range(0, max_size + 2))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(team_sizes, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_title("Team Size Distribution (selected non-null agents)")
    ax.set_xlabel("Team Size")
    ax.set_ylabel("Task Count")
    ax.set_xticks(list(range(0, max_size + 1)))
    ax.grid(axis="y", alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "team_size_hist.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def _plot_family_teaming(rows: List[Dict[str, object]], output_dir: Path) -> Path | None:
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

    families = sorted({str(row["family"]) for row in rows})
    family_to_rows: Dict[str, List[Dict[str, object]]] = {f: [] for f in families}
    for row in rows:
        family_to_rows[str(row["family"])].append(row)

    avg_team = [mean([int(item["team_size"]) for item in family_to_rows[f]]) for f in families]

    domain_counts_by_family: Dict[str, Dict[str, int]] = {f: {} for f in families}
    for family in families:
        for item in family_to_rows[family]:
            for domain in item.get("selected_domains") or []:
                domain_text = str(domain)
                domain_counts_by_family[family][domain_text] = domain_counts_by_family[family].get(domain_text, 0) + 1

    domain_totals: Dict[str, int] = {}
    for family in families:
        for domain, count in domain_counts_by_family[family].items():
            domain_totals[domain] = domain_totals.get(domain, 0) + count
    top_domains = [d for d, _ in sorted(domain_totals.items(), key=lambda kv: kv[1], reverse=True)[:6]]
    if not top_domains:
        top_domains = ["unknown"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.bar(families, avg_team)
    ax1.set_title("Avg Team Size by Task Family")
    ax1.set_xlabel("Task Family")
    ax1.set_ylabel("Average Team Size")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    bottoms = [0.0] * len(families)
    for domain in top_domains:
        vals: List[float] = []
        for family in families:
            total = sum(domain_counts_by_family[family].values())
            count = domain_counts_by_family[family].get(domain, 0)
            vals.append((count / total) if total > 0 else 0.0)
        ax2.bar(families, vals, bottom=bottoms, label=domain)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax2.set_title("Selected Agent Domain Mix by Task Family")
    ax2.set_xlabel("Task Family")
    ax2.set_ylabel("Proportion")
    ax2.tick_params(axis="x", rotation=20)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "family_teaming.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return fig_path


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    tasks_path = Path(args.tasks).resolve()
    db_path = Path(args.registry_db).resolve()

    if args.run_eval and not args.local_model_dir:
        raise SystemExit("--local_model_dir is required when --run_eval is set.")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    if args.run_eval:
        report_path = _run_eval(args, output_dir=output_dir)
    else:
        if not args.report_path:
            raise SystemExit("Provide --report_path when --run_eval is not set.")
        report_path = Path(args.report_path).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    rows = _extract_task_rows(report_path=report_path, tasks_path=tasks_path, db_path=db_path)
    if not rows:
        raise RuntimeError("No task rows extracted from report.")
    task_csv_path = output_dir / "task_teaming_rows.csv"
    family_csv_path = output_dir / "family_summary.csv"
    _save_task_csv(rows, task_csv_path)
    family_summary = _save_family_summary(rows, family_csv_path)
    hist_path = _plot_team_size_hist(rows, output_dir=output_dir)
    family_plot_path = _plot_family_teaming(rows, output_dir=output_dir)

    summary = {
        "report_path": str(report_path),
        "task_rows_csv": str(task_csv_path),
        "family_summary_csv": str(family_csv_path),
        "team_size_hist": str(hist_path) if hist_path else "",
        "family_teaming_plot": str(family_plot_path) if family_plot_path else "",
        "num_tasks": len(rows),
        "family_summary": family_summary,
    }
    summary_path = output_dir / "teaming_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("task_rows_csv", "family_summary_csv", "team_size_hist", "family_teaming_plot")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
