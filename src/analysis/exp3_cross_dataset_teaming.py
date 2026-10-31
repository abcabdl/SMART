from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
DEFAULT_DB = REPO_ROOT / "demo_registry.sqlite"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp3_cross_dataset"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exp3 cross-dataset task-oriented teaming: compare Team Size and selected Agent-category distribution "
            "across datasets (e.g., MBPP vs MATH/HumanEval)."
        )
    )
    parser.add_argument(
        "--dataset_specs",
        nargs="+",
        required=True,
        help=(
            "Dataset spec format: '<dataset_label>|<report_path1>[,<report_path2>...]'. "
            "Example: \"MBPP|data/MBPP/mbpp_report (19).json\" "
            "\"MATH|data/Math/report/math_report_recheck_by_eval_math_indexname_after_patch.json\""
        ),
    )
    parser.add_argument("--registry_db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--top_n_domains",
        type=int,
        default=8,
        help="Top-N domains to keep in domain-distribution stacked plot (rest merged into 'other').",
    )
    return parser.parse_args()


def _split_non_empty(raw: str, sep: str) -> List[str]:
    return [item.strip() for item in raw.split(sep) if item.strip()]


def _parse_dataset_specs(specs: Sequence[str]) -> Dict[str, List[Path]]:
    parsed: Dict[str, List[Path]] = {}
    for spec in specs:
        if "|" not in spec:
            raise ValueError(
                f"Invalid dataset spec: {spec}. Expected '<label>|<report1>[,<report2>...]'."
            )
        label, reports_raw = spec.split("|", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Dataset label is empty in spec: {spec}")
        report_items = _split_non_empty(reports_raw, ",")
        if not report_items:
            raise ValueError(f"No report path provided in spec: {spec}")
        paths = [Path(item).resolve() for item in report_items]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Report not found for dataset '{label}': {path}")
        parsed.setdefault(label, [])
        parsed[label].extend(paths)

    deduped: Dict[str, List[Path]] = {}
    for label, paths in parsed.items():
        seen = set()
        uniq: List[Path] = []
        for path in paths:
            text = str(path)
            if text in seen:
                continue
            seen.add(text)
            uniq.append(path)
        deduped[label] = uniq
    return deduped


def _load_agent_domain_map(db_path: Path) -> Dict[str, str]:
    if not db_path.exists():
        return {}
    import sqlite3

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
                    else:
                        domain = raw_domain.strip() or "unknown"
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
    if "math" in text or "number" in text or "algorithm" in text:
        return "math_number"
    if "checker" in text or "test" in text:
        return "checker"
    if "planner" in text or "architect" in text:
        return "planner"
    if "builder" in text or "assemble" in text:
        return "builder"
    if "research" in text:
        return "research"
    if "refactor" in text:
        return "refactor"
    return "other"


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_rows_for_dataset(
    dataset_label: str,
    report_paths: Sequence[Path],
    agent_domain_map: Dict[str, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for report_path in report_paths:
        payload = _read_report(report_path)
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            ok = bool(item.get("ok"))
            topology = item.get("topology") or {}
            topology_roles = topology.get("roles") or []
            selected_agents = item.get("selected_agents") or {}
            if not isinstance(selected_agents, dict):
                selected_agents = {}
            selected_pairs = [(str(role), str(agent)) for role, agent in selected_agents.items() if agent]
            selected_roles = [role for role, _ in selected_pairs]
            selected_agent_names = [agent for _, agent in selected_pairs]
            selected_domains = [
                agent_domain_map.get(agent_name) or _fallback_domain_from_name(agent_name)
                for agent_name in selected_agent_names
            ]
            rows.append(
                {
                    "dataset": dataset_label,
                    "report_path": str(report_path),
                    "task_name": name,
                    "ok": ok,
                    "team_size": len(selected_pairs),
                    "declared_team_size": len(topology_roles) if isinstance(topology_roles, list) else 0,
                    "selected_roles": selected_roles,
                    "selected_agents": selected_agent_names,
                    "selected_domains": selected_domains,
                }
            )
    return rows


def _normalize_counter(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _entropy_from_probs(probs: Iterable[float]) -> float:
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def _max_share_from_counter(counter: Counter[int] | Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return max(counter.values()) / total


def _to_prob_vector(counter: Counter[int] | Counter[str], keys: Sequence[int | str]) -> List[float]:
    total = sum(counter.values())
    if total <= 0:
        return [0.0 for _ in keys]
    return [float(counter.get(key, 0)) / float(total) for key in keys]


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    value = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            continue
        value += pi * math.log(pi / qi, 2)
    return value


def _tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


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


def _save_task_rows(rows: List[Dict[str, object]], path: Path) -> None:
    serialized: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["selected_roles"] = json.dumps(item.get("selected_roles") or [], ensure_ascii=True)
        item["selected_agents"] = json.dumps(item.get("selected_agents") or [], ensure_ascii=True)
        item["selected_domains"] = json.dumps(item.get("selected_domains") or [], ensure_ascii=True)
        serialized.append(item)
    _write_csv(serialized, path)


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


def _plot_team_size_distribution(
    dataset_to_team_size_counter: Dict[str, Counter[int]],
    output_dir: Path,
) -> Path | None:
    plt = _maybe_import_plt()
    if plt is None:
        return None

    datasets = sorted(dataset_to_team_size_counter.keys())
    all_sizes = sorted({size for c in dataset_to_team_size_counter.values() for size in c.keys()})
    if not datasets or not all_sizes:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(datasets))
    x = list(range(len(all_sizes)))
    for idx, dataset in enumerate(datasets):
        counter = dataset_to_team_size_counter[dataset]
        total = sum(counter.values())
        vals = [(counter.get(size, 0) / total) if total > 0 else 0.0 for size in all_sizes]
        offsets = [v + (idx - (len(datasets) - 1) / 2.0) * width for v in x]
        ax.bar(offsets, vals, width=width, label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in all_sizes])
    ax.set_xlabel("Team Size")
    ax.set_ylabel("Proportion")
    ax.set_title("Team Size Distribution Across Datasets")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "cross_dataset_team_size_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def _plot_stacked_distribution(
    dataset_to_counter: Dict[str, Counter[str]],
    output_dir: Path,
    top_n: int,
    title: str,
    y_label: str,
    out_name: str,
) -> Path | None:
    plt = _maybe_import_plt()
    if plt is None:
        return None

    datasets = sorted(dataset_to_counter.keys())
    total_counter: Counter[str] = Counter()
    for dataset in datasets:
        total_counter.update(dataset_to_counter[dataset])
    if not total_counter:
        return None
    top_keys = [k for k, _ in total_counter.most_common(max(1, top_n))]
    keys = list(top_keys) + ["other"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = [0.0 for _ in datasets]
    x = list(range(len(datasets)))
    for key in keys:
        vals: List[float] = []
        for dataset in datasets:
            counter = dataset_to_counter[dataset]
            total = sum(counter.values())
            if total <= 0:
                vals.append(0.0)
                continue
            if key == "other":
                val = sum(v for k, v in counter.items() if k not in top_keys) / total
            else:
                val = counter.get(key, 0) / total
            vals.append(val)
        ax.bar(x, vals, bottom=bottoms, label=key)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / out_name
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def _build_findings_markdown(
    dataset_summary_rows: List[Dict[str, object]],
    pairwise_rows: List[Dict[str, object]],
    path: Path,
) -> None:
    if not dataset_summary_rows:
        path.write_text("# Cross-Dataset Teaming Findings\n\nNo data.\n", encoding="utf-8")
        return
    by_team = sorted(dataset_summary_rows, key=lambda r: float(r["avg_team_size"]), reverse=True)
    by_fixed = sorted(dataset_summary_rows, key=lambda r: float(r["fixed_team_size_ratio"]), reverse=True)
    by_entropy = sorted(dataset_summary_rows, key=lambda r: float(r["domain_entropy_bits"]), reverse=True)

    lines: List[str] = []
    lines.append("# Cross-Dataset Teaming Findings")
    lines.append("")
    lines.append("## Key Statistics")
    lines.append(
        f"- Highest average team size: `{by_team[0]['dataset']}` ({float(by_team[0]['avg_team_size']):.3f})."
    )
    lines.append(
        f"- Most fixed team-size pattern (higher means more template-like): "
        f"`{by_fixed[0]['dataset']}` ({float(by_fixed[0]['fixed_team_size_ratio']):.3f})."
    )
    lines.append(
        f"- Highest selected-domain entropy: `{by_entropy[0]['dataset']}` "
        f"({float(by_entropy[0]['domain_entropy_bits']):.3f} bits)."
    )

    if pairwise_rows:
        strongest_team = max(pairwise_rows, key=lambda r: float(r["team_size_js_divergence"]))
        strongest_domain = max(pairwise_rows, key=lambda r: float(r["domain_js_divergence"]))
        strongest_role = max(pairwise_rows, key=lambda r: float(r["role_js_divergence"]))
        lines.append("")
        lines.append("## Pairwise Distribution Gaps")
        lines.append(
            f"- Largest team-size gap: `{strongest_team['dataset_a']}` vs `{strongest_team['dataset_b']}` "
            f"(JS={float(strongest_team['team_size_js_divergence']):.4f})."
        )
        lines.append(
            f"- Largest domain-mix gap: `{strongest_domain['dataset_a']}` vs `{strongest_domain['dataset_b']}` "
            f"(JS={float(strongest_domain['domain_js_divergence']):.4f})."
        )
        lines.append(
            f"- Largest role-mix gap: `{strongest_role['dataset_a']}` vs `{strongest_role['dataset_b']}` "
            f"(JS={float(strongest_role['role_js_divergence']):.4f})."
        )

        adaptive_score = max(
            float(strongest_team["team_size_js_divergence"]),
            float(strongest_domain["domain_js_divergence"]),
            float(strongest_role["role_js_divergence"]),
        )
        lines.append("")
        lines.append("## Interpretation")
        if adaptive_score >= 0.08:
            lines.append(
                "- Distribution divergence is clearly non-trivial, supporting task-oriented dynamic teaming."
            )
        elif adaptive_score >= 0.03:
            lines.append(
                "- Distribution divergence exists but is moderate; dynamic teaming signal is present."
            )
        else:
            lines.append(
                "- Distribution divergence is weak; current runs show limited evidence against fixed templates."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    dataset_to_reports = _parse_dataset_specs(args.dataset_specs)
    db_path = Path(args.registry_db).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_domain_map = _load_agent_domain_map(db_path)

    all_rows: List[Dict[str, object]] = []
    for dataset, report_paths in dataset_to_reports.items():
        all_rows.extend(
            _extract_rows_for_dataset(
                dataset_label=dataset,
                report_paths=report_paths,
                agent_domain_map=agent_domain_map,
            )
        )
    if not all_rows:
        raise RuntimeError("No task rows extracted from provided reports.")

    task_rows_path = output_dir / "cross_dataset_task_rows.csv"
    _save_task_rows(all_rows, task_rows_path)

    dataset_to_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        dataset_to_rows[str(row["dataset"])].append(row)

    dataset_summary_rows: List[Dict[str, object]] = []
    dataset_to_team_size_counter: Dict[str, Counter[int]] = {}
    dataset_to_domain_counter: Dict[str, Counter[str]] = {}
    dataset_to_role_counter: Dict[str, Counter[str]] = {}

    team_size_distribution_rows: List[Dict[str, object]] = []
    domain_distribution_rows: List[Dict[str, object]] = []
    role_distribution_rows: List[Dict[str, object]] = []

    for dataset in sorted(dataset_to_rows.keys()):
        rows = dataset_to_rows[dataset]
        team_sizes = [int(r["team_size"]) for r in rows]
        declared_sizes = [int(r["declared_team_size"]) for r in rows]
        ok_flags = [1 if bool(r["ok"]) else 0 for r in rows]

        team_size_counter: Counter[int] = Counter(team_sizes)
        domain_counter: Counter[str] = Counter()
        role_counter: Counter[str] = Counter()
        for row in rows:
            for domain in row.get("selected_domains") or []:
                domain_counter[str(domain)] += 1
            for role in row.get("selected_roles") or []:
                role_counter[str(role)] += 1

        dataset_to_team_size_counter[dataset] = team_size_counter
        dataset_to_domain_counter[dataset] = domain_counter
        dataset_to_role_counter[dataset] = role_counter

        team_probs = _normalize_counter(Counter({str(k): v for k, v in team_size_counter.items()}))
        domain_probs = _normalize_counter(domain_counter)
        role_probs = _normalize_counter(role_counter)

        dataset_summary_rows.append(
            {
                "dataset": dataset,
                "num_reports": len(dataset_to_reports.get(dataset) or []),
                "num_tasks": len(rows),
                "pass_rate": round(mean(ok_flags), 6) if ok_flags else 0.0,
                "avg_team_size": round(mean(team_sizes), 6) if team_sizes else 0.0,
                "avg_declared_team_size": round(mean(declared_sizes), 6) if declared_sizes else 0.0,
                "fixed_team_size_ratio": round(_max_share_from_counter(team_size_counter), 6),
                "team_size_entropy_bits": round(_entropy_from_probs(team_probs.values()), 6),
                "domain_entropy_bits": round(_entropy_from_probs(domain_probs.values()), 6),
                "role_entropy_bits": round(_entropy_from_probs(role_probs.values()), 6),
                "num_unique_domains": len(domain_counter),
                "num_unique_roles": len(role_counter),
            }
        )

        total_team = sum(team_size_counter.values())
        for team_size, count in sorted(team_size_counter.items(), key=lambda kv: kv[0]):
            team_size_distribution_rows.append(
                {
                    "dataset": dataset,
                    "team_size": team_size,
                    "count": count,
                    "proportion": round((count / total_team) if total_team > 0 else 0.0, 6),
                }
            )

        total_domain = sum(domain_counter.values())
        for domain, count in sorted(domain_counter.items(), key=lambda kv: kv[1], reverse=True):
            domain_distribution_rows.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "count": count,
                    "proportion": round((count / total_domain) if total_domain > 0 else 0.0, 6),
                }
            )

        total_role = sum(role_counter.values())
        for role, count in sorted(role_counter.items(), key=lambda kv: kv[1], reverse=True):
            role_distribution_rows.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "count": count,
                    "proportion": round((count / total_role) if total_role > 0 else 0.0, 6),
                }
            )

    pairwise_rows: List[Dict[str, object]] = []
    datasets = sorted(dataset_to_rows.keys())
    for a, b in combinations(datasets, 2):
        team_keys = sorted(
            set(dataset_to_team_size_counter[a].keys()) | set(dataset_to_team_size_counter[b].keys())
        )
        team_p = _to_prob_vector(dataset_to_team_size_counter[a], team_keys)
        team_q = _to_prob_vector(dataset_to_team_size_counter[b], team_keys)

        domain_keys = sorted(
            set(dataset_to_domain_counter[a].keys()) | set(dataset_to_domain_counter[b].keys())
        )
        domain_p = _to_prob_vector(dataset_to_domain_counter[a], domain_keys)
        domain_q = _to_prob_vector(dataset_to_domain_counter[b], domain_keys)

        role_keys = sorted(set(dataset_to_role_counter[a].keys()) | set(dataset_to_role_counter[b].keys()))
        role_p = _to_prob_vector(dataset_to_role_counter[a], role_keys)
        role_q = _to_prob_vector(dataset_to_role_counter[b], role_keys)

        pairwise_rows.append(
            {
                "dataset_a": a,
                "dataset_b": b,
                "team_size_js_divergence": round(_js_divergence(team_p, team_q), 6),
                "team_size_tv_distance": round(_tv_distance(team_p, team_q), 6),
                "domain_js_divergence": round(_js_divergence(domain_p, domain_q), 6),
                "domain_tv_distance": round(_tv_distance(domain_p, domain_q), 6),
                "role_js_divergence": round(_js_divergence(role_p, role_q), 6),
                "role_tv_distance": round(_tv_distance(role_p, role_q), 6),
            }
        )

    dataset_summary_path = output_dir / "dataset_summary.csv"
    team_dist_path = output_dir / "team_size_distribution.csv"
    domain_dist_path = output_dir / "agent_domain_distribution.csv"
    role_dist_path = output_dir / "selected_role_distribution.csv"
    pairwise_path = output_dir / "pairwise_divergence.csv"
    findings_path = output_dir / "cross_dataset_findings.md"

    _write_csv(dataset_summary_rows, dataset_summary_path)
    _write_csv(team_size_distribution_rows, team_dist_path)
    _write_csv(domain_distribution_rows, domain_dist_path)
    _write_csv(role_distribution_rows, role_dist_path)
    _write_csv(pairwise_rows, pairwise_path)

    team_size_plot = _plot_team_size_distribution(dataset_to_team_size_counter, output_dir=output_dir)
    domain_plot = _plot_stacked_distribution(
        dataset_to_counter=dataset_to_domain_counter,
        output_dir=output_dir,
        top_n=args.top_n_domains,
        title="Selected Agent Category (Domain) Mix Across Datasets",
        y_label="Proportion",
        out_name="cross_dataset_agent_domain_mix.png",
    )
    role_plot = _plot_stacked_distribution(
        dataset_to_counter=dataset_to_role_counter,
        output_dir=output_dir,
        top_n=12,
        title="Selected Role Mix Across Datasets",
        y_label="Proportion",
        out_name="cross_dataset_role_mix.png",
    )

    _build_findings_markdown(dataset_summary_rows, pairwise_rows, findings_path)

    summary = {
        "dataset_specs": {
            dataset: [str(path) for path in paths] for dataset, paths in dataset_to_reports.items()
        },
        "registry_db": str(db_path),
        "num_rows": len(all_rows),
        "task_rows_csv": str(task_rows_path),
        "dataset_summary_csv": str(dataset_summary_path),
        "team_size_distribution_csv": str(team_dist_path),
        "domain_distribution_csv": str(domain_dist_path),
        "role_distribution_csv": str(role_dist_path),
        "pairwise_divergence_csv": str(pairwise_path),
        "team_size_plot": str(team_size_plot) if team_size_plot else "",
        "domain_mix_plot": str(domain_plot) if domain_plot else "",
        "role_mix_plot": str(role_plot) if role_plot else "",
        "findings_markdown": str(findings_path),
    }
    summary_path = output_dir / "cross_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset_summary_csv": summary["dataset_summary_csv"],
                "pairwise_divergence_csv": summary["pairwise_divergence_csv"],
                "team_size_plot": summary["team_size_plot"],
                "domain_mix_plot": summary["domain_mix_plot"],
                "role_mix_plot": summary["role_mix_plot"],
                "findings_markdown": summary["findings_markdown"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
