from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.smart_pdf_preset import load_smart_pdf_preset


def _resolve_output_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "repo":  # legacy "" prefix
        raw = Path(*raw.parts[1:])
    return ROOT / raw


def _registry_summary(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cards WHERE kind='agent'")
    agent_count = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM cards WHERE kind='tool'")
    tool_count = int(cur.fetchone()[0])
    cur.execute("SELECT dataset_family, COUNT(*) FROM cards WHERE kind='agent' GROUP BY dataset_family ORDER BY dataset_family")
    dataset_distribution = {row[0]: int(row[1]) for row in cur.fetchall()}
    cur.execute("SELECT primary_role, COUNT(*) FROM cards WHERE kind='agent' GROUP BY primary_role ORDER BY primary_role")
    role_distribution = {row[0]: int(row[1]) for row in cur.fetchall()}
    conn.close()
    return {
        "agent_count": agent_count,
        "tool_count": tool_count,
        "dataset_distribution": dataset_distribution,
        "role_distribution": role_distribution,
    }


def generate_alignment_report() -> Dict[str, Any]:
    preset = load_smart_pdf_preset()
    retrieval = dict(preset.get("retrieval") or {})
    mcts = dict(preset.get("mcts") or {})
    paths = dict(preset.get("paths") or {})
    db_path = _resolve_output_path(str(paths.get("benchmark_db") or "paper_registry.sqlite"))
    out_path = _resolve_output_path(str(paths.get("alignment_report") or "reports/paper_alignment_report.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = _registry_summary(db_path)
    report = {
        "source_of_truth": preset.get("source_of_truth"),
        "registry_counts": registry,
        "expected": {
            "agent_count": preset.get("agent_pool_size"),
            "tool_count": preset.get("total_tools"),
            "tools_per_agent": preset.get("tools_per_agent"),
            "datasets": preset.get("datasets"),
            "roles": preset.get("roles"),
        },
        "retrieval_flags": {
            "top_n": retrieval.get("top_n"),
            "top_k": retrieval.get("top_k"),
            "mmr_lambda": retrieval.get("mmr_lambda"),
            "artifact_aware_query": retrieval.get("artifact_aware_query"),
            "experimental_reranker_after_mmr": retrieval.get("experimental_reranker_after_mmr"),
        },
        "mcts_flags": {
            "iterations": mcts.get("iterations"),
            "rollout_depth": mcts.get("rollout_depth"),
            "exploration": mcts.get("exploration"),
            "discount": mcts.get("discount"),
            "max_candidates": mcts.get("max_candidates"),
            "reward_variant": mcts.get("reward_variant"),
        },
        "checks": {
            "agent_count_matches": registry["agent_count"] == int(preset.get("agent_pool_size") or 0),
            "tool_count_matches": registry["tool_count"] == int(preset.get("total_tools") or 0),
            "hotpotqa_excluded_from_main_registry": "hotpotqa" not in registry["dataset_distribution"],
            "reranker_disabled": retrieval.get("experimental_reranker_after_mmr") is False,
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate_alignment_report(), ensure_ascii=True, indent=2))
