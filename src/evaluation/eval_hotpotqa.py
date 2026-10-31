from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from evaluation.smart_pdf_preset import apply_smart_pdf_preset
from execution.orchestrator import run_workflow
from retrieval.embedder import build_embedder
from retrieval.faiss_index import HNSWIndex


ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _resolve_repo_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())
    return str((REPO_ROOT / path).resolve())


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit and len(rows) >= limit:
                break
    return rows


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = ARTICLES_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _support_title_coverage(result: Dict[str, Any], task: Dict[str, Any]) -> float:
    gold_titles = {
        _normalize_answer(item[0])
        for item in task.get("supporting_facts", [])
        if isinstance(item, list) and item
    }
    if not gold_titles:
        return 0.0
    seen_titles = set()
    for branch in (result.get("parallel") or {}).get("branches", []):
        artifact = branch.get("artifact_bag") if isinstance(branch, dict) else {}
        for point in (artifact or {}).get("evidence_points", []):
            if isinstance(point, dict) and point.get("title"):
                seen_titles.add(_normalize_answer(point.get("title")))
    for trace in result.get("execution_trace") or []:
        artifact = trace.get("artifact_bag") if isinstance(trace, dict) else {}
        for point in (artifact or {}).get("evidence_points", []):
            if isinstance(point, dict) and point.get("title"):
                seen_titles.add(_normalize_answer(point.get("title")))
    return len(gold_titles & seen_titles) / max(1, len(gold_titles))


def _predicted_supporting_facts(result: Dict[str, Any]) -> set[tuple[str, int]]:
    predicted: set[tuple[str, int]] = set()
    containers: List[Any] = []
    containers.extend((result.get("parallel") or {}).get("branches", []) if isinstance(result.get("parallel"), dict) else [])
    containers.extend(result.get("execution_trace") or [])
    for item in containers:
        artifact = item.get("artifact_bag") if isinstance(item, dict) else {}
        for point in (artifact or {}).get("evidence_points", []):
            if not isinstance(point, dict) or not point.get("title"):
                continue
            try:
                sent_idx = int(point.get("sentence_index") or 0)
            except Exception:
                sent_idx = 0
            predicted.add((_normalize_answer(point.get("title")), sent_idx))
    return predicted


def _gold_supporting_facts(task: Dict[str, Any]) -> set[tuple[str, int]]:
    facts: set[tuple[str, int]] = set()
    for item in task.get("supporting_facts", []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            sent_idx = int(item[1])
        except Exception:
            sent_idx = 0
        facts.add((_normalize_answer(item[0]), sent_idx))
    return facts


def _support_metrics(result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, float]:
    pred = _predicted_supporting_facts(result)
    gold = _gold_supporting_facts(task)
    if not pred and not gold:
        return {"supporting_fact_em": 1.0, "supporting_fact_f1": 1.0}
    if not pred or not gold:
        return {"supporting_fact_em": 0.0, "supporting_fact_f1": 0.0}
    overlap = len(pred & gold)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"supporting_fact_em": 1.0 if pred == gold else 0.0, "supporting_fact_f1": f1}


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = _resolve_repo_path(args.data)
    db_path = _resolve_repo_path(args.db)
    index_path = Path(_resolve_repo_path(args.index_dir)) / "faiss.index"
    tasks = _load_jsonl(data_path, limit=args.limit if args.limit > 0 else None)

    registry = SQLiteRegistry(db_path)
    index = HNSWIndex.load(str(index_path), use_faiss=False)
    embedder = build_embedder(kind=args.embedder, dim=index.dim, seed=args.seed)
    records: List[Dict[str, Any]] = []
    try:
        for idx, task in enumerate(tasks):
            started = time.perf_counter()
            constraints = {
                role: {"domain_tags": ["hotpotqa"], "role_tags": [role]}
                for role in ("planner", "builder", "checker")
            }
            inference_visibility = "public_only" if args.hidden_test_clean else args.inference_visibility
            result = run_workflow(
                task_text=str(task.get("question") or ""),
                roles=["planner", "builder", "checker"],
                constraints_per_role=constraints,
                registry=registry,
                index=index,
                embedder=embedder,
                dataset="hotpotqa",
                task_family="qa",
                task_context={**task, "dataset": "hotpotqa"},
                inference_visibility=inference_visibility,
                use_cross_instance_cache=bool(args.use_cross_instance_cache),
                update_posteriors_across_instances=bool(args.update_posteriors_across_instances),
                use_mmr=not bool(args.no_mmr),
                use_iochain=not bool(args.no_iochain),
                use_transition_term=not bool(args.no_transition_term),
                use_gap_term=not bool(args.no_gap_term),
                reward_variant=args.reward_variant,
                use_parallel_frontier=bool(args.use_parallel_frontier),
                parallel_mode=args.parallel_mode,
                parallel_execution=bool(args.parallel),
                parallel_width=int(args.parallel_width),
                execute_tools=bool(args.execute_tools),
                tool_timeout_s=float(args.tool_timeout),
                tool_only=bool(args.tool_only),
                paper_mode=bool(args.paper_mode),
                runs_dir=args.runs_dir,
                parallel_fail_fast=False,
            )
            prediction = result.get("answer", "")
            gold = task.get("answer", "")
            exact = _normalize_answer(prediction) == _normalize_answer(gold)
            support = _support_metrics(result, task)
            records.append(
                {
                    "idx": idx,
                    "id": task.get("_id") or task.get("id"),
                    "question": task.get("question"),
                    "prediction": prediction,
                    "gold": gold,
                    "exact_match": exact,
                    "f1": _token_f1(prediction, gold),
                    "supporting_fact_em": support["supporting_fact_em"],
                    "supporting_fact_f1": support["supporting_fact_f1"],
                    "support_title_coverage": _support_title_coverage(result, task),
                    "wall_clock_s": round(time.perf_counter() - started, 4),
                    "task_id": result.get("task_id"),
                }
            )
    finally:
        registry.close()

    count = max(1, len(records))
    summary = {
        "data": data_path,
        "count": len(records),
        "exact_match": sum(1 for item in records if item["exact_match"]) / count,
        "f1": sum(float(item["f1"]) for item in records) / count,
        "supporting_fact_em": sum(float(item["supporting_fact_em"]) for item in records) / count,
        "supporting_fact_f1": sum(float(item["supporting_fact_f1"]) for item in records) / count,
        "support_title_coverage": sum(float(item["support_title_coverage"]) for item in records) / count,
        "avg_wall_clock_s": sum(float(item["wall_clock_s"]) for item in records) / count,
        "records": records,
    }
    if args.out:
        out_path = Path(_resolve_repo_path(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA appendix diagnostic through Agent-router")
    parser.add_argument("--data", default="data/datas/hotpotqa_validate.jsonl")
    parser.add_argument("--db", default="real_dataset_registry.sqlite")
    parser.add_argument("--index_dir", default="index_real_dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--use_parallel_frontier", action="store_true")
    parser.add_argument("--parallel_mode", default="frontier")
    parser.add_argument("--parallel_width", type=int, default=4)
    parser.add_argument("--execute_tools", action="store_true")
    parser.add_argument("--tool_timeout", type=float, default=60.0)
    parser.add_argument("--tool_only", action="store_true")
    parser.add_argument("--paper_mode", action="store_true", help="appendix diagnostic mode only; HotpotQA is not a paper main benchmark")
    parser.add_argument("--out", default="")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--embedder", default="dummy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inference_visibility", default="tool_visible", choices=["tool_visible", "public_only"])
    parser.add_argument("--hidden_test_clean", action="store_true")
    parser.add_argument("--use_cross_instance_cache", action="store_true")
    parser.add_argument("--update_posteriors_across_instances", action="store_true")
    parser.add_argument("--no_mmr", action="store_true")
    parser.add_argument("--no_iochain", action="store_true")
    parser.add_argument("--no_transition_term", action="store_true")
    parser.add_argument("--no_gap_term", action="store_true")
    parser.add_argument("--reward_variant", default="mean_active")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_smart_pdf_preset(args, benchmark="hotpotqa_appendix")
    summary = evaluate(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
