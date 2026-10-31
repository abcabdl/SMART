from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure `core`, `retrieval`, etc. are importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from execution.orchestrator import run_workflow
from generation.llm_client import LLMClient
from retrieval.build_index import build_index
from retrieval.embedder import build_embedder
from retrieval.faiss_index import HNSWIndex
from routing.export_training_data import export_training_data
from routing.train_router import train_reranker


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_id_map(index_dir: str) -> List[str]:
    path = os.path.join(index_dir, "id_map.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return []
    try:
        ordered_keys = sorted(raw.keys(), key=lambda k: int(k))
    except ValueError:
        ordered_keys = sorted(raw.keys())
    return [raw[key] for key in ordered_keys]


def _load_queries(path: Optional[str], inline: Optional[str]) -> List[str]:
    queries: List[str] = []
    if inline:
        queries.extend(_parse_list(inline))
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.endswith(".jsonl"):
                    record = json.loads(line)
                    if isinstance(record, dict):
                        text = record.get("query") or record.get("task") or record.get("text")
                        if text:
                            queries.append(str(text))
                            continue
                queries.append(line)
    return queries


def _normalize_constraints(raw: str) -> Dict[str, Dict[str, object]]:
    if not raw:
        return {}
    return json.loads(raw)


def _clear_runs(runs_dir: str) -> None:
    os.makedirs(runs_dir, exist_ok=True)
    for name in os.listdir(runs_dir):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(runs_dir, name)
        try:
            os.remove(path)
        except OSError:
            continue


def _load_checkpoint(path: str) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items() if str(k).isdigit()}


def _save_checkpoint(path: str, checkpoint: Dict[str, int]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate runs and train reranker")
    parser.add_argument("--queries_file", type=str, default=None, help="txt or jsonl with queries")
    parser.add_argument("--queries", type=str, default="", help="comma-separated queries")
    parser.add_argument("--max_queries", type=int, default=0, help="limit number of queries")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--db", type=str, default="demo_registry.sqlite")
    parser.add_argument("--index_dir", type=str, default="index")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", type=str, default="sentence-transformer", help="dummy|sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embedder_device", type=str, default=None)
    parser.add_argument("--embedder_normalize", action="store_true")
    parser.add_argument("--roles", type=str, default="planner,builder,checker,refactor")
    parser.add_argument("--constraints", type=str, default="", help="JSON string per role")
    parser.add_argument("--workflow_version", type=str, default="v1")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rerank_top_m", type=int, default=3)
    parser.add_argument("--mmr_lambda", type=float, default=0.5)
    parser.add_argument("--execute_tools", action="store_true")
    parser.add_argument("--tool_timeout", type=float, default=1.0)
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_timeout", type=float, default=60.0)
    parser.add_argument("--llm_retries", type=int, default=5)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--train_out", type=str, default="models/reranker.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--use_prev_reranker", action="store_true")
    parser.add_argument("--no_prev_reranker", action="store_true")
    parser.add_argument("--reset_runs", action="store_true", help="clear runs/*.jsonl before each iteration")
    parser.add_argument("--query_retries", type=int, default=5, help="retries per query on failure")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="base backoff seconds between retries")
    parser.add_argument("--skip_failures", action="store_true", help="skip failed queries instead of exiting")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint_file", type=str, default="runs/batch_checkpoint.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = _load_queries(args.queries_file, args.queries)
    if not queries:
        raise ValueError("No queries provided. Use --queries or --queries_file.")
    if args.shuffle:
        random.shuffle(queries)
    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]

    if args.rebuild_index or not os.path.exists(os.path.join(args.index_dir, "faiss.index")):
        build_index(
            db_path=args.db,
            kind="agent",
            out_dir=args.index_dir,
            dim=args.dim,
            seed=args.seed,
            embedder_kind=args.embedder,
            embedder_model=args.embedder_model,
            embedder_device=args.embedder_device,
            embedder_normalize=args.embedder_normalize,
        )

    index = HNSWIndex.load(os.path.join(args.index_dir, "faiss.index"))
    id_map = _load_id_map(args.index_dir)
    if id_map:
        index.ids = list(id_map)
    embedder = build_embedder(
        kind=args.embedder,
        dim=args.dim,
        seed=args.seed,
        model_name=args.embedder_model,
        device=args.embedder_device,
        normalize=args.embedder_normalize,
    )
    if embedder.dim != index.dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} != index dim {index.dim}. "
            "Rebuild index with matching embedder/model."
        )

    llm = LLMClient(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout_s=args.llm_timeout,
        max_retries=args.llm_retries,
    )

    roles = _parse_list(args.roles)
    constraints = _normalize_constraints(args.constraints)

    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    use_prev = (args.use_prev_reranker or args.iterations > 1) and not args.no_prev_reranker

    checkpoint = _load_checkpoint(args.checkpoint_file) if args.resume else {}

    for iteration in range(1, max(1, args.iterations) + 1):
        if args.reset_runs:
            _clear_runs(args.runs_dir)

        reranker_path = args.train_out if use_prev and os.path.exists(args.train_out) else None
        with SQLiteRegistry(args.db) as registry:
            for idx, query in enumerate(queries, start=1):
                if args.resume and idx <= int(checkpoint.get(str(iteration), 0)):
                    continue
                print(
                    f"[batch] iter {iteration}/{args.iterations} {idx}/{len(queries)}: {query}",
                    flush=True,
                )
                attempt = 0
                while True:
                    try:
                        run_workflow(
                            task_text=query,
                            roles=roles,
                            constraints_per_role=constraints,
                            workflow_version=args.workflow_version,
                            registry=registry,
                            index=index,
                            embedder=embedder,
                            top_n=args.top_n,
                            top_k=args.top_k,
                            rerank_top_m=args.rerank_top_m,
                            mmr_lambda=args.mmr_lambda,
                            execute_tools=args.execute_tools,
                            tool_timeout_s=args.tool_timeout,
                            llm_client=llm,
                            reranker_model_path=reranker_path,
                            runs_dir=args.runs_dir,
                        )
                        break
                    except Exception as exc:
                        attempt += 1
                        print(
                            f"[batch] query failed (attempt {attempt}/{args.query_retries + 1}): {exc}",
                            flush=True,
                        )
                        if attempt > args.query_retries:
                            if args.skip_failures:
                                break
                            raise
                        time.sleep(args.retry_backoff * attempt)
                if args.resume:
                    checkpoint[str(iteration)] = idx
                    _save_checkpoint(args.checkpoint_file, checkpoint)

        data_path = os.path.join(os.path.dirname(args.train_out) or ".", "train.jsonl")
        count = export_training_data(args.runs_dir, args.db, data_path)
        if count <= 0:
            raise ValueError("No training data exported; check runs directory.")
        train_reranker(data_path, args.train_out, epochs=args.epochs, lr=args.lr)
        print(f"[batch] trained reranker -> {args.train_out}")


if __name__ == "__main__":
    main()
