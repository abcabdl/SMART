from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from scalability_plotting import enrich_scalability_rows, plot_scalability_latency, plot_scalability_passrate


THIS_FILE = Path(__file__).resolve()
ANALYSIS_DIR = THIS_FILE.parent
SRC_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "datas" / "mbpp_test.jsonl"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "exp2_scalability"
DEFAULT_DB = REPO_ROOT / "demo_registry.sqlite"
DEFAULT_INDEX_DIR = REPO_ROOT / "index"

CARD_COLUMNS = [
    "id",
    "name",
    "kind",
    "version",
    "updated_at",
    "domain_tags",
    "role_tags",
    "tool_tags",
    "modalities",
    "output_formats",
    "permissions",
    "cost_tier",
    "latency_tier",
    "reliability_prior",
    "description",
    "examples",
    "embedding_text",
    "available_tool_ids",
]

ROLE_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "builder": ("builder", "code-generation"),
    "planner": ("planner", "code-planner", "manager"),
    "checker": ("checker", "tester", "code-testing"),
    "refactor": ("refactor", "refractor", "code-refactoring"),
}


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
        description="Dynamic workflow + LLM-free MCTS",
    ),
    "dynamic_router_local": MethodPreset(
        name="dynamic_router_local",
        cli_args=(
            "--use_orchestrator",
            "--dynamic_topology",
        ),
        description="Dynamic workflow + router only (no MCTS)",
    ),
    "fixed_bcb_mcts": MethodPreset(
        name="fixed_bcb_mcts",
        cli_args=(
            "--use_orchestrator",
            "--baseline_fixed_bcb_mcts",
        ),
        description="Fixed B->C->B + MCTS",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp2: Candidate-pool scalability on MBPP."
    )
    parser.add_argument("--tasks", type=str, default=str(DEFAULT_TASKS))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--methods",
        type=str,
        default="dynamic_mcts,dynamic_router_local",
        help=f"Comma-separated methods. Available: {','.join(METHOD_PRESETS.keys())}",
    )
    parser.add_argument("--pool_sizes", type=str, default="10,50,100,200,300")
    parser.add_argument("--skip_run", action="store_true")

    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--index_dir", type=str, default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", type=str, default="sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedder_device", type=str, default="")
    parser.add_argument("--embedder_normalize", action="store_true")

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

    parser.add_argument("--disable_stratified_pool_sampling", action="store_true")
    parser.add_argument(
        "--pool_roles",
        type=str,
        default="builder,planner,checker,refactor",
        help="Role categories used for balanced random sampling into each pool size.",
    )
    parser.add_argument("--pool_seed", type=int, default=13)

    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--postprocess_rounds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--top_k_ratio",
        type=float,
        default=0.2,
        help="Scale top_k with pool_size by this ratio (top_k ~= pool_size * ratio).",
    )
    parser.add_argument(
        "--top_k_min",
        type=int,
        default=5,
        help="Minimum top_k before clamping to top_n.",
    )
    parser.add_argument(
        "--top_k_max",
        type=int,
        default=0,
        help="Optional hard max for top_k (<=0 disables cap).",
    )
    parser.add_argument("--rerank_top_m", type=int, default=3)
    parser.add_argument(
        "--rerank_top_m_ratio",
        type=float,
        default=0.1,
        help="Scale rerank_top_m with pool_size (min is --rerank_top_m).",
    )
    parser.add_argument(
        "--rerank_top_m_max",
        type=int,
        default=0,
        help="Optional hard max for rerank_top_m (<=0 disables cap).",
    )
    parser.add_argument(
        "--router_top_m_min",
        type=int,
        default=5,
        help="Minimum router_top_m before clamping to top_k.",
    )
    parser.add_argument(
        "--router_top_m_ratio",
        type=float,
        default=0.1,
        help="Scale router_top_m with pool_size by this ratio.",
    )
    parser.add_argument(
        "--router_top_m_max",
        type=int,
        default=0,
        help="Optional hard max for router_top_m (<=0 disables cap).",
    )
    parser.add_argument("--mcts_iterations", type=int, default=64)
    parser.add_argument(
        "--mcts_iteration_base_pool",
        type=int,
        default=50,
        help=(
            "Linear scaling anchor for MCTS iterations: "
            "effective_iterations = mcts_iterations * pool_size / mcts_iteration_base_pool"
        ),
    )
    parser.add_argument("--mcts_rollout_depth", type=int, default=4)
    parser.add_argument("--mcts_exploration", type=float, default=1.414)
    parser.add_argument("--mcts_discount", type=float, default=0.95)
    parser.add_argument("--mcts_max_candidates", type=int, default=8)
    parser.add_argument(
        "--mcts_max_candidates_ratio",
        type=float,
        default=0.1,
        help="Scale mcts_max_candidates with pool_size (min is --mcts_max_candidates).",
    )
    parser.add_argument(
        "--mcts_max_candidates_max",
        type=int,
        default=0,
        help="Optional hard max for mcts_max_candidates (<=0 disables cap).",
    )
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
        "--sample_size",
        type=int,
        default=0,
        help="If >0, sample this many tasks from --tasks (MBPP) for faster sweeps.",
    )
    parser.add_argument("--sample_seed", type=int, default=7)
    return parser.parse_args()


def _split_methods(raw: str) -> List[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [m for m in methods if m not in METHOD_PRESETS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not methods:
        raise ValueError("No methods selected")
    return methods


def _parse_pool_sizes(raw: str) -> List[int]:
    pools = [int(item.strip()) for item in raw.split(",") if item.strip()]
    pools = [value for value in pools if value > 0]
    if not pools:
        raise ValueError("No valid pool sizes")
    return sorted(set(pools))


def _parse_roles(raw: str) -> List[str]:
    roles = [item.strip() for item in raw.split(",") if item.strip()]
    if not roles:
        raise ValueError("No roles provided")
    return roles


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _materialize_sampled_tasks(tasks_path: Path, output_dir: Path, sample_size: int, seed: int) -> Path:
    if sample_size <= 0:
        return tasks_path
    rows = _read_jsonl(tasks_path)
    if sample_size >= len(rows):
        return tasks_path
    rng = random.Random(seed)
    sampled = rng.sample(rows, sample_size)
    sampled_path = output_dir / "sampled_tasks.jsonl"
    _write_jsonl(sampled_path, sampled)
    return sampled_path


def _normalize_role_name(role: str) -> str:
    text = (role or "").strip().lower()
    if text == "refractor":
        return "refactor"
    return text


def _extract_aliases(role: str) -> Tuple[str, ...]:
    normalized = _normalize_role_name(role)
    aliases = ROLE_ALIAS_MAP.get(normalized)
    if aliases:
        return aliases
    return (normalized,)


def _load_agents_for_sampling(db_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, name, role_tags FROM cards WHERE kind = 'agent' ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()

    agents: List[Dict[str, Any]] = []
    for row in rows:
        role_tags_raw = row["role_tags"]
        role_tags: List[str] = []
        if isinstance(role_tags_raw, str) and role_tags_raw.strip():
            try:
                parsed = json.loads(role_tags_raw)
                if isinstance(parsed, list):
                    role_tags = [str(item).strip().lower() for item in parsed if str(item).strip()]
            except Exception:
                role_tags = [item.strip().lower() for item in role_tags_raw.split(",") if item.strip()]
        agents.append(
            {
                "id": str(row["id"]),
                "name": str(row["name"] or ""),
                "role_tags": role_tags,
            }
        )
    return agents


def _agent_matches_role(agent: Dict[str, Any], role: str) -> bool:
    aliases = {alias.strip().lower() for alias in _extract_aliases(role)}
    role_tags = {str(tag).strip().lower() for tag in (agent.get("role_tags") or []) if str(tag).strip()}
    if role_tags.intersection(aliases):
        return True
    name = str(agent.get("name") or "").lower()
    if any(alias in name for alias in aliases):
        return True
    return False


def _stratified_sample_agent_ids(
    *,
    agents: List[Dict[str, Any]],
    pool_roles: List[str],
    pool_size: int,
    seed: int,
) -> Tuple[List[str], Dict[str, int]]:
    if not agents:
        return [], {}
    all_agent_ids = [str(agent["id"]) for agent in agents]
    total_agents = len(all_agent_ids)
    target = min(max(1, int(pool_size)), total_agents)
    canonical_roles = [_normalize_role_name(role) for role in pool_roles]
    canonical_roles = [role for role in canonical_roles if role]
    if not canonical_roles:
        canonical_roles = ["builder", "planner", "checker", "refactor"]

    rng = random.Random(int(seed))
    role_to_candidates: Dict[str, List[str]] = {}
    for role in canonical_roles:
        matched = [str(agent["id"]) for agent in agents if _agent_matches_role(agent, role)]
        role_to_candidates[role] = matched

    quota_base = target // len(canonical_roles)
    remainder = target % len(canonical_roles)
    selected: List[str] = []
    selected_set = set()
    role_counts: Dict[str, int] = {}

    for idx, role in enumerate(canonical_roles):
        quota = quota_base + (1 if idx < remainder else 0)
        candidates = [cid for cid in role_to_candidates.get(role, []) if cid not in selected_set]
        if quota <= 0 or not candidates:
            role_counts[role] = 0
            continue
        take = min(quota, len(candidates))
        picked = rng.sample(candidates, take) if len(candidates) > take else list(candidates)
        for cid in picked:
            if cid not in selected_set:
                selected_set.add(cid)
                selected.append(cid)
        role_counts[role] = len(picked)

    if len(selected) < target:
        remaining = [cid for cid in all_agent_ids if cid not in selected_set]
        need = target - len(selected)
        if need > 0 and remaining:
            extra = rng.sample(remaining, min(need, len(remaining)))
            for cid in extra:
                if cid not in selected_set:
                    selected_set.add(cid)
                    selected.append(cid)

    return sorted(selected), role_counts


def _ensure_registry_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            domain_tags TEXT,
            role_tags TEXT,
            tool_tags TEXT,
            modalities TEXT,
            output_formats TEXT,
            permissions TEXT,
            cost_tier TEXT,
            latency_tier TEXT,
            reliability_prior REAL,
            description TEXT,
            examples TEXT,
            embedding_text TEXT,
            available_tool_ids TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_code (
            id TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _copy_rows_in_chunks(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    *,
    where_sql: str,
    where_args: Tuple[Any, ...],
) -> int:
    columns_sql = ", ".join(CARD_COLUMNS)
    query = f"SELECT {columns_sql} FROM cards WHERE {where_sql}"
    rows = src_conn.execute(query, where_args).fetchall()
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(CARD_COLUMNS))
    insert_sql = f"INSERT OR REPLACE INTO cards ({columns_sql}) VALUES ({placeholders})"
    dst_conn.executemany(insert_sql, rows)
    dst_conn.commit()
    return len(rows)


def _materialize_sampled_registry(
    *,
    base_db_path: Path,
    sampled_db_path: Path,
    selected_agent_ids: List[str],
) -> Dict[str, int]:
    sampled_db_path.parent.mkdir(parents=True, exist_ok=True)
    if sampled_db_path.exists():
        sampled_db_path.unlink()

    src_conn = sqlite3.connect(str(base_db_path))
    dst_conn = sqlite3.connect(str(sampled_db_path))
    try:
        _ensure_registry_schema(dst_conn)

        copied_agents = 0
        if selected_agent_ids:
            placeholders = ", ".join(["?"] * len(selected_agent_ids))
            copied_agents = _copy_rows_in_chunks(
                src_conn,
                dst_conn,
                where_sql=f"kind = 'agent' AND id IN ({placeholders})",
                where_args=tuple(selected_agent_ids),
            )

        copied_non_agents = _copy_rows_in_chunks(
            src_conn,
            dst_conn,
            where_sql="kind != 'agent'",
            where_args=tuple(),
        )

        tool_rows = src_conn.execute("SELECT id, code, updated_at FROM tool_code").fetchall()
        if tool_rows:
            dst_conn.executemany(
                "INSERT OR REPLACE INTO tool_code (id, code, updated_at) VALUES (?, ?, ?)",
                tool_rows,
            )
            dst_conn.commit()
    finally:
        src_conn.close()
        dst_conn.close()

    return {
        "copied_agents": copied_agents,
        "copied_non_agents": copied_non_agents,
    }


def _prepare_pool_resources(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    pool_size: int,
    pool_roles: List[str],
) -> Dict[str, Any]:
    if args.disable_stratified_pool_sampling:
        return {
            "db_path": str(Path(args.db).resolve()),
            "index_dir": str(Path(args.index_dir).resolve()),
            "sampled_agents": -1,
            "manifest_path": "",
            "role_counts": {},
        }

    base_db_path = Path(args.db).resolve()
    if not base_db_path.exists():
        raise FileNotFoundError(f"Base registry db not found: {base_db_path}")

    pool_dir = output_dir / "pool_resources" / f"pool_{pool_size}"
    sampled_db_path = pool_dir / "registry.sqlite"
    manifest_path = pool_dir / "pool_manifest.json"
    pool_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents_for_sampling(base_db_path)
    sampled_ids, role_counts = _stratified_sample_agent_ids(
        agents=agents,
        pool_roles=pool_roles,
        pool_size=pool_size,
        # Keep one fixed random seed across pool sizes for fairer pool-size comparison.
        seed=int(args.pool_seed),
    )
    copy_info = _materialize_sampled_registry(
        base_db_path=base_db_path,
        sampled_db_path=sampled_db_path,
        selected_agent_ids=sampled_ids,
    )

    id_to_name = {str(agent["id"]): str(agent.get("name") or "") for agent in agents}
    manifest = {
        "pool_size_requested": int(pool_size),
        "pool_seed": int(args.pool_seed),
        "pool_roles": pool_roles,
        "sampled_agent_count": len(sampled_ids),
        "sampled_agent_ids": sampled_ids,
        "sampled_agent_names": [id_to_name.get(cid, "") for cid in sampled_ids],
        "role_sample_counts": role_counts,
        "copied_agents": int(copy_info.get("copied_agents", 0)),
        "copied_non_agents": int(copy_info.get("copied_non_agents", 0)),
        "sampled_db_path": str(sampled_db_path.resolve()),
        "index_dir_reused": str(Path(args.index_dir).resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "db_path": str(sampled_db_path.resolve()),
        "index_dir": str(Path(args.index_dir).resolve()),
        "sampled_agents": len(sampled_ids),
        "manifest_path": str(manifest_path.resolve()),
        "role_counts": role_counts,
    }


def _build_eval_cmd(
    *,
    args: argparse.Namespace,
    tasks_path: Path,
    report_path: Path,
    solutions_path: Path,
    pool_size: int,
    pool_resource: Dict[str, Any],
) -> List[str]:
    def _scaled_limit(
        *,
        pool_n: int,
        ratio: float,
        minimum: int,
        maximum: int,
    ) -> int:
        value = max(1, int(minimum))
        if float(ratio) > 0:
            value = max(value, int(round(float(pool_n) * float(ratio))))
        if int(maximum) > 0:
            value = min(value, int(maximum))
        return max(1, value)

    top_n = max(1, int(pool_size))
    top_k = _scaled_limit(
        pool_n=top_n,
        ratio=float(args.top_k_ratio),
        minimum=int(args.top_k_min),
        maximum=int(args.top_k_max),
    )
    top_k = max(1, min(top_k, top_n))
    rerank_top_m = _scaled_limit(
        pool_n=top_n,
        ratio=float(args.rerank_top_m_ratio),
        minimum=int(args.rerank_top_m),
        maximum=int(args.rerank_top_m_max),
    )
    rerank_top_m = max(1, min(rerank_top_m, top_k))
    router_top_m = _scaled_limit(
        pool_n=top_n,
        ratio=float(args.router_top_m_ratio),
        minimum=int(args.router_top_m_min),
        maximum=int(args.router_top_m_max),
    )
    router_top_m = max(1, min(router_top_m, top_k))
    mcts_max_candidates = _scaled_limit(
        pool_n=top_n,
        ratio=float(args.mcts_max_candidates_ratio),
        minimum=int(args.mcts_max_candidates),
        maximum=int(args.mcts_max_candidates_max),
    )
    mcts_max_candidates = max(1, min(mcts_max_candidates, top_k))
    base_pool = max(1, int(args.mcts_iteration_base_pool))
    scaled_mcts_iterations = max(
        1,
        int(round(float(args.mcts_iterations) * float(pool_size) / float(base_pool))),
    )
    cmd = [
        sys.executable,
        "-m",
        "evaluation.eval_mbpp",
        "--mbpp_tasks",
        str(tasks_path.resolve()),
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
        "--db",
        str(pool_resource["db_path"]),
        "--index_dir",
        str(pool_resource["index_dir"]),
        "--dim",
        str(args.dim),
        "--seed",
        str(args.seed),
        "--embedder",
        str(args.embedder),
        "--embedder_model",
        str(args.embedder_model),
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
        "--top_n",
        str(top_n),
        "--top_k",
        str(top_k),
        "--rerank_top_m",
        str(rerank_top_m),
        "--router_top_m",
        str(router_top_m),
        "--mcts_iterations",
        str(scaled_mcts_iterations),
        "--mcts_rollout_depth",
        str(args.mcts_rollout_depth),
        "--mcts_exploration",
        str(args.mcts_exploration),
        "--mcts_discount",
        str(args.mcts_discount),
        "--mcts_max_candidates",
        str(mcts_max_candidates),
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
    if args.embedder_device:
        cmd += ["--embedder_device", str(args.embedder_device)]
    if args.embedder_normalize:
        cmd.append("--embedder_normalize")
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


def _run_eval(
    args: argparse.Namespace,
    method: str,
    pool_size: int,
    output_dir: Path,
    tasks_path: Path,
    pool_resource: Dict[str, Any],
) -> Path:
    report_path = output_dir / "reports" / f"{method}_pool{pool_size}.json"
    solutions_path = output_dir / "solutions" / f"{method}_pool{pool_size}.solutions.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    solutions_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_eval_cmd(
        args=args,
        tasks_path=tasks_path,
        report_path=report_path,
        solutions_path=solutions_path,
        pool_size=pool_size,
        pool_resource=pool_resource,
    )
    cmd.extend(METHOD_PRESETS[method].cli_args)
    subprocess.run(cmd, cwd=str(SRC_ROOT), check=True)
    return report_path


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _summarize_report(report_path: Path) -> Dict[str, float]:
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    results = payload.get("results") or []
    generation_times: List[float] = []
    workflow_times: List[float] = []
    eval_times: List[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if "generation_time_s" in item:
            generation_times.append(float(item.get("generation_time_s") or 0.0))
        if "workflow_time_s" in item:
            workflow_times.append(float(item.get("workflow_time_s") or 0.0))
        if "eval_time_s" in item:
            eval_times.append(float(item.get("eval_time_s") or 0.0))
    return {
        "total": float(payload.get("total") or 0),
        "passed": float(payload.get("passed") or 0),
        "pass_rate": float(payload.get("pass_rate") or 0.0),
        "avg_generation_time_s": _safe_mean(generation_times),
        "avg_workflow_time_s": _safe_mean(workflow_times),
        "avg_eval_time_s": _safe_mean(eval_times),
    }


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "pool_size",
        "pass_rate",
        "passed",
        "total",
        "avg_generation_time_s",
        "avg_workflow_time_s",
        "avg_eval_time_s",
        "pass_rate_se",
        "workflow_time_std_s",
        "workflow_time_se_s",
        "sampled_agents",
        "pool_manifest_path",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_scalability(rows: List[Dict[str, Any]], output_dir: Path) -> tuple[Path | None, Path | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore  # noqa: F401
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import matplotlib.pyplot as plt  # type: ignore  # noqa: F401
        except Exception:
            return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    passrate_path = output_dir / "passrate_lines.png"
    latency_path = output_dir / "latency_lines.png"
    plot_scalability_passrate(rows, output_path=passrate_path)
    plot_scalability_latency(rows, output_path=latency_path)
    return passrate_path, latency_path


def main() -> None:
    args = _parse_args()
    methods = _split_methods(args.methods)
    pool_sizes = _parse_pool_sizes(args.pool_sizes)
    pool_roles = _parse_roles(args.pool_roles)
    output_dir = Path(args.output_dir).resolve()
    reports_dir = output_dir / "reports"
    tasks_path = Path(args.tasks).resolve()

    if not args.skip_run and not args.local_model_dir:
        raise SystemExit("--local_model_dir is required unless --skip_run is set.")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    effective_tasks_path = _materialize_sampled_tasks(
        tasks_path=tasks_path,
        output_dir=output_dir,
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )

    pool_resources: Dict[int, Dict[str, Any]] = {}
    if not args.skip_run:
        for pool_size in pool_sizes:
            pool_resources[pool_size] = _prepare_pool_resources(
                args=args,
                output_dir=output_dir,
                pool_size=pool_size,
                pool_roles=pool_roles,
            )

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for pool_size in pool_sizes:
            pool_resource = pool_resources.get(pool_size) or {
                "db_path": str(Path(args.db).resolve()),
                "index_dir": str(Path(args.index_dir).resolve()),
                "sampled_agents": -1,
                "manifest_path": "",
                "role_counts": {},
            }
            if args.skip_run:
                report_path = reports_dir / f"{method}_pool{pool_size}.json"
                if not report_path.exists():
                    raise FileNotFoundError(f"Missing report: {report_path}")
            else:
                report_path = _run_eval(
                    args=args,
                    method=method,
                    pool_size=pool_size,
                    output_dir=output_dir,
                    tasks_path=effective_tasks_path,
                    pool_resource=pool_resource,
                )

            metrics = _summarize_report(report_path)
            row: Dict[str, Any] = dict(metrics)
            row["method"] = method
            row["pool_size"] = int(pool_size)
            row["sampled_agents"] = int(pool_resource.get("sampled_agents", -1))
            row["pool_manifest_path"] = str(pool_resource.get("manifest_path") or "")
            row["report_path"] = str(report_path)
            rows.append(row)

    rows = enrich_scalability_rows(rows)
    csv_path = output_dir / "scalability_summary.csv"
    _save_csv(rows, csv_path)
    passrate_fig_path, latency_fig_path = _plot_scalability(rows, output_dir=output_dir)

    summary = {
        "methods": methods,
        "pool_sizes": pool_sizes,
        "pool_roles": pool_roles,
        "stratified_pool_sampling": not args.disable_stratified_pool_sampling,
        "tasks_path": str(effective_tasks_path),
        "csv_path": str(csv_path),
        "passrate_figure_path": str(passrate_fig_path) if passrate_fig_path else "",
        "latency_figure_path": str(latency_fig_path) if latency_fig_path else "",
        "pool_resources": pool_resources,
        "rows": rows,
    }
    summary_path = output_dir / "scalability_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "csv_path": summary["csv_path"],
                "passrate_figure_path": summary["passrate_figure_path"],
                "latency_figure_path": summary["latency_figure_path"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
