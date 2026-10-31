from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
DEFAULT_PRESET_PATH = REPO_ROOT / "configs" / "smart_pdf_benchmark.json"


def load_smart_pdf_preset(path: str | None = None) -> Dict[str, Any]:
    preset_path = Path(path).expanduser() if path else DEFAULT_PRESET_PATH
    if not preset_path.is_absolute():
        preset_path = (REPO_ROOT / preset_path).resolve()
    with open(preset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "repo":  # legacy "repo/" prefix
        path = Path(*path.parts[1:])
    return str((REPO_ROOT / path).resolve())


def apply_smart_pdf_preset(args: Any, *, benchmark: str) -> Any:
    preset = load_smart_pdf_preset()
    retrieval = dict(preset.get("retrieval") or {})
    mcts = dict(preset.get("mcts") or {})
    runtime = dict(preset.get("runtime") or {})
    paths = dict(preset.get("paths") or {})

    old_db_defaults = {
        "demo_registry.sqlite",
        "real_dataset_registry.sqlite",
        "registry.sqlite",
    }
    old_index_defaults = {"index", "index_real_dataset"}
    old_role_defaults = {
        "code-generation,code-planner,code-testing,code-refactoring",
        "builder,checker,refactor",
        "builder,checker",
        "planner,builder,checker",
    }

    if getattr(args, "db", None) in old_db_defaults:
        args.db = resolve_project_path(str(paths.get("benchmark_db") or args.db))
    if getattr(args, "index_dir", None) in old_index_defaults:
        args.index_dir = resolve_project_path(str(paths.get("benchmark_index_dir") or args.index_dir))
    if hasattr(args, "roles") and (not getattr(args, "roles", "").strip() or getattr(args, "roles") in old_role_defaults):
        args.roles = ",".join(str(item) for item in (preset.get("roles") or []))

    if hasattr(args, "top_n"):
        args.top_n = int(retrieval.get("top_n", getattr(args, "top_n")))
    if hasattr(args, "top_k"):
        args.top_k = int(retrieval.get("top_k", getattr(args, "top_k")))
    if hasattr(args, "mmr_lambda"):
        args.mmr_lambda = float(retrieval.get("mmr_lambda", getattr(args, "mmr_lambda")))
    if hasattr(args, "mcts_iterations"):
        args.mcts_iterations = int(mcts.get("iterations", getattr(args, "mcts_iterations")))
    if hasattr(args, "mcts_rollout_depth"):
        args.mcts_rollout_depth = int(mcts.get("rollout_depth", getattr(args, "mcts_rollout_depth")))
    if hasattr(args, "mcts_exploration"):
        args.mcts_exploration = float(mcts.get("exploration", getattr(args, "mcts_exploration")))
    if hasattr(args, "mcts_discount"):
        args.mcts_discount = float(mcts.get("discount", getattr(args, "mcts_discount")))
    if hasattr(args, "mcts_max_candidates"):
        args.mcts_max_candidates = int(mcts.get("max_candidates", getattr(args, "mcts_max_candidates")))
    if hasattr(args, "reward_variant"):
        args.reward_variant = str(mcts.get("reward_variant", getattr(args, "reward_variant")))

    if hasattr(args, "embedder"):
        args.embedder = str(runtime.get("embedder", getattr(args, "embedder")))
    if hasattr(args, "embedder_model"):
        args.embedder_model = str(runtime.get("embedder_model", getattr(args, "embedder_model")))
    if hasattr(args, "dim"):
        args.dim = int(runtime.get("dim", getattr(args, "dim")))
    if hasattr(args, "seed"):
        args.seed = int(runtime.get("seed", getattr(args, "seed")))

    if hasattr(args, "use_cross_instance_cache"):
        args.use_cross_instance_cache = bool(runtime.get("use_cross_instance_cache", False))
    if hasattr(args, "update_posteriors_across_instances"):
        args.update_posteriors_across_instances = bool(runtime.get("update_posteriors_across_instances", False))
    if hasattr(args, "use_parallel_frontier"):
        args.use_parallel_frontier = bool(runtime.get("use_parallel_frontier", False))
    if hasattr(args, "experimental_reranker_after_mmr"):
        args.experimental_reranker_after_mmr = bool(retrieval.get("experimental_reranker_after_mmr", False))
    if hasattr(args, "disable_orchestrator_postprocess_repair"):
        args.disable_orchestrator_postprocess_repair = bool(runtime.get("disable_orchestrator_postprocess_repair", False))

    if benchmark == "hotpotqa_appendix":
        args.db = resolve_project_path(str(paths.get("hotpotqa_appendix_db") or args.db))
        args.index_dir = resolve_project_path(str(paths.get("hotpotqa_appendix_index_dir") or args.index_dir))

    return args
