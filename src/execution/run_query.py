from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure `core`, `retrieval`, etc. are importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from execution.orchestrator import run_workflow
from generation.llm_client import LLMClient
from retrieval.embedder import build_embedder
from retrieval.faiss_index import HNSWIndex


def _parse_roles(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_constraints(value: str) -> Dict[str, Dict[str, object]]:
    if not value:
        return {}
    return json.loads(value)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query through orchestrator")
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--db", default="demo_registry.sqlite", type=str)
    parser.add_argument("--index_dir", default="index", type=str)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--embedder", default="sentence-transformer", type=str, help="dummy|sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embedder_device", default=None, type=str)
    parser.add_argument("--embedder_normalize", action="store_true")
    parser.add_argument("--roles", default="code-generation,code-planner,code-testing,code-refactoring", type=str)
    parser.add_argument("--constraints", default="", type=str, help="JSON string per role")
    parser.add_argument("--workflow_version", default="v1", type=str)
    parser.add_argument("--reranker_model", default="models/reranker.json", type=str)
    parser.add_argument("--bandit_db", default="models/bandit.sqlite", type=str)
    parser.add_argument("--top_n", default=20, type=int)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--rerank_top_m", default=3, type=int)
    parser.add_argument("--mmr_lambda", default=0.5, type=float)
    parser.add_argument("--tool_timeout", default=1.0, type=float)
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_timeout", type=float, default=60.0)
    parser.add_argument("--llm_retries", type=int, default=5)
    parser.add_argument("--auto_fill", action="store_true")
    parser.add_argument("--auto_fill_model", type=str, default=None)
    parser.add_argument("--auto_fill_base_url", type=str, default=None)
    parser.add_argument("--auto_fill_api_key", type=str, default=None)
    parser.add_argument("--auto_fill_timeout", type=float, default=30.0)
    parser.add_argument("--auto_install_common_libs", action="store_true")
    parser.add_argument("--auto_install_timeout", type=float, default=300.0)
    parser.add_argument("--auto_install_user", action="store_true")
    parser.add_argument("--print_tools", action="store_true")
    parser.add_argument("--use_mock_llm", action="store_true")
    parser.add_argument("--router_model", type=str, default=None)
    parser.add_argument("--router_base_url", type=str, default=None)
    parser.add_argument("--router_api_key", type=str, default=None)
    parser.add_argument("--router_timeout", type=float, default=60.0)
    parser.add_argument("--router_retries", type=int, default=5)
    parser.add_argument("--router_top_m", type=int, default=5)
    parser.add_argument("--router_no_rerank", action="store_true")
    parser.add_argument("--dynamic_topology", action="store_true")
    parser.add_argument(
        "--topology",
        type=str,
        default="linear",
        help="linear|auto|single|centralized|decentralized|chain",
    )
    parser.add_argument("--topology_config", type=str, default="", help="JSON string or path to JSON file")
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument(
        "--disable_postprocess_repair",
        action="store_true",
        help="disable orchestrator postprocess repair chain after tool test failures",
    )
    parser.add_argument(
        "--allow_early_finish",
        action="store_true",
        help="allow router/agent to finish early even if no success signal was observed",
    )
    parser.add_argument("--allow_unknown_roles", action="store_true")
    parser.add_argument("--no_reuse_role_selection", action="store_true")
    parser.add_argument(
        "--reuse_same_role_agent_once",
        action="store_true",
        help="within one query, search/select each role once and reuse the same agent for repeated role steps",
    )
    parser.add_argument("--print_topology", action="store_true")
    parser.add_argument("--soft_connection", action="store_true")
    parser.add_argument("--mcts_dynamic_optimization", action="store_true")
    parser.add_argument("--mcts_iterations", type=int, default=64)
    parser.add_argument("--mcts_rollout_depth", type=int, default=4)
    parser.add_argument("--mcts_exploration", type=float, default=1.414)
    parser.add_argument("--mcts_discount", type=float, default=0.95)
    parser.add_argument("--mcts_max_candidates", type=int, default=8)
    parser.add_argument("--paper_mode", action="store_true")
    parser.add_argument("--task_family", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--artifact_contract_version", type=str, default="v2")
    parser.add_argument("--final_candidate_cap", type=int, default=5)
    parser.add_argument("--meta_router_model", type=str, default=None)
    parser.add_argument("--meta_router_base_url", type=str, default=None)
    parser.add_argument("--meta_router_api_key", type=str, default=None)
    parser.add_argument("--meta_router_timeout", type=float, default=60.0)
    parser.add_argument("--meta_router_retries", type=int, default=5)
    parser.add_argument("--next_role_model", type=str, default=None)
    parser.add_argument("--next_role_base_url", type=str, default=None)
    parser.add_argument("--next_role_api_key", type=str, default=None)
    parser.add_argument("--next_role_timeout", type=float, default=60.0)
    parser.add_argument("--next_role_retries", type=int, default=5)
    parser.add_argument("--local_model_dir", type=str, default=None, help="local base model path")
    parser.add_argument("--local_lora_dir", type=str, default=None, help="local LoRA adapter path")
    parser.add_argument("--local_device", type=str, default=None)
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_max_new_tokens", type=int, default=512)
    parser.add_argument("--local_temperature", type=float, default=0.2)
    parser.add_argument("--local_use_4bit", action="store_true")
    parser.add_argument("--local_use_8bit", action="store_true")
    parser.add_argument(
        "--local_use",
        type=str,
        default="",
        help="comma-separated: main,router,meta_router,next_role,all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = _parse_roles(args.roles)
    constraints_per_role = _parse_constraints(args.constraints)
    index_path = os.path.join(args.index_dir, "faiss.index")

    index = HNSWIndex.load(index_path)
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
    tool_llm = None
    if args.auto_fill:
        tool_llm = LLMClient(
            api_key=args.auto_fill_api_key,
            base_url=args.auto_fill_base_url,
            model=args.auto_fill_model,
            timeout_s=args.auto_fill_timeout,
        )
    main_llm = None
    if not args.use_mock_llm:
        main_llm = LLMClient(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            timeout_s=args.llm_timeout,
            max_retries=args.llm_retries,
        )
    router_llm = None
    if args.router_model or args.router_base_url or args.router_api_key:
        router_llm = LLMClient(
            api_key=args.router_api_key or args.llm_api_key,
            base_url=args.router_base_url or args.llm_base_url,
            model=args.router_model or args.llm_model,
            timeout_s=args.router_timeout,
            max_retries=args.router_retries,
        )
    meta_router_llm = None
    if args.meta_router_model or args.meta_router_base_url or args.meta_router_api_key:
        meta_router_llm = LLMClient(
            api_key=args.meta_router_api_key or args.llm_api_key,
            base_url=args.meta_router_base_url or args.llm_base_url,
            model=args.meta_router_model or args.llm_model,
            timeout_s=args.meta_router_timeout,
            max_retries=args.meta_router_retries,
        )
    next_role_llm = None
    if args.next_role_model or args.next_role_base_url or args.next_role_api_key:
        next_role_llm = LLMClient(
            api_key=args.next_role_api_key or args.llm_api_key,
            base_url=args.next_role_base_url or args.llm_base_url,
            model=args.next_role_model or args.llm_model,
            timeout_s=args.next_role_timeout,
            max_retries=args.next_role_retries,
        )

    local_use = {item.lower() for item in _parse_list(args.local_use)}
    if local_use:
        if not args.local_model_dir:
            raise ValueError("local_use specified but --local_model_dir is missing")
        from generation.local_llm import LocalLLMClient

        local_client = LocalLLMClient(
            model_path=args.local_model_dir,
            lora_path=args.local_lora_dir,
            device=args.local_device,
            dtype=args.local_dtype,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
            use_4bit=args.local_use_4bit,
            use_8bit=args.local_use_8bit,
        )
        if "all" in local_use or "main" in local_use:
            main_llm = local_client
        if "all" in local_use or "router" in local_use:
            router_llm = local_client
        if "all" in local_use or "meta_router" in local_use:
            meta_router_llm = local_client
        if "all" in local_use or "next_role" in local_use:
            next_role_llm = local_client

    topology_config = None
    if args.topology_config:
        if os.path.exists(args.topology_config):
            with open(args.topology_config, "r", encoding="utf-8") as f:
                topology_config = json.load(f)
        else:
            topology_config = json.loads(args.topology_config)

    with SQLiteRegistry(args.db) as registry:
        result = run_workflow(
            task_text=args.query,
            roles=roles,
            constraints_per_role=constraints_per_role,
            workflow_version=args.workflow_version,
            registry=registry,
            index=index,
            embedder=embedder,
            top_n=args.top_n,
            top_k=args.top_k,
            rerank_top_m=args.rerank_top_m,
            mmr_lambda=args.mmr_lambda,
            reranker_model_path=args.reranker_model,
            bandit_db_path=args.bandit_db,
            auto_fill_tool_inputs=args.auto_fill,
            tool_llm=tool_llm,
            tool_timeout_s=args.tool_timeout,
            auto_install_common_libs=args.auto_install_common_libs,
            auto_install_timeout_s=args.auto_install_timeout,
            auto_install_user=args.auto_install_user,
            llm_client=main_llm,
            router_llm_client=router_llm,
            router_top_m=args.router_top_m,
            router_no_rerank=args.router_no_rerank,
            dynamic_topology=args.dynamic_topology,
            topology=args.topology,
            topology_config=topology_config,
            paper_mode=args.paper_mode,
            task_family=args.task_family or None,
            dataset=args.dataset or None,
            final_candidate_cap=args.final_candidate_cap,
            artifact_contract_version=args.artifact_contract_version,
            meta_router_llm_client=meta_router_llm,
            next_role_llm_client=next_role_llm,
            max_steps=args.max_steps,
            enable_postprocess_repair=not args.disable_postprocess_repair,
            prevent_early_finish=not args.allow_early_finish,
            allow_unknown_roles=args.allow_unknown_roles,
            reuse_role_selection=not args.no_reuse_role_selection,
            reuse_same_role_agent_once=args.reuse_same_role_agent_once,
            soft_connection=args.soft_connection,
            mcts_dynamic_optimization=args.mcts_dynamic_optimization,
            mcts_iterations=args.mcts_iterations,
            mcts_rollout_depth=args.mcts_rollout_depth,
            mcts_exploration=args.mcts_exploration,
            mcts_discount=args.mcts_discount,
            mcts_max_candidates=args.mcts_max_candidates,
        )

    print(result.get("answer"))
    print("log_path:", result.get("log_path"))
    if args.print_topology and result.get("topology") is not None:
        print("topology:", json.dumps(result.get("topology"), ensure_ascii=True))
    if args.print_tools:
        tool_exec = result.get("tool_exec") or {}
        if tool_exec:
            print("tool_exec:", json.dumps(tool_exec, ensure_ascii=True))


if __name__ == "__main__":
    main()
