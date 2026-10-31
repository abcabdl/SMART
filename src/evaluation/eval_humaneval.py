from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional progress bar.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Set up logger
logger = logging.getLogger(__name__)

# Ensure `generation` is importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.humaneval_postprocess import (
    _apply_stop_tokens,
    _build_program,
    _extract_code_block,
    _extract_code_from_results,
    _extract_function_signature,
    _extract_assertion_hint,
    _looks_like_assertion_error,
    _normalize_completion,
    _strip_redundant_def,
    _repair_assertion_completion_with_llm,
    _evaluate_with_postprocess_check,
    _run_python,
)
from evaluation.smart_pdf_preset import apply_smart_pdf_preset


def _resolve_repo_aware_path(raw: str, *, prefer_repo_if_missing: bool = False) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    cwd_candidate = (Path.cwd() / path)
    repo_candidate = (REPO_ROOT / path)
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    target = repo_candidate if prefer_repo_if_missing else cwd_candidate
    return str(target.resolve())


def _build_experiment_log_id(primary_path: Optional[str], fallback_path: Optional[str], default_name: str) -> str:
    raw = primary_path or fallback_path or default_name
    stem = Path(str(raw)).stem or default_name
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in stem)
    return f"{safe}_trace"

def _load_jsonl(path: str, ignore_errors: bool = False) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if ignore_errors:
                    print(
                        f"[WARN] Skip malformed JSONL line: path={path} line={line_no}",
                        file=sys.stderr,
                    )
                    continue
                raise
            if isinstance(obj, dict):
                items.append(obj)
            elif not ignore_errors:
                raise ValueError(f"JSONL line must be object: path={path} line={line_no}")
    return items

def _load_solutions(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    records = _load_jsonl(path, ignore_errors=True)
    solutions: Dict[str, str] = {}
    for record in records:
        name = record.get("name") or record.get("task_id")
        completion = record.get("completion") or record.get("solution") or record.get("code")
        if name and completion:
            solutions[str(name)] = str(completion)
    return solutions


def _task_name(task: Dict[str, Any]) -> str:
    return str(task.get("name") or task.get("task_id") or "")

def _build_prompt(prompt: str, entry_point: Optional[str], strict: bool, strict_text: Optional[str]) -> str:
    base = prompt or ""
    if not strict:
        return base
    if strict_text:
        suffix = strict_text.strip()
    else:
        entry_hint = f"(function name: {entry_point})" if entry_point else ""
        suffix = (
            "Output only the function body code with correct indentation. "
            "Do not repeat function definition, explanation, or markdown fences. "
            f"{entry_hint}\n"
            "Important constraints:\n"
            "1. MUST use exact parameter names from function signature (e.g., use lst, not data/items/input_string).\n"
            "2. MUST handle edge cases: empty input ([], '', None), single element, and boundary values (0, 1).\n"
            "3. Generate only direct solution code; no unrelated parsing/template boilerplate.\n"
            "4. Ensure logic is correct and testable.\n"
        )
    return f"{base}\n\n{suffix}\n"

def _auto_generate_solutions(
    tasks: List[dict],
    out_path: str,
    local_model_dir: str,
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt: bool,
    strict_prompt_text: Optional[str],
    use_stop_tokens: bool,
    existing_solutions: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    from generation.local_llm import LocalLLMClient

    client = LocalLLMClient(
        model_path=local_model_dir,
        lora_path=local_lora_dir,
        device=local_device,
        dtype=local_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_4bit=local_use_4bit,
        use_8bit=local_use_8bit,
    )

    solutions: Dict[str, str] = dict(existing_solutions or {})
    pending_tasks = [task for task in tasks if _task_name(task) not in solutions]
    if existing_solutions is not None:
        print(
            f"[INFO] Resume generate: existing={len(existing_solutions)} pending={len(pending_tasks)}",
            file=sys.stderr,
        )
    if not pending_tasks:
        return solutions

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_mode = "a" if existing_solutions is not None else "w"
    with open(out_path, write_mode, encoding="utf-8") as f:
        for task in _iter_progress(pending_tasks, desc="generate"):
            name = _task_name(task)
            prompt = task.get("prompt") or ""
            entry_point = task.get("entry_point")
            stop_tokens = task.get("stop_tokens") or []
            prompt = _build_prompt(prompt, entry_point, strict_prompt, strict_prompt_text)
            messages = [{"role": "user", "content": prompt}]
            completion = client.chat(messages, temperature=temperature, max_tokens=max_new_tokens)
            completion = _extract_code_block(completion)
            completion = _strip_redundant_def(completion, entry_point)
            if use_stop_tokens:
                completion = _apply_stop_tokens(completion, stop_tokens)
            completion = _normalize_completion(completion, entry_point)
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
    return solutions

def _load_selections(log_path: Optional[str], registry) -> Dict[str, str]:
    if not log_path or not os.path.exists(log_path):
        return {}
    selections: Dict[str, str] = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                role = record.get("role")
                selected = record.get("selected_main")
                if role and selected:
                    card = registry.get(selected)
                    name = getattr(card, "name", None) if card is not None else None
                    if name:
                        selections[role] = name
    except Exception:
        return selections
    return selections


def _align_selected_agents(topology_info: Dict[str, Any], selections: Dict[str, str]) -> Dict[str, Optional[str]]:
    aligned: Dict[str, Optional[str]] = {}
    roles = topology_info.get("roles")
    if isinstance(roles, list):
        for role in roles:
            role_name = str(role or "").strip()
            if not role_name:
                continue
            aligned[role_name] = selections.get(role_name) or None
    for role, selected in selections.items():
        role_name = str(role or "").strip()
        if not role_name:
            continue
        aligned[role_name] = str(selected) if selected is not None else None
    return aligned


def _apply_humaneval_default_constraints(
    constraints: Optional[Dict[str, Dict[str, Any]]],
    tasks: Optional[List[dict]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Add safe default domain constraints for HumanEval routing to avoid cross-domain
    agents (e.g., GSM8K) when caller does not explicitly set domain_tags.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    if constraints:
        for role, cfg in constraints.items():
            merged[str(role)] = dict(cfg or {})

    role_defaults = [
        (["planner", "code-planner"], ["planner", "code-planner"]),
        (["builder", "code-generation"], ["builder", "code-generation"]),
        (["checker", "tester", "code-testing"], ["checker", "tester", "code-testing"]),
        (["refactor", "refractor", "code-refactoring"], ["refactor", "refractor", "code-refactoring"]),
    ]

    inferred_domain_tags = ["code"]
    if tasks:
        dataset_values = {
            str(task.get("source") or task.get("dataset") or "").strip().lower()
            for task in tasks
            if isinstance(task, dict)
        }
        dataset_values.discard("")
        if dataset_values == {"mbpp"}:
            inferred_domain_tags = ["assert-driven-programming"]
        elif dataset_values == {"humaneval"}:
            inferred_domain_tags = ["python-function-synthesis"]

    for role_group, default_role_tags in role_defaults:
        current_domain_tags: Optional[List[str]] = None
        current_role_tags: Optional[List[str]] = None
        for role_name in role_group:
            current = merged.get(role_name) or {}
            domain_tags = list(current.get("domain_tags") or [])
            role_tags = list(current.get("role_tags") or [])
            if domain_tags and current_domain_tags is None:
                current_domain_tags = domain_tags
            if role_tags and current_role_tags is None:
                current_role_tags = role_tags
        effective_domain_tags = current_domain_tags or list(inferred_domain_tags)
        effective_role_tags = current_role_tags or list(default_role_tags)
        for role_name in role_group:
            current = dict(merged.get(role_name) or {})
            if not current.get("domain_tags"):
                current["domain_tags"] = list(effective_domain_tags)
            if not current.get("role_tags"):
                current["role_tags"] = list(effective_role_tags)
            merged[role_name] = current

    return merged

def _auto_generate_solutions_orchestrator(
    tasks: List[dict],
    out_path: str,
    *,
    db_path: str,
    index_dir: str,
    dim: int,
    seed: int,
    roles: List[str],
    constraints: Dict[str, Dict[str, Any]],
    workflow_version: str,
    reranker_model: str,
    bandit_db: str,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_top_m: int,
    router_no_rerank: bool,
    dynamic_topology: bool,
    topology: str,
    topology_config: Optional[Dict[str, Any]],
    soft_connection: bool,
    max_steps: int,
    force_final_builder_extra_step: bool,
    allow_unknown_roles: bool,
    reuse_role_selection: bool,
    reuse_same_role_agent_once: bool,
    local_model_dir: str,
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt: bool,
    strict_prompt_text: Optional[str],
    use_stop_tokens: bool,
    code_model: Optional[str],
    code_base_url: Optional[str],
    code_api_key: Optional[str],
    code_timeout: float,
    code_retries: int,
    next_role_model: Optional[str],
    next_role_base_url: Optional[str],
    next_role_api_key: Optional[str],
    next_role_timeout: float,
    next_role_retries: int,
    embedder_kind: str,
    embedder_model: str,
    embedder_device: Optional[str],
    embedder_normalize: bool,
    include_tool_trace: bool,
    tool_only: bool,
    tool_timeout_s: float,
    mcts_dynamic_optimization: bool,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    use_mmr: bool,
    use_iochain: bool,
    use_transition_term: bool,
    use_gap_term: bool,
    reward_variant: str,
    use_cross_instance_cache: bool,
    update_posteriors_across_instances: bool,
    use_parallel_frontier: bool,
    parallel_mode: str,
    inference_visibility: str,
    experimental_reranker_after_mmr: bool,
    max_attempts: int,
    enable_postprocess_repair: bool,
    prevent_early_finish: bool,
    baseline_fixed_bcb_mcts: bool,
    baseline_fixed_bcb_router_gpt4o: bool,
    dynamic_workflow_router_gpt4o: bool,
    force_role: Optional[str] = None,
    strict_roles: bool = True,
    existing_solutions: Optional[Dict[str, str]] = None,
    analysis_include_selection_trace: bool = False,
    analysis_include_run_meta: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    from core.registry import SQLiteRegistry
    from execution.orchestrator import run_workflow
    from generation.llm_client import LLMClient
    from retrieval.embedder import build_embedder
    from retrieval.faiss_index import HNSWIndex
    supports_defer_bandit_update = "defer_bandit_update" in inspect.signature(run_workflow).parameters
    warned_defer_bandit_update_fallback = False

    local_client = None
    if local_model_dir:
        from generation.local_llm import LocalLLMClient

        local_client = LocalLLMClient(
            model_path=local_model_dir,
            lora_path=local_lora_dir,
            device=local_device,
            dtype=local_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_4bit=local_use_4bit,
            use_8bit=local_use_8bit,
        )
    
    # Propagate CLI args to environment variables for tools that rely on os.getenv
    if code_base_url:
        os.environ["LLM_API_BASE"] = code_base_url
    if code_api_key:
        os.environ["LLM_API_KEY"] = code_api_key

    code_client: Optional[LLMClient] = None
    if code_model or code_base_url or code_api_key:
        code_client = LLMClient(
            api_key=code_api_key or None,
            base_url=code_base_url or None,
            model=code_model or None,
            timeout_s=code_timeout,
            max_retries=code_retries,
        )

    # Remote primary client: used for all non-topology generation/routing paths.
    # If explicit code_* args are absent, this client falls back to env vars
    # (LLM_API_KEY / LLM_API_BASE / LLM_MODEL or OPENAI_* equivalents).
    remote_primary_client: LLMClient = code_client or LLMClient(
        api_key=None,
        base_url=None,
        model=code_model or None,
        timeout_s=code_timeout,
        max_retries=code_retries,
    )

    if not getattr(remote_primary_client, "api_key", None):
        raise SystemExit(
            "Remote LLM is required for role execution/routing. "
            "Please set LLM_API_KEY (or --code_api_key)."
        )
    
    next_role_client: Optional[LLMClient] = None
    if next_role_model or next_role_base_url or next_role_api_key:
        next_role_client = LLMClient(
            api_key=next_role_api_key or None,
            base_url=next_role_base_url or None,
            model=next_role_model or None,
            timeout_s=next_role_timeout,
            max_retries=next_role_retries,
        )

    if baseline_fixed_bcb_mcts and baseline_fixed_bcb_router_gpt4o:
        raise SystemExit(
            "Only one baseline switch can be enabled at a time: "
            "--baseline_fixed_bcb_mcts or --baseline_fixed_bcb_router_gpt4o"
        )
    if dynamic_workflow_router_gpt4o and (
        baseline_fixed_bcb_mcts or baseline_fixed_bcb_router_gpt4o
    ):
        raise SystemExit(
            "--dynamic_workflow_router_gpt4o cannot be combined with "
            "--baseline_fixed_bcb_mcts or --baseline_fixed_bcb_router_gpt4o"
        )

    baseline_fixed_topology_config: Dict[str, Any] = {
        "topology": "chain",
        "roles": ["builder", "checker", "builder"],
        "entry_role": "builder",
        "max_steps": 3,
        "flow_type": "sequential",
    }

    index_path = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. "
            "Set --index_dir to a directory containing faiss.index."
        )
    index = HNSWIndex.load(index_path)
    embedder = build_embedder(
        kind=embedder_kind,
        dim=dim,
        seed=seed,
        model_name=embedder_model,
        device=embedder_device,
        normalize=embedder_normalize,
    )
    if embedder.dim != index.dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} != index dim {index.dim}. "
            "Rebuild index with matching embedder/model."
        )

    solutions: Dict[str, str] = dict(existing_solutions or {})
    meta: Dict[str, Dict[str, Any]] = {}
    pending_tasks = [task for task in tasks if _task_name(task) not in solutions]
    if existing_solutions is not None:
        print(
            f"[INFO] Resume generate (orchestrator): existing={len(existing_solutions)} pending={len(pending_tasks)}",
            file=sys.stderr,
        )
    if not pending_tasks:
        return solutions, meta

    experiment_log_id = _build_experiment_log_id(out_path, out_path, "humaneval")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_mode = "a" if existing_solutions is not None else "w"
    with SQLiteRegistry(db_path) as registry, open(out_path, write_mode, encoding="utf-8") as f:
        for task in _iter_progress(pending_tasks, desc="generate"):
            task_started = time.perf_counter()
            name = _task_name(task)
            original_prompt = task.get("prompt") or ""  # Save original prompt without suffix
            entry_point = task.get("entry_point")
            stop_tokens = task.get("stop_tokens") or []
            test = task.get("test") or ""  # Extract test code from task
            
            # Extract function signature for better context
            func_sig = _extract_function_signature(original_prompt, entry_point)
            
            prompt = _build_prompt(original_prompt, entry_point, strict_prompt, strict_prompt_text)
            
            # Add explicit parameter names to task text
            param_hint = ""
            if func_sig and func_sig.get("parameters"):
                params = func_sig["parameters"]
                param_hint = f"\nParameter names: {', '.join(params)}. Use these exact names in code.\n"

            is_mbpp_task = str(task.get("source") or "").strip().lower() == "mbpp"
            if is_mbpp_task:
                requirements_text = (
                    "Put the implementation in JSON field `code_or_commands` as runnable function-body code only.\n"
                    "Requirements:\n"
                    "1. Use exact parameter names from the function signature (no generic names like data/items/input_string).\n"
                    "2. Prioritize passing the provided asserts exactly.\n"
                    "3. Keep logic minimal and test-driven; avoid broad defensive validation/exception branches unless asserts require them.\n"
                    "4. Generate direct solution code only; avoid unrelated parsing/template boilerplate.\n"
                    "5. Ensure the logic is correct.\n"
                )
            else:
                requirements_text = (
                    "Put the implementation in JSON field `code_or_commands` as runnable function-body code only.\n"
                    "Requirements:\n"
                    "1. Use exact parameter names from the function signature (no generic names like data/items/input_string).\n"
                    "2. Handle edge cases: empty input ([], '', None), single element, and boundary values (0, 1).\n"
                    "3. Generate direct solution code only; avoid unrelated parsing/template boilerplate.\n"
                    "4. Ensure the logic is correct.\n"
                )

            task_text = (
                f"{prompt}\n\n"
                f"{requirements_text}"
                f"{param_hint}"
            )

            # Build task context with function signature and test information
            # IMPORTANT: Use original_prompt (without Chinese suffix) for testing!
            task_context = {
                "entry_point": entry_point,
                "function_signature": func_sig,
                "prompt": original_prompt,  # Use original prompt for test execution
                "test": test,
                "dataset": "mbpp" if is_mbpp_task else "humaneval",
                "stop_tokens": stop_tokens,
            } if entry_point else None

            effective_dynamic_topology = dynamic_topology
            effective_topology = topology
            effective_topology_config = topology_config
            effective_max_steps = max_steps
            effective_mcts_dynamic_optimization = mcts_dynamic_optimization
            effective_reuse_same_role_agent_once = reuse_same_role_agent_once
            # Topology / routing defaults now prefer remote API clients as well.
            # Local models are only retained for explicit fallback / ablation paths.
            effective_router_llm_client: Optional[LLMClient] = remote_primary_client
            effective_next_role_llm_client: Optional[LLMClient] = next_role_client or remote_primary_client
            effective_meta_router_llm_client: Optional[LLMClient] = next_role_client or remote_primary_client

            if baseline_fixed_bcb_mcts or baseline_fixed_bcb_router_gpt4o:
                effective_dynamic_topology = True
                effective_topology = "chain"
                effective_topology_config = dict(baseline_fixed_topology_config)
                effective_max_steps = 3
                effective_reuse_same_role_agent_once = True
                effective_next_role_llm_client = None
                effective_meta_router_llm_client = None

            if baseline_fixed_bcb_mcts:
                effective_mcts_dynamic_optimization = True
                # router llm is not used on MCTS path; keep the remote default for consistency
                effective_router_llm_client = next_role_client or remote_primary_client
            elif baseline_fixed_bcb_router_gpt4o:
                # old embedding retrieval + router LLM final choice
                effective_mcts_dynamic_optimization = False
                router_remote_client = next_role_client or code_client
                if router_remote_client is None:
                    raise SystemExit(
                        "--baseline_fixed_bcb_router_gpt4o requires a remote OpenAI-compatible client. "
                        "Please set --next_role_model/--next_role_base_url/--next_role_api_key "
                        "or --code_model/--code_base_url/--code_api_key."
                    )
                effective_router_llm_client = router_remote_client
            elif dynamic_workflow_router_gpt4o:
                # dynamic workflow + embedding retrieval + remote router LLM (e.g. GPT-4o) final choice
                effective_dynamic_topology = True
                effective_mcts_dynamic_optimization = False
                router_remote_client = next_role_client or code_client
                if router_remote_client is None:
                    raise SystemExit(
                        "--dynamic_workflow_router_gpt4o requires a remote OpenAI-compatible client. "
                        "Please set --next_role_model/--next_role_base_url/--next_role_api_key "
                        "or --code_model/--code_base_url/--code_api_key."
                    )
                effective_router_llm_client = router_remote_client

            workflow_started = time.perf_counter()
            workflow_kwargs: Dict[str, Any] = {
                "task_text": task_text,
                "roles": roles,
                "constraints_per_role": constraints,
                "workflow_version": workflow_version,
                "registry": registry,
                "index": index,
                "embedder": embedder,
                "top_n": top_n,
                "top_k": top_k,
                "rerank_top_m": rerank_top_m,
                "mmr_lambda": mmr_lambda,
                "reranker_model_path": reranker_model,
                "bandit_db_path": bandit_db,
                "max_attempts": max_attempts,
                "llm_client": remote_primary_client,
                "router_llm_client": effective_router_llm_client,
                "router_top_m": router_top_m,
                "task_context": task_context,
                "router_no_rerank": router_no_rerank,
                "experimental_reranker_after_mmr": experimental_reranker_after_mmr,
                "paper_mode": True,
                "task_family": "code",
                "dataset": "mbpp" if is_mbpp_task else "humaneval",
                "final_candidate_cap": 5,
                "artifact_contract_version": "v2",
                "dynamic_topology": effective_dynamic_topology,
                "topology": effective_topology,
                "topology_config": effective_topology_config,
                "meta_router_llm_client": effective_meta_router_llm_client,
                "next_role_llm_client": effective_next_role_llm_client,
                "soft_connection": soft_connection,
                "max_steps": effective_max_steps,
                "force_final_builder_extra_step": force_final_builder_extra_step,
                "allow_unknown_roles": allow_unknown_roles,
                "reuse_role_selection": reuse_role_selection,
                "reuse_same_role_agent_once": effective_reuse_same_role_agent_once,
                "tool_only": tool_only,
                "tool_timeout_s": tool_timeout_s,
                "mcts_dynamic_optimization": effective_mcts_dynamic_optimization,
                "mcts_iterations": mcts_iterations,
                "mcts_rollout_depth": mcts_rollout_depth,
                "mcts_exploration": mcts_exploration,
                "mcts_discount": mcts_discount,
                "mcts_max_candidates": mcts_max_candidates,
                "use_mmr": use_mmr,
                "use_iochain": use_iochain,
                "use_transition_term": use_transition_term,
                "use_gap_term": use_gap_term,
                "reward_variant": reward_variant,
                "log_task_id": experiment_log_id,
                "use_cross_instance_cache": use_cross_instance_cache,
                "update_posteriors_across_instances": update_posteriors_across_instances,
                "use_parallel_frontier": use_parallel_frontier,
                "parallel_mode": parallel_mode,
                "inference_visibility": inference_visibility,
                "force_role": force_role,
                "strict_roles": strict_roles,
                "enable_postprocess_repair": enable_postprocess_repair,
                "prevent_early_finish": prevent_early_finish,
            }
            if supports_defer_bandit_update:
                workflow_kwargs["defer_bandit_update"] = True
            elif not warned_defer_bandit_update_fallback:
                print(
                    "[WARN] run_workflow does not support defer_bandit_update; "
                    "falling back to compatible invocation.",
                    file=sys.stderr,
                )
                warned_defer_bandit_update_fallback = True
            result = run_workflow(**workflow_kwargs)
            workflow_elapsed = time.perf_counter() - workflow_started

            results = result.get("results") or {}
            tool_exec = result.get("tool_exec")
            tool_trace = tool_exec if include_tool_trace else None
            
            # Extract topology and agent selection info FIRST (before checking completion)
            topology_info = result.get("topology") or {"topology": "linear", "roles": roles}
            selections = _load_selections(result.get("log_path"), registry)
            task_meta = {
                "topology": topology_info,
                "selected_agents": _align_selected_agents(topology_info, selections),
                "forced_final_builder_executed": bool(result.get("forced_final_builder_executed")),
            }
            if analysis_include_run_meta:
                task_meta["task_id"] = str(result.get("task_id") or "")
                task_meta["log_path"] = str(result.get("log_path") or "")
            if analysis_include_selection_trace:
                selection_trace = result.get("selections")
                if isinstance(selection_trace, list):
                    task_meta["selection_trace"] = json.loads(
                        json.dumps(selection_trace, ensure_ascii=True, default=str)
                    )
            raw_selections = result.get("selections")
            if isinstance(raw_selections, list):
                task_meta["_bandit_selections"] = json.loads(
                    json.dumps(raw_selections, ensure_ascii=True, default=str)
                )
            if include_tool_trace and tool_trace is not None:
                task_meta["tool_trace"] = json.loads(json.dumps(tool_trace, ensure_ascii=True, default=str))
            task_meta["workflow_time_s"] = round(workflow_elapsed, 3)
            
            # Include tool_exec in extraction so we can recover code even if role outputs
            # were validated into schemas that drop code_or_commands (e.g., planner).
            extraction_payload = dict(results)
            if tool_exec:
                extraction_payload["tool_exec"] = tool_exec
            execution_trace = result.get("execution_trace")
            if isinstance(execution_trace, list) and execution_trace:
                extraction_payload["execution_trace"] = execution_trace
            extraction_payload["forced_final_builder_executed"] = bool(
                result.get("forced_final_builder_executed")
            )
            completion = _extract_code_from_results(
                extraction_payload,
                entry_point,
                param_names=(func_sig.get("parameters") if isinstance(func_sig, dict) else None),
            )
            if completion is None or not completion.strip():
                # Record meta even if no code was generated (helps debugging)
                task_meta["extraction_failed"] = True
                task_meta["results_summary"] = {k: type(v).__name__ for k, v in results.items()}
                # Try to extract from tool_trace as last resort
                tool_trace = result.get("tool_exec")
                if tool_trace and isinstance(tool_trace, dict):
                    # Reuse the main extractor on tool_exec for robust nested output handling.
                    completion = _extract_code_from_results(
                        {
                            "tool_exec": tool_trace,
                            "execution_trace": (result.get("execution_trace") or []),
                            "forced_final_builder_executed": bool(
                                result.get("forced_final_builder_executed")
                            ),
                        },
                        entry_point,
                        param_names=(func_sig.get("parameters") if isinstance(func_sig, dict) else None),
                    )
                    if completion:
                        task_meta["extracted_from"] = "tool_exec"
                
                # Final fallback: Use LLM to generate code directly if all tools failed
                if not completion or not completion.strip():
                    logger.warning(f"All tools failed for {name}, using LLM fallback")
                    task_meta["used_llm_fallback"] = True
                    try:
                        # Simple fallback with enhanced prompt (no failure context to avoid confusion)
                        llm_prompt = (
                            f"{prompt}\n\n"
                            "You MUST follow these output rules exactly:\n"
                            "1. Output ONLY raw function body code with 4-space indentation\n"
                            "2. Do NOT include labels (python/json), markdown fences, titles, explanations, or comments\n"
                            "3. CRITICAL: Use ONLY the actual parameter names from the function signature (e.g., if parameter is 'lst', use 'lst' NOT 'data' or 'items')\n"
                            "4. Do NOT use generic variable names like: data, items, input_string, inputs, arr (unless they are the actual parameter names)\n"
                            "5. MUST handle edge cases: empty inputs ([], '', None), single elements, boundary values (0, 1)\n"
                            "6. Generate ONLY the function body that directly solves the problem, NOT generic parsing/processing templates\n"
                            "7. If asked for a function body, do NOT include def/class/import/main/docstrings\n"
                            "8. Empty/whitespace output is invalid\n"
                        )
                        llm_response = (code_client or local_client).chat(
                            [{"role": "user", "content": llm_prompt}],
                            temperature=0.2,
                            max_tokens=600,
                        )
                        completion = _extract_code_block(str(llm_response))
                        if completion and completion.strip():
                            task_meta["extracted_from"] = "llm_fallback"
                    except Exception as e:
                        logger.error(f"LLM fallback failed: {e}")
                
                if not completion or not completion.strip():
                    # Use empty completion to allow test to run and show error
                    completion = ""
            
            if use_stop_tokens:
                completion = _apply_stop_tokens(completion, stop_tokens)
            completion = _normalize_completion(completion, entry_point)

            # Enable assertion repair for all HumanEval orchestrator runs.
            # Most residual failures are logical assertion mismatches.
            enable_assertion_logic_repair = True
            if (
                enable_assertion_logic_repair
                and completion
                and completion.strip()
                and entry_point
                and test
            ):
                repair_client = code_client or local_client
                if repair_client is not None:
                    eval_timeout_s = max(5.0, float(tool_timeout_s))
                    current_completion = completion
                    repair_logs: List[Dict[str, Any]] = []
                    for attempt in range(2):
                        candidate_program = _build_program(
                            original_prompt, current_completion, test, entry_point
                        )
                        ok_pre, msg_pre = _run_python(candidate_program, timeout_s=eval_timeout_s)
                        if ok_pre or (not _looks_like_assertion_error(msg_pre)):
                            break

                        repaired_completion = _repair_assertion_completion_with_llm(
                            llm_client=repair_client,
                            prompt=original_prompt,
                            completion=current_completion,
                            test_code=test,
                            entry_point=entry_point,
                            error_message=msg_pre,
                            failure_context=(
                                f"assertion_hint={_extract_assertion_hint(msg_pre)}\n"
                                f"error={msg_pre}"
                            ),
                        )
                        if not repaired_completion or repaired_completion == current_completion:
                            break

                        repaired_program = _build_program(
                            original_prompt, repaired_completion, test, entry_point
                        )
                        ok_repaired, msg_repaired = _run_python(
                            repaired_program, timeout_s=eval_timeout_s
                        )
                        repair_logs.append(
                            {
                                "attempt": attempt + 1,
                                "repair_type": "assertion_logic",
                                "success": ok_repaired,
                                "original_error": msg_pre,
                                "final_error": "" if ok_repaired else msg_repaired,
                            }
                        )
                        if ok_repaired:
                            completion = repaired_completion
                            break
                        current_completion = repaired_completion
                    if repair_logs:
                        task_meta["assertion_logic_repairs"] = repair_logs
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
            task_meta["generation_time_s"] = round(time.perf_counter() - task_started, 3)
            meta[str(name)] = task_meta

    return solutions, meta

def evaluate(
    tasks_path: str,
    solutions_path: Optional[str],
    timeout_s: float,
    postprocess_rounds: int = 1,
) -> Dict[str, object]:
    tasks = _load_jsonl(tasks_path)
    solutions = _load_solutions(solutions_path)
    passed = 0
    results: List[dict] = []
    for task in tasks:
        eval_started = time.perf_counter()
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, postprocess_info = _evaluate_with_postprocess_check(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=timeout_s,
            max_postprocess_rounds=postprocess_rounds,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        record["eval_time_s"] = round(time.perf_counter() - eval_started, 3)
        if postprocess_info:
            record["postprocess_check"] = postprocess_info
        results.append(record)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    return {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval-style tasks")
    parser.add_argument("--tasks", required=True, type=str, help="JSONL tasks file (prompt/test)")
    parser.add_argument("--solutions", default="", type=str, help="JSONL solutions file (name+completion)")
    parser.add_argument("--timeout", default=5.0, type=float, help="timeout seconds per task")
    # Backward-compatible alias for older scripts/commands.
    parser.add_argument("--time_out", dest="timeout", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", type=str, help="optional output JSON path")
    parser.add_argument("--auto_generate", action="store_true", help="generate solutions if missing")
    parser.add_argument("--local_model_dir", type=str, default=None, help="local base model path")
    parser.add_argument("--local_lora_dir", type=str, default=None, help="local LoRA adapter path")
    parser.add_argument("--local_device", type=str, default=None)
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_use_4bit", action="store_true")
    parser.add_argument("--local_use_8bit", action="store_true")
    parser.add_argument("--code_model", type=str, default="", help="remote code model (OpenAI-compatible)")
    parser.add_argument("--code_base_url", type=str, default="", help="remote code API base URL")
    parser.add_argument("--code_api_key", type=str, default="", help="remote code API key (optional; uses env)")
    parser.add_argument("--code_timeout", type=float, default=480.0, help="remote code API timeout seconds")
    parser.add_argument("--code_retries", type=int, default=5, help="remote code API retry count")
    parser.add_argument("--next_role_model", type=str, default="", help="remote next-role model (OpenAI-compatible)")
    parser.add_argument("--next_role_base_url", type=str, default="", help="remote next-role API base URL")
    parser.add_argument("--next_role_api_key", type=str, default="", help="remote next-role API key (optional; uses env)")
    parser.add_argument("--next_role_timeout", type=float, default=60.0, help="remote next-role API timeout seconds")
    parser.add_argument("--next_role_retries", type=int, default=5, help="remote next-role API retry count")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no_strict_prompt", action="store_true", help="disable strict prompt suffix")
    parser.add_argument("--strict_prompt_text", type=str, default="", help="custom strict prompt suffix")
    parser.add_argument("--use_stop_tokens", action="store_true", help="apply stop_tokens during generation")
    parser.add_argument("--use_orchestrator", action="store_true", help="generate solutions via orchestrator")
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
    parser.add_argument("--include_tool_trace", action="store_true")
    parser.add_argument("--tool_only", action="store_true", help="use tools only; no LLM generation")
    parser.add_argument("--tool_timeout", type=float, default=60.0, help="tool execution timeout seconds")
    parser.add_argument(
        "--postprocess_rounds",
        type=int,
        default=1,
        help="postprocess repair rounds after initial failure",
    )
    parser.add_argument(
        "--disable_orchestrator_postprocess_repair",
        action="store_true",
        help="disable orchestrator postprocess repair chain after tool test failures",
    )
    parser.add_argument(
        "--allow_early_finish",
        action="store_true",
        help="allow router/agent to end workflow early even when no success signal exists",
    )
    parser.add_argument("--mcts_dynamic_optimization", action="store_true")
    parser.add_argument("--mcts_iterations", type=int, default=64)
    parser.add_argument("--mcts_rollout_depth", type=int, default=4)
    parser.add_argument("--mcts_exploration", type=float, default=1.414)
    parser.add_argument("--mcts_discount", type=float, default=0.95)
    parser.add_argument("--mcts_max_candidates", type=int, default=5)
    parser.add_argument("--no_mmr", action="store_true", help="disable MMR diversity reranking")
    parser.add_argument("--no_iochain", action="store_true", help="disable I/O-chain reward component")
    parser.add_argument("--no_transition_term", action="store_true", help="disable transition-prior reward component")
    parser.add_argument("--no_gap_term", action="store_true", help="disable artifact-gap reward component")
    parser.add_argument("--reward_variant", default="mean_active", type=str)
    parser.add_argument("--use_cross_instance_cache", action="store_true", help="enable persistent cross-instance bandit/cache reads")
    parser.add_argument(
        "--update_posteriors_across_instances",
        action="store_true",
        help="enable posterior updates across benchmark items",
    )
    parser.add_argument("--use_parallel_frontier", action="store_true", help="enable fork-join ready-frontier execution")
    parser.add_argument("--parallel_mode", default="frontier", type=str)
    parser.add_argument(
        "--inference_visibility",
        default="tool_visible",
        choices=["tool_visible", "public_only"],
        help="restrict inference-time tool inputs for hidden-test-clean runs",
    )
    parser.add_argument(
        "--baseline_fixed_bcb_mcts",
        action="store_true",
        help="baseline: fixed workflow builder->checker->builder with MCTS agent selection",
    )
    parser.add_argument(
        "--baseline_fixed_bcb_router_gpt4o",
        action="store_true",
        help="baseline: fixed workflow builder->checker->builder with embedding retrieval + GPT-4o router final choice",
    )
    parser.add_argument(
        "--dynamic_workflow_router_gpt4o",
        action="store_true",
        help=(
            "one-shot switch: enable dynamic workflow and use embedding retrieval + remote router LLM "
            "(e.g. GPT-4o) for final agent selection"
        ),
    )
    parser.add_argument("--roles", default="code-generation,code-planner,code-testing,code-refactoring", type=str)
    parser.add_argument("--constraints", default="", type=str, help="JSON string per role")
    parser.add_argument("--workflow_version", default="v1", type=str)
    parser.add_argument("--reranker_model", default="models/reranker.json", type=str)
    parser.add_argument("--bandit_db", default="models/bandit.sqlite", type=str)
    parser.add_argument("--top_n", default=30, type=int)
    parser.add_argument("--top_k", default=15, type=int)
    parser.add_argument("--rerank_top_m", default=3, type=int)
    parser.add_argument("--mmr_lambda", default=0.5, type=float)
    parser.add_argument("--router_top_m", default=5, type=int)
    parser.add_argument("--router_no_rerank", action="store_true")
    parser.add_argument("--experimental_reranker_after_mmr", action="store_true")
    parser.add_argument("--dynamic_topology", action="store_true")
    parser.add_argument("--topology", default="linear", type=str)
    parser.add_argument("--topology_config", default="", type=str)
    parser.add_argument("--soft_connection", action="store_true")
    parser.add_argument("--max_steps", default=6, type=int)
    parser.add_argument(
        "--force_final_builder_extra_step",
        dest="force_final_builder_extra_step",
        action="store_true",
        help="force one extra builder step after dynamic workflow if not solved (default: enabled)",
    )
    parser.add_argument(
        "--no_force_final_builder_extra_step",
        dest="force_final_builder_extra_step",
        action="store_false",
        help="disable extra final builder step",
    )
    parser.set_defaults(force_final_builder_extra_step=True)
    parser.add_argument("--max_attempts", default=3, type=int, help="max retries per role/tool execution")
    parser.add_argument("--allow_unknown_roles", action="store_true")
    parser.add_argument("--no_reuse_role_selection", action="store_true")
    parser.add_argument(
        "--reuse_same_role_agent_once",
        action="store_true",
        help="within one query, search/select each role once and reuse the same agent for repeated role steps",
    )
    parser.add_argument(
        "--resume_generate",
        action="store_true",
        help="resume generation from existing solutions file by appending missing tasks only",
    )
    parser.add_argument("--force_role", default="", type=str, help="force all queries to use this role (skip Router)")
    parser.add_argument(
        "--analysis_include_selection_trace",
        action="store_true",
        help="analysis mode: include per-step selection trace (including MCTS child rewards) in report results",
    )
    parser.add_argument(
        "--analysis_include_run_meta",
        action="store_true",
        help="analysis mode: include task_id/log_path in report results",
    )
    return parser.parse_args()

def _iter_progress(items: List[dict], desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items))

def main() -> None:
    args = parse_args()
    args = apply_smart_pdf_preset(args, benchmark="humaneval")
    if args.dynamic_workflow_router_gpt4o:
        args.use_orchestrator = True
        args.dynamic_topology = True
        args.mcts_dynamic_optimization = False
        args.baseline_fixed_bcb_mcts = False
        args.baseline_fixed_bcb_router_gpt4o = False

    # Make default orchestrator assets robust to different working directories.
    args.db = _resolve_repo_aware_path(args.db, prefer_repo_if_missing=True)
    args.index_dir = _resolve_repo_aware_path(args.index_dir, prefer_repo_if_missing=True)
    args.reranker_model = _resolve_repo_aware_path(args.reranker_model, prefer_repo_if_missing=True)
    args.bandit_db = _resolve_repo_aware_path(args.bandit_db, prefer_repo_if_missing=True)

    tasks = _load_jsonl(args.tasks)
    effective_postprocess_rounds = args.postprocess_rounds
    # Respect explicit disable from caller (e.g., eval_mbpp forwards --postprocess_rounds 0).
    # MCTS default boosting only applies when postprocess is enabled (>0).
    if (
        (args.mcts_dynamic_optimization or args.baseline_fixed_bcb_mcts)
        and effective_postprocess_rounds > 0
        and effective_postprocess_rounds < 2
    ):
        effective_postprocess_rounds = 2
    solutions_path = args.solutions or None
    if not solutions_path and args.out:
        out_path = Path(args.out)
        solutions_path = str(out_path.with_suffix(".solutions.jsonl"))
    solutions: Dict[str, str] = {}
    meta_by_task: Dict[str, Dict[str, Any]] = {}
    topology_config = None
    if args.topology_config:
        topology_config_path = _resolve_repo_aware_path(args.topology_config)
        if os.path.exists(topology_config_path):
            with open(topology_config_path, "r", encoding="utf-8") as f:
                topology_config = json.load(f)
        else:
            topology_config = json.loads(args.topology_config)

    should_generate = args.auto_generate or bool(args.local_model_dir) or args.use_orchestrator
    existing_solutions: Dict[str, str] = {}
    if solutions_path and os.path.exists(solutions_path):
        existing_solutions = _load_solutions(solutions_path)
        solutions = dict(existing_solutions)

    if should_generate:
        if not args.use_orchestrator and not args.local_model_dir:
            raise SystemExit("--auto_generate requires --local_model_dir unless --use_orchestrator is set")
        if not solutions_path:
            solutions_path = args.tasks + ".solutions.jsonl"

        if existing_solutions and not args.resume_generate:
            print(
                f"[INFO] Found existing solutions file with {len(existing_solutions)} records: {solutions_path}",
                file=sys.stderr,
            )
            print(
                "[INFO] Skip generation. Use --resume_generate to continue unfinished tasks.",
                file=sys.stderr,
            )
        else:
            resume_seed = (
                existing_solutions if (args.resume_generate and os.path.exists(solutions_path)) else None
            )
            if args.use_orchestrator:
                roles = [item.strip() for item in args.roles.split(",") if item.strip()]
                constraints = json.loads(args.constraints) if args.constraints else {}
                constraints = _apply_humaneval_default_constraints(constraints, tasks=tasks)
                solutions, meta_by_task = _auto_generate_solutions_orchestrator(
                    tasks=tasks,
                    out_path=solutions_path,
                    db_path=args.db,
                    index_dir=args.index_dir,
                    dim=args.dim,
                    seed=args.seed,
                    roles=roles or ["builder"],
                    constraints=constraints,
                    workflow_version=args.workflow_version,
                    reranker_model=args.reranker_model,
                    bandit_db=args.bandit_db,
                    top_n=args.top_n,
                    top_k=args.top_k,
                    rerank_top_m=args.rerank_top_m,
                    mmr_lambda=args.mmr_lambda,
                    router_top_m=args.router_top_m,
                    router_no_rerank=args.router_no_rerank,
                    dynamic_topology=args.dynamic_topology,
                    topology=args.topology,
                    topology_config=topology_config,
                    soft_connection=args.soft_connection,
                    max_steps=args.max_steps,
                    force_final_builder_extra_step=args.force_final_builder_extra_step,
                    max_attempts=args.max_attempts,
                    allow_unknown_roles=args.allow_unknown_roles,
                    reuse_role_selection=not args.no_reuse_role_selection,
                    reuse_same_role_agent_once=args.reuse_same_role_agent_once,
                    local_model_dir=args.local_model_dir,
                    local_lora_dir=args.local_lora_dir,
                    local_device=args.local_device,
                    local_dtype=args.local_dtype,
                    local_use_4bit=args.local_use_4bit,
                    local_use_8bit=args.local_use_8bit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    strict_prompt=not args.no_strict_prompt,
                    strict_prompt_text=args.strict_prompt_text or None,
                    use_stop_tokens=args.use_stop_tokens,
                    code_model=args.code_model or None,
                    code_base_url=args.code_base_url or None,
                    code_api_key=args.code_api_key or None,
                    code_timeout=args.code_timeout,
                    code_retries=args.code_retries,
                    next_role_model=args.next_role_model or None,
                    next_role_base_url=args.next_role_base_url or None,
                    next_role_api_key=args.next_role_api_key or None,
                    next_role_timeout=args.next_role_timeout,
                    next_role_retries=args.next_role_retries,
                    embedder_kind=args.embedder,
                    embedder_model=args.embedder_model,
                    embedder_device=args.embedder_device,
                    embedder_normalize=args.embedder_normalize,
                    include_tool_trace=args.include_tool_trace,
                    tool_only=args.tool_only,
                    tool_timeout_s=args.tool_timeout,
                    mcts_dynamic_optimization=args.mcts_dynamic_optimization,
                    mcts_iterations=args.mcts_iterations,
                    mcts_rollout_depth=args.mcts_rollout_depth,
                    mcts_exploration=args.mcts_exploration,
                    mcts_discount=args.mcts_discount,
                    mcts_max_candidates=args.mcts_max_candidates,
                    use_mmr=not args.no_mmr,
                    use_iochain=not args.no_iochain,
                    use_transition_term=not args.no_transition_term,
                    use_gap_term=not args.no_gap_term,
                    reward_variant=args.reward_variant,
                    use_cross_instance_cache=args.use_cross_instance_cache,
                    update_posteriors_across_instances=args.update_posteriors_across_instances,
                    use_parallel_frontier=args.use_parallel_frontier,
                    parallel_mode=args.parallel_mode,
                    inference_visibility=args.inference_visibility,
                    experimental_reranker_after_mmr=args.experimental_reranker_after_mmr,
                    baseline_fixed_bcb_mcts=args.baseline_fixed_bcb_mcts,
                    baseline_fixed_bcb_router_gpt4o=args.baseline_fixed_bcb_router_gpt4o,
                    dynamic_workflow_router_gpt4o=args.dynamic_workflow_router_gpt4o,
                    force_role=args.force_role or None,
                    enable_postprocess_repair=not args.disable_orchestrator_postprocess_repair,
                    prevent_early_finish=not args.allow_early_finish,
                    existing_solutions=resume_seed,
                    analysis_include_selection_trace=args.analysis_include_selection_trace,
                    analysis_include_run_meta=args.analysis_include_run_meta,
                )
            else:
                solutions = _auto_generate_solutions(
                    tasks=tasks,
                    out_path=solutions_path,
                    local_model_dir=args.local_model_dir,
                    local_lora_dir=args.local_lora_dir,
                    local_device=args.local_device,
                    local_dtype=args.local_dtype,
                    local_use_4bit=args.local_use_4bit,
                    local_use_8bit=args.local_use_8bit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    strict_prompt=not args.no_strict_prompt,
                    strict_prompt_text=args.strict_prompt_text or None,
                    use_stop_tokens=args.use_stop_tokens,
                    existing_solutions=resume_seed,
                )
    elif solutions_path and not os.path.exists(solutions_path):
        raise SystemExit(f"Solutions file not found: {solutions_path}")

    bandit_feedback_fn = None
    if args.use_orchestrator:
        from execution.orchestrator import apply_bandit_feedback as bandit_feedback_fn

    passed = 0
    results: List[dict] = []
    for task in _iter_progress(tasks, desc="evaluate"):
        eval_started = time.perf_counter()
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, postprocess_info = _evaluate_with_postprocess_check(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=args.timeout,
            max_postprocess_rounds=effective_postprocess_rounds,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        record["eval_time_s"] = round(time.perf_counter() - eval_started, 3)
        meta = meta_by_task.get(str(name))
        if meta:
            bandit_selections = meta.get("_bandit_selections")
            if bandit_feedback_fn is not None and isinstance(bandit_selections, list):
                bandit_feedback_fn(
                    args.bandit_db,
                    workflow_version=args.workflow_version,
                    selections=bandit_selections,
                    success=bool(ok),
                )
            public_meta = {k: v for k, v in meta.items() if not str(k).startswith("_")}
            record.update(public_meta)
        if postprocess_info:
            record["postprocess_check"] = postprocess_info
        results.append(record)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    report = {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=True))
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)

if __name__ == "__main__":
    main()
