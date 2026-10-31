from __future__ import annotations

import json
import os
import re
import sys
import math
import hashlib
import sqlite3
import subprocess
import tempfile
import textwrap
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.contracts import validate_output
from core.failure_router import route_failure
from core.logging import write_event
from core.query_builder import build_role_query
from core.registry import SQLiteRegistry
from core.topology import TopologyConfig, TopologyType
from execution.parallel_aggregation import aggregate_parallel_results
from execution.parallel_executor import ParallelBranchResult, ParallelBranchSpec, run_parallel_branches
from execution.probe_commit import probe_commit
from execution.tool_executor import ToolExecutor
from generation.llm_client import LLMClient
from retrieval.parallel_search import ParallelCandidateRequest, get_candidates_batch
from retrieval.search_service import get_candidates
from retrieval.search_text import card_search_text, sanitize_search_text
from routing.bandit_store import BanditStore
from routing.reranker_model import TfidfLinearReranker
from routing.smart_mcts import select_with_smart_mcts
from routing.smart_reward import build_smart_candidates, io_chain_score as _smart_io_chain_score, score_smart_reward


class MockLLMClient:
    """Mock LLM client that returns structured outputs per role."""

    def generate(self, role: str, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        agent = context.get("agent", {})
        agent_name = agent.get("name") or agent.get("id") or "agent"
        agent_domain = ",".join(agent.get("domain_tags") or [])
        agent_tools = ",".join(agent.get("tool_tags") or [])
        role_key = _normalize_role_name(role)
        if role_key == "planner":
            return {
                "steps": [
                    f"{agent_name} clarify scope for: {task_text}",
                    "Break down milestones",
                    "Sequence tasks with dependencies",
                ],
                "acceptance_criteria": [
                    "Plan covers core tasks",
                    f"Dependencies explicit for {agent_domain or 'general'}",
                ],
            }
        if role_key == "evidence_finder":
            return {
                "evidence_targets": [f"{task_text} provided-context evidence", f"{task_text} {agent_domain}"],
                "sources": ["provided_context", "retrieved_passages"],
                "evidence_points": [f"Relevant evidence summarized with {agent_tools or 'tools'}"],
            }
        if role_key == "builder":
            return {
                "runnable_plan": ["Implement core logic", "Wire dependencies", "Run tests"],
                "code_or_commands": f"run build; use {agent_tools or 'default tools'}",
                "self_test": ["unit tests pass", "smoke test pass"],
            }
        if role_key == "refactor":
            return {
                "runnable_plan": ["Refactor implementation", "Preserve behavior", "Re-run regression tests"],
                "code_or_commands": f"run refactor; use {agent_tools or 'default tools'}",
                "self_test": ["regression tests pass", "smoke test pass"],
            }
        if role_key == "checker":
            return {
                "test_cases": ["happy path", "edge case"],
                "verdicts": ["pass", "needs review"],
                "failure_localization": f"review edge case for {agent_name}",
            }
        if role_key == "manager":
            roles = context.get("available_roles") or []
            next_role = roles[0] if roles else "builder"
            return {
                "status": "delegate",
                "next_role": next_role,
                "instruction": f"Handle: {task_text}",
                "summary": f"Delegated to {next_role}",
                "notes": ["Use upstream context for continuity."],
            }
        return {}

    def rewrite_format(self, role: str, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return self.generate(role, "format_fix", context)

    def clarify(self, role: str, task_text: str) -> Dict[str, Any]:
        return {"clarifier": f"Provide missing details for {role} about {task_text}"}

    def plan_tool(
        self,
        role: str,
        task_text: str,
        context: Dict[str, Any],
        tools: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        benchmark_decision = _benchmark_tool_decision(role, context, tools, tool_history)
        if benchmark_decision is not None:
            return benchmark_decision
        if context.get("tool_only") and tools:
            return {
                "action": "tool",
                "tool_id": tools[0].get("id"),
                "tool_input": {},
                "reason": "tool_only_mock",
            }
        return {"action": "final", "reason": "mock"}


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_PURE_NUMERIC_ANSWER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))(?:/[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+)))?$"
)
_PLACEHOLDER_FINAL_ANSWER_RE = re.compile(
    r"^<\s*number\s*>(?:\s*\.)?(?:\s*without\b.*)?$",
    re.IGNORECASE,
)
_CHAT_ROLE_LINE_RE = re.compile(r"^\s*(?:assistant|user|system|tool)\s*:?\s*$", re.IGNORECASE)
_CHAT_ROLE_TAG_RE = re.compile(r"^\s*</?(?:assistant|user|system|tool)\s*>\s*$", re.IGNORECASE)
_CHAT_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|user|system|tool)\s*:\s*(.*)$", re.IGNORECASE)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _safe_load_json(text: str, fallback: Any) -> Any:
    blob = _extract_json_blob(text) or text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return fallback


def _normalize_dataset_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text else "unknown"


def _infer_task_family(dataset: Any, task_context: Optional[Dict[str, Any]]) -> str:
    dataset_name = _normalize_dataset_name(dataset)
    if dataset_name in {"humaneval", "mbpp"}:
        return "code"
    if dataset_name == "gsm8k":
        return "gsm8k"
    if dataset_name == "math":
        return "math"
    if dataset_name == "hotpotqa":
        return "qa"
    if isinstance(task_context, dict):
        if task_context.get("question") and task_context.get("context"):
            return "qa"
        if task_context.get("entry_point") or task_context.get("test") or task_context.get("test_list"):
            return "code"
    return "generic"


def _is_benchmark_task_family(task_family: Any) -> bool:
    return str(task_family or "").strip().lower() in {"code", "gsm8k", "math", "qa", "hotpotqa"}


def _extract_primary_task_text(task_context: Optional[Dict[str, Any]], task_text: str) -> str:
    if not isinstance(task_context, dict):
        return str(task_text or "")
    for key in ("prompt", "question", "problem", "query", "task"):
        value = task_context.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return str(task_text or "")


def _normalize_inference_visibility(value: Optional[str]) -> str:
    text = str(value or "tool_visible").strip().lower().replace("-", "_")
    if text in {"public", "public_only", "hidden_test_clean", "clean"}:
        return "public_only"
    if text in {"tool_visible", "debug", "diagnostic", "full"}:
        return "tool_visible"
    return "tool_visible"


def _sanitize_task_context_for_inference(
    task_context: Optional[Dict[str, Any]],
    inference_visibility: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(task_context, dict):
        return task_context
    visibility = _normalize_inference_visibility(inference_visibility)
    if visibility != "public_only":
        sanitized = dict(task_context)
        sanitized["inference_visibility"] = visibility
        return sanitized

    sanitized = dict(task_context)
    for key in (
        "answer",
        "expected",
        "expected_answer",
        "gold",
        "gold_answer",
        "label",
        "labels",
        "supporting_facts",
        "supporting_fact_labels",
        "hidden_tests",
        "challenge_tests",
    ):
        sanitized.pop(key, None)

    # In HumanEval and similar code benchmarks, `test` commonly denotes the
    # evaluator-only test program. Keep only explicitly public/prompt-visible
    # tests in the inference context.
    if "public_test" in sanitized:
        sanitized["test"] = sanitized.get("public_test")
    elif "public_tests" in sanitized:
        public_tests = sanitized.get("public_tests")
        sanitized["test_list"] = list(public_tests) if isinstance(public_tests, list) else [str(public_tests)]
        sanitized.pop("test", None)
    elif "prompt_visible_tests" in sanitized:
        prompt_tests = sanitized.get("prompt_visible_tests")
        sanitized["test_list"] = list(prompt_tests) if isinstance(prompt_tests, list) else [str(prompt_tests)]
        sanitized.pop("test", None)
    else:
        sanitized.pop("test", None)
    sanitized["inference_visibility"] = visibility
    return sanitized


def _initialize_artifact_bag(
    *,
    task_text: str,
    task_context: Optional[Dict[str, Any]],
    extra_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    extra_context = extra_context or {}
    prior_artifact = extra_context.get("artifact_bag") if isinstance(extra_context.get("artifact_bag"), dict) else {}
    dataset = _normalize_dataset_name(extra_context.get("dataset") or (task_context or {}).get("dataset"))
    task_family = str(extra_context.get("task_family") or _infer_task_family(dataset, task_context)).strip().lower()
    tests_public: List[str] = []
    if isinstance(task_context, dict):
        raw_test_list = task_context.get("test_list") or []
        tests_public.extend(str(item) for item in raw_test_list if str(item).strip())
        raw_test = task_context.get("test")
        if raw_test:
            tests_public.append(str(raw_test))
    function_signature = (task_context or {}).get("function_signature") if isinstance(task_context, dict) else None
    param_names = []
    if isinstance(function_signature, dict):
        param_names = [str(item).strip() for item in (function_signature.get("parameters") or []) if str(item).strip()]
    bag: Dict[str, Any] = {
        "task_family": task_family or "generic",
        "dataset": dataset,
        "entry_point": (task_context or {}).get("entry_point") if isinstance(task_context, dict) else None,
        "function_signature": function_signature if isinstance(function_signature, dict) else {},
        "param_names": param_names,
        "tests_public": tests_public,
        "draft_code": "",
        "normalized_code": "",
        "code_shape": "",
        "assert_score_passed": 0,
        "assert_score_total": 0,
        "first_failure_kind": "",
        "cross_dataset_fallback": False,
        "candidate_answer": "",
        "formatted_answer": "",
        "reasoning_trace": [],
        "verification": {},
        "diagnostics": {},
        "final_answer": "",
        "context": (task_context or {}).get("context") if isinstance(task_context, dict) else [],
        "supporting_facts": (task_context or {}).get("supporting_facts") if isinstance(task_context, dict) else [],
        "evidence_points": [],
        "task_text": _extract_primary_task_text(task_context, task_text),
        "expected_answer": (task_context or {}).get("answer") if isinstance(task_context, dict) else None,
        "inference_visibility": (task_context or {}).get("inference_visibility", "tool_visible")
        if isinstance(task_context, dict)
        else "tool_visible",
    }
    if prior_artifact:
        for key, value in prior_artifact.items():
            if key not in bag:
                bag[key] = value
                continue
            if key in {"reasoning_trace", "tests_public", "evidence_points", "supporting_facts"}:
                prior_values = list(value) if isinstance(value, list) else [value]
                merged = list(bag.get(key) or [])
                for item in prior_values:
                    if item not in merged and _artifact_has_value(item):
                        merged.append(item)
                bag[key] = merged
                continue
            if key in {"verification", "diagnostics"} and isinstance(value, dict):
                merged = dict(value)
                merged.update(dict(bag.get(key) or {}))
                bag[key] = merged
                continue
            if not _artifact_has_value(bag.get(key)) and _artifact_has_value(value):
                bag[key] = value
    return bag


def _append_reasoning_trace(artifact_bag: Dict[str, Any], values: Any) -> None:
    if values is None:
        return
    trace = artifact_bag.setdefault("reasoning_trace", [])
    if not isinstance(trace, list):
        trace = artifact_bag["reasoning_trace"] = [str(trace)]
    if isinstance(values, str):
        if values.strip():
            trace.append(values.strip())
        return
    if isinstance(values, list):
        for item in values:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                trace.append(text)
        return
    text = str(values).strip()
    if text:
        trace.append(text)


def _merge_role_output_into_artifact_bag(
    artifact_bag: Dict[str, Any],
    role: str,
    output: Any,
) -> None:
    if not isinstance(output, dict):
        return
    role_key = _normalize_role_name(role)
    task_family = str(artifact_bag.get("task_family") or "generic").lower()
    if output.get("diagnostics") and isinstance(output.get("diagnostics"), dict):
        artifact_bag["diagnostics"] = dict(output.get("diagnostics") or {})
    if output.get("final_answer"):
        artifact_bag["final_answer"] = str(output.get("final_answer") or "")
    if task_family == "code":
        draft_code = output.get("draft_code") or output.get("code_or_commands")
        if isinstance(draft_code, str) and draft_code.strip():
            artifact_bag["draft_code"] = draft_code
        if output.get("normalized_code"):
            artifact_bag["normalized_code"] = str(output.get("normalized_code") or "")
        if output.get("code_shape"):
            artifact_bag["code_shape"] = str(output.get("code_shape") or "")
        if isinstance(output.get("function_signature"), dict):
            artifact_bag["function_signature"] = dict(output.get("function_signature") or {})
        if isinstance(output.get("param_names"), list) and output.get("param_names"):
            artifact_bag["param_names"] = [str(item).strip() for item in output.get("param_names") if str(item).strip()]
        if output.get("assert_score_passed") is not None:
            artifact_bag["assert_score_passed"] = int(output.get("assert_score_passed") or 0)
        if output.get("assert_score_total") is not None:
            artifact_bag["assert_score_total"] = int(output.get("assert_score_total") or 0)
        if output.get("first_failure_kind"):
            artifact_bag["first_failure_kind"] = str(output.get("first_failure_kind") or "")
        if output.get("cross_dataset_fallback") is not None:
            artifact_bag["cross_dataset_fallback"] = bool(output.get("cross_dataset_fallback"))
        _append_reasoning_trace(artifact_bag, output.get("runnable_plan"))
        _append_reasoning_trace(artifact_bag, output.get("self_test"))
        if role_key == "checker":
            verification = dict(artifact_bag.get("verification") or {})
            verification["verdicts"] = list(output.get("verdicts") or [])
            verification["failure_localization"] = str(output.get("failure_localization") or "")
            artifact_bag["verification"] = verification
        return

    if task_family in {"gsm8k", "math"}:
        candidate = output.get("candidate_answer") or output.get("final_answer") or output.get("code_or_commands")
        if isinstance(candidate, str) and candidate.strip():
            artifact_bag["candidate_answer"] = candidate.strip()
        _append_reasoning_trace(
            artifact_bag,
            output.get("reasoning_trace")
            or output.get("runnable_plan")
            or output.get("steps")
            or output.get("evidence_points"),
        )
        if role_key == "checker":
            verification = dict(artifact_bag.get("verification") or {})
            verification["verdicts"] = list(output.get("verdicts") or [])
            verification["failure_localization"] = str(output.get("failure_localization") or "")
            artifact_bag["verification"] = verification
        return

    if task_family in {"qa", "hotpotqa"}:
        candidate = output.get("candidate_answer") or output.get("final_answer") or output.get("code_or_commands")
        if isinstance(candidate, str) and candidate.strip():
            artifact_bag["candidate_answer"] = candidate.strip()
        if output.get("final_answer"):
            artifact_bag["final_answer"] = str(output.get("final_answer") or "").strip()
        if isinstance(output.get("evidence_points"), list):
            artifact_bag["evidence_points"] = list(output.get("evidence_points") or [])
        _append_reasoning_trace(
            artifact_bag,
            output.get("reasoning_trace")
            or output.get("runnable_plan")
            or output.get("steps")
            or output.get("evidence_points"),
        )
        if role_key == "checker":
            verification = dict(artifact_bag.get("verification") or {})
            verification["verdicts"] = list(output.get("verdicts") or [])
            verification["failure_localization"] = str(output.get("failure_localization") or "")
            artifact_bag["verification"] = verification


def _merge_tool_payload_into_artifact_bag(
    artifact_bag: Dict[str, Any],
    payload: Any,
    tool_info: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(payload, dict):
        candidate = _extract_tool_final_answer_candidate(payload)
        if candidate:
            artifact_bag["final_answer"] = str(candidate)
        return
    inner = payload.get("output") if isinstance(payload.get("output"), dict) else payload
    if not isinstance(inner, dict):
        inner = payload
    for key in (
        "entry_point",
        "normalized_code",
        "formatted_answer",
        "final_answer",
        "candidate_answer",
        "normalized_answer",
        "code_shape",
        "first_failure_kind",
    ):
        value = inner.get(key)
        if isinstance(value, str) and value.strip():
            artifact_bag[key] = value.strip()
    if inner.get("assert_score_passed") is not None:
        artifact_bag["assert_score_passed"] = int(inner.get("assert_score_passed") or 0)
    if inner.get("assert_score_total") is not None:
        artifact_bag["assert_score_total"] = int(inner.get("assert_score_total") or 0)
    if inner.get("cross_dataset_fallback") is not None:
        artifact_bag["cross_dataset_fallback"] = bool(inner.get("cross_dataset_fallback"))
    if isinstance(inner.get("function_signature"), dict):
        artifact_bag["function_signature"] = dict(inner.get("function_signature") or {})
    if isinstance(inner.get("param_names"), list) and inner.get("param_names"):
        artifact_bag["param_names"] = [str(item).strip() for item in inner.get("param_names") if str(item).strip()]
    elif isinstance(inner.get("parameters"), list) and inner.get("parameters"):
        artifact_bag["param_names"] = [str(item).strip() for item in inner.get("parameters") if str(item).strip()]
    if isinstance(inner.get("evidence_points"), list):
        artifact_bag["evidence_points"] = list(inner.get("evidence_points") or [])
    if inner.get("reasoning_trace"):
        _append_reasoning_trace(artifact_bag, inner.get("reasoning_trace"))
    if inner.get("extracted_function") and not artifact_bag.get("normalized_code"):
        artifact_bag["normalized_code"] = str(inner.get("extracted_function") or "")
    if inner.get("canonical_answer") and not artifact_bag.get("formatted_answer"):
        artifact_bag["formatted_answer"] = str(inner.get("canonical_answer") or "")
    if inner.get("candidate_value") is not None and not artifact_bag.get("candidate_answer"):
        artifact_bag["candidate_answer"] = str(inner.get("candidate_value"))
    if inner.get("equation") or inner.get("expression") or inner.get("numbers"):
        _append_reasoning_trace(
            artifact_bag,
            json.dumps(
                {
                    key: inner.get(key)
                    for key in ("equation", "expression", "numbers", "fractions", "units", "solutions")
                    if _artifact_has_value(inner.get(key))
                },
                ensure_ascii=True,
                default=str,
            ),
        )
    verification = dict(artifact_bag.get("verification") or {})
    for key in (
        "passed",
        "matches_expected",
        "expected_value",
        "returncode",
        "stdout",
        "stderr",
        "syntax_ok",
        "syntax_error",
        "supporting_fact_coverage",
        "answer_in_evidence",
    ):
        if key in inner:
            verification[key] = inner.get(key)
    if isinstance(inner.get("verification"), dict):
        verification.update(dict(inner.get("verification") or {}))
    if verification:
        artifact_bag["verification"] = verification
    diagnostics = dict(artifact_bag.get("diagnostics") or {})
    for key in (
        "error_summary",
        "focus_lines",
        "cue_hits",
        "observed_calls",
        "parameters",
        "signature_text",
        "question_type",
        "answer_type_hint",
        "entities",
        "support_titles",
        "context_title_count",
        "evidence_count",
    ):
        if key in inner and _artifact_has_value(inner.get(key)):
            diagnostics[key] = inner.get(key)
    if isinstance(inner.get("diagnostics"), dict):
        diagnostics.update(dict(inner.get("diagnostics") or {}))
    if diagnostics:
        artifact_bag["diagnostics"] = diagnostics
    if tool_info:
        produces = list(tool_info.get("produces") or [])
        if "formatted_answer" in produces and artifact_bag.get("formatted_answer") and not artifact_bag.get("final_answer"):
            artifact_bag["final_answer"] = str(artifact_bag.get("formatted_answer") or "")


def _artifact_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _compute_gap_state(
    artifact_bag: Optional[Dict[str, Any]],
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    artifact = dict(artifact_bag or {})
    resolved_task_family = str(task_family or artifact.get("task_family") or "generic").strip().lower()
    resolved_dataset = _normalize_dataset_name(dataset or artifact.get("dataset"))
    verification = dict(artifact.get("verification") or {})
    diagnostics = dict(artifact.get("diagnostics") or {})
    reasoning_trace = list(artifact.get("reasoning_trace") or [])
    reasoning_text = " ".join(str(item).strip().lower() for item in reasoning_trace if str(item).strip())
    failure_text = " ".join(
        str(item).strip().lower()
        for item in [
            diagnostics.get("error_summary"),
            verification.get("stderr"),
            verification.get("syntax_error"),
            verification.get("failure_localization"),
        ]
        if str(item or "").strip()
    )
    has_candidate_answer = _artifact_has_value(artifact.get("candidate_answer")) or _artifact_has_value(artifact.get("final_answer"))
    has_verification = any(
        key in verification and _artifact_has_value(verification.get(key))
        for key in ("passed", "matches_expected", "verdicts", "syntax_ok", "expected_value", "stdout")
    )
    has_failure_trace = any(token in failure_text for token in ("error", "traceback", "failed", "exception", "syntax"))

    if resolved_task_family == "code":
        gap_vector = {
            "missing_entry_point": not _artifact_has_value(artifact.get("entry_point")),
            "missing_draft_code": not _artifact_has_value(artifact.get("draft_code")),
            "missing_normalized_code": not _artifact_has_value(artifact.get("normalized_code")),
            "missing_verification": not has_verification,
            "has_failure_trace": has_failure_trace,
        }
    elif resolved_task_family == "gsm8k":
        has_quantity_signal = bool(reasoning_text) or _artifact_has_value(diagnostics.get("cue_hits"))
        has_relation_signal = any(
            token in reasoning_text
            for token in ("equation", "expression", "relation", "ratio", "fraction", "sum", "difference", "product")
        ) or _artifact_has_value(diagnostics.get("parameters"))
        gap_vector = {
            "missing_quantities": not has_quantity_signal,
            "missing_relations": not has_relation_signal,
            "missing_candidate_answer": not has_candidate_answer,
            "missing_verification": not has_verification,
            "missing_format": not (_artifact_has_value(artifact.get("formatted_answer")) or _artifact_has_value(artifact.get("final_answer"))),
        }
    elif resolved_task_family == "math":
        has_structure_signal = bool(reasoning_text) or _artifact_has_value(diagnostics.get("observed_calls"))
        has_exact_relation = any(
            token in reasoning_text
            for token in ("equation", "expression", "solution", "integral", "derivative", "proof", "system")
        ) or _artifact_has_value(diagnostics.get("parameters"))
        gap_vector = {
            "missing_structure_parse": not has_structure_signal,
            "missing_exact_relation": not has_exact_relation,
            "missing_candidate_answer": not has_candidate_answer,
            "missing_verification": not has_verification,
            "missing_format": not (_artifact_has_value(artifact.get("formatted_answer")) or _artifact_has_value(artifact.get("final_answer"))),
        }
    elif resolved_task_family in {"qa", "hotpotqa"} or resolved_dataset == "hotpotqa":
        has_decomposition = bool(reasoning_text) or _artifact_has_value(diagnostics.get("question_decomposition")) or _artifact_has_value(diagnostics.get("question_type"))
        has_evidence = _artifact_has_value(artifact.get("evidence_points")) or _artifact_has_value(diagnostics.get("evidence_count"))
        gap_vector = {
            "missing_question_decomposition": not has_decomposition,
            "missing_evidence": not has_evidence,
            "missing_candidate_answer": not has_candidate_answer,
            "missing_verification": not has_verification,
            "missing_final_answer": not (_artifact_has_value(artifact.get("formatted_answer")) or _artifact_has_value(artifact.get("final_answer"))),
        }
    else:
        gap_vector = {
            "missing_candidate_answer": not has_candidate_answer,
            "missing_verification": not has_verification,
        }

    stage_progress = {
        "has_reasoning_trace": bool(reasoning_text),
        "has_candidate_answer": has_candidate_answer,
        "has_verification": has_verification,
        "has_final_answer": _artifact_has_value(artifact.get("final_answer")) or _artifact_has_value(artifact.get("formatted_answer")),
    }
    failure_flags = {"has_failure_trace": has_failure_trace}
    gap_count = sum(1 for value in gap_vector.values() if bool(value))
    return {
        "task_family": resolved_task_family,
        "dataset": resolved_dataset,
        "gap_vector": gap_vector,
        "stage_progress": stage_progress,
        "failure_flags": failure_flags,
        "gap_count": gap_count,
    }


def _predict_next_gap(
    gap_state: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]],
    task_family: Optional[str],
) -> Optional[str]:
    if not isinstance(gap_state, dict):
        return None
    gap_vector = gap_state.get("gap_vector") if isinstance(gap_state.get("gap_vector"), dict) else {}
    failure_flags = gap_state.get("failure_flags") if isinstance(gap_state.get("failure_flags"), dict) else {}
    task_family_key = str(task_family or gap_state.get("task_family") or "").strip().lower()
    priorities = {
        "code": [
            "has_failure_trace",
            "missing_entry_point",
            "missing_draft_code",
            "missing_normalized_code",
            "missing_verification",
        ],
        "gsm8k": [
            "missing_quantities",
            "missing_relations",
            "missing_candidate_answer",
            "missing_verification",
            "missing_format",
        ],
        "math": [
            "missing_structure_parse",
            "missing_exact_relation",
            "missing_candidate_answer",
            "missing_verification",
            "missing_format",
        ],
        "qa": [
            "missing_question_decomposition",
            "missing_evidence",
            "missing_candidate_answer",
            "missing_verification",
            "missing_final_answer",
        ],
        "hotpotqa": [
            "missing_question_decomposition",
            "missing_evidence",
            "missing_candidate_answer",
            "missing_verification",
            "missing_final_answer",
        ],
    }
    ordered = priorities.get(task_family_key, list(gap_vector.keys()))
    if failure_flags.get("has_failure_trace") and "has_failure_trace" in gap_vector and gap_vector.get("has_failure_trace"):
        return "has_failure_trace"
    for gap_name in ordered:
        if gap_vector.get(gap_name):
            return gap_name
    return None


def _infer_agent_capability_coverage(agent_card: Any, registry) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    if agent_card is None:
        return coverage
    tool_ids = list(getattr(agent_card, "available_tool_ids", []) or [])
    num_tools = max(1, len(tool_ids))
    for tool_id in tool_ids:
        tool_card = registry.get(tool_id)
        if tool_card is None:
            continue
        stage = str(getattr(tool_card, "stage", "") or "").strip().lower()
        capability = str(getattr(tool_card, "capability_kind", "") or "").strip().lower()
        if stage:
            coverage[f"stage:{stage}"] = coverage.get(f"stage:{stage}", 0.0) + 1.0
        if capability:
            coverage[f"capability:{capability}"] = coverage.get(f"capability:{capability}", 0.0) + 1.0
        for produced in list(getattr(tool_card, "produces", []) or []):
            produced_key = str(produced or "").strip().lower()
            if produced_key:
                coverage[f"produces:{produced_key}"] = coverage.get(f"produces:{produced_key}", 0.0) + 1.0
    for role_tag in list(getattr(agent_card, "role_tags", []) or []):
        normalized = _normalize_role_name(role_tag)
        if normalized:
            coverage[f"role:{normalized}"] = 1.0
    for key, value in list(coverage.items()):
        if key.startswith("stage:") or key.startswith("capability:") or key.startswith("produces:"):
            coverage[key] = float(value) / float(num_tools)
    return coverage


def _gap_reduction_score(
    gap_state: Optional[Dict[str, Any]],
    agent_card: Any,
    role: str,
    tool_coverage: Dict[str, float],
) -> float:
    if not isinstance(gap_state, dict):
        return 0.0
    gap_vector = gap_state.get("gap_vector") if isinstance(gap_state.get("gap_vector"), dict) else {}
    role_key = _normalize_role_name(role)
    role_targets = {
        "missing_entry_point": {"stages": ["analyze"], "capabilities": ["signature-extractor"], "produces": ["entry_point"], "roles": ["planner", "checker"]},
        "missing_draft_code": {"roles": ["builder", "refactor"]},
        "missing_normalized_code": {"stages": ["prepare"], "capabilities": ["code-normalizer"], "produces": ["normalized_code"], "roles": ["builder", "refactor"]},
        "missing_verification": {"stages": ["verify", "execute"], "capabilities": ["public-test-runner", "structure-checker", "answer-verifier", "support-verifier"], "produces": ["verification"], "roles": ["checker"]},
        "has_failure_trace": {"stages": ["diagnose"], "capabilities": ["traceback-localizer"], "produces": ["diagnostics"], "roles": ["checker", "refactor"]},
        "missing_quantities": {"stages": ["analyze"], "capabilities": ["quantity-extractor", "signal-profiler"], "roles": ["planner", "checker"]},
        "missing_relations": {"stages": ["solve_support"], "capabilities": ["relation-calculator"], "roles": ["planner", "checker"]},
        "missing_candidate_answer": {"roles": ["builder", "planner", "refactor"]},
        "missing_format": {"stages": ["format"], "capabilities": ["answer-formatter"], "produces": ["formatted_answer"], "roles": ["checker", "planner"]},
        "missing_structure_parse": {"stages": ["analyze"], "capabilities": ["signal-profiler"], "roles": ["planner", "checker"]},
        "missing_exact_relation": {"stages": ["solve_support"], "capabilities": ["relation-calculator"], "roles": ["planner", "checker"]},
        "missing_question_decomposition": {"stages": ["analyze"], "capabilities": ["question-decomposer"], "produces": ["reasoning_trace"], "roles": ["planner", "checker"]},
        "missing_evidence": {"stages": ["retrieve"], "capabilities": ["context-indexer"], "produces": ["evidence_points"], "roles": ["planner", "checker"]},
        "missing_final_answer": {"stages": ["answer", "verify"], "capabilities": ["answer-extractor", "support-verifier"], "produces": ["final_answer"], "roles": ["builder", "checker"]},
    }
    total = 0.0
    total_weight = 0.0
    for gap_name, active in gap_vector.items():
        if not active:
            continue
        spec = role_targets.get(gap_name, {})
        gap_score = 0.0
        if role_key and role_key in spec.get("roles", []):
            gap_score = max(gap_score, 0.65)
        for stage in spec.get("stages", []):
            gap_score = max(gap_score, float(tool_coverage.get(f"stage:{stage}", 0.0)))
        for capability in spec.get("capabilities", []):
            gap_score = max(gap_score, float(tool_coverage.get(f"capability:{capability}", 0.0)))
        for produced in spec.get("produces", []):
            gap_score = max(gap_score, float(tool_coverage.get(f"produces:{produced}", 0.0)))
        total_weight += 1.0
        total += gap_score
    if total_weight <= 0.0:
        return 0.0
    return _clamp01(total / total_weight)


def _simulate_gap_transition(
    gap_state: Optional[Dict[str, Any]],
    role: str,
    agent_card: Any,
    tool_coverage: Dict[str, float],
) -> Dict[str, Any]:
    state = dict(gap_state or {})
    state["gap_vector"] = dict(state.get("gap_vector") or {})
    state["stage_progress"] = dict(state.get("stage_progress") or {})
    state["failure_flags"] = dict(state.get("failure_flags") or {})
    primary_gap = _predict_next_gap(state, history=None, task_family=state.get("task_family"))
    if not primary_gap:
        return state
    reduction_score = _gap_reduction_score(state, agent_card, role, tool_coverage)
    if reduction_score >= 0.55:
        state["gap_vector"][primary_gap] = False
        if primary_gap == "has_failure_trace":
            state["failure_flags"]["has_failure_trace"] = False
        if primary_gap in {"missing_verification"}:
            state["stage_progress"]["has_verification"] = True
        if primary_gap in {"missing_candidate_answer", "missing_draft_code"}:
            state["stage_progress"]["has_candidate_answer"] = True
    state["gap_count"] = sum(1 for value in state["gap_vector"].values() if bool(value))
    return state


def _artifact_bag_to_role_output(role: str, artifact_bag: Dict[str, Any], tool_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    role_key = _normalize_role_name(role)
    task_family = str(artifact_bag.get("task_family") or "generic").lower()
    summary = _summarize_tool_history(tool_history)
    final_answer = (
        artifact_bag.get("final_answer")
        or artifact_bag.get("formatted_answer")
        or artifact_bag.get("candidate_answer")
        or _extract_final_answer_from_tool_history(tool_history)
    )
    if role_key in {"builder", "refactor"}:
        primary_text = artifact_bag.get("draft_code") or artifact_bag.get("normalized_code")
        if artifact_bag.get("code_shape") == "raw_full_def" and artifact_bag.get("normalized_code"):
            primary_text = artifact_bag.get("normalized_code")
        if task_family in {"gsm8k", "math", "qa", "hotpotqa"}:
            primary_text = artifact_bag.get("candidate_answer") or artifact_bag.get("formatted_answer") or primary_text
        return {
            "runnable_plan": list(artifact_bag.get("reasoning_trace") or summary or []),
            "code_or_commands": str(primary_text or ""),
            "self_test": [],
            "final_answer": final_answer,
        }
    if role_key == "planner":
        return {
            "steps": list(artifact_bag.get("reasoning_trace") or summary or []),
            "acceptance_criteria": [],
            "final_answer": final_answer,
        }
    if role_key == "checker":
        verification = artifact_bag.get("verification") or {}
        verdicts = []
        if "passed" in verification:
            verdicts.append("pass" if verification.get("passed") else "fail")
        return {
            "test_cases": list(artifact_bag.get("tests_public") or []),
            "verdicts": verdicts,
            "failure_localization": str((artifact_bag.get("diagnostics") or {}).get("error_summary") or ""),
            "final_answer": final_answer,
        }
    return _tool_history_to_output(role, tool_history)


def _has_verification_signal(artifact_bag: Dict[str, Any]) -> bool:
    verification = artifact_bag.get("verification") or {}
    if not isinstance(verification, dict):
        verification = {}
    return bool(
        verification.get("verdicts")
        or ("passed" in verification)
        or artifact_bag.get("formatted_answer")
    )


def _should_invoke_tools_on_demand(
    *,
    role: str,
    role_context: Dict[str, Any],
    output: Any,
    task_context: Optional[Dict[str, Any]],
) -> bool:
    if bool(role_context.get("tool_only")):
        return True
    artifact_bag = role_context.get("artifact_bag")
    if not isinstance(artifact_bag, dict):
        return False
    task_family = str(artifact_bag.get("task_family") or "").lower()
    if not _is_benchmark_task_family(task_family):
        return False

    role_key = _normalize_role_name(role)
    if role_key not in {"planner", "builder", "checker", "refactor"}:
        return False

    has_reasoning = bool(artifact_bag.get("reasoning_trace"))
    has_verification = _has_verification_signal(artifact_bag)
    has_code = bool(artifact_bag.get("normalized_code") or artifact_bag.get("draft_code"))
    has_candidate_answer = bool(
        artifact_bag.get("candidate_answer")
        or artifact_bag.get("final_answer")
        or artifact_bag.get("formatted_answer")
    )
    has_evidence = bool(artifact_bag.get("evidence_points"))
    has_public_tests = bool(
        artifact_bag.get("tests_public")
        or ((task_context or {}).get("test"))
        or ((task_context or {}).get("test_list"))
    )

    if task_family == "code":
        if role_key in {"builder", "refactor"}:
            return has_code and has_public_tests
        if role_key == "checker":
            return has_code
        if role_key == "planner":
            return not has_reasoning
        return False

    if task_family in {"gsm8k", "math"}:
        if role_key == "checker":
            return has_candidate_answer or has_reasoning
        if role_key in {"builder", "refactor"}:
            return has_candidate_answer and not has_verification
        if role_key == "planner":
            return not has_reasoning
        return False

    if task_family in {"qa", "hotpotqa"}:
        if role_key == "checker":
            return has_candidate_answer or has_evidence
        if role_key in {"builder", "refactor"}:
            return has_candidate_answer and (not has_evidence or not has_verification)
        if role_key == "planner":
            return (not has_reasoning) or (not has_evidence)
        return False

    return False


def _should_prefer_tool_augmented_output(
    *,
    role: str,
    role_context: Dict[str, Any],
    artifact_bag: Dict[str, Any],
    tool_history: List[Dict[str, Any]],
) -> bool:
    if not tool_history:
        return False
    diagnostics = role_context.get("diagnostics") or {}
    if isinstance(diagnostics, dict) and (
        diagnostics.get("error_message")
        or diagnostics.get("test_error")
        or diagnostics.get("latest_test_error")
        or diagnostics.get("postprocess_error")
    ):
        return False
    task_family = str(artifact_bag.get("task_family") or "").lower()
    role_key = _normalize_role_name(role)
    has_verification = _has_verification_signal(artifact_bag)

    if task_family == "code":
        return bool(
            artifact_bag.get("normalized_code")
            or artifact_bag.get("draft_code")
            or has_verification
            or artifact_bag.get("final_answer")
        )

    if task_family in {"gsm8k", "math"}:
        return bool(
            artifact_bag.get("candidate_answer")
            or artifact_bag.get("formatted_answer")
            or artifact_bag.get("final_answer")
            or has_verification
        )

    if task_family in {"qa", "hotpotqa"}:
        return bool(
            artifact_bag.get("candidate_answer")
            or artifact_bag.get("final_answer")
            or artifact_bag.get("evidence_points")
            or has_verification
        )

    return role_key == "checker" and has_verification


def _benchmark_tool_decision(
    role: str,
    context: Dict[str, Any],
    tools: List[Dict[str, Any]],
    tool_history: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    artifact_bag = context.get("artifact_bag")
    if not isinstance(artifact_bag, dict):
        return None
    task_family = str(artifact_bag.get("task_family") or "").lower()
    if not _is_benchmark_task_family(task_family):
        return None

    used_stages = {str(entry.get("stage") or "") for entry in tool_history if isinstance(entry, dict)}

    def find_tool(*stages: str) -> Optional[Dict[str, Any]]:
        stage_set = {stage for stage in stages if stage}
        for tool in tools:
            tool_stage = str(tool.get("stage") or "")
            if tool_stage in stage_set:
                return tool
        return None

    selected_tool: Optional[Dict[str, Any]] = None
    reason = ""
    if task_family == "code":
        has_code = bool(artifact_bag.get("draft_code") or artifact_bag.get("normalized_code"))
        has_tests = bool(artifact_bag.get("tests_public"))
        has_exec = "execute" in used_stages
        if not artifact_bag.get("entry_point"):
            selected_tool = find_tool("analyze")
            reason = "extract entry point/signature"
        elif has_code and not artifact_bag.get("normalized_code"):
            selected_tool = find_tool("prepare")
            reason = "normalize generated code"
        elif has_code and not has_exec and has_tests:
            selected_tool = find_tool("execute")
            reason = "run public tests on generated code"
        elif has_code and "verify" not in used_stages:
            selected_tool = find_tool("verify")
            reason = "run structure/contract verification"
        elif context.get("diagnostics") and "diagnose" not in used_stages:
            selected_tool = find_tool("diagnose")
            reason = "localize failure from diagnostics"
    elif task_family in {"gsm8k", "math"}:
        if "analyze" not in used_stages:
            selected_tool = find_tool("analyze")
            reason = "extract structured quantities and cues"
        elif "solve_support" not in used_stages:
            selected_tool = find_tool("solve_support")
            reason = "derive structured relations before final answer"
        elif artifact_bag.get("candidate_answer") and "verify" not in used_stages:
            selected_tool = find_tool("verify")
            reason = "verify the candidate answer"
        elif (artifact_bag.get("candidate_answer") or artifact_bag.get("formatted_answer")) and "format" not in used_stages:
            selected_tool = find_tool("format")
            reason = "format the final answer"
    elif task_family in {"qa", "hotpotqa"}:
        if "analyze" not in used_stages:
            selected_tool = find_tool("analyze")
            reason = "decompose multi-hop question"
        elif "retrieve" not in used_stages:
            selected_tool = find_tool("retrieve")
            reason = "rank context sentences as evidence"
        elif "solve_support" not in used_stages:
            selected_tool = find_tool("solve_support")
            reason = "build evidence chain"
        elif "answer" not in used_stages:
            selected_tool = find_tool("answer")
            reason = "extract answer from evidence"
        elif "verify" not in used_stages:
            selected_tool = find_tool("verify")
            reason = "verify answer support and evidence coverage"

    if selected_tool is None:
        return {"action": "final", "reason": "benchmark_artifact_ready"}
    return {
        "action": "tool",
        "tool_id": selected_tool.get("id"),
        "tool_input": {},
        "reason": reason or "benchmark_capability_graph",
    }


def _build_tool_exec_inputs(
    *,
    task_text: str,
    role: str,
    agent_context: Dict[str, Any],
    results: Dict[str, Any],
    task_context: Optional[Dict[str, Any]],
    artifact_bag: Dict[str, Any],
    tool_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    visibility = _normalize_inference_visibility(artifact_bag.get("inference_visibility"))
    forbidden_public_only = {
        "answer",
        "expected",
        "expected_answer",
        "gold",
        "gold_answer",
        "label",
        "labels",
        "supporting_facts",
        "supporting_fact_labels",
        "hidden_tests",
        "challenge_tests",
        "test",
    }
    exec_inputs: Dict[str, Any] = {
        "query": task_text,
        "task": task_text,
        "role": role,
        "agent": agent_context,
        "upstream": results,
    }
    if task_context:
        exec_inputs["task_context"] = task_context
        for key in ("entry_point", "prompt", "question", "context", "supporting_facts", "type", "level", "test", "test_list", "answer", "expected"):
            if visibility == "public_only" and key in forbidden_public_only:
                continue
            if key in task_context and task_context.get(key) is not None:
                exec_inputs[key] = task_context.get(key)
    for key, value in artifact_bag.items():
        if not _artifact_has_value(value):
            continue
        if visibility == "public_only" and key in forbidden_public_only:
            continue
        exec_inputs[key] = value
    prompt_text = _extract_primary_task_text(task_context, task_text)
    if prompt_text:
        exec_inputs.setdefault("prompt", prompt_text)
        exec_inputs.setdefault("question", prompt_text)
    draft_code = artifact_bag.get("normalized_code") or artifact_bag.get("draft_code")
    if draft_code:
        exec_inputs.setdefault("code", draft_code)
    candidate_answer = (
        artifact_bag.get("formatted_answer")
        or artifact_bag.get("candidate_answer")
        or artifact_bag.get("final_answer")
    )
    if candidate_answer:
        exec_inputs.setdefault("candidate", candidate_answer)
        exec_inputs.setdefault("answer", candidate_answer)
        exec_inputs.setdefault("value", candidate_answer)
    if visibility != "public_only" and artifact_bag.get("expected_answer") is not None:
        exec_inputs.setdefault("expected", artifact_bag.get("expected_answer"))
    if artifact_bag.get("tests_public"):
        tests = list(artifact_bag.get("tests_public") or [])
        exec_inputs.setdefault("test_list", tests)
        exec_inputs.setdefault("test", "\n".join(str(item) for item in tests))
    if tool_info:
        for consume_key in tool_info.get("consumes") or []:
            if consume_key in artifact_bag and _artifact_has_value(artifact_bag.get(consume_key)):
                exec_inputs.setdefault(consume_key, artifact_bag.get(consume_key))
    return exec_inputs


def _is_pseudo_final_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return False
    text = str(value).strip()
    if not text:
        return True

    hash_matches = _HASH_ANSWER_RE.findall(text)
    if hash_matches:
        return _is_pseudo_final_answer(hash_matches[-1].strip())

    lowered = re.sub(r"\s+", " ", text.lower())
    if lowered in {"true", "false", "unresolved_no_final_answer"}:
        return True
    if _PLACEHOLDER_FINAL_ANSWER_RE.fullmatch(lowered):
        return True
    return False


class RealLLMClient:
    """LLM-backed client that returns structured outputs per role."""

    _ROLE_SCHEMAS = {
        "planner": {
            "fields": {
                "steps": "list[str]",
                "acceptance_criteria": "list[str]",
                "ready_to_handoff": "bool (optional, true when planning is sufficient)",
                "next_role": "str (optional, suggested next role when handing off)",
            },
            "guidance": (
                "Provide a concise task plan and acceptance criteria. "
                "Set ready_to_handoff=true only when the plan is sufficient for execution."
            ),
        },
        "evidence_finder": {
            "fields": {"evidence_targets": "list[str]", "sources": "list[str]", "evidence_points": "list[str]"},
            "guidance": "Find useful facts in the provided context; do not browse the web.",
        },
        "builder": {
            "fields": {"runnable_plan": "list[str]", "code_or_commands": "str", "self_test": "list[str]"},
            "guidance": "Outline implementation steps and provide runnable code/commands.",
        },
        "refactor": {
            "fields": {"runnable_plan": "list[str]", "code_or_commands": "str", "self_test": "list[str]"},
            "guidance": "Refactor code with minimal behavior-preserving edits and include updated code.",
        },
        "checker": {
            "fields": {"test_cases": "list[str]", "verdicts": "list[str]", "failure_localization": "str"},
            "guidance": "Define tests, verdicts, and failure localization notes.",
        },
        "manager": {
            "fields": {
                "status": "str (delegate|finish)",
                "next_role": "str or null",
                "instruction": "str or null",
                "summary": "str or null",
                "notes": "list[str]",
            },
            "guidance": "Decide next role or finish. Provide instruction and a short summary.",
        },
    }

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _build_messages(self, role: str, task_text: str, context: Dict[str, Any], fix: bool) -> List[Dict[str, str]]:
        role_key = _normalize_role_name(role)
        schema = self._ROLE_SCHEMAS.get(role_key) or {
            "fields": {"summary": "str", "key_points": "list[str]", "next_hint": "str"},
            "guidance": "Summarize the work and suggest the next action.",
        }
        artifact_bag = context.get("artifact_bag") if isinstance(context.get("artifact_bag"), dict) else {}
        task_family = str(artifact_bag.get("task_family") or context.get("task_family") or "").lower()

        agent = context.get("agent", {})
        constraints = context.get("constraints") or {}
        upstream = context.get("upstream") or {}
        tool_history = context.get("tool_history") or []
        available_tools = context.get("available_tools") or []
        is_decentralized = context.get("is_decentralized", False)
        available_roles = context.get("available_roles") or []

        agent_desc = _truncate_text(agent.get("description"), max_chars=600)
        agent_embed = _truncate_text(agent.get("embedding_text"), max_chars=600)
        tool_history_text = _truncate_text(
            json.dumps(tool_history, ensure_ascii=True, default=str), max_chars=1200
        )
        tools_text = _truncate_text(json.dumps(available_tools, ensure_ascii=True, default=str), max_chars=1200)
        diagnostics = context.get("diagnostics") or {}
        diagnostics_text = _truncate_text(
            json.dumps(diagnostics, ensure_ascii=True, default=str), max_chars=1200
        )
        candidate_final_answer = context.get("candidate_final_answer")
        candidate_final_answer_source = context.get("candidate_final_answer_source")
        candidate_final_answer_round = context.get("candidate_final_answer_round")
        allow_tool_only_final_decision = bool(context.get("allow_tool_only_final_decision"))

        # In decentralized topology, agents can specify next_role
        if is_decentralized and role_key != "manager":
            schema = dict(schema)  # Copy to avoid modifying original
            schema["fields"] = dict(schema["fields"])
            schema["fields"]["next_role"] = "str (optional: specify next agent role, or 'finish' to end)"
            schema["guidance"] += (
                f" In decentralized topology, you can optionally specify 'next_role' to choose which agent should act next "
                f"from available roles: {available_roles}, or set it to 'finish' to end the workflow."
            )

        if task_family == "code" and role_key in {"builder", "refactor"}:
            schema = dict(schema)
            schema["fields"] = dict(schema["fields"])
            schema["fields"]["draft_code"] = "str (optional, raw candidate code before deterministic tool normalization)"
            schema["guidance"] += (
                " This is a benchmark code task. You are responsible for producing the initial candidate code. "
                "Set draft_code to the raw candidate implementation and also mirror it in code_or_commands."
            )
        elif task_family in {"gsm8k", "math"} and role_key in {"planner", "builder", "checker", "refactor"}:
            schema = dict(schema)
            schema["fields"] = dict(schema["fields"])
            schema["fields"]["candidate_answer"] = "str (optional, the current best answer candidate)"
            schema["fields"]["reasoning_trace"] = "list[str] (optional, concise reasoning or extracted facts)"
            schema["guidance"] += (
                " This is a math benchmark task. Produce the answer candidate yourself; deterministic tools will only "
                "extract, verify, and format. Put the answer in candidate_answer and optionally mirror it in final_answer."
            )
        elif task_family in {"qa", "hotpotqa"} and role_key in {"planner", "builder", "checker", "refactor"}:
            schema = dict(schema)
            schema["fields"] = dict(schema["fields"])
            schema["fields"]["candidate_answer"] = "str (optional, short HotpotQA-style answer candidate)"
            schema["fields"]["reasoning_trace"] = "list[str] (optional, concise multi-hop evidence reasoning)"
            schema["fields"]["evidence_points"] = "list[dict] (optional, supporting titles/sentences used)"
            schema["guidance"] += (
                " This is an evidence-grounded multi-hop QA benchmark task. Keep answers short. "
                "Use candidate_answer for the proposed answer, final_answer for the final short answer, "
                "and evidence_points for supporting facts or context sentences."
            )

        schema_lines = [f"- {key}: {value}" for key, value in schema["fields"].items()]
        schema_text = "\n".join(schema_lines)

        system_msg = (
            f"You are the {role_key} role in a multi-agent system. "
            "Return ONLY a JSON object that matches the required schema exactly. "
            "Do not include markdown or extra keys."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Agent name: {agent.get('name') or agent.get('id')}\n"
            f"Agent domain_tags: {agent.get('domain_tags')}\n"
            f"Agent role_tags: {agent.get('role_tags')}\n"
            f"Agent tool_tags: {agent.get('tool_tags')}\n"
            f"Agent description: {agent_desc}\n"
            f"Agent embedding_text: {agent_embed}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=True)}\n"
            f"Upstream outputs: {json.dumps(upstream, ensure_ascii=True)}\n"
            f"Available roles: {json.dumps(context.get('available_roles') or [], ensure_ascii=True)}\n"
            f"Available tools: {tools_text}\n"
            f"Tool history: {tool_history_text}\n"
            f"Diagnostics: {diagnostics_text}\n"
            f"Artifact bag: {json.dumps(artifact_bag, ensure_ascii=True, default=str)}\n"
            f"Schema:\n{schema_text}\n"
            f"Guidance: {schema.get('guidance','')}\n"
        )
        if fix:
            user_msg += (
                "Fix the previous output to match the schema.\n"
                f"Previous output: {json.dumps(context.get('bad_output'), ensure_ascii=True)}\n"
            )
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    def _call(self, role: str, task_text: str, context: Dict[str, Any], fix: bool) -> Dict[str, Any]:
        messages = self._build_messages(role, task_text, context, fix)
        response = self.llm.chat(
            messages,
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        return data if isinstance(data, dict) else {}

    def generate(self, role: str, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._call(role, task_text, context, fix=False)

    def rewrite_format(self, role: str, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        fix_context = dict(context)
        fix_context["bad_output"] = output
        return self._call(role, "format_fix", fix_context, fix=True)

    def clarify(self, role: str, task_text: str) -> Dict[str, Any]:
        return {"clarifier": f"Provide missing details for {role} about {task_text}"}

    def plan_tool(
        self,
        role: str,
        task_text: str,
        context: Dict[str, Any],
        tools: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        benchmark_decision = _benchmark_tool_decision(role, context, tools, tool_history)
        if benchmark_decision is not None:
            return benchmark_decision
        agent = context.get("agent", {})
        constraints = context.get("constraints") or {}
        upstream = context.get("upstream") or {}
        tool_only = bool(context.get("tool_only"))
        tool_history_text = _truncate_text(
            json.dumps(tool_history, ensure_ascii=True, default=str), max_chars=2000
        )
        tools_text = _truncate_text(json.dumps(tools, ensure_ascii=True, default=str), max_chars=2400)
        diagnostics = context.get("diagnostics") or {}
        diagnostics_text = _truncate_text(
            json.dumps(diagnostics, ensure_ascii=True, default=str), max_chars=1200
        )
        candidate_final_answer = context.get("candidate_final_answer")
        candidate_final_answer_source = context.get("candidate_final_answer_source")
        candidate_final_answer_round = context.get("candidate_final_answer_round")
        allow_tool_only_final_decision = bool(context.get("allow_tool_only_final_decision"))
        if _is_pseudo_final_answer(candidate_final_answer):
            candidate_final_answer = None
            candidate_final_answer_source = None
            candidate_final_answer_round = None
            allow_tool_only_final_decision = False

        system_msg = (
            "You are an intelligent tool-using controller. Analyze the task and tool history to select the BEST tool for the job. "
            "If previous attempts failed, choose a DIFFERENT tool that can fix the specific error. "
            "Return ONLY JSON with keys: action ('tool' or 'final'), tool_id, tool_input, reason.\n"
            "CRITICAL for MATH tasks: Tools follow a stage pipeline. You MUST progress through stages:\n"
            "  1. analyze (extract givens/constraints) - call ONCE\n"
            "  2. model (set up equations) - call ONCE\n"
            "  3. solve (compute the answer) - call ONCE, this produces final_answer\n"
            "  4. verify (check the answer) - call ONCE\n"
            "  5. format (normalize the answer) - call ONCE, this produces the formatted final answer\n"
            "NEVER call the same stage tool more than twice. After analyze, you MUST move to solve or model. "
            "The 'solve' stage is ESSENTIAL - without it, no final_answer is produced."
        )
        
        # Analyze tool history for failures and suggest better alternatives
        failure_analysis = ""
        if tool_history:
            failed_tools = []
            last_error = None
            for entry in tool_history:
                tid = entry.get("tool_id", "")
                result = entry.get("result", {})
                if isinstance(result, dict):
                    if result.get("ok") is False or result.get("error"):
                        failed_tools.append(tid)
                        if result.get("error"):
                            error_obj = result["error"]
                            if isinstance(error_obj, dict):
                                last_error = f"{error_obj.get('code', '')}: {error_obj.get('message', '')}"
            
            if failed_tools:
                failure_analysis = (
                    f"\n\nIMPORTANT - Previous failures:\n"
                    f"Failed tools: {', '.join(failed_tools)}\n"
                )
                if last_error:
                    failure_analysis += f"Last error: {last_error}\n"
                failure_analysis += (
                    "You have two strategies:\n"
                    "1. If the error can be fixed with better input/refinement, you MAY retry the same tool with improved instructions\n"
                    "2. If the tool is fundamentally unsuitable, choose a DIFFERENT tool from available tools\n"
                    "Consider whether refinement or switching is more appropriate for fixing the error.\n"
                )
        
        # Build math stage progress hint from tool history
        math_stage_hint = ""
        if tool_history:
            used_stages: Dict[str, int] = {}
            for entry in tool_history:
                tid = entry.get("tool_id") or ""
                if tid.startswith("math-"):
                    parts = tid.split("-")
                    if len(parts) > 2:
                        st = parts[2]
                        used_stages[st] = used_stages.get(st, 0) + 1
            if used_stages:
                completed = ", ".join(f"{s}({c}x)" for s, c in used_stages.items())
                has_solve = "solve" in used_stages
                has_format = "format" in used_stages
                if not has_solve:
                    math_stage_hint = (
                        f"\n⚠️ MATH STAGE PROGRESS: Completed stages: [{completed}]. "
                        "You have NOT called a 'solve' tool yet! You MUST call a solve-stage tool NOW to compute the answer.\n"
                    )
                elif not has_format:
                    math_stage_hint = (
                        f"\n⚠️ MATH STAGE PROGRESS: Completed stages: [{completed}]. "
                        "Solve is done. Now call a 'format' or 'verify' stage tool to finalize the answer.\n"
                    )
                else:
                    math_stage_hint = (
                        f"\nMATH STAGE PROGRESS: Completed stages: [{completed}]. "
                        "All key stages done. Choose action='final' if the answer is ready.\n"
                    )

        user_msg = (
            f"Role: {role}\n"
            f"Task: {task_text}\n"
            f"Agent name: {agent.get('name') or agent.get('id')}\n"
            f"Agent domain_tags: {agent.get('domain_tags')}\n"
            f"Agent role_tags: {agent.get('role_tags')}\n"
            f"Agent tool_tags: {agent.get('tool_tags')}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=True)}\n"
            f"Upstream outputs: {json.dumps(upstream, ensure_ascii=True)}\n"
            f"Available tools: {tools_text}\n"
            f"Tool history: {tool_history_text}\n"
            f"Diagnostics: {diagnostics_text}\n"
            f"{failure_analysis}"
            f"{math_stage_hint}"
            "If no tool is needed, choose action='final'."
        )
        if candidate_final_answer is not None:
            user_msg += (
                "\nA candidate final answer has already been produced by tools.\n"
                f"candidate_final_answer: {candidate_final_answer}\n"
                f"candidate_source_tool: {candidate_final_answer_source}\n"
                f"candidate_round: {candidate_final_answer_round}\n"
                "Evaluate whether the answer is sufficiently reliable and fully reasoned:\n"
                "- If reliable: choose action='final' to handoff/stop this role.\n"
                "- If uncertain: choose action='tool' to continue verification/refinement.\n"
            )
        if tool_only:
            if allow_tool_only_final_decision:
                user_msg += (
                    "\nTool-only mode is enabled, but final-answer self-evaluation is active. "
                    "You may choose action='final' (handoff/stop role) or action='tool'.\n"
                )
            else:
                user_msg += (
                    "\nTool-only mode is enabled. You MUST choose action='tool' and "
                    "set tool_id to one of the available tools.\n"
                )
            tool_ids = [tool.get("id") for tool in tools if tool.get("id")]
            if tool_ids:
                user_msg += f"Allowed tool_ids: {tool_ids}\n"
        response = self.llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        return data if isinstance(data, dict) else {"action": "final"}

def _build_candidate_texts(registry, candidates: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for candidate in candidates:
        card_id = candidate.get("card_id")
        if not card_id:
            texts.append("")
            continue
        card = registry.get(card_id)
        if card is None:
            texts.append("")
            continue
        texts.append(card_search_text(card))
    return texts


def _format_router_candidates(candidates: List[dict]) -> str:
    """Format candidates with name interpretation for better router decisions.
    
    Agent names follow the format: {OriginalName}-{CreativityLevel}-{ModelCapability}
    - Creativity levels: VeryPrecise, HighlyPrecise, Precise, Moderate, Balanced, Creative, HighlyCreative, VeryCreative
    - Model capabilities: Pro (highest quality), Adv (balanced), Std (standard)
    """
    parts: List[str] = []
    for cand in candidates:
        agent_name = cand.get('name', '')
        name_interpretation = _interpret_agent_name(agent_name)
        role_score = cand.get("role_score")
        unary_raw = cand.get("unary_raw")
        cost_penalty = cand.get("cost_penalty")
        latency_penalty = cand.get("latency_penalty")

        score_lines: List[str] = []
        if role_score is not None:
            score_lines.append(f"  role_score: {float(role_score):.4f}")
        if unary_raw is not None:
            score_lines.append(f"  unary_raw: {float(unary_raw):.4f}")
        if cost_penalty is not None:
            score_lines.append(f"  cost_penalty: {float(cost_penalty):.4f}")
        if latency_penalty is not None:
            score_lines.append(f"  latency_penalty: {float(latency_penalty):.4f}")
        parts.append(
            "\n".join(
                [
                    f"- id: {cand.get('id')}",
                    f"  name: {agent_name}",
                    f"  name_interpretation: {name_interpretation}",
                    f"  domain_tags: {cand.get('domain_tags')}",
                    f"  role_tags: {cand.get('role_tags')}",
                    f"  tool_tags: {cand.get('tool_tags')}",
                    f"  description: {cand.get('description')}",
                    *score_lines,
                ]
            )
        )
    return "\n".join(parts)


def _interpret_agent_name(name: str) -> str:
    """Interpret agent name to provide context about its configuration.
    
    Examples:
    - AlgorithmArchitect-HighlyPrecise-Pro -> Highly precise, deterministic output (temp ~0.10-0.20), top-quality models
    - StringMaster-Moderate-Std -> Moderate creativity (temp ~0.30-0.40), standard models
    - Researcher-VeryCreative-Adv -> Very creative, diverse exploration (temp ≥0.70), advanced models
    """
    if not name or '-' not in name:
        return "Standard configuration"
    
    parts = name.split('-')
    if len(parts) < 3:
        return "Standard configuration"
    
    # Extract creativity and model from the last two parts
    model_capability = parts[-1]  # Pro, Adv, or Std
    creativity_level = parts[-2]  # VeryPrecise, HighlyPrecise, etc.
    
    # Map creativity to temperature range and description
    creativity_map = {
        'VeryPrecise': ('temp <0.10', 'extremely deterministic, minimal variation'),
        'HighlyPrecise': ('temp 0.10-0.20', 'highly deterministic, very consistent'),
        'Precise': ('temp 0.20-0.30', 'precise, consistent output'),
        'Moderate': ('temp 0.30-0.40', 'moderate creativity, balanced approach'),
        'Balanced': ('temp 0.40-0.50', 'balanced creativity and consistency'),
        'Creative': ('temp 0.50-0.60', 'creative, diverse output'),
        'HighlyCreative': ('temp 0.60-0.70', 'highly creative, exploratory'),
        'VeryCreative': ('temp ≥0.70', 'very creative, maximum diversity'),
    }
    
    # Map model to capability description
    model_map = {
        'Pro': 'highest quality models (GPT-4o dominant)',
        'Adv': 'balanced model mix (advanced capabilities)',
        'Std': 'standard models (cost-efficient)',
    }
    
    creativity_desc = creativity_map.get(creativity_level, ('unknown temp', 'unknown creativity'))
    model_desc = model_map.get(model_capability, 'unknown model capability')
    
    return f"{creativity_desc[1]} ({creativity_desc[0]}), {model_desc}"


def _select_with_router_llm(
    llm: LLMClient,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    registry,
) -> Optional[str]:
    candidate_cards: List[dict] = []
    for cand in candidates:
        card_id = cand.get("card_id")
        if not card_id:
            continue
        card = registry.get(card_id)
        if card is None:
            continue
        candidate_cards.append(
            {
                "id": card.id,
                "name": card.name,
                "description": card.description,
                "domain_tags": card.domain_tags,
                "role_tags": card.role_tags,
                "tool_tags": card.tool_tags,
                "role_score": cand.get("role_score", cand.get("score")),
                "unary_raw": cand.get("unary_raw"),
                "cost_penalty": cand.get("cost_penalty"),
                "latency_penalty": cand.get("latency_penalty"),
            }
        )
    if not candidate_cards:
        return None

    candidates_text = _format_router_candidates(candidate_cards)
    system_msg = (
        "You are a routing model. Select the single best agent id from the candidate list. "
        "Each agent's name includes configuration info: {OriginalName}-{CreativityLevel}-{ModelCapability}. "
        "The 'name_interpretation' field explains the agent's creativity (temperature) and model quality. "
        "Consider both the agent's capabilities (description, tags) AND its configuration (creativity/model) when selecting. "
        "For tasks requiring precision/consistency, prefer Precise/HighlyPrecise agents. "
        "For tasks requiring exploration/diversity, prefer Creative/HighlyCreative/VeryCreative agents. "
        "Return ONLY JSON: {\"selected_id\": \"...\"}."
    )
    user_msg = (
        f"Role: {role}\n"
        f"Task: {task_text}\n"
        f"Constraints: {json.dumps(constraints or {}, ensure_ascii=True)}\n"
        f"Candidates:\n{candidates_text}\n"
    )
    response = llm.chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    data = _safe_load_json(str(response), {})
    selected_id = data.get("selected_id")
    valid_ids = {c["id"] for c in candidate_cards}
    return selected_id if selected_id in valid_ids else None


def _get_agent_context(registry, agent_id: Optional[str]) -> Dict[str, Any]:
    if not agent_id:
        print(f"[ERROR] _get_agent_context called with empty agent_id!", file=sys.stderr)
        return {"id": None, "name": None, "domain_tags": [], "tool_tags": []}
    card = registry.get(agent_id)
    if card is None:
        print(f"[ERROR] Agent '{agent_id}' not found in registry!", file=sys.stderr)
        return {"id": agent_id, "name": None, "domain_tags": [], "tool_tags": []}
    return {
        "id": card.id,
        "name": card.name,
        "domain_tags": card.domain_tags,
        "role_tags": card.role_tags,
        "tool_tags": card.tool_tags,
        "description": card.description,
        "embedding_text": card.embedding_text,
        "available_tool_ids": getattr(card, "available_tool_ids", []),
    }


def _summarize_output(output: Any, max_chars: int = 400) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        summary = output
    else:
        try:
            summary = json.dumps(output, ensure_ascii=True)
        except TypeError:
            summary = str(output)
    if len(summary) > max_chars:
        return summary[: max_chars - 3] + "..."
    return summary


def _truncate_text(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _extract_assertion_hints(message: Optional[str], max_items: int = 2) -> List[str]:
    if not message:
        return []
    hints: List[str] = []
    for line in str(message).splitlines():
        text = line.strip()
        if not text:
            continue
        if "assert" in text and "candidate" in text:
            hints.append(text)
            if len(hints) >= max_items:
                break
    return hints


def _summarize_failure(error_message: Optional[str], test_error: Optional[str], max_chars: int = 700) -> str:
    parts: List[str] = []
    if test_error:
        parts.append(f"test_error={_truncate_text(str(test_error), max_chars=max_chars)}")
    if error_message and error_message != test_error:
        parts.append(f"runtime_error={_truncate_text(str(error_message), max_chars=max_chars)}")
    return " | ".join(parts)


def _extract_tool_failure_detail(tool_result: Any, max_chars: int = 600) -> str:
    """Extract compact failure hints from heterogeneous tool payloads."""
    if not isinstance(tool_result, dict):
        return ""

    hints: List[str] = []
    error_obj = tool_result.get("error")
    if error_obj:
        hints.append(_truncate_text(str(error_obj), max_chars=max_chars))

    output_obj = tool_result.get("output")
    if isinstance(output_obj, dict):
        nested = output_obj.get("output") if isinstance(output_obj.get("output"), dict) else output_obj
        if isinstance(nested, dict):
            for key in ("error", "message", "detail", "response"):
                value = nested.get(key)
                if value:
                    hints.append(f"{key}={_truncate_text(str(value), max_chars=max_chars)}")
            candidate = nested.get("code_or_commands") or nested.get("code") or nested.get("solution")
            if isinstance(candidate, str) and _is_likely_error_text(candidate):
                hints.append(f"code={_truncate_text(candidate, max_chars=max_chars)}")

    if not hints:
        return ""

    compact: List[str] = []
    seen = set()
    for item in hints:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        compact.append(key)
        if len(compact) >= 2:
            break
    return " | ".join(compact)


def _is_likely_error_text(code: str) -> bool:
    text = str(code or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if text.startswith("# Error") or text.startswith("Error:"):
        return True
    if "traceback (most recent call last)" in lowered:
        return True
    if "error calling llm" in lowered:
        return True
    if "timed out" in lowered or "timeout" in lowered:
        return True
    if text.startswith("Exception:"):
        return True
    return False


def _looks_like_code_snippet(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return True
    lowered = stripped.lower()
    code_tokens = (
        "return ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except",
        "raise ",
        "assert ",
        "def ",
        "import ",
        "lambda ",
        " = ",
    )
    if any(token in lowered for token in code_tokens):
        return True
    if stripped.endswith(")") and "(" in stripped:
        return True
    return False


def _extract_code_candidate(value: Any, depth: int = 0, trust_text: bool = False) -> Optional[str]:
    """Best-effort recursive extraction of runnable code from tool payloads."""
    if value is None or depth > 8:
        return None

    if isinstance(value, str):
        text = _extract_code_block(str(value)).strip()
        if not text:
            return None

        # If text is JSON-wrapped, unwrap and recurse.
        unwrapped = _unwrap_json_code(text)
        if isinstance(unwrapped, str) and unwrapped != text:
            nested = _extract_code_candidate(unwrapped, depth + 1, trust_text=trust_text)
            if nested:
                return nested

        # If text is itself a serialized object/array, inspect recursively.
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if parsed is not None:
                nested = _extract_code_candidate(parsed, depth + 1, trust_text=trust_text)
                if nested:
                    return nested

        if _is_likely_error_text(text):
            return None
        if not trust_text and not _looks_like_code_snippet(text):
            return None
        return text

    if isinstance(value, dict):
        if value.get("ok") is False:
            return None
        for key in ("code_or_commands", "code", "solution", "completion"):
            if key in value:
                nested = _extract_code_candidate(value.get(key), depth + 1, trust_text=True)
                if nested:
                    return nested
        for key in ("output", "result", "payload", "data", "response"):
            if key in value:
                nested = _extract_code_candidate(value.get(key), depth + 1, trust_text=False)
                if nested:
                    return nested
        for nested_value in value.values():
            nested = _extract_code_candidate(nested_value, depth + 1, trust_text=False)
            if nested:
                return nested
        return None

    if isinstance(value, list):
        for item in reversed(value):
            nested = _extract_code_candidate(item, depth + 1, trust_text=trust_text)
            if nested:
                return nested
        return None

    return None


def _has_syntax_error_signal(value: Any, depth: int = 0) -> bool:
    if value is None or depth > 8:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in ("syntaxerror", "indentationerror", "taberror"))
    if isinstance(value, dict):
        for nested in value.values():
            if _has_syntax_error_signal(nested, depth + 1):
                return True
        return False
    if isinstance(value, (list, tuple)):
        for nested in value:
            if _has_syntax_error_signal(nested, depth + 1):
                return True
        return False
    return False


def _is_python_snippet_compilable(snippet: str) -> Optional[bool]:
    text = str(snippet or "").strip("\r\n")
    if not text.strip():
        return None

    candidates = [text]
    indented_body = "\n".join(("    " + line) if line.strip() else "    " for line in text.splitlines())
    candidates.append(f"def __mcts_syntax_probe__():\n{indented_body}\n")

    for source in candidates:
        try:
            compile(source, "<mcts-syntax-probe>", "exec")
            return True
        except (SyntaxError, IndentationError, TabError):
            continue
        except Exception:
            return None
    return False


def _infer_role_syntax_compilable(
    role: str,
    output: Any,
    executor_result: Any,
) -> Optional[bool]:
    role_key = _normalize_role_name(role)
    if role_key not in {"builder", "refactor"}:
        return None

    code_candidate = _extract_code_candidate(output)
    if code_candidate is None and isinstance(executor_result, list):
        payload = _extract_tool_output(executor_result)
        code_candidate = _extract_code_candidate(payload)

    if code_candidate:
        compiled = _is_python_snippet_compilable(code_candidate)
        if compiled is not None:
            return compiled

    if _has_syntax_error_signal(output) or _has_syntax_error_signal(executor_result):
        return False
    return None


def _append_diagnostics_to_task(task_text: str, role_context: Dict[str, Any]) -> str:
    diagnostics = role_context.get("diagnostics") or {}
    if not diagnostics:
        return task_text
    if "Diagnostics:" in task_text:
        return task_text

    error_message = diagnostics.get("error_message") or ""
    test_error = diagnostics.get("test_error") or ""
    original_test_error = diagnostics.get("original_test_error") or ""
    latest_test_error = diagnostics.get("latest_test_error") or ""
    postprocess_error = diagnostics.get("postprocess_error") or ""
    failure_summary = diagnostics.get("failure_summary") or ""
    failure_stage = diagnostics.get("failure_stage") or ""
    failure_ts = diagnostics.get("failure_ts") or ""
    failed_code = diagnostics.get("failed_code") or ""

    lines = ["Diagnostics:"]
    if failure_stage:
        if failure_ts:
            lines.append(f"Failure stage: {failure_stage} @ {failure_ts}")
        else:
            lines.append(f"Failure stage: {failure_stage}")
    if failure_summary:
        lines.append(f"Failure summary: {_truncate_text(str(failure_summary), max_chars=800)}")
    if error_message:
        lines.append(f"Error: {_truncate_text(error_message, max_chars=500)}")
    if test_error and test_error != error_message:
        lines.append(f"Test error: {_truncate_text(test_error, max_chars=500)}")
    if original_test_error and original_test_error not in {test_error, error_message}:
        lines.append(f"Original test error: {_truncate_text(original_test_error, max_chars=500)}")
    if latest_test_error and latest_test_error not in {test_error, error_message, original_test_error}:
        lines.append(f"Latest test error: {_truncate_text(latest_test_error, max_chars=500)}")
    if postprocess_error and postprocess_error not in {test_error, error_message, latest_test_error}:
        lines.append(f"Postprocess error: {_truncate_text(postprocess_error, max_chars=500)}")
    assertion_hints = _extract_assertion_hints(original_test_error or test_error)
    if assertion_hints:
        lines.append("Assertion hints:")
        for hint in assertion_hints:
            lines.append(f"- {_truncate_text(hint, max_chars=500)}")
    if failed_code:
        lines.append("Failed code:")
        lines.append(_truncate_text(str(failed_code), max_chars=1200))

    return f"{task_text}\n\n" + "\n".join(lines)


_DATASET_ROUTING_CODE_DATASETS = {"mbpp", "humaneval"}
_DATASET_ROUTING_ROLES = {"planner", "builder", "checker", "refactor"}
_DATASET_ROUTING_HARD_ROLES = {"builder", "checker", "refactor"}
_DATASET_ROUTING_MIN_CANDIDATES = 8
_DATASET_ROUTING_MARGIN = 0.03
_DATASET_ROUTING_TAGS = {
    "humaneval": ["python-function-synthesis"],
    "mbpp": ["assert-driven-programming"],
    "gsm8k": ["arithmetic-word-problem"],
    "math": ["exact-competition"],
    "hotpotqa": ["multi-hop-qa", "evidence-grounded-qa"],
}


def _dataset_routing_tags(dataset: Optional[str]) -> List[str]:
    dataset_name = _normalize_dataset_name(dataset)
    return list(_DATASET_ROUTING_TAGS.get(dataset_name, []))


def _dataset_routing_enabled(
    *,
    dataset: Optional[str],
    task_family: Optional[str],
    role: str,
) -> bool:
    dataset_name = _normalize_dataset_name(dataset)
    family_name = str(task_family or "").strip().lower()
    role_key = _normalize_role_name(role)
    return (
        family_name == "code"
        and dataset_name in _DATASET_ROUTING_CODE_DATASETS
        and role_key in _DATASET_ROUTING_ROLES
    )


def _resolve_dataset_routing_context(
    constraints: Optional[Dict[str, Any]] = None,
    artifact_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    constraint_map = constraints if isinstance(constraints, dict) else {}
    artifact_map = artifact_context if isinstance(artifact_context, dict) else {}
    dataset = _normalize_dataset_name(
        constraint_map.get("_routing_dataset")
        or artifact_map.get("dataset")
        or artifact_map.get("task_context", {}).get("dataset")
    )
    task_family = str(
        constraint_map.get("_routing_task_family")
        or artifact_map.get("task_family")
        or ""
    ).strip().lower()
    return dataset, task_family


def _candidate_matches_dataset(candidate: Dict[str, Any], dataset: str) -> bool:
    route_tags = {tag.lower() for tag in _dataset_routing_tags(dataset)}
    if not route_tags:
        return False
    brief_tags = candidate.get("brief_tags") if isinstance(candidate, dict) else {}
    domain_tags = []
    if isinstance(brief_tags, dict):
        domain_tags = list(brief_tags.get("domain_tags") or [])
    candidate_tags = {str(item).strip().lower() for item in domain_tags if str(item).strip()}
    return bool(candidate_tags.intersection(route_tags))


def _build_cross_dataset_fallback_constraints(
    constraints: Optional[Dict[str, Any]],
    *,
    task_family: Optional[str],
) -> Dict[str, Any]:
    relaxed = dict(constraints or {})
    original_domain_tags = list(relaxed.get("_original_domain_tags") or [])
    dataset = _normalize_dataset_name(relaxed.get("_routing_dataset"))
    family = str(task_family or relaxed.get("_routing_task_family") or "").strip().lower()
    route_tags = {tag.lower() for tag in _dataset_routing_tags(dataset)}
    fallback_tags = [
        str(tag).strip()
        for tag in original_domain_tags
        if str(tag).strip() and str(tag).strip().lower() not in route_tags and str(tag).strip().lower() != dataset
    ]
    if family == "code" and "code" not in {tag.lower() for tag in fallback_tags}:
        fallback_tags.insert(0, "code")
    if fallback_tags:
        relaxed["domain_tags"] = list(dict.fromkeys(fallback_tags))
    else:
        relaxed.pop("domain_tags", None)
    relaxed["_same_dataset_required"] = False
    return relaxed


def _annotate_dataset_candidates(
    candidates: List[Dict[str, Any]],
    *,
    dataset: str,
    task_family: str,
    role: str,
    margin: float = _DATASET_ROUTING_MARGIN,
) -> List[Dict[str, Any]]:
    if not _dataset_routing_enabled(dataset=dataset, task_family=task_family, role=role):
        return [dict(item) for item in candidates]

    deduped: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "").strip()
        if not card_id:
            continue
        annotated = dict(item)
        same_dataset = _candidate_matches_dataset(annotated, dataset)
        annotated["cross_dataset_fallback"] = not same_dataset
        base_score = float(annotated.get("unary_raw", annotated.get("score", 0.0)) or 0.0)
        annotated["dataset_adjusted_score"] = base_score + (margin if same_dataset else 0.0)
        existing = deduped.get(card_id)
        if existing is None:
            deduped[card_id] = annotated
            ordered_ids.append(card_id)
            continue
        existing_score = float(existing.get("dataset_adjusted_score", existing.get("score", 0.0)) or 0.0)
        if float(annotated["dataset_adjusted_score"]) > existing_score:
            deduped[card_id] = annotated

    ordered = [deduped[card_id] for card_id in ordered_ids if card_id in deduped]
    ordered.sort(
        key=lambda item: (
            float(item.get("dataset_adjusted_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ordered


def _apply_dataset_selection_tiebreak(
    selected_main: Optional[str],
    candidates: List[Dict[str, Any]],
    *,
    dataset: str,
    task_family: str,
    role: str,
    margin: float = _DATASET_ROUTING_MARGIN,
) -> Optional[str]:
    if not selected_main or not _dataset_routing_enabled(dataset=dataset, task_family=task_family, role=role):
        return selected_main

    candidate_by_id = {
        str(item.get("card_id") or "").strip(): item
        for item in candidates
        if isinstance(item, dict) and str(item.get("card_id") or "").strip()
    }
    selected = candidate_by_id.get(str(selected_main).strip())
    if not selected or not bool(selected.get("cross_dataset_fallback")):
        return selected_main

    best_same_dataset = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and not bool(item.get("cross_dataset_fallback"))
        ),
        None,
    )
    if best_same_dataset is None:
        return selected_main

    selected_score = float(selected.get("unary_raw", selected.get("score", 0.0)) or 0.0)
    same_score = float(best_same_dataset.get("unary_raw", best_same_dataset.get("score", 0.0)) or 0.0)
    if selected_score < same_score + margin:
        return str(best_same_dataset.get("card_id") or "").strip() or selected_main
    return selected_main


def _ensure_role_constraints(
    constraints: Optional[Dict[str, Any]],
    role: str,
    *,
    dataset: Optional[str] = None,
    task_family: Optional[str] = None,
    allow_cross_dataset_fallback: bool = True,
) -> Dict[str, Any]:
    merged = dict(constraints) if constraints else {}
    role_tags = list(merged.get("role_tags") or [])
    if not role_tags:
        role_key = role.strip().lower()
        tags: List[str] = [role_key] if role_key else []
        if role_key == "checker":
            tags.append("tester")
            tags.append("code-testing")
        elif role_key == "tester":
            tags.append("checker")
            tags.append("code-testing")
        elif role_key in {"refractor", "refactor"}:
            tags.append("refactor")
            tags.append("refractor")
            tags.append("code-refactoring")
        elif role_key == "code-generation":
            tags.append("builder")
            tags.append("code-generation")
        elif role_key == "code-planner":
            tags.append("planner")
            tags.append("code-planner")
        elif role_key == "code-testing":
            tags.append("tester")
            tags.append("checker")
            tags.append("code-testing")
        elif role_key == "code-refactoring":
            tags.append("refactor")
            tags.append("refractor")
            tags.append("code-refactoring")
        if tags:
            merged["role_tags"] = tags
    role_key = _normalize_role_name(role)
    dataset_name = _normalize_dataset_name(dataset)
    task_family_name = str(task_family or "").strip().lower()
    if _dataset_routing_enabled(dataset=dataset_name, task_family=task_family_name, role=role_key):
        original_domain_tags = list(merged.get("domain_tags") or [])
        merged["_routing_dataset"] = dataset_name
        merged["_routing_task_family"] = task_family_name
        merged["_allow_cross_dataset_fallback"] = bool(allow_cross_dataset_fallback)
        merged["_original_domain_tags"] = original_domain_tags
        merged["_same_dataset_min_candidates"] = _DATASET_ROUTING_MIN_CANDIDATES
        merged["_same_dataset_margin"] = _DATASET_ROUTING_MARGIN
        merged["_same_dataset_required"] = role_key in _DATASET_ROUTING_HARD_ROLES
        route_tags = _dataset_routing_tags(dataset_name)
        if route_tags:
            merged["domain_tags"] = route_tags
    return merged


def _build_selection_cache_key(*, role: str, task_text: str, step_idx: int, topology: str) -> str:
    digest = hashlib.sha256((task_text or "").encode("utf-8")).hexdigest()[:12]
    topo = (topology or "unknown").strip().lower()
    return f"{topo}|{role}|s{step_idx}|q{digest}"


def _jaccard_overlap(a: List[str], b: List[str]) -> float:
    set_a = {str(item).strip().lower() for item in (a or []) if str(item).strip()}
    set_b = {str(item).strip().lower() for item in (b or []) if str(item).strip()}
    if not set_a and not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return float(len(set_a.intersection(set_b)) / len(union))


def _tier_penalty(value: Optional[str]) -> float:
    if not value:
        return 0.25
    normalized = str(value).strip().lower()
    if normalized in {"low", "cheap", "economy", "fast"}:
        return 0.0
    if normalized in {"medium", "mid", "normal", "balanced"}:
        return 0.5
    if normalized in {"high", "expensive", "slow", "premium"}:
        return 1.0
    return 0.25


_UNARY_MODEL_MATCH_WEIGHT = 0.08
_UNARY_TEMP_MATCH_WEIGHT = 0.06

_ROLE_DIFFICULTY_PRIOR = {
    "planner": 0.45,
    "evidence_finder": 0.60,
    "builder": 0.60,
    "checker": 0.55,
    "refactor": 0.55,
    "manager": 0.40,
}

_ROLE_EXPLORATION_TARGET = {
    "planner": 0.40,
    "evidence_finder": 0.80,
    "builder": 0.20,
    "checker": 0.10,
    "refactor": 0.20,
    "manager": 0.45,
}

_CREATIVITY_EXPLORATION_LEVEL = {
    "veryprecise": 0.00,
    "highlyprecise": 0.10,
    "precise": 0.20,
    "moderate": 0.35,
    "balanced": 0.45,
    "creative": 0.60,
    "highlycreative": 0.75,
    "verycreative": 0.90,
}

_DIFFICULTY_HINT_TOKENS = (
    "hard",
    "complex",
    "multi-step",
    "prove",
    "proof",
    "optimize",
    "edge case",
    "concurrency",
    "dynamic programming",
    "security",
    "formal",
    "benchmark",
)

_EXPLORATION_HINT_TOKENS = (
    "research",
    "explore",
    "brainstorm",
    "alternatives",
    "open-ended",
    "creative",
    "diverse",
)

_PRECISION_HINT_TOKENS = (
    "exact",
    "deterministic",
    "strict",
    "assert",
    "verify",
    "unit test",
    "minimal",
    "bugfix",
    "fix",
)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _extract_model_tier_from_name(name: Optional[str]) -> str:
    raw_name = str(name or "").strip()
    if not raw_name or "-" not in raw_name:
        return "unknown"
    suffix = raw_name.split("-")[-1].strip().lower()
    if suffix in {"pro", "adv", "std"}:
        return suffix
    return "unknown"


def _extract_creativity_level_from_name(name: Optional[str]) -> str:
    raw_name = str(name or "").strip()
    if not raw_name or "-" not in raw_name:
        return "balanced"
    parts = raw_name.split("-")
    if len(parts) < 3:
        return "balanced"
    return str(parts[-2]).strip().lower()


def _count_active_constraint_items(constraints: Optional[Dict[str, Any]]) -> int:
    if not isinstance(constraints, dict):
        return 0
    active = 0
    for value in constraints.values():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                active += 1
            continue
        if isinstance(value, (int, float)):
            active += 1
            continue
        if isinstance(value, str):
            if value.strip():
                active += 1
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if len(value) > 0:
                active += 1
            continue
        active += 1
    return active


def _estimate_task_difficulty(
    *,
    role: str,
    task_text: str,
    constraints: Optional[Dict[str, Any]],
    historical_failure_count: int,
) -> float:
    role_key = _normalize_role_name(role)
    role_signal = float(_ROLE_DIFFICULTY_PRIOR.get(role_key, 0.50))

    text = str(task_text or "").strip().lower()
    token_count = len([tok for tok in re.split(r"\W+", text) if tok])
    length_signal = _clamp01(token_count / 120.0)
    hard_hint_hits = sum(1 for token in _DIFFICULTY_HINT_TOKENS if token in text)
    hard_hint_signal = _clamp01(hard_hint_hits / 4.0)
    text_signal = (0.6 * length_signal) + (0.4 * hard_hint_signal)

    constraint_signal = _clamp01(_count_active_constraint_items(constraints) / 6.0)
    failure_signal = _clamp01(float(max(0, int(historical_failure_count))) / 4.0)

    difficulty = (
        (0.35 * role_signal)
        + (0.25 * text_signal)
        + (0.20 * constraint_signal)
        + (0.20 * failure_signal)
    )
    return _clamp01(difficulty)


def _model_tier_match(difficulty: float, model_tier: str) -> float:
    diff = _clamp01(difficulty)
    tier = str(model_tier or "").strip().lower()
    if tier == "pro":
        return diff
    if tier == "adv":
        return _clamp01(1.0 - (2.0 * abs(diff - 0.5)))
    if tier == "std":
        return _clamp01(1.0 - diff)
    return 0.5


def _desired_exploration_level(role: str, task_text: str) -> float:
    role_key = _normalize_role_name(role)
    desired = float(_ROLE_EXPLORATION_TARGET.get(role_key, 0.45))
    text = str(task_text or "").strip().lower()
    if any(token in text for token in _EXPLORATION_HINT_TOKENS):
        desired += 0.15
    if any(token in text for token in _PRECISION_HINT_TOKENS):
        desired -= 0.15
    return _clamp01(desired)


def _temp_band_match(*, role: str, task_text: str, creativity_level: str) -> float:
    desired = _desired_exploration_level(role, task_text)
    observed = float(_CREATIVITY_EXPLORATION_LEVEL.get(str(creativity_level or "").strip().lower(), 0.45))
    return _clamp01(1.0 - abs(desired - observed))


_ROLE_ALIAS = {
    "code-generation": "builder",
    "code-planner": "planner",
    "code-testing": "checker",
    "code-refactoring": "refactor",
    "tester": "checker",
    "refractor": "refactor",
    "researcher": "evidence_finder",
    "evidence-finder": "evidence_finder",
    "evidence-reader": "evidence_finder",
    "context-finder": "evidence_finder",
}


def _normalize_role_name(role: Optional[str]) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIAS.get(key, key)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "ready", "done"}
    return False


def _planner_ready_to_handoff(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    if "ready_to_handoff" in output:
        return _to_bool(output.get("ready_to_handoff"))
    if str(output.get("next_role") or "").strip():
        return True
    status = str(output.get("status") or "").strip().lower()
    if status in {"delegate", "handoff", "ready_to_handoff", "plan_ready"}:
        return True
    steps = output.get("steps")
    if isinstance(steps, list) and len(steps) > 0:
        return True
    return False


def _planner_next_role(output: Any) -> Optional[str]:
    if not isinstance(output, dict):
        return None
    candidate = str(output.get("next_role") or "").strip()
    if not candidate:
        return None
    normalized = _normalize_role_name(candidate)
    if normalized in {"finish", "none", "null"}:
        return None
    return normalized


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / float(len(values)))


def _io_chain_score(prev_card: Any, curr_card: Any) -> float:
    if prev_card is None or curr_card is None:
        return 0.5

    format_overlap = _jaccard_overlap(
        list(getattr(prev_card, "output_formats", []) or []),
        list(getattr(curr_card, "output_formats", []) or []),
    )
    modality_overlap = _jaccard_overlap(
        list(getattr(prev_card, "modalities", []) or []),
        list(getattr(curr_card, "modalities", []) or []),
    )
    tool_overlap = _jaccard_overlap(
        list(getattr(prev_card, "tool_tags", []) or []),
        list(getattr(curr_card, "tool_tags", []) or []),
    )

    return _clamp01((format_overlap + modality_overlap + tool_overlap) / 3.0)


def _ensure_pair_bandit_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_bandits (
            workflow_version TEXT NOT NULL,
            prev_role TEXT NOT NULL,
            prev_card_id TEXT NOT NULL,
            role TEXT NOT NULL,
            card_id TEXT NOT NULL,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            PRIMARY KEY (workflow_version, prev_role, prev_card_id, role, card_id)
        )
        """
    )
    conn.commit()


def _pair_success_key(
    prev_role: Optional[str],
    prev_card_id: Optional[str],
    role: Optional[str],
    card_id: Optional[str],
) -> Tuple[str, str, str, str]:
    return (
        _normalize_role_name(prev_role),
        str(prev_card_id or "").strip(),
        _normalize_role_name(role),
        str(card_id or "").strip(),
    )


def _role_card_success_key(
    role: Optional[str],
    card_id: Optional[str],
) -> Tuple[str, str]:
    return (
        _normalize_role_name(role),
        str(card_id or "").strip(),
    )


def _load_pair_success_priors(
    db_path: Optional[str],
    workflow_version: str,
) -> Dict[Tuple[str, str, str, str], float]:
    priors: Dict[Tuple[str, str, str, str], float] = {}
    if not db_path or not os.path.exists(db_path):
        return priors
    conn = sqlite3.connect(db_path)
    try:
        _ensure_pair_bandit_schema(conn)
        cursor = conn.execute(
            """
            SELECT prev_role, prev_card_id, role, card_id, alpha, beta
            FROM pair_bandits
            WHERE workflow_version = ?
            """,
            (workflow_version,),
        )
        for row in cursor.fetchall():
            key = _pair_success_key(row[0], row[1], row[2], row[3])
            alpha = float(row[4] or 1.0)
            beta = float(row[5] or 1.0)
            denom = alpha + beta
            priors[key] = float(alpha / denom) if denom > 0 else 0.5
    except Exception:
        return {}
    finally:
        conn.close()
    return priors


def _load_role_success_priors(
    db_path: Optional[str],
    workflow_version: str,
) -> Dict[Tuple[str, str], float]:
    priors: Dict[Tuple[str, str], float] = {}
    if not db_path or not os.path.exists(db_path):
        return priors
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT role, card_id, alpha, beta
            FROM bandits
            WHERE workflow_version = ?
            """,
            (workflow_version,),
        )
        for row in cursor.fetchall():
            key = _role_card_success_key(row[0], row[1])
            alpha = float(row[2] or 1.0)
            beta = float(row[3] or 1.0)
            denom = alpha + beta
            priors[key] = float(alpha / denom) if denom > 0 else 0.5
    except Exception:
        return {}
    finally:
        conn.close()
    return priors


def _pair_success_rate(
    priors: Optional[Dict[Tuple[str, str, str, str], float]],
    *,
    prev_role: Optional[str],
    prev_card_id: Optional[str],
    role: Optional[str],
    card_id: Optional[str],
) -> float:
    if not priors:
        return 0.5
    key = _pair_success_key(prev_role, prev_card_id, role, card_id)
    return float(priors.get(key, 0.5))


def _role_card_success_rate(
    priors: Optional[Dict[Tuple[str, str], float]],
    *,
    role: Optional[str],
    card_id: Optional[str],
) -> float:
    if not priors:
        return 0.5
    key = _role_card_success_key(role, card_id)
    return float(priors.get(key, 0.5))


def _update_pair_success_stats(
    db_path: Optional[str],
    *,
    workflow_version: str,
    selections: List[Dict[str, Any]],
    reward: float,
    confidence: float = 1.0,
) -> None:
    if not db_path or not selections:
        return

    reward = max(0.0, min(1.0, float(reward)))
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence <= 0.0:
        return

    conn = sqlite3.connect(db_path)
    try:
        _ensure_pair_bandit_schema(conn)
        for idx in range(1, len(selections)):
            prev = selections[idx - 1] or {}
            curr = selections[idx] or {}

            prev_role = _normalize_role_name(prev.get("role"))
            prev_card_id = str(prev.get("selected_main") or "").strip()
            role = _normalize_role_name(curr.get("role"))
            card_id = str(curr.get("selected_main") or "").strip()
            if not prev_role or not prev_card_id or not role or not card_id:
                continue

            cursor = conn.execute(
                """
                SELECT alpha, beta FROM pair_bandits
                WHERE workflow_version = ? AND prev_role = ? AND prev_card_id = ? AND role = ? AND card_id = ?
                """,
                (workflow_version, prev_role, prev_card_id, role, card_id),
            )
            row = cursor.fetchone()
            alpha = float(row[0]) if row else 1.0
            beta = float(row[1]) if row else 1.0
            alpha += reward * confidence
            beta += (1.0 - reward) * confidence
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_bandits
                (workflow_version, prev_role, prev_card_id, role, card_id, alpha, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (workflow_version, prev_role, prev_card_id, role, card_id, alpha, beta),
            )
        conn.commit()
    finally:
        conn.close()


def apply_bandit_feedback(
    db_path: Optional[str],
    *,
    workflow_version: str,
    selections: List[Dict[str, Any]],
    success: bool,
    main_confidence: float = 1.0,
    shadow_confidence: float = 0.5,
) -> None:
    """Update bandit/pair priors using a final task-level success label."""
    if not db_path or not selections:
        return

    reward = 1.0 if bool(success) else 0.0
    main_conf = max(0.0, min(1.0, float(main_confidence)))
    shadow_conf = max(0.0, min(1.0, float(shadow_confidence)))
    if main_conf <= 0.0 and shadow_conf <= 0.0:
        return

    pair_selections: List[Dict[str, Any]] = []
    with BanditStore(db_path) as store:
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            role = str(selection.get("role") or "").strip()
            if not role:
                continue

            main_id = str(selection.get("selected_main") or selection.get("main") or "").strip()
            raw_shadows = selection.get("selected_shadows")
            if raw_shadows is None:
                raw_shadows = selection.get("shadows")
            shadow_ids = [str(item).strip() for item in (raw_shadows or []) if str(item).strip()]

            if main_id and main_conf > 0.0:
                store.update(
                    workflow_version,
                    role,
                    main_id,
                    reward=reward,
                    confidence=main_conf,
                )
                pair_selections.append({"role": role, "selected_main": main_id})

            if shadow_conf > 0.0:
                for shadow_id in shadow_ids:
                    store.update(
                        workflow_version,
                        role,
                        shadow_id,
                        reward=reward,
                        confidence=shadow_conf,
                    )

    if pair_selections and main_conf > 0.0:
        _update_pair_success_stats(
            db_path,
            workflow_version=workflow_version,
            selections=pair_selections,
            reward=reward,
            confidence=main_conf,
        )


def _apply_unary_raw_scoring(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    registry,
    mu_cost: float,
    nu_latency: float,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    difficulty = _estimate_task_difficulty(
        role=role,
        task_text=task_text,
        constraints=constraints,
        historical_failure_count=0,
    )

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        card_id = candidate.get("card_id")
        card = registry.get(card_id) if card_id else None

        role_score = float(candidate.get("score", 0.0) or 0.0)
        cost_penalty = _tier_penalty(getattr(card, "cost_tier", None)) if card is not None else 0.25
        latency_penalty = _tier_penalty(getattr(card, "latency_tier", None)) if card is not None else 0.25
        unary_raw = _unary_score(
            candidate,
            card,
            mu_cost,
            nu_latency,
            difficulty=difficulty,
            role=role,
            task_text=task_text,
        )

        enriched["role_score"] = role_score
        enriched["cost_penalty"] = cost_penalty
        enriched["latency_penalty"] = latency_penalty
        enriched["unary_raw"] = unary_raw
        scored.append(enriched)

    scored.sort(key=lambda item: float(item.get("unary_raw", item.get("score", 0.0)) or 0.0), reverse=True)
    return scored


def _unary_score(
    candidate: Dict[str, Any],
    card: Any,
    mu_cost: float,
    nu_latency: float,
    *,
    difficulty: float,
    role: str,
    task_text: str,
) -> float:
    base = float(candidate.get("score", 0.0) or 0.0)
    cost_penalty = _tier_penalty(getattr(card, "cost_tier", None)) if card is not None else 0.25
    latency_penalty = _tier_penalty(getattr(card, "latency_tier", None)) if card is not None else 0.25
    card_name = str(getattr(card, "name", "") or "")
    model_tier = _extract_model_tier_from_name(card_name)
    creativity_level = _extract_creativity_level_from_name(card_name)

    model_match = _model_tier_match(difficulty, model_tier)
    temp_match = _temp_band_match(role=role, task_text=task_text, creativity_level=creativity_level)
    capability_match = (_UNARY_MODEL_MATCH_WEIGHT * model_match) + (_UNARY_TEMP_MATCH_WEIGHT * temp_match)

    return float(base + capability_match - (mu_cost * cost_penalty) - (nu_latency * latency_penalty))


def _prepare_role_candidates(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    artifact_context: Optional[Dict[str, Any]],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    use_mmr: bool,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool = False,
    selected_agent_ids: Optional[List[str]] = None,
    team_lambda_compat: float = 1.0,
    team_mu_cost: float = 0.05,
    team_nu_latency: float = 0.05,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset, task_family = _resolve_dataset_routing_context(constraints, artifact_context)
    min_candidates = int((constraints or {}).get("_same_dataset_min_candidates") or _DATASET_ROUTING_MIN_CANDIDATES)
    allow_cross_dataset_fallback = bool((constraints or {}).get("_allow_cross_dataset_fallback", True))

    candidates = get_candidates(
        task_text=task_text,
        role=role,
        constraints=constraints,
        kind="agent",
        top_n=top_n,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        registry=registry,
        index=index,
        embedder=embedder,
        use_mmr=use_mmr,
        artifact_context=artifact_context,
    )
    if (
        _dataset_routing_enabled(dataset=dataset, task_family=task_family, role=role)
        and allow_cross_dataset_fallback
        and len(candidates) < min_candidates
    ):
        fallback_constraints = _build_cross_dataset_fallback_constraints(
            constraints,
            task_family=task_family,
        )
        fallback_candidates = get_candidates(
            task_text=task_text,
            role=role,
            constraints=fallback_constraints,
            kind="agent",
            top_n=top_n,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
            registry=registry,
            index=index,
            embedder=embedder,
            use_mmr=use_mmr,
            artifact_context=artifact_context,
        )
        merged_candidates: List[Dict[str, Any]] = []
        seen_ids = set()
        for item in list(candidates) + list(fallback_candidates):
            card_id = str(item.get("card_id") or "").strip() if isinstance(item, dict) else ""
            if not card_id or card_id in seen_ids:
                continue
            seen_ids.add(card_id)
            merged_candidates.append(item)
        candidates = merged_candidates

    if candidates:
        if router_no_rerank or not experimental_reranker_after_mmr:
            reranked = list(candidates)
        else:
            candidate_texts = _build_candidate_texts(registry, candidates)
            query_text = sanitize_search_text(
                build_role_query(task_text, role, constraints, artifact_context=artifact_context)
            )
            rerank_indices, _ = reranker.rank(query_text, candidate_texts, top_m=min(rerank_top_m, len(candidates)))
            reranked = [candidates[i] for i in rerank_indices]
    else:
        reranked = []

    base_ranked = reranked if reranked else candidates
    _ = selected_agent_ids  # retained for backward-compatible public signatures.
    _ = team_lambda_compat  # retained for backward-compatible public signatures.
    scored = _apply_unary_raw_scoring(
        task_text=task_text,
        role=role,
        constraints=constraints,
        candidates=base_ranked,
        registry=registry,
        mu_cost=team_mu_cost,
        nu_latency=team_nu_latency,
    )
    ranked = scored if scored else base_ranked
    annotated_candidates = _annotate_dataset_candidates(
        list(candidates),
        dataset=dataset,
        task_family=task_family,
        role=role,
    )
    annotated_ranked = _annotate_dataset_candidates(
        list(ranked),
        dataset=dataset,
        task_family=task_family,
        role=role,
    )
    return annotated_candidates, annotated_ranked


class _MCTSNode:
    def __init__(
        self,
        *,
        role: Optional[str],
        depth: int,
        selected_ids: Tuple[str, ...],
        selected_roles: Tuple[str, ...],
        gap_state: Optional[Dict[str, Any]] = None,
        stage_progress: Optional[Dict[str, bool]] = None,
        failure_flags: Optional[Dict[str, bool]] = None,
        unary_score: float = 0.5,
        transition_score: float = 0.5,
        io_score: float = 0.5,
        gap_reduction: float = 0.0,
        parent: Optional["_MCTSNode"] = None,
        action_card_id: Optional[str] = None,
        action_reward: float = 0.0,
    ) -> None:
        self.role = role
        self.depth = depth
        self.selected_ids = selected_ids
        self.selected_roles = selected_roles
        self.gap_state = dict(gap_state or {})
        self.stage_progress = dict(stage_progress or self.gap_state.get("stage_progress") or {})
        self.failure_flags = dict(failure_flags or self.gap_state.get("failure_flags") or {})
        self.unary_score = float(unary_score)
        self.transition_score = float(transition_score)
        self.io_score = float(io_score)
        self.posterior_score = float(transition_score)
        self.uncertainty_score = 0.0
        self.gap_reduction = float(gap_reduction)
        self.parent = parent
        self.action_card_id = action_card_id
        self.action_reward = action_reward
        self.children: Dict[str, _MCTSNode] = {}
        self.untried_actions: List[str] = []
        self.visits: int = 0
        self.total_reward: float = 0.0

    @property
    def value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_reward / float(self.visits)


def _select_agent_with_mcts_dynamic(
    *,
    task_text: str,
    role: str,
    step_idx: int,
    config: TopologyConfig,
    constraints_per_role: Dict[str, Dict[str, Any]],
    selected_agent_ids: List[str],
    selected_agent_roles: List[str],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool = False,
    team_lambda_compat: float = 1.0,
    team_mu_cost: float = 0.05,
    team_nu_latency: float = 0.05,
    mcts_iterations: int = 64,
    mcts_rollout_depth: int = 4,
    mcts_exploration: float = 1.414,
    mcts_discount: float = 0.95,
    mcts_max_candidates: int = 5,
    pair_success_priors: Optional[Dict[Tuple[str, str, str, str], float]] = None,
    role_success_priors: Optional[Dict[Tuple[str, str], float]] = None,
    use_mmr: bool = True,
    use_iochain: bool = True,
    use_transition_term: bool = True,
    use_gap_term: bool = True,
    reward_variant: str = "mean_active",
    history: Optional[List[Dict[str, Any]]] = None,
    artifact_bag: Optional[Dict[str, Any]] = None,
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    available_roles = list(config.roles or [role])
    if role not in available_roles:
        available_roles.insert(0, role)

    role_cache: Dict[str, List[Dict[str, Any]]] = {}
    card_cache: Dict[str, Any] = {}
    role_candidate_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    role_constraints_cache: Dict[str, Dict[str, Any]] = {}
    difficulty_cache: Dict[str, float] = {}
    capability_cache: Dict[str, Dict[str, float]] = {}
    unary_score_cache: Dict[str, Dict[str, float]] = {}
    history_entries = list(history or [])
    resolved_task_family = str(task_family or (artifact_bag or {}).get("task_family") or "generic").strip().lower()
    resolved_dataset = _normalize_dataset_name(dataset or (artifact_bag or {}).get("dataset"))
    root_gap_state = _compute_gap_state(
        artifact_bag,
        task_family=resolved_task_family,
        dataset=resolved_dataset,
    )

    def _gap_signature(gap_state_value: Optional[Dict[str, Any]]) -> str:
        gap_vector = gap_state_value.get("gap_vector") if isinstance(gap_state_value, dict) else {}
        active = sorted(key for key, value in dict(gap_vector or {}).items() if bool(value))
        return "|".join(active)

    def _get_candidates_for_role(target_role: str) -> List[Dict[str, Any]]:
        if target_role in role_cache:
            return role_cache[target_role]
        role_constraints = _ensure_role_constraints(
            constraints_per_role.get(target_role),
            target_role,
            dataset=resolved_dataset,
            task_family=resolved_task_family,
        )
        role_constraints_cache[target_role] = dict(role_constraints)
        _, ranked = _prepare_role_candidates(
            task_text=task_text,
            role=target_role,
            constraints=role_constraints,
            artifact_context=artifact_bag,
            registry=registry,
            index=index,
            embedder=embedder,
            reranker=reranker,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            use_mmr=use_mmr,
            router_no_rerank=router_no_rerank,
            experimental_reranker_after_mmr=experimental_reranker_after_mmr,
            selected_agent_ids=selected_agent_ids,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
        )
        limited = list(ranked[: max(1, mcts_max_candidates)])
        role_cache[target_role] = limited
        role_candidate_map[target_role] = {}
        for item in limited:
            cid = str(item.get("card_id") or "")
            if not cid:
                continue
            role_candidate_map[target_role][cid] = item
            if cid not in card_cache:
                card_cache[cid] = registry.get(cid)
        return limited

    explicit_failure_statuses = {
        "failed",
        "failure",
        "error",
        "invalid",
        "timeout",
        "tool_error",
        "no_valid_output",
        "exception",
    }
    explicit_success_statuses = {"success", "ok", "passed", "done"}

    def _history_failure_count_for_role(target_role: str) -> int:
        normalized_target = _normalize_role_name(target_role)
        failures = 0
        for item in history_entries:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            has_test_passed = "test_passed" in item
            test_passed = _to_bool(item.get("test_passed")) if has_test_passed else None
            is_explicit_success = (status in explicit_success_statuses) or (has_test_passed and test_passed is True)
            is_explicit_failure = (
                (status in explicit_failure_statuses)
                or (has_test_passed and test_passed is False)
                or (item.get("syntax_compilable") is False)
            )
            if is_explicit_success or not is_explicit_failure:
                continue
            hist_role = _normalize_role_name(item.get("role"))
            if hist_role and hist_role == normalized_target:
                failures += 1
        return failures

    def _history_syntax_failures_for_card(target_role: str, card_id: str) -> int:
        normalized_target = _normalize_role_name(target_role)
        if normalized_target not in {"builder", "refactor"}:
            return 0
        target_card_id = str(card_id or "").strip()
        if not target_card_id:
            return 0
        failures = 0
        for item in history_entries:
            if not isinstance(item, dict):
                continue
            if _normalize_role_name(item.get("role")) != normalized_target:
                continue
            if str(item.get("selected_main") or "").strip() != target_card_id:
                continue
            if item.get("syntax_compilable") is False:
                failures += 1
        return failures

    def _difficulty_for_role(target_role: str) -> float:
        normalized_target = _normalize_role_name(target_role)
        if normalized_target in difficulty_cache:
            return float(difficulty_cache[normalized_target])
        role_constraints = role_constraints_cache.get(target_role)
        if role_constraints is None:
            role_constraints = _ensure_role_constraints(
                constraints_per_role.get(target_role),
                target_role,
                dataset=resolved_dataset,
                task_family=resolved_task_family,
            )
            role_constraints_cache[target_role] = dict(role_constraints)
        failure_count = _history_failure_count_for_role(target_role)
        difficulty = _estimate_task_difficulty(
            role=target_role,
            task_text=task_text,
            constraints=role_constraints,
            historical_failure_count=failure_count,
        )
        difficulty_cache[normalized_target] = float(difficulty)
        return float(difficulty)

    def _capability_for_card(card_id: str) -> Dict[str, float]:
        if card_id not in capability_cache:
            capability_cache[card_id] = _infer_agent_capability_coverage(card_cache.get(card_id), registry)
        return capability_cache[card_id]

    def _raw_unary_scores_for_role(target_role: str) -> Dict[str, float]:
        normalized_target = _normalize_role_name(target_role)
        if normalized_target in unary_score_cache:
            return unary_score_cache[normalized_target]
        scores: Dict[str, float] = {}
        for cand in _get_candidates_for_role(target_role):
            cand_id = str(cand.get("card_id") or "")
            if not cand_id:
                continue
            card = card_cache.get(cand_id)
            score = _unary_score(
                cand,
                card,
                team_mu_cost,
                team_nu_latency,
                difficulty=_difficulty_for_role(target_role),
                role=target_role,
                task_text=task_text,
            )
            syntax_failures = _history_syntax_failures_for_card(target_role, cand_id)
            if syntax_failures > 0:
                score -= 1.5 * float(min(3, syntax_failures))
            scores[cand_id] = float(score)
        unary_score_cache[normalized_target] = scores
        return scores

    def _normalized_unary_score(target_role: str, card_id: str) -> float:
        scores = _raw_unary_scores_for_role(target_role)
        if card_id not in scores:
            return 0.5
        values = list(scores.values())
        if not values:
            return 0.5
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return 0.5
        return _clamp01((scores[card_id] - lo) / (hi - lo))

    def _action_metrics(
        prev_ids: Tuple[str, ...],
        prev_roles: Tuple[str, ...],
        target_role: str,
        card_id: str,
        gap_state_value: Optional[Dict[str, Any]],
        seed_marker: Any = "root",
    ) -> Dict[str, Any]:
        card = card_cache.get(card_id)
        coverage = _capability_for_card(card_id)
        unary_component = _normalized_unary_score(target_role, card_id)

        unary_success = (
            _role_card_success_rate(
                role_success_priors,
                role=target_role,
                card_id=card_id,
            )
            if use_transition_term
            else 0.5
        )
        transition_component = unary_success
        io_chain = 0.0
        io_component = 0.5
        if not use_transition_term:
            transition_component = 0.5
        elif not prev_ids:
            transition_component = 0.5
        else:
            success_values: List[float] = []
            for idx, prev_id in enumerate(prev_ids):
                prev_card = card_cache.get(prev_id) or registry.get(prev_id)
                if prev_card is not None and prev_id not in card_cache:
                    card_cache[prev_id] = prev_card
                prev_role = prev_roles[idx] if idx < len(prev_roles) else ""
                success_rate = _pair_success_rate(
                    pair_success_priors,
                    prev_role=prev_role,
                    prev_card_id=prev_id,
                    role=target_role,
                    card_id=card_id,
                )
                success_values.append(float(success_rate))

            transition_component = _safe_mean(success_values, default=0.5)
        if use_iochain and prev_ids:
            prev_last_id = prev_ids[-1]
            prev_last_card = card_cache.get(prev_last_id) or registry.get(prev_last_id)
            if prev_last_card is not None and prev_last_id not in card_cache:
                card_cache[prev_last_id] = prev_last_card
            io_chain = _io_chain_score(prev_last_card, card)
            io_component = _clamp01(float(io_chain))

        gap_reduction = _gap_reduction_score(gap_state_value, card, target_role, coverage) if use_gap_term else 0.0
        next_gap_state = (
            _simulate_gap_transition(gap_state_value, target_role, card, coverage)
            if use_gap_term
            else dict(gap_state_value or {})
        )

        breakdown = score_smart_reward(
            unary_score=float(unary_component),
            transition_score=float(transition_component),
            io_score=float(io_component),
            gap_reduction=float(gap_reduction),
            gap_state=next_gap_state,
            use_transition_term=use_transition_term,
            use_iochain=use_iochain,
            use_gap_term=use_gap_term,
            reward_variant=reward_variant,
        ).as_dict()
        breakdown["io_chain"] = float(io_chain)
        return breakdown

    def _predict_next_role(current_role: Optional[str], depth: int, gap_state_value: Optional[Dict[str, Any]]) -> Optional[str]:
        if current_role is None:
            return None
        if depth >= max(1, mcts_rollout_depth):
            return None
        neighbors = _neighbors_for_role(current_role, config)
        candidates = neighbors if neighbors else available_roles
        if not candidates:
            return None
        primary_gap = _predict_next_gap(gap_state_value, history_entries, gap_state_value.get("task_family") if isinstance(gap_state_value, dict) else resolved_task_family)
        gap_role_preferences = {
            "missing_entry_point": ["checker", "planner"],
            "missing_draft_code": ["builder", "refactor"],
            "missing_normalized_code": ["builder", "refactor"],
            "missing_verification": ["checker"],
            "has_failure_trace": ["checker", "refactor"],
            "missing_quantities": ["planner", "checker"],
            "missing_relations": ["planner", "checker"],
            "missing_candidate_answer": ["builder", "planner", "refactor"],
            "missing_format": ["checker", "planner"],
            "missing_structure_parse": ["planner", "checker"],
            "missing_exact_relation": ["planner", "checker"],
        }
        preferred_roles = gap_role_preferences.get(primary_gap or "", [])
        normalized_candidate_map = {_normalize_role_name(item): item for item in candidates}
        for preferred in preferred_roles:
            preferred_role = normalized_candidate_map.get(_normalize_role_name(preferred))
            if preferred_role and not (preferred_role == current_role and len(candidates) > 1):
                return preferred_role
        avoid_role = current_role if len(candidates) > 1 else None
        return _select_next_role(
            task_text=task_text,
            available_roles=candidates,
            history=history_entries,
            llm=None,
            avoid_role=avoid_role,
        )

    def _init_untried_actions(node: _MCTSNode) -> List[str]:
        if not node.role:
            return []
        candidates = _get_candidates_for_role(node.role)
        if not candidates:
            return []
        budget = max(1, min(int(mcts_max_candidates), len(candidates)))
        scored_candidates: List[Tuple[float, str]] = []
        for cand in candidates:
            cand_id = str(cand.get("card_id") or "")
            if not cand_id:
                continue
            metrics = _action_metrics(
                node.selected_ids,
                node.selected_roles,
                node.role,
                cand_id,
                node.gap_state,
                seed_marker=f"expand:{node.depth}:{cand_id}:{_gap_signature(node.gap_state)}",
            )
            local_prior = float(metrics["reward"])
            scored_candidates.append((local_prior, cand_id))
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [card_id for _, card_id in scored_candidates[:budget]]

    root_candidates = _get_candidates_for_role(role)
    if not root_candidates:
        return None

    normalized_role_history = tuple(_normalize_role_name(item) for item in (selected_agent_roles or []))
    iterations = max(1, int(mcts_iterations))
    depth_limit = max(1, int(mcts_rollout_depth))
    exploration = max(0.0, float(mcts_exploration))
    discount = min(1.0, max(0.0, float(mcts_discount)))

    def _action_ids_for_node(
        target_role: str,
        prev_ids: Tuple[str, ...],
        prev_roles: Tuple[str, ...],
        gap_state_value: Dict[str, Any],
        depth: int,
    ) -> List[str]:
        candidates = _get_candidates_for_role(target_role)
        if not candidates:
            return []
        smart_candidates = build_smart_candidates(
            candidates,
            raw_unary_scores=_raw_unary_scores_for_role(target_role),
            final_k=mcts_max_candidates,
        )
        scored_candidates: List[Tuple[float, float, str]] = []
        for candidate in smart_candidates:
            metrics = _action_metrics(
                prev_ids,
                prev_roles,
                target_role,
                candidate.card_id,
                gap_state_value,
                seed_marker=f"mcts:{depth}:{target_role}:{candidate.card_id}:{_gap_signature(gap_state_value)}",
            )
            role_prior = _role_card_success_rate(
                role_success_priors,
                role=target_role,
                card_id=candidate.card_id,
            )
            scored_candidates.append((float(metrics.get("reward", 0.0)), float(role_prior), candidate.card_id))
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [card_id for _, _, card_id in scored_candidates]

    def _score_action_for_mcts(
        prev_ids: Tuple[str, ...],
        prev_roles: Tuple[str, ...],
        target_role: str,
        card_id: str,
        gap_state_value: Dict[str, Any],
        depth: int,
    ) -> Dict[str, Any]:
        normalized_prev_roles = tuple(_normalize_role_name(item) for item in prev_roles)
        return _action_metrics(
            prev_ids,
            normalized_prev_roles,
            target_role,
            card_id,
            gap_state_value,
            seed_marker=f"score:{depth}:{target_role}:{card_id}",
        )

    mcts_result = select_with_smart_mcts(
        initial_role=role,
        initial_selected_ids=tuple(selected_agent_ids),
        initial_selected_roles=normalized_role_history,
        initial_gap_state=root_gap_state,
        iterations=iterations,
        rollout_depth=depth_limit,
        exploration=exploration,
        discount=discount,
        max_candidates=mcts_max_candidates,
        action_ids_for_node=_action_ids_for_node,
        score_action=_score_action_for_mcts,
        predict_next_role=_predict_next_role,
    )
    if not mcts_result:
        return None

    reranked = _get_candidates_for_role(role)
    selection = {
        "selected_main": mcts_result["selected_main"],
        "selected_shadows": list(mcts_result.get("selected_shadows") or []),
        "selected_shadow": mcts_result.get("selected_shadow"),
        "candidates": list(reranked),
        "reranked": list(reranked),
        "mcts": {
            "iterations": iterations,
            "depth_limit": depth_limit,
            "use_iochain": bool(use_iochain),
            "use_transition_term": bool(use_transition_term),
            "use_gap_term": bool(use_gap_term),
            "reward_variant": str(reward_variant or "mean_active"),
            "use_mmr": bool(use_mmr),
            "current_gap_state": root_gap_state,
            "primary_gap": _predict_next_gap(root_gap_state, history_entries, resolved_task_family),
            "children": dict(mcts_result.get("children") or {}),
        },
    }
    return selection


def _append_role_result(results: Dict[str, Any], role: str, output: Any) -> None:
    if role not in results:
        results[role] = output
        return
    existing = results[role]
    if isinstance(existing, list):
        existing.append(output)
        return
    results[role] = [existing, output]


def _append_tool_exec(tool_exec: Dict[str, Any], role: str, tool_results: Any) -> None:
    if role not in tool_exec:
        tool_exec[role] = tool_results
        return
    existing = tool_exec[role]
    if isinstance(existing, list) and isinstance(tool_results, list):
        existing.extend(tool_results)
        return
    tool_exec[role] = [existing, tool_results]


def _describe_tools(registry, tool_ids: List[str]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool_id in tool_ids:
        card = registry.get(tool_id)
        if card is None:
            continue
        tools.append(
            {
                "id": card.id,
                "name": card.name,
                "description": card.description,
                "dataset_family": getattr(card, "dataset_family", ""),
                "focus_slug": getattr(card, "focus_slug", ""),
                "primary_role": getattr(card, "primary_role", ""),
                "capability_kind": getattr(card, "capability_kind", ""),
                "stage": getattr(card, "stage", ""),
                "deterministic": bool(getattr(card, "deterministic", True)),
                "consumes": list(getattr(card, "consumes", []) or []),
                "produces": list(getattr(card, "produces", []) or []),
            }
        )
    return tools


def _normalize_tool_input(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _safe_load_json(value, None)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_tool_id(
    decision: Dict[str, Any], tool_ids: List[str], tools_info: List[Dict[str, Any]]
) -> Optional[str]:
    raw = (
        decision.get("tool_id")
        or decision.get("tool")
        or decision.get("tool_name")
        or decision.get("name")
        or decision.get("id")
    )
    if isinstance(raw, dict):
        raw = raw.get("id") or raw.get("tool_id") or raw.get("name")
    if isinstance(raw, int):
        idx = raw
        if 0 <= idx < len(tools_info):
            return tools_info[idx].get("id")
        if 1 <= idx <= len(tools_info):
            return tools_info[idx - 1].get("id")
        return None
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate in tool_ids:
            return candidate
        if ":" in candidate:
            tail = candidate.split(":")[-1].strip()
            if tail in tool_ids:
                return tail
            candidate = tail
        if candidate.isdigit():
            idx = int(candidate)
            if 0 <= idx < len(tools_info):
                return tools_info[idx].get("id")
            if 1 <= idx <= len(tools_info):
                return tools_info[idx - 1].get("id")
        lower = candidate.lower()
        for tool in tools_info:
            name = str(tool.get("name") or "").strip().lower()
            tool_id = str(tool.get("id") or "").strip().lower()
            if lower == name or lower == tool_id:
                return tool.get("id")
    return None


def _plan_and_run_tools(
    *,
    llm: Any,
    role: str,
    task_text: str,
    role_context: Dict[str, Any],
    agent_context: Dict[str, Any],
    registry,
    tool_executor: Optional[ToolExecutor],
    results: Dict[str, Any],
    tool_max_rounds: int,
    tool_only: bool,
    enable_postprocess_repair: bool = True,
    task_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tool_history: List[Dict[str, Any]] = []
    artifact_bag = role_context.setdefault("artifact_bag", {})
    # Ensure no stale success flag leaks into this run
    role_context.pop("test_passed", None)
    if _is_pseudo_final_answer(role_context.get("candidate_final_answer")):
        role_context.pop("candidate_final_answer", None)
        role_context.pop("candidate_final_answer_source", None)
        role_context.pop("candidate_final_answer_round", None)
        role_context.pop("candidate_final_answer_role", None)
        role_context.pop("allow_tool_only_final_decision", None)
    if tool_executor is None:
        return tool_history
    tool_ids = list(agent_context.get("available_tool_ids") or [])
    if not tool_ids:
        tool_tags = list(agent_context.get("tool_tags") or [])
        if tool_tags:
            try:
                tool_cards = registry.list({"kind": "tool", "tool_tags": tool_tags})
                tool_ids = [card.id for card in tool_cards]
            except Exception:
                tool_ids = []
    if not tool_ids:
        agent_name = agent_context.get("name") or agent_context.get("id") or "unknown"
        agent_tags = agent_context.get("tool_tags") or []
        print(f"[WARNING] Agent '{agent_name}' has NO tools! tool_tags={agent_tags}, role={role}", file=sys.stderr)
        return tool_history
    
    agent_name = agent_context.get("name") or agent_context.get("id") or "unknown"
    print(f"[DEBUG] Agent '{agent_name}' has {len(tool_ids)} tools for role '{role}': {tool_ids[:3]}...", file=sys.stderr)
    
    tools_info = _describe_tools(registry, tool_ids)
    role_context["available_tools"] = tools_info
    if tool_only:
        role_context["tool_only"] = True
    
    # Initialize tracking variables
    used_tool_ids = set()  # Track which tools have been tried
    tool_failure_counts = {}  # Track failures per tool for switching strategy
    tool_category_counts = {}  # Track failures per tool category (e.g., 'assemblesnippets', 'generatemath')
    successful_code = None  # Track if we got non-placeholder code
    test_passed = False  # Track if code passed actual test
    has_errors = False  # Track if last tool had errors
    failed_code = None  # Track the last failed code for refinement
    last_error = None  # Track the last error message
    last_test_error = None  # Track the last test failure message
    original_test_error = None  # First test failure in current refinement chain
    latest_test_error = None  # Most recent test/postprocess failure
    postprocess_error = None  # Most recent postprocess-specific failure
    consecutive_same_category_failures = 0  # Track consecutive failures in same category
    consecutive_failures = 0  # Track consecutive failures across tools
    
    # Early stopping detection
    last_code_length = None
    same_code_length_count = 0
    tried_categories = set()  # Track which categories we've tried
    
    last_tool_category = None  # Track previous tool category
    tool_id_call_counts: Dict[str, int] = {}  # Track per-tool call counts
    stage_call_counts: Dict[str, int] = {}  # Track per-stage call counts for math tools
    _MATH_STAGE_ORDER = ["analyze", "model", "solve_support", "solve", "verify", "format"]

    for round_idx in range(max(0, tool_max_rounds)):
        round_ts = datetime.utcnow().isoformat()
        # Early termination: if code passed test, stop immediately
        if test_passed and not tool_only:
            break
            
        decision = llm.plan_tool(role, task_text, role_context, tools_info, tool_history)
        action = str(decision.get("action") or "final").strip().lower()
        tool_id = _resolve_tool_id(decision, tool_ids, tools_info)
        forced = False
        
        # Check if LLM wants to stop
        if action != "tool":
            if tool_only:
                # In tool_only mode, allow role-level stop only after a candidate final answer is detected.
                allow_final_decision = bool(role_context.get("allow_tool_only_final_decision"))
                if allow_final_decision and action in {"final", "finish", "stop", "handoff"}:
                    role_context["stop_tool_loop"] = True
                    role_context["allow_llm_reason"] = "tool_only_self_eval_stop"
                    print(
                        f"[DEBUG] tool_only self-eval stop: role={role}, round={round_idx}, action={action}",
                        file=sys.stderr,
                    )
                    break
                # Otherwise keep forcing tool calls.
                action = "tool"
                forced = True
                # If LLM chose final but we still have errors, suggest an unused tool
                if has_errors and used_tool_ids:
                    # Try to find an unused tool for LLM to consider
                    unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids]
                    if unused_tools:
                        # Don't force a specific tool, let LLM choose in next iteration
                        pass
            else:
                # Not tool_only mode
                # If we have errors and haven't tried all tools, give LLM another chance
                if has_errors and len(used_tool_ids) < len(tool_ids):
                    # Suggest to LLM that there are more tools available
                    unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids]
                    if unused_tools:
                        # Add hint to context for next iteration
                        role_context["suggested_tools"] = unused_tools
                        # Let LLM reconsider
                        continue
                # Otherwise, stop
                break
        
        # CRITICAL: If we forced tool diversity, override LLM's choice
        if role_context.get("force_different_category") or role_context.get("force_different_tool"):
            suggested = role_context.get("suggested_tools", [])
            if suggested:
                # Check if LLM chose a tool from the suggested list
                if tool_id not in suggested:
                    # LLM ignored our suggestion, force the first suggested tool
                    old_tool = tool_id
                    tool_id = suggested[0]
                    print(f"[DEBUG] ⚠️ FORCING tool switch: {old_tool} → {tool_id} (LLM ignored suggestion)", file=sys.stderr)
                    forced = True
                else:
                    print(f"[DEBUG] ✓ LLM accepted suggestion: {tool_id}", file=sys.stderr)
        
        if tool_id not in tool_ids:
            if tool_only and tool_ids:
                # Tool ID is invalid, select first unused tool or fallback to first tool
                for tid in tool_ids:
                    if tid not in used_tool_ids:
                        tool_id = tid
                        break
                else:
                    tool_id = tool_ids[0]
                forced = True
            else:
                tool_history.append(
                    {
                        "tool_id": tool_id,
                        "ok": False,
                        "error": {"code": "tool_not_allowed", "message": "Tool not in agent tool list"},
                    }
                )
                break
        
        used_tool_ids.add(tool_id)
        selected_tool_info = next((tool for tool in tools_info if tool.get("id") == tool_id), None)

        # --- Math tool stage progression: detect repeated stage and force next stage ---
        if tool_id and not (selected_tool_info and selected_tool_info.get("stage")):
            parts = tool_id.split("-")
            current_stage = _math_tool_stage_from_id(tool_id)
            tool_id_call_counts[tool_id] = tool_id_call_counts.get(tool_id, 0) + 1
            stage_call_counts[current_stage] = stage_call_counts.get(current_stage, 0) + 1

            if tool_id_call_counts[tool_id] > 2 or stage_call_counts.get(current_stage, 0) > 2:
                next_stage = None
                if current_stage in _MATH_STAGE_ORDER:
                    idx = _MATH_STAGE_ORDER.index(current_stage)
                    for ns in _MATH_STAGE_ORDER[idx + 1:]:
                        if stage_call_counts.get(ns, 0) < 2:
                            next_stage = ns
                            break
                if next_stage is None and stage_call_counts.get("solve_support", 0) == 0:
                    next_stage = "solve_support"

                if next_stage:
                    candidates = [
                        t
                        for t in tool_ids
                        if _math_tool_stage_from_id(t) == next_stage or f"-{next_stage}-" in t
                    ]
                    if candidates:
                        old_id = tool_id
                        tool_id = candidates[0]
                        forced = True
                        print(
                            f"[DEBUG] Math stage forced: {old_id} (stage={current_stage}, "
                            f"calls={tool_id_call_counts.get(old_id, 0)}) -> {tool_id} (stage={next_stage})",
                            file=sys.stderr,
                        )

        tool_input = _normalize_tool_input(decision.get("tool_input"))
        exec_inputs = _build_tool_exec_inputs(
            task_text=task_text,
            role=role,
            agent_context=agent_context,
            results=results,
            task_context=task_context,
            artifact_bag=artifact_bag,
            tool_info=selected_tool_info,
        )
        
        # Pass failed code and error to next tool for refinement
        if last_error:
            if failed_code:
                exec_inputs["failed_code"] = failed_code
            exec_inputs["error_message"] = last_error
            
            # Build detailed diagnostic message
            diagnostic_hints = []
            
            if last_test_error:
                exec_inputs["test_error"] = last_test_error
                if original_test_error:
                    exec_inputs["original_test_error"] = original_test_error
                if latest_test_error:
                    exec_inputs["latest_test_error"] = latest_test_error
                if postprocess_error:
                    exec_inputs["postprocess_error"] = postprocess_error
                failure_summary = _summarize_failure(
                    last_error,
                    original_test_error or last_test_error,
                )
                if failure_summary:
                    exec_inputs["failure_summary"] = failure_summary
                
                # Analyze specific error patterns and provide targeted hints
                focus_error_text = original_test_error or last_test_error
                error_lower = focus_error_text.lower()
                
                # Type checking issues - be very aggressive about isinstance(x, int) bug
                if failed_code and "isinstance" in failed_code and "int" in failed_code:
                    if "5.0" in focus_error_text or "float" in error_lower or "assertionerror" in error_lower:
                        diagnostic_hints.append(
                            "❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0 that are mathematically integers.\n"
                            "✅ CRITICAL FIX: Replace isinstance(x, int) with: isinstance(x, (int, float)) and x == int(x)\n"
                            "   This accepts both 5 and 5.0 as valid integers."
                        )
                
                # Assertion errors with specific values
                if "assertionerror" in error_lower:
                    # Extract expected vs actual if present
                    if "==" in focus_error_text:
                        diagnostic_hints.append(
                            f"❌ ASSERTION FAILED: The test expectation was not met.\n"
                            f"✅ HINT: Check the logic carefully - the output doesn't match expected result."
                        )
                
                # Index/Key errors
                if "indexerror" in error_lower or "keyerror" in error_lower:
                    diagnostic_hints.append(
                        "❌ ACCESS ERROR: Trying to access invalid index/key.\n"
                        "✅ FIX: Add boundary checks or validate indices before access."
                    )
                
                # Name errors
                if "nameerror" in error_lower:
                    diagnostic_hints.append(
                        "❌ NAME ERROR: Variable not defined.\n"
                        "✅ FIX: Check variable names and ensure they're defined before use."
                    )
                
                # Type errors (wrong argument types)
                if "typeerror" in error_lower and "argument" in error_lower:
                    diagnostic_hints.append(
                        "❌ ARGUMENT ERROR: Wrong type passed to function.\n"
                        "✅ FIX: Check parameter types and convert if needed."
                    )
                
                # Build refinement request with diagnostics
                refinement_parts = [
                    "🔍 PREVIOUS ATTEMPT FAILED - DETAILED ANALYSIS:",
                    f"\n📋 Primary Test Error: {focus_error_text[:300]}",
                ]
                if postprocess_error and postprocess_error != focus_error_text:
                    refinement_parts.append(f"\n📋 Latest Postprocess Error: {postprocess_error[:300]}")
                
                if diagnostic_hints:
                    refinement_parts.append("\n💡 SPECIFIC ISSUES IDENTIFIED:")
                    refinement_parts.extend(f"\n{hint}" for hint in diagnostic_hints)
                
                failure_code_text = failed_code or "(no code)"
                refinement_parts.append(
                    f"\n\n❌ Failed Code:\n{failure_code_text}\n"
                    f"\n⚠️ CRITICAL: Do NOT repeat the same logic error. "
                    f"Analyze the error carefully and implement a DIFFERENT approach to fix it."
                )
                
                exec_inputs["refinement_request"] = "\n".join(refinement_parts)
            else:
                failure_code_text = failed_code or "(no code)"
                exec_inputs["refinement_request"] = (
                    f"🔍 PREVIOUS ATTEMPT FAILED:\n"
                    f"Error: {last_error}\n\n"
                    f"❌ Failed Code:\n{failure_code_text}\n\n"
                    f"⚠️ Please fix the error above and generate corrected code."
                )
            
            refinement_request = exec_inputs.get("refinement_request")
            if refinement_request:
                exec_inputs["task"] = f"{task_text}\n\n{refinement_request}"
                exec_inputs["query"] = exec_inputs["task"]

            # Force tool diversity after consecutive failures
            # NOTE: Only switches within current agent's available tools (tool_ids is from agent_context)
            if consecutive_same_category_failures >= 3:
                # Get tools from DIFFERENT categories within this agent's tool list
                different_category_tools = []
                for tid in tool_ids:  # tool_ids = current agent's available tools only
                    parts = tid.split('-')
                    if len(parts) >= 3:
                        tool_family = parts[2]
                        import re
                        tid_category = re.sub(r'\d+$', '', tool_family)
                    else:
                        tid_category = tid.rsplit('-', 1)[0] if '-' in tid else tid
                    
                    if tid_category != tool_category:
                        different_category_tools.append(tid)
                
                if different_category_tools:
                    # Only suggest tools from current agent's pool, just different category
                    role_context["suggested_tools"] = different_category_tools
                    role_context["force_different_category"] = True
                    role_context["failed_category"] = tool_category
                    # Extract categories from different_category_tools
                    alt_categories = list(set([re.sub(r'\d+$', '', t.split('-')[2]) if len(t.split('-')) >= 3 else t for t in different_category_tools[:5]]))
                    role_context["reason"] = (
                        f"Tool category '{tool_category}' failed {consecutive_same_category_failures} times consecutively. "
                        f"MUST switch to a DIFFERENT tool category (within current agent's tools) from: {alt_categories}"
                    )
                    print(f"[DEBUG] Forcing tool category switch from '{tool_category}' after {consecutive_same_category_failures} failures", file=sys.stderr)
                    print(f"[DEBUG] Available different-category tools in current agent: {different_category_tools[:3]}", file=sys.stderr)
                else:
                    # No different category tools available - force switch to DIFFERENT tool within same category
                    same_category_different_tools = [tid for tid in tool_ids if tid != tool_id]
                    if same_category_different_tools:
                        role_context["suggested_tools"] = same_category_different_tools
                        role_context["force_different_tool"] = True
                        role_context["failed_tool"] = tool_id
                        role_context["reason"] = (
                            f"Tool '{tool_id}' failed {consecutive_same_category_failures} times consecutively. "
                            f"Current agent only has '{tool_category}' category tools. "
                            f"MUST switch to a DIFFERENT tool from: {same_category_different_tools[:5]}"
                        )
                        print(f"[DEBUG] Agent only has '{tool_category}' tools - forcing switch from '{tool_id}' to different tool", file=sys.stderr)
                        print(f"[DEBUG] Available alternative tools: {same_category_different_tools[:3]}", file=sys.stderr)
                    else:
                        print(f"[DEBUG] Cannot switch: agent only has one tool '{tool_id}'", file=sys.stderr)
            
            # If same specific tool failed 3+ times, suggest alternatives
            elif tool_failure_counts.get(tool_id, 0) >= 3:
                unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids or tool_failure_counts.get(tid, 0) < 3]
                if unused_tools:
                    role_context["suggested_tools"] = unused_tools
                    role_context["reason"] = f"Tool {tool_id} failed {tool_failure_counts[tool_id]} times, consider alternative"
        
        exec_inputs.update(tool_input)
        
        # Debug output: verify refinement_request is being passed
        if "refinement_request" in exec_inputs:
            print(f"\n[DEBUG] Round {round_idx}: Passing refinement_request to tool {tool_id}", file=sys.stderr)
            print(f"[DEBUG] Refinement request (first 300 chars):", file=sys.stderr)
            print(f"[DEBUG] {exec_inputs['refinement_request'][:300]}...", file=sys.stderr)
        if "failed_code" in exec_inputs:
            print(f"[DEBUG] Passing failed_code (length: {len(exec_inputs['failed_code'])})", file=sys.stderr)
        if "test_error" in exec_inputs:
            print(f"[DEBUG] Passing test_error: {exec_inputs['test_error'][:100]}...", file=sys.stderr)
        
        tool_result = tool_executor.run_tool(tool_id, exec_inputs)
        stage_name = str((selected_tool_info or {}).get("stage") or "")
        if isinstance(tool_result, dict) and tool_result.get("ok"):
            _merge_tool_payload_into_artifact_bag(artifact_bag, tool_result.get("output"), selected_tool_info)
            role_context["artifact_bag"] = artifact_bag
        
        # Extract tool category for diversity tracking (use the specific tool family, not the broad category)
        # Tool IDs are like: code-generation-assemblesnippets, code-generation-generatemath6
        # We want to extract: assemblesnippets, generatemath (without the number suffix)
        parts = tool_id.split('-')
        if len(parts) >= 3:
            # Extract the tool family name (e.g., 'assemblesnippets', 'generatemath')
            tool_family = parts[2]
            # Remove trailing numbers (e.g., 'generatemath6' -> 'generatemath')
            import re
            tool_category = re.sub(r'\d+$', '', tool_family)
        else:
            tool_category = tool_id.rsplit('-', 1)[0] if '-' in tool_id else tool_id
        
        # Note: We track category change but DON'T reset consecutive_same_category_failures here
        # The counter should only reset on SUCCESS, not just category change
        # This allows the force-switch logic (at 3 failures) to work properly
        if last_tool_category and last_tool_category != tool_category:
            print(f"[DEBUG] Tool category changed from '{last_tool_category}' to '{tool_category}'", file=sys.stderr)
        last_tool_category = tool_category
        
        # Check if this tool produced useful code or had errors
        has_errors = False
        failure_stage = ""
        failure_detail = ""
        if isinstance(tool_result, dict):
            # Check for execution failure (ok=False or has error)
            if tool_result.get("ok") is False:
                has_errors = True
                failure_stage = "tool_executor_failed"
                # Capture the failed code and error for next tool
                if tool_result.get("error"):
                    error_obj = tool_result["error"]
                    if isinstance(error_obj, dict):
                        last_error = f"{error_obj.get('code', 'Error')}: {error_obj.get('message', 'Unknown error')}"
                    else:
                        last_error = str(error_obj)
                failure_detail = _extract_tool_failure_detail(tool_result)
            elif tool_result.get("error"):
                # Has error even with ok=True
                has_errors = True
                failure_stage = "tool_executor_error_field"
                error_obj = tool_result["error"]
                if isinstance(error_obj, dict):
                    last_error = f"{error_obj.get('code', 'Error')}: {error_obj.get('message', 'Unknown error')}"
                else:
                    last_error = str(error_obj)
                failure_detail = _extract_tool_failure_detail(tool_result)
            elif tool_result.get("ok") and "output" in tool_result:
                # Tool executed successfully, check if code is useful
                output = tool_result["output"]
                if isinstance(output, dict):
                    # Handle nested output structure: output.output.code
                    inner_output = None
                    success_flag = True  # Default to true if not specified
                    
                    if "output" in output and isinstance(output["output"], dict):
                        inner_output = output["output"]
                        code = inner_output.get("code_or_commands") or inner_output.get("code") or inner_output.get("solution")
                        # Check success flag in nested structure
                        if "success" in inner_output:
                            success_flag = bool(inner_output.get("success"))
                    else:
                        code = output.get("code_or_commands") or output.get("code") or output.get("solution")
                        # Check success flag in flat structure
                        if "success" in output:
                            success_flag = bool(output.get("success"))
                    
                    # Validate code: must exist, be string, not be error message, and have success=true
                    if code and isinstance(code, str) and success_flag:
                        code_length = len(code)
                        print(f"[DEBUG] Round {round_idx}: Extracted code (len={code_length}), success_flag={success_flag}, is_placeholder={_is_placeholder_code(code)}", file=sys.stderr)
                        
                        # Early stopping detection: check if code length hasn't changed
                        if code_length == last_code_length:
                            same_code_length_count += 1
                            if same_code_length_count >= 3:
                                print(f"[DEBUG] ⚠️ Early stop: Same code length ({code_length}) for {same_code_length_count} rounds, agent stuck!", file=sys.stderr)
                                has_errors = True  # Mark as having errors to stop loop
                                break
                        else:
                            same_code_length_count = 0
                            last_code_length = code_length
                        
                        # Check if code is actually an error message
                        if _is_likely_error_text(code):
                            has_errors = True
                            failure_stage = "tool_returned_error_text"
                            failed_code = None
                            last_error = code.strip()
                        elif not _is_placeholder_code(code):
                            # Got valid code, now test it if test context available
                            if task_context and task_context.get("test"):
                                print(f"[DEBUG] Running test for tool {tool_id}, round {round_idx}", file=sys.stderr)
                                
                                test_ok, test_error = _test_code(code, task_context)
                                
                                print(f"[DEBUG] Test result: ok={test_ok}, error={test_error[:100] if test_error else 'None'}", file=sys.stderr)
                                
                                if test_ok:
                                    # Test passed! Set success and stop immediately
                                    successful_code = code
                                    test_passed = True
                                    has_errors = False
                                    failed_code = None
                                    last_error = None
                                    last_test_error = None
                                    original_test_error = None
                                    latest_test_error = None
                                    postprocess_error = None
                                    role_context["test_passed"] = True
                                    role_context.pop("diagnostics", None)
                                    artifact_bag["draft_code"] = code
                                    artifact_bag["normalized_code"] = code
                                    
                                    print(f"[DEBUG] Test PASSED! Stopping immediately at round {round_idx}", file=sys.stderr)
                                    
                                    # Record successful result and break immediately
                                    tool_history.append(
                                        {
                                            "tool_id": tool_id,
                                            "stage": stage_name,
                                            "input": tool_input,
                                            "result": tool_result,
                                            "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                                        }
                                    )
                                    if tool_history:
                                        role_context["tool_history"] = tool_history
                                    return tool_history  # Stop immediately on test pass
                                else:
                                    # Test failed, treat as error and retry
                                    has_errors = True
                                    failure_stage = "unit_test_failed"
                                    failed_code = code
                                    last_error = f"Test failed: {test_error}"
                                    last_test_error = test_error
                                    latest_test_error = test_error
                                    postprocess_error = None
                                    if original_test_error is None:
                                        original_test_error = test_error
                                    failure_summary = _summarize_failure(last_error, test_error)
                                    role_context["diagnostics"] = {
                                        "test_error": test_error,
                                        "error_message": last_error,
                                        "failed_code": code,
                                        "original_test_error": original_test_error,
                                        "latest_test_error": latest_test_error,
                                        "failure_summary": failure_summary,
                                    }
                                    # Track failure count for this tool and category
                                    tool_failure_counts[tool_id] = tool_failure_counts.get(tool_id, 0) + 1
                                    tool_category_counts[tool_category] = tool_category_counts.get(tool_category, 0) + 1
                                    consecutive_same_category_failures += 1
                                    
                                    print(f"[DEBUG] Test FAILED: {test_error[:200]}, retrying...", file=sys.stderr)
                                    print(f"[DEBUG] Tool category '{tool_category}' failures: {tool_category_counts[tool_category]}, consecutive: {consecutive_same_category_failures}", file=sys.stderr)
                                    
                                    if enable_postprocess_repair and _should_attempt_postprocess_repair(test_error):
                                        base_test_error = original_test_error or last_test_error or test_error
                                        repair_llm = _resolve_repair_llm(llm, tool_executor)
                                        post_ok, post_error, post_code = _postprocess_tool_failure(
                                            code,
                                            task_context,
                                            test_error,
                                            repair_llm,
                                            timeout_s=5.0,
                                        )
                                        if post_ok:
                                            successful_code = post_code
                                            test_passed = True
                                            has_errors = False
                                            failed_code = None
                                            last_error = None
                                            last_test_error = None
                                            original_test_error = None
                                            latest_test_error = None
                                            postprocess_error = None
                                            role_context["test_passed"] = True
                                            role_context.pop("diagnostics", None)
                                            artifact_bag["draft_code"] = post_code
                                            artifact_bag["normalized_code"] = post_code
                                            
                                            print(
                                                f"[DEBUG] Postprocess+LLM repair fixed tool {tool_id} at round {round_idx}",
                                                file=sys.stderr,
                                            )
                                            
                                            tool_history.append(
                                                {
                                                    "tool_id": tool_id,
                                                    "stage": stage_name,
                                                    "input": tool_input,
                                                    "result": tool_result,
                                                    "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                                                }
                                            )
                                            if tool_history:
                                                role_context["tool_history"] = tool_history
                                            return tool_history

                                        postprocess_error = post_error
                                        latest_test_error = post_error
                                        failure_stage = "postprocess_or_remote_repair_failed"
                                        if _is_postprocess_degradation(base_test_error, post_error):
                                            print(
                                                "[DEBUG] Postprocess degraded failure type; keep original failure context",
                                                file=sys.stderr,
                                            )
                                            failed_code = failed_code or code
                                            last_error = f"Test failed: {base_test_error}"
                                        else:
                                            failed_code = post_code or failed_code
                                            last_error = f"Postprocess failed: {post_error}"
                                        failure_summary = _summarize_failure(last_error, original_test_error or last_test_error)
                                        role_context["diagnostics"] = {
                                            "test_error": original_test_error or last_test_error or post_error,
                                            "error_message": last_error,
                                            "failed_code": failed_code,
                                            "original_test_error": original_test_error,
                                            "latest_test_error": latest_test_error,
                                            "postprocess_error": postprocess_error,
                                            "failure_stage": failure_stage,
                                            "failure_ts": round_ts,
                                            "failure_summary": failure_summary,
                                        }
                                    elif enable_postprocess_repair:
                                        print(
                                            f"[DEBUG] Skipping postprocess repair for failure type: {_classify_error_message(test_error)}",
                                            file=sys.stderr,
                                        )
                                    else:
                                        print("[DEBUG] Postprocess repair disabled by configuration", file=sys.stderr)
                            else:
                                # No test available, accept code based on success flag
                                # Mark as successful but continue to allow LLM to decide when to stop
                                print(f"[DEBUG] No test context available (task_context={bool(task_context)}, has_test={bool(task_context.get('test') if task_context else False)})", file=sys.stderr)
                                
                                successful_code = code
                                has_errors = False
                                failed_code = None
                                last_error = None
                                original_test_error = None
                                latest_test_error = None
                                postprocess_error = None
                                consecutive_same_category_failures = 0  # Reset on success
                        else:
                            # Placeholder code counts as "no useful result"
                            has_errors = True
                            failure_stage = "tool_returned_placeholder_code"
                            failed_code = code  # Save for refinement
                            last_error = "Generated placeholder code (e.g., 'return None')"
                    else:
                        # No valid code or success=false
                        has_errors = True
                        failure_stage = "tool_output_invalid_or_success_false"
                        if not success_flag and inner_output:
                            last_error = inner_output.get("error") or "Tool returned success=false"
                        failure_detail = _extract_tool_failure_detail(tool_result)

        if has_errors:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
            if tool_only:
                resolved_answer = _extract_tool_final_answer_candidate(tool_result)
                if resolved_answer:
                    role_context["resolved_final_answer"] = resolved_answer
                    role_context["candidate_final_answer"] = resolved_answer
                    role_context["candidate_final_answer_source"] = tool_id
                    role_context["candidate_final_answer_round"] = round_idx
                    role_context["allow_tool_only_final_decision"] = True
                    role_context["allow_llm_reason"] = "tool_result_contains_candidate_final_answer"
                    print(
                        f"[DEBUG] tool_only candidate final answer detected at round {round_idx}: "
                        f"tool={tool_id} final_answer={_truncate_text(str(resolved_answer), max_chars=120)}; "
                        "agent will self-evaluate in next planning step",
                        file=sys.stderr,
                    )

        if has_errors and (last_error or failed_code or last_test_error):
            failure_summary = _summarize_failure(
                last_error,
                original_test_error or last_test_error,
            )
            diagnostics_payload = {
                "error_message": last_error,
                "test_error": original_test_error or last_test_error,
                "failed_code": failed_code,
                "original_test_error": original_test_error,
                "latest_test_error": latest_test_error,
                "postprocess_error": postprocess_error,
                "failure_stage": failure_stage or "unknown",
                "failure_ts": round_ts,
                "failure_summary": failure_summary,
            }
            role_context["diagnostics"] = {k: v for k, v in diagnostics_payload.items() if v}
            print(
                f"[DEBUG][FailureTrace] ts={round_ts} round={round_idx} role={role} tool={tool_id} "
                f"category={tool_category} stage={failure_stage or 'unknown'}",
                file=sys.stderr,
            )
            if last_error:
                print(f"[DEBUG][FailureTrace] error={_truncate_text(str(last_error), max_chars=500)}", file=sys.stderr)
            if failure_detail:
                print(f"[DEBUG][FailureTrace] detail={failure_detail}", file=sys.stderr)
            if isinstance(tool_result, dict):
                top_keys = list(tool_result.keys())
                print(f"[DEBUG][FailureTrace] tool_result_keys={top_keys}", file=sys.stderr)
                output_obj = tool_result.get("output")
                if isinstance(output_obj, dict):
                    print(f"[DEBUG][FailureTrace] output_keys={list(output_obj.keys())}", file=sys.stderr)

        # Track last tool category for next iteration
        last_tool_category = tool_category
        
        # Track which categories we've tried
        if tool_category:
            tried_categories.add(tool_category)
        
        # Early stopping: if we've tried multiple categories and all failed
        if len(tried_categories) >= 3 and consecutive_same_category_failures >= 5:
            print(f"[DEBUG] ⚠️ Early stop: Tried {len(tried_categories)} categories, all failing. Agent exhausted!", file=sys.stderr)
            break
        
        # Clear force flags after each round (they should only affect one round)
        role_context.pop("force_different_category", None)
        role_context.pop("force_different_tool", None)
        role_context.pop("suggested_tools", None)
        role_context.pop("reason", None)
        role_context.pop("failed_category", None)
        role_context.pop("failed_tool", None)
        
        # Only append to history if we didn't already return (test didn't pass)
        if not (test_passed and successful_code):
            tool_history.append(
                {
                    "tool_id": tool_id,
                    "stage": stage_name,
                    "input": tool_input,
                    "result": tool_result,
                    "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                }
            )

        if role_context.get("stop_tool_loop"):
            break

        # Stop conditions:
        # 1. Got successful code AND no errors AND not forced to use all tools
        # 2. OR reached max rounds (removed the "tried all tools" condition to allow refinement)
        if successful_code and not has_errors and not tool_only:
            break
        # Note: Removed "if len(used_tool_ids) >= len(tool_ids): break" to allow tools
        # to be retried for refinement. The loop will continue up to tool_max_rounds.
    if tool_history:
        role_context["tool_history"] = tool_history
    return tool_history


def _extract_tool_output(tool_history: List[Dict[str, Any]]) -> Optional[Any]:
    for entry in reversed(tool_history):
        result = entry.get("result")
        if isinstance(result, dict) and result.get("ok") and "output" in result:
            return result.get("output")
    return None


def _normalize_math_tool_stage(stage: Any) -> str:
    normalized = str(stage or "").strip().lower().replace("-", "_")
    if normalized in {"solve_support", "solve_supporting", "support"}:
        return "solve_support"
    if normalized in {"solve", "verify", "format", "model", "analyze"}:
        return normalized
    return ""


def _math_tool_stage_from_id(tool_id: str, stage: Any = None) -> str:
    stage_from_metadata = _normalize_math_tool_stage(stage)
    if stage_from_metadata:
        return stage_from_metadata
    source_text = str(tool_id or "").strip()
    match = re.search(r"(?:^|[|;,\s])stage=([A-Za-z0-9_-]+)", source_text)
    if match:
        stage_from_source = _normalize_math_tool_stage(match.group(1))
        if stage_from_source:
            return stage_from_source
    parts = source_text.split("|", 1)[0].split("-")
    if len(parts) >= 3 and parts[0] == "math":
        return _normalize_math_tool_stage(parts[2])
    return ""


def _normalize_math_candidate_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    matches = _HASH_ANSWER_RE.findall(text)
    if matches:
        text = matches[-1].strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    text = text.replace("\\,", "").replace("\\ ", "")
    text = re.sub(r"\s+", "", text)
    return text


def _parse_math_candidate_number(value: Any) -> Optional[Decimal]:
    text = _normalize_math_candidate_text(value)
    if not text or not _PURE_NUMERIC_ANSWER_RE.fullmatch(text):
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return Decimal(num) / Decimal(den)
        return Decimal(text)
    except (InvalidOperation, ZeroDivisionError):
        return None


def _math_candidate_answers_equivalent(lhs: Any, rhs: Any) -> bool:
    left_norm = _normalize_math_candidate_text(lhs)
    right_norm = _normalize_math_candidate_text(rhs)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_num = _parse_math_candidate_number(left_norm)
    right_num = _parse_math_candidate_number(right_norm)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) <= Decimal("1e-8")
    return False


def _select_preferred_math_tool_candidate(
    tool_history: List[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    if not isinstance(tool_history, list):
        return None

    records: List[Dict[str, Any]] = []
    for idx, entry in enumerate(tool_history):
        if not isinstance(entry, dict):
            continue
        tool_id = str(entry.get("tool_id") or "").strip()
        stage = _math_tool_stage_from_id(tool_id, entry.get("stage"))
        if not stage:
            continue
        answer = _extract_tool_final_answer_candidate(entry.get("result"))
        if not answer or _is_pseudo_final_answer(answer):
            continue
        source = tool_id or "tool_history"
        if stage and not _math_tool_stage_from_id(source):
            source = f"{source}|stage={stage}"
        records.append(
            {
                "answer": str(answer).strip(),
                "source": source,
                "stage": stage,
                "index": idx,
            }
        )

    if not records:
        return None

    def _pick_stage_candidate(stage_name: str) -> Optional[Tuple[str, str]]:
        stage_records = [record for record in records if record.get("stage") == stage_name]
        if not stage_records:
            return None

        groups: List[Dict[str, Any]] = []
        for record in stage_records:
            placed = False
            for group in groups:
                if _math_candidate_answers_equivalent(record["answer"], group["answer"]):
                    group["items"].append(record)
                    placed = True
                    break
            if not placed:
                groups.append({"answer": record["answer"], "items": [record]})

        best_group = min(
            groups,
            key=lambda group: (
                -len(group.get("items") or []),
                min(int(item.get("index", 10**9)) for item in group.get("items") or []),
            ),
        )
        best_item = min(best_group.get("items") or [], key=lambda item: int(item.get("index", 10**9)))
        return str(best_item.get("answer") or "").strip(), str(best_item.get("source") or "tool_history")

    for stage_name in ("solve_support", "solve", "verify", "format", "model", "analyze"):
        preferred = _pick_stage_candidate(stage_name)
        if preferred:
            return preferred
    return None


def _extract_tool_final_answer_candidate(
    value: Any,
    depth: int = 0,
    *,
    allow_numeric_scalar: bool = False,
) -> Optional[str]:
    """Extract a final answer signal from heterogeneous tool outputs."""
    if value is None or depth > 12:
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value) if allow_numeric_scalar else None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        matches = _HASH_ANSWER_RE.findall(text)
        if matches:
            candidate = matches[-1].strip()
            if _is_pseudo_final_answer(candidate):
                return None
            return candidate
        if _is_pseudo_final_answer(text):
            return None
        if allow_numeric_scalar and re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?", text):
            return text
        return None

    if isinstance(value, dict):
        if value.get("ok") is False:
            return None

        for key in ("final_answer", "numeric_answer", "recomputed_answer"):
            if key in value:
                found = _extract_tool_final_answer_candidate(
                    value.get(key),
                    depth + 1,
                    allow_numeric_scalar=True,
                )
                if found:
                    return found

        for key in ("formatted", "result", "response", "content", "message", "summary"):
            if key in value:
                found = _extract_tool_final_answer_candidate(value.get(key), depth + 1)
                if found:
                    return found

        for key in ("payload", "output", "data", "results", "tool_exec", "execution_trace"):
            if key in value:
                found = _extract_tool_final_answer_candidate(value.get(key), depth + 1)
                if found:
                    return found

        for nested in value.values():
            found = _extract_tool_final_answer_candidate(nested, depth + 1)
            if found:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            found = _extract_tool_final_answer_candidate(
                item,
                depth + 1,
                allow_numeric_scalar=allow_numeric_scalar,
            )
            if found:
                return found
        return None

    return None


def _extract_candidate_final_answer_from_tool_history(
    tool_history: Any,
) -> Optional[Tuple[str, str]]:
    if not isinstance(tool_history, list):
        return None
    preferred_math = _select_preferred_math_tool_candidate(tool_history)
    if preferred_math:
        return preferred_math
    for entry in reversed(tool_history):
        if not isinstance(entry, dict):
            continue
        answer = _extract_tool_final_answer_candidate(entry.get("result"))
        if answer:
            source = str(entry.get("tool_id") or "tool_history")
            return answer, source
    return None


def _extract_candidate_final_answer_from_output(output: Any) -> Optional[str]:
    return _extract_tool_final_answer_candidate(output)


def _should_replace_candidate_final_answer(
    current_state: Dict[str, Any],
    new_answer: Any,
    new_source: Any,
) -> bool:
    current_answer = current_state.get("answer")
    current_source = current_state.get("source")

    if not new_answer or _is_pseudo_final_answer(new_answer):
        return False
    if not current_answer or _is_pseudo_final_answer(current_answer):
        return True
    if _math_candidate_answers_equivalent(current_answer, new_answer):
        current_stage = _math_tool_stage_from_id(str(current_source or ""))
        new_stage = _math_tool_stage_from_id(str(new_source or ""))
        return current_stage not in {"solve_support", "solve"} and new_stage in {"solve_support", "solve"}

    current_stage = _math_tool_stage_from_id(str(current_source or ""))
    new_stage = _math_tool_stage_from_id(str(new_source or ""))
    if current_stage in {"solve_support", "solve"}:
        return False
    if new_stage in {"solve_support", "solve"}:
        return True

    stage_rank = {"solve_support": 0, "solve": 0, "verify": 1, "format": 2, "model": 3, "analyze": 4, "": 5}
    return stage_rank.get(new_stage, 5) < stage_rank.get(current_stage, 5)


def _is_placeholder_code(code: str) -> bool:
    """Check if the code is a placeholder like 'return None' or empty."""
    if not code or not isinstance(code, str):
        return True
    normalized = code.strip().lower()
    # Empty or trivial code
    if not normalized or normalized == "pass":
        return True
    # Common placeholder patterns
    placeholder_patterns = [
        "return none",
        "return null",
        "# todo",
        "# placeholder",
        "not implemented",
        "raise notimplementederror",
    ]
    for pattern in placeholder_patterns:
        if pattern in normalized:
            return True
    # Very short code that's likely just a placeholder
    if len(normalized.split('\n')) == 1 and len(normalized) < 20:
        if normalized.startswith('return') or normalized == '...':
            return True
    return False


def _apply_stop_tokens(text: str, stop_tokens: List[str]) -> str:
    if not text or not stop_tokens:
        return text
    stop_pos = None
    for token in stop_tokens:
        if not token:
            continue
        idx = text.find(token)
        if idx == -1:
            continue
        if stop_pos is None or idx < stop_pos:
            stop_pos = idx
    if stop_pos is None:
        return text
    return text[:stop_pos]


def _unwrap_json_code(text: str) -> str:
    if not text or "{" not in text:
        return text
    try:
        start_idx = text.find("{")
        if start_idx == -1:
            return text
        json_text = text[start_idx:]
        data = json.loads(json_text)
        if isinstance(data, dict):
            for key in ["code_or_commands", "code", "solution", "output"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and value.strip():
                        return value
                    if isinstance(value, dict):
                        for nested_key in ["code_or_commands", "code", "solution"]:
                            if nested_key in value and isinstance(value[nested_key], str):
                                return value[nested_key]
    except Exception:
        return text
    return text


def _extract_code_block(text: str) -> str:
    if not text:
        return text
    text = _strip_chat_role_artifacts(_unwrap_json_code(text))
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 3:
        return text
    code = parts[1]
    lines = code.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "json"}:
        lines = lines[1:]
    extracted = "\n".join(lines).strip() or text
    return _strip_chat_role_artifacts(extracted)


def _strip_chat_role_artifacts(text: str) -> str:
    if not text:
        return text
    cleaned_lines: List[str] = []
    for raw_line in str(text).replace("\r\n", "\n").splitlines():
        stripped = raw_line.strip()
        if _CHAT_ROLE_TAG_RE.fullmatch(stripped):
            continue
        if _CHAT_ROLE_LINE_RE.fullmatch(stripped):
            continue
        prefix_match = _CHAT_ROLE_PREFIX_RE.match(raw_line)
        if prefix_match:
            remainder = prefix_match.group(2)
            if not remainder.strip():
                continue
            cleaned_lines.append(remainder)
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines).strip("\n")


def _strip_redundant_def(text: str, entry_point: Optional[str]) -> str:
    if not entry_point or not text:
        return text
    target = f"def {entry_point}"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(target):
            body_lines = lines[idx + 1 :]
            if not body_lines:
                return text
            return "\n".join(body_lines).lstrip("\n")
    return text


def _normalize_completion(text: str, entry_point: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\\n", "\n")
    if not entry_point:
        return text
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return text
    if lines[0].lstrip().startswith(("def ", "class ", "@")):
        return "\n".join(lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    min_pos_indent = min((i for i in indents if i > 0), default=0)
    if min_pos_indent > 0:
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            leading = len(line) - len(line.lstrip())
            trim = min(leading, min_pos_indent)
            new_lines.append(line[trim:])
        lines = new_lines
    lines = [("    " + line) if line.strip() else line for line in lines]
    return "\n".join(lines)


def _build_program(
    prompt: str, completion: str, test_code: str, entry_point: Optional[str] = None
) -> str:
    prompt_text = prompt or ""
    completion_text = completion or ""
    completion_text = _extract_code_block(completion_text)
    completion_text = _strip_redundant_def(completion_text, entry_point)
    completion_text = _normalize_completion(completion_text, entry_point)
    if not completion_text.endswith("\n"):
        completion_text += "\n"
    return f"{prompt_text}{completion_text}\n{test_code}\n"


def _looks_like_syntax_error(message: str) -> bool:
    if not message:
        return False
    return any(token in message for token in ("SyntaxError", "IndentationError", "TabError"))


def _classify_error_message(message: Optional[str]) -> str:
    text = str(message or "").strip().lower()
    if not text:
        return "unknown"
    if "assertionerror" in text or " assert " in f" {text} ":
        return "assertion"
    if any(token in text for token in ("syntaxerror", "indentationerror", "taberror")):
        return "syntax"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "nameerror" in text:
        return "name"
    if "typeerror" in text:
        return "type"
    if "indexerror" in text:
        return "index"
    if "keyerror" in text:
        return "key"
    if "attributeerror" in text:
        return "attribute"
    if "valueerror" in text:
        return "value"
    return "runtime"


def _should_attempt_postprocess_repair(error_message: Optional[str]) -> bool:
    kind = _classify_error_message(error_message)
    return kind != "unknown"


def _is_postprocess_degradation(original_error: Optional[str], postprocess_error: Optional[str]) -> bool:
    original_kind = _classify_error_message(original_error)
    repaired_kind = _classify_error_message(postprocess_error)
    if repaired_kind == "unknown":
        return False
    if repaired_kind == "syntax" and original_kind != "syntax":
        return True
    if repaired_kind == "timeout" and original_kind not in {"timeout", "unknown"}:
        return True
    return False


def _repair_completion(completion: str, entry_point: Optional[str]) -> str:
    text = str(completion or "")
    text = text.replace("\r\n", "\n")
    text = _extract_code_block(text)
    text = _strip_chat_role_artifacts(text)
    text = text.expandtabs(4)
    text = _strip_redundant_def(text, entry_point)
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    text = "\n".join(lines)
    text = textwrap.dedent(text).strip("\n")
    if entry_point:
        lines = text.splitlines()
        lines = [("    " + line) if line.strip() else line for line in lines]
        text = "\n".join(lines)
    return text


def _run_python(code: str, timeout_s: float) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "eval_task.py")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return False, f"File write error: {str(e)}"
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "Test execution timeout"
        except Exception as e:
            return False, f"Test execution error: {str(e)}"
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"exit_code={result.returncode}"
    return False, message


def _run_with_repair(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[bool, str]:
    code = _build_program(prompt, str(completion), test_code, entry_point)
    ok, message = _run_python(code, timeout_s=timeout_s)
    if ok or not _looks_like_syntax_error(message):
        return ok, message
    repaired = _repair_completion(str(completion), entry_point)
    if not repaired or repaired == str(completion):
        return ok, message
    repaired_code = _build_program(prompt, repaired, test_code, entry_point)
    ok2, message2 = _run_python(repaired_code, timeout_s=timeout_s)
    return ok2, ("" if ok2 else message2)


def _test_code(code: str, task_context: Optional[Dict[str, Any]], timeout_s: float = 5.0) -> Tuple[bool, str]:
    """Test generated code if task_context contains test information.
    
    Args:
        code: Generated code to test
        task_context: Context containing prompt, test, entry_point
        timeout_s: Timeout for test execution
        
    Returns:
        (test_passed, error_message)
    """
    if not task_context:
        return True, ""  # No test available, assume pass
    
    prompt = task_context.get("prompt", "")
    test_code = task_context.get("test", "")
    entry_point = task_context.get("entry_point")
    stop_tokens = task_context.get("stop_tokens") or []
    
    if not test_code:
        return True, ""  # No test available
    
    completion = str(code)
    if stop_tokens:
        completion = _apply_stop_tokens(_extract_code_block(completion), stop_tokens)
    ok, message = _run_with_repair(prompt, completion, test_code, entry_point, timeout_s=timeout_s)
    return ok, message


def _resolve_repair_llm(llm: Any, tool_executor: Optional[ToolExecutor]) -> Optional[Any]:
    if tool_executor is not None:
        tool_llm = getattr(tool_executor, "llm", None)
        if tool_llm is not None:
            return tool_llm
    if hasattr(llm, "llm"):
        inner = getattr(llm, "llm", None)
        if inner is not None:
            return inner
    if hasattr(llm, "chat"):
        return llm
    return None


def _postprocess_tool_failure(
    code: str,
    task_context: Optional[Dict[str, Any]],
    error_message: str,
    llm_client: Optional[Any],
    timeout_s: float = 5.0,
) -> Tuple[bool, str, str]:
    if not task_context:
        return False, error_message, code

    prompt = task_context.get("prompt", "")
    test_code = task_context.get("test", "")
    entry_point = task_context.get("entry_point")
    stop_tokens = task_context.get("stop_tokens") or []

    if not test_code:
        return False, error_message, code

    candidate = str(code or "")
    try:
        from evaluation import humaneval_postprocess as _hp

        param_names = _hp._extract_param_names_from_prompt(prompt, entry_point)
        repaired_text, _ = _hp._run_postprocess_tool_agent(
            completion=candidate,
            prompt=prompt,
            entry_point=entry_point,
            stop_tokens=stop_tokens,
            use_stop_tokens=bool(stop_tokens),
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=error_message,
        )
        if repaired_text and str(repaired_text).strip():
            candidate = str(repaired_text)
        candidate = _hp._normalize_completion(
            _hp._strip_redundant_def(_hp._extract_code_block(candidate), entry_point),
            entry_point,
        )
        candidate = _hp._fix_missing_indents(candidate)
        if llm_client is not None:
            repaired_llm = _hp._repair_assertion_completion_with_llm(
                llm_client=llm_client,
                prompt=prompt,
                completion=candidate,
                test_code=test_code,
                entry_point=entry_point,
                error_message=error_message,
            )
            if repaired_llm and str(repaired_llm).strip():
                candidate = repaired_llm
    except Exception as exc:
        print(
            f"[DEBUG] Postprocess repair pipeline exception: {type(exc).__name__}: "
            f"{_truncate_text(str(exc), max_chars=500)}",
            file=sys.stderr,
        )

    ok, message = _run_with_repair(prompt, candidate, test_code, entry_point, timeout_s=timeout_s)
    return ok, message, candidate


def _summarize_tool_history(tool_history: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for entry in tool_history:
        tool_id = entry.get("tool_id") or "tool"
        reason = entry.get("reason")
        result = entry.get("result")
        ok = result.get("ok") if isinstance(result, dict) else None
        if reason:
            summaries.append(f"{tool_id} ok={ok} reason={reason}")
        else:
            summaries.append(f"{tool_id} ok={ok}")
    return summaries


def _extract_final_answer_from_tool_history(tool_history: List[Dict[str, Any]]) -> Optional[str]:
    """Walk tool_history in reverse and pull final_answer from nested payloads."""
    preferred_math = _select_preferred_math_tool_candidate(tool_history)
    if preferred_math:
        return preferred_math[0]
    for entry in reversed(tool_history):
        result = entry.get("result")
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        output = result.get("output")
        if not isinstance(output, dict):
            continue
        inner = output.get("payload") or output
        if isinstance(inner, dict):
            for key in ("final_answer", "numeric_answer", "recomputed_answer", "formatted"):
                val = inner.get(key)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    hash_match = _HASH_ANSWER_RE.findall(text)
                    if hash_match:
                        return hash_match[-1].strip()
                    return text
                if isinstance(val, (int, float)):
                    return str(val)
        direct = _extract_tool_final_answer_candidate(output)
        if direct:
            return direct
    return None


def _tool_history_to_output(role: str, tool_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tool_history:
        return {"error": "no_tool_used"}
    payload = _extract_tool_output(tool_history)
    payload_text = _extract_code_candidate(payload) or ""
    if not payload_text and isinstance(payload, str):
        payload_text = _extract_code_block(payload).strip()
    if not payload_text and payload is not None and not isinstance(payload, (dict, list, tuple)):
        payload_text = json.dumps(payload, ensure_ascii=True, default=str)
    summary = _summarize_tool_history(tool_history)
    final_answer = _extract_final_answer_from_tool_history(tool_history)
    role_key = _normalize_role_name(role)
    if role_key == "builder":
        return {"runnable_plan": summary, "code_or_commands": payload_text, "self_test": [], "final_answer": final_answer}
    if role_key == "refactor":
        return {"runnable_plan": summary, "code_or_commands": payload_text, "self_test": [], "final_answer": final_answer}
    if role_key == "planner":
        return {"steps": summary, "acceptance_criteria": [], "code_or_commands": payload_text, "final_answer": final_answer}
    if role_key == "evidence_finder":
        return {"evidence_targets": [], "sources": [], "evidence_points": summary, "code_or_commands": payload_text}
    if role_key == "checker":
        return {"test_cases": [], "verdicts": [], "failure_localization": payload_text or "tool_only", "final_answer": final_answer}
    if role_key == "manager":
        return {
            "status": "finish",
            "next_role": None,
            "instruction": None,
            "summary": payload_text or "tool_only",
            "notes": summary,
            "steps": summary if isinstance(summary, list) else [summary] if summary else [],
            "code_or_commands": payload_text,
        }
    return {"output": payload_text, "final_answer": final_answer}


def _select_agent_for_role(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    use_mmr: bool,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool,
    reuse_cache: bool,
    cache: Dict[str, Dict[str, Any]],
    cache_key: Optional[str],
    artifact_context: Optional[Dict[str, Any]],
    selected_agent_ids: Optional[List[str]],
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
) -> Dict[str, Any]:
    resolved_dataset, resolved_task_family = _resolve_dataset_routing_context(constraints, artifact_context)
    constraints = _ensure_role_constraints(
        constraints,
        role,
        dataset=resolved_dataset,
        task_family=resolved_task_family,
    )
    effective_cache_key = cache_key or role
    if reuse_cache and effective_cache_key in cache:
        cached = cache[effective_cache_key]
        return dict(cached)

    candidates, router_candidates = _prepare_role_candidates(
        task_text=task_text,
        role=role,
        constraints=constraints,
        artifact_context=artifact_context,
        registry=registry,
        index=index,
        embedder=embedder,
        reranker=reranker,
        top_n=top_n,
        top_k=top_k,
        rerank_top_m=rerank_top_m,
        mmr_lambda=mmr_lambda,
        use_mmr=use_mmr,
        router_no_rerank=router_no_rerank,
        experimental_reranker_after_mmr=experimental_reranker_after_mmr,
        selected_agent_ids=selected_agent_ids,
        team_lambda_compat=team_lambda_compat,
        team_mu_cost=team_mu_cost,
        team_nu_latency=team_nu_latency,
    )

    selected_main = None
    selected_shadows: List[str] = []
    selected_shadow = None

    if router_llm_client and router_candidates:
        trimmed = router_candidates[: max(1, router_top_m)]
        selected_main = _select_with_router_llm(
            router_llm_client,
            task_text,
            role,
            constraints,
            trimmed,
            registry,
        )
        if selected_main:
            for cand in trimmed:
                cand_id = cand.get("card_id")
                if not cand_id or cand_id == selected_main:
                    continue
                selected_shadows.append(cand_id)
                if len(selected_shadows) >= 2:
                    break
            selected_shadow = selected_shadows[0] if selected_shadows else None

    if selected_main is None:
        probe_result = probe_commit(
            task_text=task_text,
            role=role,
            constraints=constraints,
            candidates=router_candidates,
            top_probe=min(5, len(router_candidates)),
            max_shadows=2,
        )
        selected_main = probe_result.get("selected_main")
        selected_shadows = probe_result.get("selected_shadows", [])
        if selected_main is None and router_candidates:
            selected_main = router_candidates[0]["card_id"]
        selected_shadow = selected_shadows[0] if selected_shadows else None

    selected_main = _apply_dataset_selection_tiebreak(
        selected_main,
        router_candidates,
        dataset=resolved_dataset,
        task_family=resolved_task_family,
        role=role,
        margin=float((constraints or {}).get("_same_dataset_margin") or _DATASET_ROUTING_MARGIN),
    )
    candidate_by_id = {
        str(item.get("card_id") or "").strip(): item
        for item in router_candidates
        if isinstance(item, dict) and str(item.get("card_id") or "").strip()
    }
    selected_candidate = candidate_by_id.get(str(selected_main or "").strip())
    selected_cross_dataset_fallback = bool(selected_candidate.get("cross_dataset_fallback")) if selected_candidate else False
    if selected_main:
        selected_shadows = [
            str(item.get("card_id") or "").strip()
            for item in router_candidates
            if isinstance(item, dict)
            and str(item.get("card_id") or "").strip()
            and str(item.get("card_id") or "").strip() != str(selected_main).strip()
        ][:2]
        selected_shadow = selected_shadows[0] if selected_shadows else None

    selection = {
        "selected_main": selected_main,
        "selected_shadows": selected_shadows,
        "selected_shadow": selected_shadow,
        "selected_cross_dataset_fallback": selected_cross_dataset_fallback,
        "candidates": candidates,
        "reranked": router_candidates,
        "base_reranked": list(router_candidates),
    }
    cache[effective_cache_key] = dict(selection)
    return selection


def _execute_role_with_selection(
    *,
    role: str,
    task_text: str,
    selection: Dict[str, Any],
    registry,
    llm: Any,
    tool_executor: Optional[ToolExecutor],
    max_attempts: int,
    results: Dict[str, Any],
    constraints: Optional[Dict[str, Any]],
    allow_unknown_roles: bool,
    runs_dir: str,
    workflow_version: str,
    task_id: str,
    log_task_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    tool_max_rounds: int = 10,
    tool_only: bool = False,
    enable_postprocess_repair: bool = True,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_main = selection.get("selected_main")
    selected_shadow = selection.get("selected_shadow")
    reranked = selection.get("reranked") or []
    event_task_id = str(log_task_id or task_id)
    
    if not selected_main:
        print(f"[ERROR] No agent selected for role '{role}'! selection={selection}", file=sys.stderr)

    attempt = 0
    role_key = role.strip().lower()
    role_context: Dict[str, Any] = {"upstream": results, "constraints": constraints}
    agent_context = _get_agent_context(registry, selected_main)
    role_context["agent"] = agent_context
    last_output_obj: Any = None
    last_validation_errors: List[str] = []
    last_failure_type: Optional[str] = None
    last_action: Optional[str] = None
    if extra_context:
        role_context.update(extra_context)
    artifact_bag = _initialize_artifact_bag(
        task_text=task_text,
        task_context=task_context,
        extra_context=extra_context,
    )
    artifact_bag["cross_dataset_fallback"] = bool(selection.get("selected_cross_dataset_fallback"))
    role_context["artifact_bag"] = artifact_bag
    prefetched_output: Optional[Dict[str, Any]] = None
    if _is_benchmark_task_family(artifact_bag.get("task_family")) and role_key in {"planner", "builder", "checker", "refactor"}:
        prefetched_output = llm.generate(role, task_text, role_context)
        _merge_role_output_into_artifact_bag(artifact_bag, role, prefetched_output)
        role_context["artifact_bag"] = artifact_bag

    tool_history: List[Dict[str, Any]] = []
    tools_invoked = False

    def _return_tool_output(*, action_label: str, failure_type: Optional[str]) -> Dict[str, Any]:
        tool_output = _artifact_bag_to_role_output(role, artifact_bag, tool_history)
        if isinstance(tool_output, dict) and tool_output.get("error"):
            validation = {"ok": False, "errors": [str(tool_output.get("error"))]}
            stored_output = tool_output
        else:
            valid, errors, model = validate_output(role, tool_output, allow_unknown=allow_unknown_roles)
            validation = {"ok": valid, "errors": errors}
            if valid and model is not None:
                stored_output = model.model_dump()
            else:
                stored_output = tool_output
        if action_label == "tool_test_passed" and isinstance(stored_output, dict):
            stored_output["test_passed"] = True
        log_path = write_event(
            task_id=event_task_id,
            workflow_version=workflow_version,
            role=role,
            selected_main=selected_main,
            selected_shadow=selected_shadow,
            candidates_topk=reranked,
            output=tool_output,
            validation=validation,
            executor_result=tool_history or None,
            failure_type=failure_type,
            action=action_label,
            meta=meta,
            runs_dir=runs_dir,
        )
        return {
            "output": stored_output,
            "raw_output": tool_output,
            "executor_result": tool_history or None,
            "artifact_bag": dict(artifact_bag),
            "log_path": log_path,
            "selected_main": selected_main,
            "selected_shadows": selection.get("selected_shadows", []),
        }

    if tool_only:
        tool_history = _plan_and_run_tools(
            llm=llm,
            role=role,
            task_text=task_text,
            role_context=role_context,
            agent_context=agent_context,
            registry=registry,
            tool_executor=tool_executor,
            results=results,
            tool_max_rounds=tool_max_rounds,
            tool_only=tool_only,
            enable_postprocess_repair=enable_postprocess_repair,
            task_context=task_context,
        )
        if bool(role_context.get("test_passed")):
            return _return_tool_output(action_label="tool_test_passed", failure_type=None)
        return _return_tool_output(action_label="tool_only", failure_type="tool_only")

    while attempt < max_attempts:
        llm_task_text = _append_diagnostics_to_task(task_text, role_context)
        if attempt == 0 and prefetched_output is not None:
            output = prefetched_output
        else:
            output = llm.generate(role, llm_task_text, role_context)
        _merge_role_output_into_artifact_bag(artifact_bag, role, output)
        role_context["artifact_bag"] = artifact_bag
        last_output_obj = output
        valid, errors, model = validate_output(role, output, allow_unknown=allow_unknown_roles)
        last_validation_errors = list(errors or [])
        validation = {"ok": valid, "errors": errors}
        executor_result = tool_history or None
        failure_type = None
        action = None
        last_failure_type = None
        last_action = None

        if (
            not tools_invoked
            and tool_executor is not None
            and _should_invoke_tools_on_demand(
                role=role,
                role_context=role_context,
                output=output,
                task_context=task_context,
            )
        ):
            tool_history = _plan_and_run_tools(
                llm=llm,
                role=role,
                task_text=task_text,
                role_context=role_context,
                agent_context=agent_context,
                registry=registry,
                tool_executor=tool_executor,
                results=results,
                tool_max_rounds=tool_max_rounds,
                tool_only=False,
                enable_postprocess_repair=enable_postprocess_repair,
                task_context=task_context,
            )
            tools_invoked = True
            executor_result = tool_history or None

            if bool(role_context.get("test_passed")):
                return _return_tool_output(action_label="tool_test_passed", failure_type=None)

            if _should_prefer_tool_augmented_output(
                role=role,
                role_context=role_context,
                artifact_bag=artifact_bag,
                tool_history=tool_history,
            ):
                tool_output = _artifact_bag_to_role_output(role, artifact_bag, tool_history)
                if not (isinstance(tool_output, dict) and tool_output.get("error")):
                    tool_valid, tool_errors, _tool_model = validate_output(
                        role,
                        tool_output,
                        allow_unknown=allow_unknown_roles,
                    )
                    if tool_valid:
                        return _return_tool_output(
                            action_label="tool_augmented",
                            failure_type="tool_on_demand",
                        )

        if valid:
            if model is not None:
                stored_output = model.model_dump()
            elif isinstance(output, dict):
                stored_output = output
            else:
                stored_output = {"raw": output}

            log_path = write_event(
                task_id=event_task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            return {
                "output": stored_output,
                "raw_output": output,
                "executor_result": executor_result,
                "artifact_bag": dict(artifact_bag),
                "log_path": log_path,
                "selected_main": selected_main,
                "selected_shadows": selection.get("selected_shadows", []),
            }

        route = route_failure(
            role,
            output,
            errors,
            executor_result if isinstance(executor_result, dict) else None,
        )
        failure_type = route["failure_type"]
        action = route["action"]
        last_failure_type = failure_type
        last_action = action

        if failure_type == "A_contract":
            repaired = llm.rewrite_format(role, output, role_context)
            last_output_obj = repaired
            valid_fix, errors_fix, model_fix = validate_output(role, repaired, allow_unknown=allow_unknown_roles)
            last_validation_errors = list(errors_fix or [])
            validation = {"ok": valid_fix, "errors": errors_fix}
            if valid_fix:
                stored_output = model_fix.model_dump() if model_fix is not None else repaired
                log_path = write_event(
                    task_id=event_task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=repaired,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    meta=meta,
                    runs_dir=runs_dir,
                )
                return {
                    "output": stored_output,
                    "raw_output": repaired,
                    "executor_result": executor_result,
                    "artifact_bag": dict(artifact_bag),
                    "log_path": log_path,
                    "selected_main": selected_main,
                    "selected_shadows": selection.get("selected_shadows", []),
                }
            log_path = write_event(
                task_id=event_task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=repaired,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            attempt += 1
            continue

        if failure_type == "B_missing_info":
            role_context.update(llm.clarify(role, task_text))
            log_path = write_event(
                task_id=event_task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            attempt += 1
            continue

        if failure_type == "C_capability":
            log_path = write_event(
                task_id=event_task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            if selected_shadow and selected_shadow != selected_main:
                selected_main = selected_shadow
                selected_shadow = None
                agent_context = _get_agent_context(registry, selected_main)
                role_context["agent"] = agent_context
                attempt += 1
                continue
            break

        attempt += 1

    role_norm = _normalize_role_name(role)
    print(
        f"[DEBUG] Role '{role}' exhausted max_attempts={max_attempts}; returning no_valid_output",
        file=sys.stderr,
    )
    if last_failure_type or last_action:
        print(
            f"[DEBUG] Last failure route: failure_type={last_failure_type}, action={last_action}",
            file=sys.stderr,
        )
    if last_validation_errors:
        print(
            f"[DEBUG] Last validation errors: {last_validation_errors}",
            file=sys.stderr,
        )
    if last_output_obj is not None:
        try:
            preview = _truncate_text(
                json.dumps(last_output_obj, ensure_ascii=True, default=str),
                max_chars=700,
            )
        except Exception:
            preview = _truncate_text(str(last_output_obj), max_chars=700)
        print(f"[DEBUG] Last raw output preview: {preview}", file=sys.stderr)
    if role_norm in {"planner", "manager"} and isinstance(last_output_obj, dict):
        steps_obj = last_output_obj.get("steps") or last_output_obj.get("plan")
        has_steps = isinstance(steps_obj, list) and len(steps_obj) > 0
        has_acceptance = bool(last_output_obj.get("acceptance_criteria"))
        print(
            "[DEBUG] Planner/Manager output diagnostics: "
            f"has_steps={has_steps}, has_acceptance_criteria={has_acceptance}, "
            f"keys={list(last_output_obj.keys())}",
            file=sys.stderr,
        )

    if tool_history:
        print(
            f"[DEBUG] Role '{role}' LLM failed; attempting tool_history fallback with {len(tool_history)} entries",
            file=sys.stderr,
        )
        fallback_output = _tool_history_to_output(role, tool_history)
        if _is_benchmark_task_family(artifact_bag.get("task_family")):
            fallback_output = _artifact_bag_to_role_output(role, artifact_bag, tool_history)
        if isinstance(fallback_output, dict) and not fallback_output.get("error"):
            fb_valid, fb_errors, fb_model = validate_output(role, fallback_output, allow_unknown=allow_unknown_roles)
            if fb_valid:
                stored_fb = fb_model.model_dump() if fb_model is not None else fallback_output
                print(
                    f"[DEBUG] Role '{role}' tool_history fallback succeeded",
                    file=sys.stderr,
                )
                log_path = write_event(
                    task_id=event_task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=fallback_output,
                    validation={"ok": True, "errors": []},
                    executor_result=tool_history,
                    failure_type="tool_history_fallback",
                    action="tool_history_fallback",
                    meta=meta,
                    runs_dir=runs_dir,
                )
                return {
                    "output": stored_fb,
                    "raw_output": fallback_output,
                    "executor_result": tool_history,
                    "artifact_bag": dict(artifact_bag),
                    "log_path": log_path,
                    "selected_main": selected_main,
                    "selected_shadows": selection.get("selected_shadows", []),
                }
            else:
                print(
                    f"[DEBUG] Role '{role}' tool_history fallback also failed validation: {fb_errors}",
                    file=sys.stderr,
                )

    return {
        "output": {"error": "no_valid_output"},
        "raw_output": last_output_obj,
        "executor_result": tool_history or None,
        "artifact_bag": dict(artifact_bag),
        "log_path": None,
    }


def _heuristic_topology(task_text: str) -> TopologyType:
    text = task_text.lower()
    if any(token in text for token in ["brainstorm", "multi-angle", "debate", "discussion", "对比", "多角度", "讨论"]):
        return TopologyType.DECENTRALIZED
    if any(token in text for token in ["then", "first", "next", "after", "并且", "然后", "最后", "步骤", "流程"]):
        return TopologyType.CENTRALIZED
    # Check for code generation tasks
    if any(token in text for token in ["def ", "function", "implement", "write code", "算法", "函数", "实现", "代码"]):
        return TopologyType.CENTRALIZED
    return TopologyType.SINGLE


def _plan_topology(
    *,
    task_text: str,
    roles: List[str],
    topology: Optional[str],
    topology_config: Optional[Dict[str, Any]],
    meta_router_llm_client: Optional[LLMClient],
    max_steps: int,
    strict_roles: bool = False,
) -> TopologyConfig:
    if topology_config:
        return TopologyConfig(**topology_config).normalized(default_roles=roles)

    topology_value = (topology or "").strip().lower()
    if topology_value and topology_value not in {"auto", "dynamic"}:
        try:
            topo = TopologyType(topology_value)
        except ValueError:
            topo = _heuristic_topology(task_text)
        return TopologyConfig(topology=topo, roles=roles, max_steps=max_steps).normalized(default_roles=roles)

    if meta_router_llm_client:
        system_msg = (
            "You are a meta-router. Decide the best agent topology for the task. "
            "Return ONLY JSON with keys: topology, roles, manager_role, entry_role, max_steps, flow_type."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Available roles: {json.dumps(roles, ensure_ascii=True)}\n"
            "Topology options: single, centralized, decentralized, chain.\n"
            "Use 'single' for straightforward one-function implementations where the shown asserts are enough for direct coding.\n"
            "Use 'centralized' when algorithm design, edge cases, or verification are genuinely needed.\n"
            "Do not include 'refactor' in the initial roles unless the task explicitly asks for cleanup/optimization or a checker has surfaced a concrete failure to repair.\n"
        )
        response = meta_router_llm_client.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        try:
            os.makedirs("runs", exist_ok=True)
            debug_path = os.path.join("runs", "meta_router_raw.jsonl")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "task_text": task_text,
                            "available_roles": roles,
                            "raw_response": response,
                        },
                        ensure_ascii=True,
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass
        data = _safe_load_json(str(response), {})
        if isinstance(data, dict) and data:
            # Ensure max_steps is a valid integer (not None or missing)
            if "max_steps" not in data or data["max_steps"] is None:
                data["max_steps"] = max_steps
            else:
                try:
                    llm_steps = int(data["max_steps"])
                except Exception:
                    llm_steps = max_steps
                # Never allow meta-router to exceed CLI max_steps budget.
                llm_steps = max(1, min(llm_steps, int(max_steps)))
                data["max_steps"] = llm_steps
            
            # Trust LoRA role selection, but robustly normalize/validate role formats.
            # Accept roles as list, string, or {"manager": ..., "workers": [...]}.
            available_role_map: Dict[str, str] = {}
            for raw_role in roles:
                key = _normalize_role_name(raw_role)
                if key and key not in available_role_map:
                    available_role_map[key] = str(raw_role)

            raw_roles = data.get("roles")
            llm_roles_raw: List[Any] = []
            if isinstance(raw_roles, dict):
                mgr = raw_roles.get("manager")
                workers = raw_roles.get("workers")
                if mgr is not None:
                    llm_roles_raw.append(mgr)
                if isinstance(workers, list):
                    llm_roles_raw.extend(workers)
            elif isinstance(raw_roles, list):
                llm_roles_raw = list(raw_roles)
            elif isinstance(raw_roles, str):
                llm_roles_raw = [raw_roles]

            valid_roles: List[str] = []
            seen_roles: set[str] = set()
            for item in llm_roles_raw:
                key = _normalize_role_name(str(item) if item is not None else "")
                mapped = available_role_map.get(key)
                if not mapped or mapped in seen_roles:
                    continue
                seen_roles.add(mapped)
                valid_roles.append(mapped)

            data["roles"] = valid_roles if valid_roles else roles

            # Normalize optional role fields from LLM output.
            if "entry_role" in data:
                entry_norm = _normalize_role_name(str(data.get("entry_role") or ""))
                mapped_entry = available_role_map.get(entry_norm)
                if mapped_entry:
                    data["entry_role"] = mapped_entry
                else:
                    data.pop("entry_role", None)
            if "manager_role" in data:
                manager_norm = _normalize_role_name(str(data.get("manager_role") or ""))
                mapped_manager = available_role_map.get(manager_norm)
                if mapped_manager:
                    data["manager_role"] = mapped_manager
                else:
                    data.pop("manager_role", None)

            # If strict_roles=True and LLM set entry_role to something not in selected roles, clear it.
            if strict_roles and "entry_role" in data and data["entry_role"] not in data["roles"]:
                del data["entry_role"]  # Let normalized() pick from roles list
            return TopologyConfig(**data).normalized(default_roles=roles)

    return TopologyConfig(topology=_heuristic_topology(task_text), roles=roles, max_steps=max_steps).normalized(
        default_roles=roles
    )


def _select_next_role(
    *,
    task_text: str,
    available_roles: List[str],
    history: List[Dict[str, Any]],
    llm: Optional[LLMClient],
    avoid_role: Optional[str] = None,
    force_role: Optional[str] = None,
    prevent_early_finish: bool = True,
) -> Optional[str]:
    if not available_roles:
        return None
    # If force_role is specified, skip LLM and directly return it
    if force_role:
        if force_role in available_roles:
            print(f"[DEBUG] Force using role: {force_role}", file=sys.stderr)
            return force_role
        else:
            print(f"[WARNING] force_role '{force_role}' not in available_roles {available_roles}, falling back", file=sys.stderr)
    if history and any(item.get("test_passed") or item.get("status") == "success" for item in history):
        print("[DEBUG] Task already completed, stopping role selection", file=sys.stderr)
        return None
    if llm:
        recent = history[-3:] if history else []
        recent_summary = []
        for item in recent:
            if not item.get("summary"):
                continue
            row: Dict[str, Any] = {"role": item.get("role"), "summary": item.get("summary")}
            if "planner_ready_to_handoff" in item:
                row["planner_ready_to_handoff"] = bool(item.get("planner_ready_to_handoff"))
            if item.get("planner_next_role"):
                row["planner_next_role"] = item.get("planner_next_role")
            recent_summary.append(row)
        # Check if task is already completed (test passed in recent history)
        task_completed = any(
            item.get("test_passed") or item.get("status") == "success" 
            for item in recent
        )
        system_msg = (
            "You are a dynamic router. Choose the next role from the list to continue working on the task. "
            "ONLY respond with 'finish' if the task is already completed successfully. "
            "If the task is not completed, you MUST choose a role from the available list to try a different approach. "
            "Return ONLY JSON: {\"next_role\": \"...\"}."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Task completed: {task_completed}\n"
            f"Available roles: {json.dumps(available_roles, ensure_ascii=True)}\n"
            f"Recent history: {json.dumps(recent_summary, ensure_ascii=True)}\n"
        )
        response = llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        candidate = data.get("next_role")
        print(f"[DEBUG] LLM returned next_role='{candidate}' (available: {available_roles})", file=sys.stderr)
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate.lower() in ("finish", "none"):
                if task_completed or not prevent_early_finish:
                    return None
                print(
                    "[WARNING] Router requested early finish before success; ignoring and continuing",
                    file=sys.stderr,
                )
                candidate = ""
            if candidate in available_roles:
                if avoid_role and candidate == avoid_role and len(available_roles) > 1:
                    for role in available_roles:
                        if role != avoid_role:
                            return role
                return candidate
            else:
                # LLM returned invalid role, warn and fall back
                print(f"[WARNING] LLM returned invalid role '{candidate}', not in {available_roles}. Using fallback.", file=sys.stderr)

    text = task_text.lower()
    role_priority = [
        ("evidence_finder", ["evidence", "context", "passage", "find", "查找", "证据", "上下文", "资料"]),
        ("planner", ["plan", "planning", "拆解", "规划", "路线"]),
        ("builder", ["build", "implement", "code", "开发", "实现", "写代码", "生成"]),
        ("refactor", ["refactor", "refractor", "polish", "cleanup", "重构", "优化代码"]),
        ("checker", ["test", "verify", "check", "验证", "检查", "评估"]),
    ]
    for role_name, keywords in role_priority:
        if role_name in available_roles and any(token in text for token in keywords):
            if avoid_role and role_name == avoid_role and len(available_roles) > 1:
                continue
            return role_name
    for role in available_roles:
        if role != avoid_role:
            return role
    return available_roles[0]


def _fallback_next_role(available_roles: List[str], avoid_role: Optional[str] = None) -> Optional[str]:
    if not available_roles:
        return None
    preferred = ("checker", "builder")
    for wanted in preferred:
        for role in available_roles:
            if _normalize_role_name(role) != wanted:
                continue
            if avoid_role and role == avoid_role and len(available_roles) > 1:
                continue
            return role
    for role in available_roles:
        if role != avoid_role:
            return role
    return available_roles[0]


def _build_route_query(
    *,
    task_text: str,
    history: List[Dict[str, Any]],
    last_output: Optional[Any],
    llm: Optional[LLMClient],
    fallback_max_chars: int,
) -> str:
    summary = _summarize_output(last_output, max_chars=fallback_max_chars)
    if llm:
        recent = history[-3:] if history else []
        system_msg = (
            "You are a router planner. Produce a short capability-oriented query for the next agent. "
            "Return ONLY JSON: {\"route_query\": \"...\"}."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Last output summary: {summary}\n"
            f"Recent history: {json.dumps(recent, ensure_ascii=True)}\n"
        )
        response = llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        candidate = data.get("route_query")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return summary or task_text


def _neighbors_for_role(role: str, config: TopologyConfig) -> List[str]:
    if role in config.routing_table:
        return list(config.routing_table.get(role) or [])
    if config.edges:
        return [dst for src, dst in config.edges if src == role]
    return []


def _checker_ok(results: Dict[str, Any]) -> bool:
    checker_output = results.get("checker")
    if checker_output is None:
        return True
    if isinstance(checker_output, list):
        return all(not (isinstance(item, dict) and item.get("error")) for item in checker_output)
    if isinstance(checker_output, dict) and checker_output.get("error"):
        return False
    return True


def _resolve_workflow_dataset(
    *,
    dataset: Optional[str],
    task_context: Optional[Dict[str, Any]],
) -> str:
    if dataset:
        return _normalize_dataset_name(dataset)
    if isinstance(task_context, dict):
        for key in ("dataset", "dataset_name"):
            value = task_context.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_dataset_name(value)
    return "unknown"


def _apply_paper_mode_defaults(
    *,
    paper_mode: bool,
    task_family: str,
    top_n: int,
    top_k: int,
    topology: Optional[str],
    dynamic_topology: bool,
    mcts_dynamic_optimization: bool,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    team_mu_cost: float,
    team_nu_latency: float,
) -> Dict[str, Any]:
    if not paper_mode:
        return {
            "top_n": top_n,
            "top_k": top_k,
            "topology": topology,
            "dynamic_topology": dynamic_topology,
            "mcts_dynamic_optimization": mcts_dynamic_optimization,
            "mcts_iterations": mcts_iterations,
            "mcts_rollout_depth": mcts_rollout_depth,
            "mcts_exploration": mcts_exploration,
            "mcts_discount": mcts_discount,
            "mcts_max_candidates": mcts_max_candidates,
            "team_mu_cost": team_mu_cost,
            "team_nu_latency": team_nu_latency,
        }
    benchmark_mode = _is_benchmark_task_family(task_family)
    return {
        "top_n": 30 if benchmark_mode else top_n,
        "top_k": 15 if benchmark_mode else top_k,
        "topology": "auto" if benchmark_mode else (topology or "auto"),
        "dynamic_topology": True if benchmark_mode else True,
        "mcts_dynamic_optimization": True if benchmark_mode else True,
        "mcts_iterations": 64,
        "mcts_rollout_depth": 4,
        "mcts_exploration": 1.414,
        "mcts_discount": 0.95,
        "mcts_max_candidates": 5 if benchmark_mode else max(1, mcts_max_candidates),
        "team_mu_cost": 0.05,
        "team_nu_latency": 0.05,
    }


def _default_parallel_branch_roles(
    *,
    roles: List[str],
    parallel_width: int,
    task_family: str,
    dataset: str,
    explicit_roles: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    if explicit_roles:
        branch_roles = list(explicit_roles)
        hints = [f"independent {role} branch" for role in branch_roles]
    else:
        dataset_key = _normalize_dataset_name(dataset)
        task_key = str(task_family or "").strip().lower()
        if task_key == "code" or dataset_key in {"humaneval", "mbpp"}:
            branch_roles = ["builder", "builder", "refactor", "checker"]
            hints = [
                "precise minimal implementation",
                "alternative algorithmic implementation",
                "repair/refinement implementation",
                "edge-case verification and candidate validation",
            ]
        elif task_key in {"gsm8k", "math"} or dataset_key in {"gsm8k", "math"}:
            branch_roles = ["planner", "builder", "checker", "refactor"]
            hints = [
                "extract quantities and relations",
                "solve directly",
                "verify independently",
                "repair or reformat candidate reasoning",
            ]
        elif task_key in {"qa", "hotpotqa"} or dataset_key == "hotpotqa":
            branch_roles = ["planner", "builder", "builder", "checker"]
            hints = [
                "decompose question and identify bridge/comparison structure",
                "answer from direct evidence chain",
                "answer through alternative entity linking path",
                "verify answer against supporting facts and evidence coverage",
            ]
        else:
            branch_roles = list(roles[:parallel_width] or ["planner", "evidence_finder", "builder", "checker"])
            hints = [f"independent {role} branch" for role in branch_roles]

    width = max(1, int(parallel_width or 1))
    if len(branch_roles) < width:
        base = list(branch_roles or roles or ["builder"])
        idx = 0
        while len(branch_roles) < width:
            branch_roles.append(base[idx % len(base)])
            hints.append(f"independent {branch_roles[-1]} branch")
            idx += 1
    return branch_roles[:width], hints[:width]


def _parallel_branch_task_text(task_text: str, diversity_hint: str) -> str:
    return (
        f"{task_text}\n\n"
        f"Parallel branch objective: {diversity_hint}\n"
        "Return an independent candidate. Do not assume other branches are correct."
    )


def _dedupe_parallel_selection(selection: Dict[str, Any], selected_agent_ids: List[str]) -> Dict[str, Any]:
    selected_main = selection.get("selected_main")
    if selected_main and selected_main not in selected_agent_ids:
        return selection
    for candidate in selection.get("reranked") or selection.get("candidates") or []:
        cand_id = candidate.get("card_id") if isinstance(candidate, dict) else None
        if cand_id and cand_id not in selected_agent_ids:
            updated = dict(selection)
            updated["selected_main"] = cand_id
            shadows = [item.get("card_id") for item in (selection.get("reranked") or []) if isinstance(item, dict)]
            updated["selected_shadows"] = [item for item in shadows if item and item != cand_id][:2]
            updated["selected_shadow"] = updated["selected_shadows"][0] if updated["selected_shadows"] else None
            return updated
    return selection


def _selection_from_parallel_candidates(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    registry,
    reranker: TfidfLinearReranker,
    rerank_top_m: int,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool,
    selected_agent_ids: List[str],
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
) -> Dict[str, Any]:
    dataset, task_family = _resolve_dataset_routing_context(constraints, None)
    if candidates:
        if router_no_rerank or not experimental_reranker_after_mmr:
            reranked = list(candidates)
        else:
            candidate_texts = _build_candidate_texts(registry, candidates)
            query_text = sanitize_search_text(build_role_query(task_text, role, constraints))
            rerank_indices, _ = reranker.rank(query_text, candidate_texts, top_m=min(rerank_top_m, len(candidates)))
            reranked = [candidates[i] for i in rerank_indices]
    else:
        reranked = []

    _ = selected_agent_ids  # retained for backward-compatible public signatures.
    _ = team_lambda_compat  # retained for backward-compatible public signatures.
    router_candidates = _apply_unary_raw_scoring(
        task_text=task_text,
        role=role,
        constraints=constraints,
        candidates=reranked if reranked else candidates,
        registry=registry,
        mu_cost=team_mu_cost,
        nu_latency=team_nu_latency,
    )
    router_candidates = _annotate_dataset_candidates(
        list(router_candidates),
        dataset=dataset,
        task_family=task_family,
        role=role,
    )
    selected_main = None
    selected_shadows: List[str] = []
    selected_shadow = None

    if router_llm_client and router_candidates:
        trimmed = router_candidates[: max(1, router_top_m)]
        selected_main = _select_with_router_llm(
            router_llm_client,
            task_text,
            role,
            constraints,
            trimmed,
            registry,
        )
        if selected_main:
            for cand in trimmed:
                cand_id = cand.get("card_id")
                if not cand_id or cand_id == selected_main:
                    continue
                selected_shadows.append(cand_id)
                if len(selected_shadows) >= 2:
                    break
            selected_shadow = selected_shadows[0] if selected_shadows else None

    if selected_main is None:
        probe_result = probe_commit(
            task_text=task_text,
            role=role,
            constraints=constraints,
            candidates=router_candidates,
            top_probe=min(5, len(router_candidates)),
            max_shadows=2,
        )
        selected_main = probe_result.get("selected_main")
        selected_shadows = probe_result.get("selected_shadows", [])
        if selected_main is None and router_candidates:
            selected_main = router_candidates[0]["card_id"]
        selected_shadow = selected_shadows[0] if selected_shadows else None

    selected_main = _apply_dataset_selection_tiebreak(
        selected_main,
        router_candidates,
        dataset=dataset,
        task_family=task_family,
        role=role,
        margin=float((constraints or {}).get("_same_dataset_margin") or _DATASET_ROUTING_MARGIN),
    )
    candidate_by_id = {
        str(item.get("card_id") or "").strip(): item
        for item in router_candidates
        if isinstance(item, dict) and str(item.get("card_id") or "").strip()
    }
    selected_candidate = candidate_by_id.get(str(selected_main or "").strip())
    selected_cross_dataset_fallback = bool(selected_candidate.get("cross_dataset_fallback")) if selected_candidate else False
    if selected_main:
        selected_shadows = [
            str(item.get("card_id") or "").strip()
            for item in router_candidates
            if isinstance(item, dict)
            and str(item.get("card_id") or "").strip()
            and str(item.get("card_id") or "").strip() != str(selected_main).strip()
        ][:2]
        selected_shadow = selected_shadows[0] if selected_shadows else None

    return {
        "selected_main": selected_main,
        "selected_shadows": selected_shadows,
        "selected_shadow": selected_shadow,
        "selected_cross_dataset_fallback": selected_cross_dataset_fallback,
        "candidates": candidates,
        "reranked": router_candidates,
        "base_reranked": list(router_candidates),
    }


def _run_parallel_workflow(
    *,
    task_text: str,
    role_list: List[str],
    constraints_per_role: Dict[str, Dict[str, Any]],
    workflow_version: str,
    registry,
    index,
    embedder,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    use_mmr: bool,
    max_attempts: int,
    tool_max_rounds: int,
    reranker: TfidfLinearReranker,
    execute_tools: bool,
    tool_timeout_s: float,
    auto_fill_tool_inputs: bool,
    tool_llm: Optional[Any],
    auto_install_common_libs: bool,
    auto_install_timeout_s: float,
    auto_install_user: bool,
    llm: Any,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool,
    runs_dir: str,
    task_id: str,
    log_task_id: Optional[str],
    task_context: Optional[Dict[str, Any]],
    task_family: str,
    dataset: str,
    artifact_contract_version: str,
    paper_mode: bool,
    allow_unknown_roles: bool,
    enable_postprocess_repair: bool,
    tool_only: bool,
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
    parallel_width: int,
    parallel_branch_roles: Optional[List[str]],
    parallel_consensus: str,
    parallel_branch_timeout_s: Optional[float],
    parallel_fail_fast: bool,
    parallel_search: bool,
) -> Dict[str, Any]:
    event_task_id = str(log_task_id or task_id)
    branch_roles, hints = _default_parallel_branch_roles(
        roles=role_list,
        parallel_width=parallel_width,
        task_family=task_family,
        dataset=dataset,
        explicit_roles=parallel_branch_roles,
    )
    group_id = f"{task_id}_parallel"
    selected_agent_ids: List[str] = []
    specs: List[ParallelBranchSpec] = []
    selections: List[Dict[str, Any]] = []
    selection_cache: Dict[str, Dict[str, Any]] = {}
    branch_plans: List[Dict[str, Any]] = []

    for idx, role in enumerate(branch_roles):
        hint = hints[idx] if idx < len(hints) else f"independent {role} branch"
        branch_id = f"{_normalize_role_name(role) or role}-{idx}"
        branch_task = _parallel_branch_task_text(task_text, hint)
        constraints = _ensure_role_constraints(
            constraints_per_role.get(role),
            role,
            dataset=dataset,
            task_family=task_family,
        )
        branch_plans.append(
            {
                "idx": idx,
                "role": role,
                "hint": hint,
                "branch_id": branch_id,
                "branch_task": branch_task,
                "constraints": constraints,
            }
        )

    batched_candidates: Dict[str, List[Dict[str, Any]]] = {}
    if parallel_search:
        batched_candidates = get_candidates_batch(
            [
                ParallelCandidateRequest(
                    branch_id=plan["branch_id"],
                    task_text=plan["branch_task"],
                    role=plan["role"],
                    constraints=plan["constraints"],
                    artifact_context=task_context,
                    kind="agent",
                    top_n=top_n,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                    use_mmr=use_mmr,
                )
                for plan in branch_plans
            ],
            registry=registry,
            index=index,
            embedder=embedder,
        )

    for plan in branch_plans:
        role = plan["role"]
        hint = plan["hint"]
        branch_id = plan["branch_id"]
        branch_task = plan["branch_task"]
        constraints = plan["constraints"]
        if (
            parallel_search
            and branch_id in batched_candidates
            and len(batched_candidates.get(branch_id) or [])
            >= int((constraints or {}).get("_same_dataset_min_candidates") or _DATASET_ROUTING_MIN_CANDIDATES)
        ):
            selection = _selection_from_parallel_candidates(
                task_text=branch_task,
                role=role,
                constraints=constraints,
                candidates=batched_candidates.get(branch_id) or [],
                registry=registry,
                reranker=reranker,
                rerank_top_m=rerank_top_m,
                router_llm_client=router_llm_client,
                router_top_m=router_top_m,
                router_no_rerank=router_no_rerank,
                experimental_reranker_after_mmr=experimental_reranker_after_mmr,
                selected_agent_ids=selected_agent_ids,
                team_lambda_compat=team_lambda_compat,
                team_mu_cost=team_mu_cost,
                team_nu_latency=team_nu_latency,
            )
        else:
            selection = _select_agent_for_role(
                task_text=branch_task,
                role=role,
                constraints=constraints,
                registry=registry,
                index=index,
                embedder=embedder,
                reranker=reranker,
                top_n=top_n,
                top_k=top_k,
                rerank_top_m=rerank_top_m,
                mmr_lambda=mmr_lambda,
                use_mmr=use_mmr,
                router_llm_client=router_llm_client,
                router_top_m=router_top_m,
                router_no_rerank=router_no_rerank,
                experimental_reranker_after_mmr=experimental_reranker_after_mmr,
                reuse_cache=False,
                cache=selection_cache,
                cache_key=f"parallel|{branch_id}",
                artifact_context=task_context,
                selected_agent_ids=selected_agent_ids,
                team_lambda_compat=team_lambda_compat,
                team_mu_cost=team_mu_cost,
                team_nu_latency=team_nu_latency,
            )
        selection = _dedupe_parallel_selection(selection, selected_agent_ids)
        selected_main = selection.get("selected_main")
        if selected_main:
            selected_agent_ids.append(str(selected_main))
        selections.append(
            {
                "role": role,
                "branch_id": branch_id,
                "selected_main": selected_main,
                "selected_shadows": selection.get("selected_shadows", []),
            }
        )
        specs.append(
            ParallelBranchSpec(
                branch_id=branch_id,
                role=role,
                task_text=branch_task,
                constraints=constraints,
                selection=selection,
                reason="parallel_branch",
                diversity_hint=hint,
            )
        )

    def _run_branch(spec: ParallelBranchSpec) -> ParallelBranchResult:
        started = datetime.utcnow()
        started_ts = started.timestamp()
        with SQLiteRegistry(registry.db_path) as branch_registry:
            branch_tool_executor = (
                ToolExecutor(
                    branch_registry,
                    timeout_s=tool_timeout_s,
                    auto_fill=auto_fill_tool_inputs,
                    llm=tool_llm,
                    auto_install_common_libs=auto_install_common_libs,
                    auto_install_timeout_s=auto_install_timeout_s,
                    auto_install_user=auto_install_user,
                )
                if execute_tools
                else None
            )
            extra_context = {
                "available_roles": role_list,
                "topology": {"topology": "parallel", "roles": branch_roles, "parallel_width": len(branch_roles)},
                "task_family": task_family,
                "dataset": dataset,
                "artifact_contract_version": artifact_contract_version,
                "paper_mode": paper_mode,
                "is_decentralized": False,
                "parallel_group_id": group_id,
                "branch_id": spec.branch_id,
                "branch_role": spec.role,
                "diversity_hint": spec.diversity_hint,
            }
            meta = {
                "step": 0,
                "topology": "parallel",
                "flow_type": "fan_out",
                "reason": spec.reason,
                "parallel_group_id": group_id,
                "branch_id": spec.branch_id,
                "branch_role": spec.role,
                "diversity_hint": spec.diversity_hint,
            }
            exec_result = _execute_role_with_selection(
                role=spec.role,
                task_text=spec.task_text,
                selection=spec.selection,
                registry=branch_registry,
                llm=llm,
                tool_executor=branch_tool_executor,
                max_attempts=max_attempts,
                tool_max_rounds=tool_max_rounds,
                results={},
                constraints=spec.constraints,
                allow_unknown_roles=allow_unknown_roles,
                runs_dir=runs_dir,
                workflow_version=workflow_version,
                task_id=task_id,
                log_task_id=event_task_id,
                meta=meta,
                extra_context=extra_context,
                tool_only=tool_only,
                enable_postprocess_repair=enable_postprocess_repair,
                task_context=task_context,
            )
        output = exec_result.get("output")
        error = None
        if isinstance(output, dict) and output.get("error"):
            error = {"code": str(output.get("error")), "message": str(output.get("error"))}
        elapsed_s = max(0.0, datetime.utcnow().timestamp() - started_ts)
        return ParallelBranchResult(
            branch_id=spec.branch_id,
            role=spec.role,
            selected_main=exec_result.get("selected_main") or spec.selection.get("selected_main"),
            output=output if isinstance(output, dict) else {"raw": output},
            raw_output=exec_result.get("raw_output"),
            executor_result=exec_result.get("executor_result"),
            artifact_bag=exec_result.get("artifact_bag") or {},
            log_path=exec_result.get("log_path"),
            error=error,
            elapsed_s=elapsed_s,
            selection=spec.selection,
        )

    branch_results = run_parallel_branches(
        specs,
        branch_runner=_run_branch,
        parallel_width=parallel_width,
        branch_timeout_s=parallel_branch_timeout_s,
        fail_fast_on_test_pass=parallel_fail_fast,
    )
    aggregation = aggregate_parallel_results(
        branch_results,
        task_family=task_family,
        dataset=dataset,
        consensus_policy=parallel_consensus,
    )
    log_path = write_event(
        task_id=event_task_id,
        workflow_version=workflow_version,
        role="parallel_aggregator",
        selected_main="agent-parallel-consensus-aggregator",
        selected_shadow=None,
        candidates_topk=[],
        output=aggregation,
        validation={"ok": True, "errors": []},
        executor_result=[result.executor_result for result in branch_results],
        failure_type=None,
        action="parallel_aggregate",
        meta={
            "topology": "parallel",
            "parallel_group_id": group_id,
            "branch_count": len(branch_results),
            "winner_branch_id": aggregation.get("winner_branch_id"),
        },
        runs_dir=runs_dir,
    )
    branch_payloads = aggregation.get("branches") or []
    results = {payload.get("branch_id"): payload.get("output") for payload in branch_payloads if payload.get("branch_id")}
    results["parallel_aggregator"] = aggregation
    tool_exec = {
        result.branch_id: result.executor_result
        for result in branch_results
        if result.executor_result is not None
    }
    execution_trace = [
        {
            "step": idx,
            "role": result.role,
            "branch_id": result.branch_id,
            "reason": "parallel",
            "selected_main": result.selected_main,
            "tool_history": result.executor_result,
            "artifact_bag": result.artifact_bag,
            "output": result.output,
            "error": result.error,
        }
        for idx, result in enumerate(branch_results)
    ]
    execution_trace.append(
        {
            "step": len(execution_trace),
            "role": "parallel_aggregator",
            "branch_id": "aggregate",
            "reason": "parallel_aggregate",
            "selected_main": "agent-parallel-consensus-aggregator",
            "output": aggregation,
        }
    )
    checker_ok = not any(
        result.branch_id == aggregation.get("winner_branch_id") and result.error
        for result in branch_results
    )
    return {
        "results": results,
        "tool_exec": tool_exec,
        "execution_trace": execution_trace,
        "log_path": log_path,
        "selections": selections,
        "checker_ok": checker_ok,
        "topology": {
            "topology": "parallel",
            "parallel_mode": "frontier",
            "roles": branch_roles,
            "parallel_width": len(branch_roles),
            "consensus_policy": parallel_consensus,
            "fail_fast_on_test_pass": parallel_fail_fast,
        },
        "parallel": {
            "enabled": True,
            "mode": "frontier",
            "width": len(branch_roles),
            "branches": branch_payloads,
            "winner_branch_id": aggregation.get("winner_branch_id"),
            "consensus": aggregation.get("consensus"),
        },
        "answer": str(aggregation.get("final_answer") or ""),
    }


def _run_dynamic_workflow(
    *,
    task_text: str,
    role_list: List[str],
    constraints_per_role: Dict[str, Dict[str, Any]],
    workflow_version: str,
    registry,
    index,
    embedder,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    use_mmr: bool,
    max_attempts: int,
    tool_max_rounds: int,
    reranker: TfidfLinearReranker,
    execute_tools: bool,
    tool_timeout_s: float,
    auto_fill_tool_inputs: bool,
    tool_llm: Optional[Any],
    auto_install_common_libs: bool,
    auto_install_timeout_s: float,
    auto_install_user: bool,
    llm: Any,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    experimental_reranker_after_mmr: bool,
    runs_dir: str,
    task_id: str,
    log_task_id: Optional[str],
    topology: Optional[str],
    topology_config: Optional[Dict[str, Any]],
    meta_router_llm_client: Optional[LLMClient],
    next_role_llm_client: Optional[LLMClient],
    route_query_llm_client: Optional[LLMClient],
    max_steps: int,
    force_final_builder_extra_step: bool,
    allow_unknown_roles: bool,
    enable_postprocess_repair: bool,
    prevent_early_finish: bool,
    reuse_role_selection: bool,
    reuse_same_role_agent_once: bool,
    summary_max_chars: int,
    soft_connection: bool,
    tool_only: bool,
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
    mcts_dynamic_optimization: bool,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    use_iochain: bool,
    use_transition_term: bool,
    use_gap_term: bool,
    reward_variant: str,
    pair_success_priors: Optional[Dict[Tuple[str, str, str, str], float]],
    role_success_priors: Optional[Dict[Tuple[str, str], float]],
    task_context: Optional[Dict[str, Any]] = None,
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
    artifact_contract_version: str = "v2",
    paper_mode: bool = False,
    force_role: Optional[str] = None,
    strict_roles: bool = False,
) -> Dict[str, Any]:
    event_task_id = str(log_task_id or task_id)
    config = _plan_topology(
        task_text=task_text,
        roles=role_list,
        topology=topology,
        topology_config=topology_config,
        meta_router_llm_client=meta_router_llm_client,
        max_steps=max_steps,
        strict_roles=strict_roles,
    )

    results: Dict[str, Any] = {}
    tool_exec: Dict[str, Any] = {}
    log_path: Optional[str] = None
    selection_log: List[Dict[str, Any]] = []
    selection_cache: Dict[str, Dict[str, Any]] = {}
    role_once_selection_cache: Dict[str, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    execution_trace: List[Dict[str, Any]] = []
    selected_agent_ids: List[str] = []
    selected_agent_roles: List[str] = []
    workflow_succeeded = False
    resolved_dataset = _resolve_workflow_dataset(dataset=dataset, task_context=task_context)
    resolved_task_family = str(task_family or _infer_task_family(resolved_dataset, task_context)).strip().lower()
    forced_final_builder_executed = False
    candidate_final_answer_state: Dict[str, Any] = {}
    current_artifact_bag = _initialize_artifact_bag(
        task_text=task_text,
        task_context=task_context,
        extra_context={
            "task_family": resolved_task_family,
            "dataset": resolved_dataset,
            "artifact_contract_version": artifact_contract_version,
        },
    )

    tool_executor = (
        ToolExecutor(
            registry,
            timeout_s=tool_timeout_s,
            auto_fill=auto_fill_tool_inputs,
            llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
        )
        if execute_tools
        else None
    )

    def _execute_role(role: str, task_for_role: str, step_idx: int, reason: str, override_tool_only: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        nonlocal log_path, current_artifact_bag
        role_constraints = _ensure_role_constraints(
            constraints_per_role.get(role),
            role,
            dataset=resolved_dataset,
            task_family=resolved_task_family,
        )
        effective_cache_key = _build_selection_cache_key(
            role=role,
            task_text=task_for_role,
            step_idx=step_idx,
            topology=config.topology.value,
        )
        role_once_key = _normalize_role_name(role)
        reused_same_role_agent_once = False

        if reuse_same_role_agent_once and role_once_key in role_once_selection_cache:
            selection = dict(role_once_selection_cache[role_once_key])
            reused_same_role_agent_once = True
        elif reuse_role_selection and effective_cache_key in selection_cache:
            selection = dict(selection_cache[effective_cache_key])
        else:
            selection = None
            if mcts_dynamic_optimization:
                selection = _select_agent_with_mcts_dynamic(
                    task_text=task_for_role,
                    role=role,
                    step_idx=step_idx,
                    config=config,
                    constraints_per_role=constraints_per_role,
                    selected_agent_ids=selected_agent_ids,
                    selected_agent_roles=selected_agent_roles,
                    registry=registry,
                    index=index,
                    embedder=embedder,
                    reranker=reranker,
                    top_n=top_n,
                    top_k=top_k,
                    rerank_top_m=rerank_top_m,
                    mmr_lambda=mmr_lambda,
                    use_mmr=use_mmr,
                    router_no_rerank=router_no_rerank,
                    experimental_reranker_after_mmr=experimental_reranker_after_mmr,
                    team_lambda_compat=team_lambda_compat,
                    team_mu_cost=team_mu_cost,
                    team_nu_latency=team_nu_latency,
                    mcts_iterations=mcts_iterations,
                    mcts_rollout_depth=mcts_rollout_depth,
                    mcts_exploration=mcts_exploration,
                    mcts_discount=mcts_discount,
                    mcts_max_candidates=mcts_max_candidates,
                    pair_success_priors=pair_success_priors,
                    role_success_priors=role_success_priors,
                    use_iochain=use_iochain,
                    use_transition_term=use_transition_term,
                    use_gap_term=use_gap_term,
                    reward_variant=reward_variant,
                    history=history,
                    artifact_bag=current_artifact_bag,
                    task_family=resolved_task_family,
                    dataset=resolved_dataset,
                )

            if selection is None:
                selection = _select_agent_for_role(
                    task_text=task_for_role,
                    role=role,
                    constraints=role_constraints,
                    registry=registry,
                    index=index,
                    embedder=embedder,
                    reranker=reranker,
                    top_n=top_n,
                    top_k=top_k,
                    rerank_top_m=rerank_top_m,
                    mmr_lambda=mmr_lambda,
                    use_mmr=use_mmr,
                    router_llm_client=router_llm_client,
                    router_top_m=router_top_m,
                    router_no_rerank=router_no_rerank,
                    experimental_reranker_after_mmr=experimental_reranker_after_mmr,
                    reuse_cache=False,
                    cache=selection_cache,
                    cache_key=effective_cache_key,
                    artifact_context=current_artifact_bag,
                    selected_agent_ids=selected_agent_ids,
                    team_lambda_compat=team_lambda_compat,
                    team_mu_cost=team_mu_cost,
                    team_nu_latency=team_nu_latency,
                )

            if reuse_role_selection:
                selection_cache[effective_cache_key] = dict(selection)
            if reuse_same_role_agent_once:
                role_once_selection_cache[role_once_key] = dict(selection)
        selected_main = selection.get("selected_main")
        if selected_main:
            selected_agent_ids.append(str(selected_main))
            selected_agent_roles.append(_normalize_role_name(role))
        selection_item = {
            "role": role,
            "selected_main": selected_main,
            "selected_shadows": selection.get("selected_shadows"),
        }
        if reused_same_role_agent_once:
            selection_item["reused_same_role_agent_once"] = True
        if mcts_dynamic_optimization and selection.get("mcts") is not None:
            selection_item["mcts"] = selection.get("mcts")
        selection_item["gap_state"] = (
            selection.get("mcts", {}).get("current_gap_state")
            if isinstance(selection.get("mcts"), dict)
            else _compute_gap_state(current_artifact_bag, resolved_task_family, resolved_dataset)
        )
        selection_log.append(selection_item)
        meta = {
            "step": step_idx,
            "topology": config.topology.value,
            "flow_type": config.flow_type,
            "reason": reason,
        }
        extra_context = {
            "history": history,
            "available_roles": config.roles,
            "topology": config.model_dump(),
            "is_decentralized": (config.topology == TopologyType.DECENTRALIZED or config.topology == TopologyType.CHAIN),
            "task_family": resolved_task_family,
            "dataset": resolved_dataset,
            "artifact_contract_version": artifact_contract_version,
            "paper_mode": paper_mode,
            "artifact_bag": dict(current_artifact_bag),
        }
        
        # Add previous agent's failure context if available (for multi-agent collaboration)
        if history and len(history) > 0:
            last_history = history[-1]
            # If previous agent failed (no test_passed or status != success), pass failure info
            if not last_history.get("test_passed") and last_history.get("status") != "success":
                extra_context["previous_agent_failed"] = True
                extra_context["previous_agent_role"] = last_history.get("role")
                extra_context["previous_attempt_summary"] = last_history.get("summary", "")[:500]

        # Use override if provided, otherwise use the global tool_only setting
        use_tool_only = override_tool_only if override_tool_only is not None else tool_only
        if candidate_final_answer_state.get("answer") and not _is_pseudo_final_answer(
            candidate_final_answer_state.get("answer")
        ):
            extra_context["candidate_final_answer"] = candidate_final_answer_state.get("answer")
            extra_context["candidate_final_answer_source"] = candidate_final_answer_state.get("source")
            extra_context["candidate_final_answer_round"] = candidate_final_answer_state.get("step")
            extra_context["candidate_final_answer_role"] = candidate_final_answer_state.get("role")
            if use_tool_only:
                extra_context["allow_tool_only_final_decision"] = True
        exec_result = _execute_role_with_selection(
            role=role,
            task_text=task_for_role,
            selection=selection,
            registry=registry,
            llm=llm,
            tool_executor=tool_executor,
            max_attempts=max_attempts,
            tool_max_rounds=tool_max_rounds,
            results=results,
            constraints=role_constraints,
            allow_unknown_roles=allow_unknown_roles,
            runs_dir=runs_dir,
            workflow_version=workflow_version,
            task_id=task_id,
            log_task_id=event_task_id,
            meta=meta,
            extra_context=extra_context,
            tool_only=use_tool_only,
            enable_postprocess_repair=enable_postprocess_repair,
            task_context=task_context,
        )
        log_path_local = exec_result.get("log_path")
        if log_path_local:
            log_path = log_path_local
        output = exec_result.get("output")
        _append_role_result(results, role, output)
        if exec_result.get("executor_result") is not None:
            _append_tool_exec(tool_exec, role, exec_result.get("executor_result"))
        if isinstance(exec_result.get("artifact_bag"), dict):
            current_artifact_bag = dict(exec_result.get("artifact_bag") or {})
        candidate_detected = _extract_candidate_final_answer_from_tool_history(exec_result.get("executor_result"))
        if candidate_detected is None and output is not None:
            output_candidate = _extract_candidate_final_answer_from_output(output)
            if output_candidate:
                candidate_detected = (output_candidate, f"{_normalize_role_name(role) or role}.output")
        if candidate_detected is not None:
            detected_answer, detected_source = candidate_detected
            if _should_replace_candidate_final_answer(
                candidate_final_answer_state,
                detected_answer,
                detected_source,
            ):
                candidate_final_answer_state.clear()
                candidate_final_answer_state.update(
                    {
                        "answer": str(detected_answer),
                        "source": str(detected_source),
                        "role": role,
                        "step": step_idx,
                    }
                )

        trace_entry: Dict[str, Any] = {
            "step": step_idx,
            "role": role,
            "reason": reason,
            "selected_main": exec_result.get("selected_main"),
        }
        post_gap_state = _compute_gap_state(current_artifact_bag, resolved_task_family, resolved_dataset)
        if exec_result.get("executor_result") is not None:
            trace_entry["tool_history"] = exec_result.get("executor_result")
        if exec_result.get("artifact_bag") is not None:
            trace_entry["artifact_bag"] = exec_result.get("artifact_bag")
        trace_entry["gap_state"] = post_gap_state
        if output is not None:
            trace_entry["output"] = output
        execution_trace.append(trace_entry)
        
        # Record history with test status for next agent to see
        history_entry = {
            "step": step_idx,
            "role": role,
            "summary": _summarize_output(output, max_chars=summary_max_chars),
            "selected_main": exec_result.get("selected_main"),
            "gap_state": post_gap_state,
        }
        syntax_compilable = _infer_role_syntax_compilable(
            role,
            output,
            exec_result.get("executor_result"),
        )
        if syntax_compilable is not None:
            history_entry["syntax_compilable"] = bool(syntax_compilable)

        # Add test status if available (for next agent to know if task completed)
        if isinstance(output, dict):
            if "test_passed" in output:
                history_entry["test_passed"] = output["test_passed"]
            elif "status" in output and output["status"] == "success":
                history_entry["status"] = "success"
            if _normalize_role_name(role) == "planner":
                history_entry["planner_ready_to_handoff"] = _planner_ready_to_handoff(output)
                planner_hint = _planner_next_role(output)
                if planner_hint:
                    history_entry["planner_next_role"] = planner_hint

        history.append(history_entry)
        
        return output

    roles_available = config.roles or role_list
    if not roles_available:
        raise ValueError("No roles available for dynamic workflow")

    if config.topology == TopologyType.SINGLE:
        single_output = _execute_role(config.entry_role or roles_available[0], task_text, 0, "single")
        if isinstance(single_output, dict) and (
            single_output.get("test_passed") or single_output.get("status") == "success"
        ):
            workflow_succeeded = True
    elif config.topology == TopologyType.CENTRALIZED:
        manager = config.manager_role or (roles_available[0] if roles_available else "planner")
        # In tool_only mode, manager/planner should also stay on the tool path so
        # role execution is handled by the tool backend rather than direct role LLM output.
        manager_output = _execute_role(manager, task_text, 0, "manager")
        manager_is_planner = _normalize_role_name(manager) == "planner"
        planner_ready = _planner_ready_to_handoff(manager_output) if manager_is_planner else True
        
        # Extract steps from manager output (for logging/planning purposes)
        steps = []
        if isinstance(manager_output, dict):
            steps = manager_output.get("steps") or manager_output.get("plan") or []
        print(f"[DEBUG] Manager output keys: {list(manager_output.keys()) if isinstance(manager_output, dict) else 'not a dict'}", file=sys.stderr)
        print(f"[DEBUG] Manager generated {len(steps)} steps: {steps[:3] if len(steps) > 3 else steps}", file=sys.stderr)
        
        # In CENTRALIZED mode, workers collaborate on the same task (not separate steps)
        # Keep trying different workers until task is solved or no more workers available
        worker_roles = [role for role in roles_available if role != manager] or roles_available
        last_worker = None
        for worker_idx in range(config.max_steps):
            candidate_roles = list(worker_roles)
            avoid_role = last_worker
            if manager_is_planner and not planner_ready and not force_role:
                # Planner has not handed off yet: include manager as a routable option.
                # The concrete next role is still chosen by the centralized routing LLM.
                candidate_roles = [manager] + [role for role in worker_roles if role != manager]
                avoid_role = None
                print("[DEBUG] Centralized planner_ready=false, allowing manager replanning in role routing", file=sys.stderr)
            checker_role = next(
                (role for role in candidate_roles if _normalize_role_name(role) == "checker"),
                None,
            )
            checker_seen = any(
                _normalize_role_name(item.get("role")) == "checker"
                for item in history
                if isinstance(item, dict)
            )
            candidate_answer_available = bool(candidate_final_answer_state.get("answer")) and not _is_pseudo_final_answer(
                candidate_final_answer_state.get("answer")
            )
            force_checker_after_candidate = (
                candidate_answer_available
                and checker_role is not None
                and not checker_seen
                and not workflow_succeeded
            )
            force_checker_before_exit = (
                checker_role is not None
                and not checker_seen
                and not workflow_succeeded
                and worker_idx >= config.max_steps - 1
            )
            if force_checker_after_candidate:
                next_role = checker_role
                print(
                    "[DEBUG] Candidate final answer detected; routing to checker before any further roles",
                    file=sys.stderr,
                )
            elif force_checker_before_exit:
                next_role = checker_role
                print(
                    "[DEBUG] Forcing checker on last centralized step before exit",
                    file=sys.stderr,
                )
            else:
                next_role = _select_next_role(
                    task_text=task_text,
                    available_roles=candidate_roles,
                    history=history,
                    llm=next_role_llm_client,
                    force_role=force_role,
                    avoid_role=avoid_role,  # Try to avoid repeating the same worker unless planner isn't ready
                    prevent_early_finish=prevent_early_finish,
                )
            if not next_role:
                task_completed = workflow_succeeded or any(
                    item.get("test_passed") or item.get("status") == "success" for item in history
                )
                if prevent_early_finish and not task_completed:
                    fallback_role = _fallback_next_role(candidate_roles, avoid_role=avoid_role)
                    if fallback_role:
                        print(
                            f"[DEBUG] Router returned finish early; fallback to role '{fallback_role}'",
                            file=sys.stderr,
                        )
                        next_role = fallback_role
                if not next_role:
                    print(f"[DEBUG] No more workers to try, stopping CENTRALIZED workflow", file=sys.stderr)
                    break
            if next_role == manager:
                manager_output = _execute_role(
                    manager,
                    task_text,
                    worker_idx + 1,
                    "centralized_manager_replan",
                )
                if isinstance(manager_output, dict) and (
                    manager_output.get("test_passed") or manager_output.get("status") == "success"
                ):
                    print("[DEBUG] Manager reported success, stopping CENTRALIZED workflow", file=sys.stderr)
                    workflow_succeeded = True
                    break
                planner_ready = _planner_ready_to_handoff(manager_output) if manager_is_planner else True
                continue
            worker_output = _execute_role(next_role, task_text, worker_idx + 1, "centralized")
            last_worker = next_role
            if _normalize_role_name(next_role) == "checker":
                candidate_role = candidate_final_answer_state.get("role")
                candidate_step = candidate_final_answer_state.get("step")
                candidate_answer = candidate_final_answer_state.get("answer")
                if (
                    candidate_answer
                    and not _is_pseudo_final_answer(candidate_answer)
                    and candidate_role == next_role
                    and candidate_step == worker_idx + 1
                ):
                    print(
                        "[DEBUG] Checker produced candidate final answer, stopping CENTRALIZED workflow early",
                        file=sys.stderr,
                    )
                    workflow_succeeded = True
                    break
            if isinstance(worker_output, dict) and (worker_output.get("test_passed") or worker_output.get("status") == "success"):
                print("[DEBUG] Test passed, stopping CENTRALIZED workflow", file=sys.stderr)
                workflow_succeeded = True
                break
    else:
        current_role = config.entry_role or roles_available[0]
        visited: Dict[str, int] = {}
        last_output: Optional[Any] = None
        route_query = task_text
        for step_idx in range(config.max_steps):
            task_for_role = task_text
            if soft_connection:
                task_for_role = f"{task_text}\nRouting focus: {route_query}"
            last_output = _execute_role(current_role, task_for_role, step_idx, "dynamic_soft" if soft_connection else "dynamic")
            if _normalize_role_name(current_role) == "checker":
                candidate_role = candidate_final_answer_state.get("role")
                candidate_step = candidate_final_answer_state.get("step")
                candidate_answer = candidate_final_answer_state.get("answer")
                if (
                    candidate_answer
                    and not _is_pseudo_final_answer(candidate_answer)
                    and candidate_role == current_role
                    and candidate_step == step_idx
                ):
                    print(
                        "[DEBUG] Checker produced candidate final answer, stopping dynamic workflow early",
                        file=sys.stderr,
                    )
                    workflow_succeeded = True
                    break
            if isinstance(last_output, dict) and (last_output.get("test_passed") or last_output.get("status") == "success"):
                print("[DEBUG] Test passed, stopping dynamic workflow", file=sys.stderr)
                workflow_succeeded = True
                break
            visited[current_role] = visited.get(current_role, 0) + 1
            neighbors = _neighbors_for_role(current_role, config)
            candidates = neighbors if neighbors else roles_available
            avoid_role = current_role if visited.get(current_role, 0) > 1 else None

            # Planner controls handoff timing in non-centralized flows.
            # If the planner says it's not ready, keep planner for next step.
            current_role_key = _normalize_role_name(current_role)
            if current_role_key == "planner" and not force_role:
                planner_ready = _planner_ready_to_handoff(last_output)
                if not planner_ready:
                    print("[DEBUG] Planner not ready_to_handoff, continuing planner", file=sys.stderr)
                    continue

                planner_suggested = _planner_next_role(last_output)
                resolved_planner_suggested = None
                if planner_suggested:
                    resolved_planner_suggested = next(
                        (candidate for candidate in candidates if _normalize_role_name(candidate) == planner_suggested),
                        None,
                    )
                if resolved_planner_suggested:
                    print(
                        f"[DEBUG] Planner ready_to_handoff, suggested next_role={resolved_planner_suggested}",
                        file=sys.stderr,
                    )
                    next_role = resolved_planner_suggested
                    current_role = next_role
                    continue

                builder_fallback = next(
                    (candidate for candidate in candidates if _normalize_role_name(candidate) == "builder"),
                    None,
                )
                if builder_fallback:
                    print(
                        "[DEBUG] Planner ready_to_handoff, no valid suggestion; fallback next_role=builder",
                        file=sys.stderr,
                    )
                    current_role = builder_fallback
                    continue
            
            # Check if agent specified next_role in output (agent-driven routing for decentralized)
            agent_suggested_role = None
            if isinstance(last_output, dict):
                agent_suggested_role = last_output.get("next_role")
                if agent_suggested_role:
                    agent_suggested_role = str(agent_suggested_role).strip().lower()
                    # Validate that suggested role is available
                    if agent_suggested_role in candidates:
                        print(f"[DEBUG] Agent {current_role} suggested next_role: {agent_suggested_role}", file=sys.stderr)
                        next_role = agent_suggested_role
                    elif agent_suggested_role == "finish":
                        task_completed = workflow_succeeded or any(
                            item.get("test_passed") or item.get("status") == "success" for item in history
                        )
                        if prevent_early_finish and not task_completed:
                            print(
                                f"[WARNING] Agent {current_role} requested finish before success; ignoring",
                                file=sys.stderr,
                            )
                            agent_suggested_role = None
                        else:
                            print(f"[DEBUG] Agent {current_role} requested finish", file=sys.stderr)
                            next_role = None
                    else:
                        print(f"[WARNING] Agent suggested invalid next_role '{agent_suggested_role}', not in {candidates}. Using router.", file=sys.stderr)
                        agent_suggested_role = None
            
            # If agent didn't specify next_role, use the router
            if agent_suggested_role is None or agent_suggested_role not in candidates:
                checker_candidate = next(
                    (candidate for candidate in candidates if _normalize_role_name(candidate) == "checker"),
                    None,
                )
                checker_seen = any(
                    _normalize_role_name(item.get("role")) == "checker"
                    for item in history
                    if isinstance(item, dict)
                )
                candidate_answer_available = bool(candidate_final_answer_state.get("answer")) and not _is_pseudo_final_answer(
                    candidate_final_answer_state.get("answer")
                )
                if (
                    candidate_answer_available
                    and checker_candidate is not None
                    and not checker_seen
                    and current_role != checker_candidate
                ):
                    next_role = checker_candidate
                    print(
                        "[DEBUG] Candidate final answer detected; routing to checker before further dynamic routing",
                        file=sys.stderr,
                    )
                    current_role = next_role
                    continue
                if soft_connection:
                    route_query = _build_route_query(
                        task_text=task_text,
                        history=history,
                        last_output=last_output,
                        llm=route_query_llm_client,
                        fallback_max_chars=summary_max_chars,
                    )
                next_role = _select_next_role(
                    task_text=route_query if soft_connection else task_text,
                    available_roles=candidates,
                    history=history,
                    llm=next_role_llm_client,
                    avoid_role=avoid_role,
                    force_role=force_role,
                    prevent_early_finish=prevent_early_finish,
                )
            
            if not next_role:
                task_completed = workflow_succeeded or any(
                    item.get("test_passed") or item.get("status") == "success" for item in history
                )
                if prevent_early_finish and not task_completed:
                    fallback_role = _fallback_next_role(candidates, avoid_role=avoid_role)
                    if fallback_role:
                        print(
                            f"[DEBUG] Router returned finish early; fallback to role '{fallback_role}'",
                            file=sys.stderr,
                        )
                        next_role = fallback_role
                if not next_role:
                    break
            current_role = next_role

    last_history = history[-1] if history else {}
    last_role = _normalize_role_name(last_history.get("role"))
    last_step_failed = not (
        bool(last_history.get("test_passed")) or str(last_history.get("status") or "") == "success"
    )
    should_force_final_builder = (
        force_final_builder_extra_step
        and not workflow_succeeded
        and last_role == "checker"
        and last_step_failed
    )
    if should_force_final_builder:
        builder_role = next(
            (role for role in roles_available if _normalize_role_name(role) == "builder"),
            None,
        )
        if builder_role:
            _execute_role(
                builder_role,
                task_text,
                config.max_steps + 1,
                "forced_final_builder_extra",
            )
            forced_final_builder_executed = True

    checker_ok = _checker_ok(results)
    return {
        "results": results,
        "tool_exec": tool_exec,
        "execution_trace": execution_trace,
        "log_path": log_path,
        "selections": selection_log,
        "checker_ok": checker_ok,
        "topology": config.model_dump(),
        "forced_final_builder_executed": forced_final_builder_executed,
    }


def run_workflow(
    task_text: str,
    roles: Optional[List[str]] = None,
    constraints_per_role: Optional[Dict[str, Dict[str, Any]]] = None,
    workflow_version: str = "v1",
    registry=None,
    index=None,
    embedder=None,
    top_n: int = 20,
    top_k: int = 5,
    rerank_top_m: int = 3,
    mmr_lambda: float = 0.5,
    use_mmr: bool = True,
    max_attempts: int = 3,
    tool_max_rounds: int = 5,
    bandit_db_path: Optional[str] = None,
    defer_bandit_update: bool = False,
    use_cross_instance_cache: bool = True,
    update_posteriors_across_instances: bool = True,
    reranker_model_path: Optional[str] = None,
    execute_tools: bool = True,
    tool_timeout_s: float = 1.0,
    auto_fill_tool_inputs: bool = False,
    tool_llm: Optional[Any] = None,
    auto_install_common_libs: bool = False,
    auto_install_timeout_s: float = 300.0,
    auto_install_user: bool = False,
    llm_client: Optional[LLMClient] = None,
    router_llm_client: Optional[LLMClient] = None,
    router_top_m: int = 5,
    router_no_rerank: bool = False,
    experimental_reranker_after_mmr: bool = False,
    runs_dir: str = "runs",
    log_task_id: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    paper_mode: bool = False,
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
    final_candidate_cap: int = 5,
    artifact_contract_version: str = "v2",
    dynamic_topology: bool = False,
    topology: Optional[str] = None,
    topology_config: Optional[Dict[str, Any]] = None,
    meta_router_llm_client: Optional[LLMClient] = None,
    next_role_llm_client: Optional[LLMClient] = None,
    max_steps: int = 6,
    force_final_builder_extra_step: bool = True,
    allow_unknown_roles: bool = False,
    enable_postprocess_repair: bool = True,
    prevent_early_finish: bool = True,
    reuse_role_selection: bool = True,
    reuse_same_role_agent_once: bool = False,
    summary_max_chars: int = 400,
    soft_connection: bool = False,
    tool_only: bool = False,
    team_lambda_compat: float = 0.2,
    team_mu_cost: float = 0.05,
    team_nu_latency: float = 0.05,
    mcts_dynamic_optimization: bool = False,
    mcts_iterations: int = 64,
    mcts_rollout_depth: int = 4,
    mcts_exploration: float = 1.414,
    mcts_discount: float = 0.95,
    mcts_max_candidates: int = 8,
    use_iochain: bool = True,
    use_transition_term: bool = True,
    use_gap_term: bool = True,
    reward_variant: str = "mean_active",
    force_role: Optional[str] = None,
    strict_roles: bool = False,
    parallel_execution: bool = False,
    use_parallel_frontier: bool = False,
    parallel_mode: str = "frontier",
    parallel_width: int = 4,
    parallel_branch_roles: Optional[List[str]] = None,
    parallel_consensus: str = "test_first_then_majority",
    parallel_branch_timeout_s: Optional[float] = None,
    parallel_fail_fast: bool = True,
    parallel_search: bool = True,
    inference_visibility: str = "tool_visible",
) -> Dict[str, Any]:
    if registry is None or index is None or embedder is None:
        raise ValueError("registry, index, embedder are required")

    role_list = roles or ["planner", "builder", "checker", "refactor"]
    constraints_per_role = constraints_per_role or {}
    task_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    event_task_id = str(log_task_id or task_id)
    llm = RealLLMClient(llm_client) if llm_client else MockLLMClient()
    if reranker_model_path and os.path.exists(reranker_model_path):
        reranker = TfidfLinearReranker.load(reranker_model_path)
    else:
        reranker = TfidfLinearReranker()
    results: Dict[str, Any] = {}
    tool_exec: Dict[str, Any] = {}
    log_path: Optional[str] = None
    selections: Dict[str, Dict[str, Any]] = {}
    tool_executor = (
        ToolExecutor(
            registry,
            timeout_s=tool_timeout_s,
            auto_fill=auto_fill_tool_inputs,
            llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
        )
        if execute_tools
        else None
    )

    resolved_dataset = _resolve_workflow_dataset(dataset=dataset, task_context=task_context)
    resolved_task_family = str(task_family or _infer_task_family(resolved_dataset, task_context)).strip().lower()
    inference_task_context = _sanitize_task_context_for_inference(task_context, inference_visibility)
    effective_bandit_db_path = bandit_db_path if use_cross_instance_cache else None
    paper_defaults = _apply_paper_mode_defaults(
        paper_mode=paper_mode,
        task_family=resolved_task_family,
        top_n=top_n,
        top_k=top_k,
        topology=topology,
        dynamic_topology=dynamic_topology,
        mcts_dynamic_optimization=mcts_dynamic_optimization,
        mcts_iterations=mcts_iterations,
        mcts_rollout_depth=mcts_rollout_depth,
        mcts_exploration=mcts_exploration,
        mcts_discount=mcts_discount,
        mcts_max_candidates=final_candidate_cap if final_candidate_cap > 0 else mcts_max_candidates,
        team_mu_cost=team_mu_cost,
        team_nu_latency=team_nu_latency,
    )
    top_n = int(paper_defaults["top_n"])
    top_k = int(paper_defaults["top_k"])
    topology = paper_defaults["topology"]
    dynamic_topology = bool(paper_defaults["dynamic_topology"])
    mcts_dynamic_optimization = bool(paper_defaults["mcts_dynamic_optimization"])
    mcts_iterations = int(paper_defaults["mcts_iterations"])
    mcts_rollout_depth = int(paper_defaults["mcts_rollout_depth"])
    mcts_exploration = float(paper_defaults["mcts_exploration"])
    mcts_discount = float(paper_defaults["mcts_discount"])
    mcts_max_candidates = int(paper_defaults["mcts_max_candidates"])
    team_mu_cost = float(paper_defaults["team_mu_cost"])
    team_nu_latency = float(paper_defaults["team_nu_latency"])

    parallel_mode_key = str(parallel_mode or "frontier").strip().lower().replace("-", "_")
    parallel_enabled = (
        bool(parallel_execution)
        or bool(use_parallel_frontier)
        or (str(topology or "").strip().lower() in {"parallel", "fork_join", "fork-join"})
    ) and parallel_mode_key not in {"off", "none", "sequential"}
    if parallel_enabled:
        parallel_result = _run_parallel_workflow(
            task_text=task_text,
            role_list=role_list,
            constraints_per_role=constraints_per_role,
            workflow_version=workflow_version,
            registry=registry,
            index=index,
            embedder=embedder,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            use_mmr=use_mmr,
            max_attempts=max_attempts,
            tool_max_rounds=tool_max_rounds,
            reranker=reranker,
            execute_tools=execute_tools,
            tool_timeout_s=tool_timeout_s,
            auto_fill_tool_inputs=auto_fill_tool_inputs,
            tool_llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
            llm=llm,
            router_llm_client=router_llm_client,
            router_top_m=router_top_m,
            router_no_rerank=router_no_rerank,
            experimental_reranker_after_mmr=experimental_reranker_after_mmr,
            runs_dir=runs_dir,
            task_id=task_id,
            log_task_id=event_task_id,
            task_context=inference_task_context,
            task_family=resolved_task_family,
            dataset=resolved_dataset,
            artifact_contract_version=artifact_contract_version,
            paper_mode=paper_mode,
            allow_unknown_roles=allow_unknown_roles,
            enable_postprocess_repair=enable_postprocess_repair,
            tool_only=tool_only,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
            parallel_width=parallel_width,
            parallel_branch_roles=parallel_branch_roles,
            parallel_consensus=parallel_consensus,
            parallel_branch_timeout_s=parallel_branch_timeout_s,
            parallel_fail_fast=parallel_fail_fast,
            parallel_search=parallel_search,
        )
        if effective_bandit_db_path and update_posteriors_across_instances and not defer_bandit_update:
            apply_bandit_feedback(
                effective_bandit_db_path,
                workflow_version=workflow_version,
                selections=parallel_result.get("selections") or [],
                success=bool(parallel_result.get("checker_ok", True)),
            )
        return {
            "task_id": task_id,
            "answer": parallel_result.get("answer", ""),
            "log_path": parallel_result.get("log_path"),
            "results": parallel_result.get("results", {}),
            "tool_exec": parallel_result.get("tool_exec", {}),
            "execution_trace": parallel_result.get("execution_trace", []),
            "selections": parallel_result.get("selections", []),
            "topology": parallel_result.get("topology"),
            "checker_ok": parallel_result.get("checker_ok", True),
            "parallel": parallel_result.get("parallel"),
        }

    topology_value = (topology or "").strip().lower()
    dynamic_mode = dynamic_topology or topology_config is not None or (
        topology_value and topology_value not in {"linear", "fixed"}
    )
    if dynamic_mode:
        pair_success_priors = _load_pair_success_priors(effective_bandit_db_path, workflow_version)
        role_success_priors = _load_role_success_priors(effective_bandit_db_path, workflow_version)
        dynamic_result = _run_dynamic_workflow(
            task_text=task_text,
            role_list=role_list,
            constraints_per_role=constraints_per_role,
            workflow_version=workflow_version,
            registry=registry,
            index=index,
            embedder=embedder,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            use_mmr=use_mmr,
            max_attempts=max_attempts,
            tool_max_rounds=tool_max_rounds,
            reranker=reranker,
            execute_tools=execute_tools,
            tool_timeout_s=tool_timeout_s,
            auto_fill_tool_inputs=auto_fill_tool_inputs,
            tool_llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
            llm=llm,
            router_llm_client=router_llm_client,
            router_top_m=router_top_m,
            router_no_rerank=router_no_rerank,
            experimental_reranker_after_mmr=experimental_reranker_after_mmr,
            runs_dir=runs_dir,
            task_id=task_id,
            log_task_id=event_task_id,
            topology=topology,
            topology_config=topology_config,
            meta_router_llm_client=meta_router_llm_client or router_llm_client,
            next_role_llm_client=next_role_llm_client or router_llm_client,
            route_query_llm_client=next_role_llm_client or router_llm_client,
            max_steps=max_steps,
            force_final_builder_extra_step=force_final_builder_extra_step,
            allow_unknown_roles=allow_unknown_roles or dynamic_mode,
            enable_postprocess_repair=enable_postprocess_repair,
            prevent_early_finish=prevent_early_finish,
            force_role=force_role,
            reuse_role_selection=reuse_role_selection,
            reuse_same_role_agent_once=reuse_same_role_agent_once,
            summary_max_chars=summary_max_chars,
            soft_connection=soft_connection,
            tool_only=tool_only,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
            mcts_dynamic_optimization=mcts_dynamic_optimization,
            mcts_iterations=mcts_iterations,
            mcts_rollout_depth=mcts_rollout_depth,
            mcts_exploration=mcts_exploration,
            mcts_discount=mcts_discount,
            mcts_max_candidates=mcts_max_candidates,
            use_iochain=use_iochain,
            use_transition_term=use_transition_term,
            use_gap_term=use_gap_term,
            reward_variant=reward_variant,
            pair_success_priors=pair_success_priors,
            role_success_priors=role_success_priors,
            task_context=inference_task_context,
            task_family=resolved_task_family,
            dataset=resolved_dataset,
            artifact_contract_version=artifact_contract_version,
            paper_mode=paper_mode,
            strict_roles=strict_roles,
        )

        if effective_bandit_db_path and update_posteriors_across_instances and not defer_bandit_update:
            checker_ok = dynamic_result.get("checker_ok", True)
            selections_list = dynamic_result.get("selections") or []
            apply_bandit_feedback(
                effective_bandit_db_path,
                workflow_version=workflow_version,
                selections=selections_list,
                success=bool(checker_ok),
            )

        results = dynamic_result.get("results", {})
        answer_lines = [f"[{role}] {results.get(role)}" for role in results.keys()]
        answer = "\n".join(answer_lines)
        return {
            "task_id": task_id,
            "answer": answer,
            "log_path": dynamic_result.get("log_path"),
            "results": results,
            "tool_exec": dynamic_result.get("tool_exec"),
            "execution_trace": dynamic_result.get("execution_trace"),
            "selections": dynamic_result.get("selections"),
            "topology": dynamic_result.get("topology"),
            "forced_final_builder_executed": bool(dynamic_result.get("forced_final_builder_executed")),
        }

    selected_agent_ids: List[str] = []
    selection_cache: Dict[str, Dict[str, Any]] = {}
    execution_trace: List[Dict[str, Any]] = []
    candidate_final_answer_state: Dict[str, Any] = {}
    for role in role_list:
        role_key = _normalize_role_name(role)
        constraints = _ensure_role_constraints(
            constraints_per_role.get(role),
            role,
            dataset=dataset,
            task_family=task_family,
        )
        selection = _select_agent_for_role(
            task_text=task_text,
            role=role,
            constraints=constraints,
            registry=registry,
            index=index,
            embedder=embedder,
            reranker=reranker,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            use_mmr=use_mmr,
            router_llm_client=router_llm_client,
            router_top_m=router_top_m,
            router_no_rerank=router_no_rerank,
            experimental_reranker_after_mmr=experimental_reranker_after_mmr,
            reuse_cache=reuse_role_selection,
            cache=selection_cache,
            cache_key=(
                _normalize_role_name(role)
                if reuse_same_role_agent_once
                else _build_selection_cache_key(
                    role=role,
                    task_text=task_text,
                    step_idx=0,
                    topology="linear",
                )
            ),
            artifact_context=None,
            selected_agent_ids=selected_agent_ids,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
        )

        selected_main = selection.get("selected_main")
        selected_shadows = list(selection.get("selected_shadows") or [])
        selected_shadow = selection.get("selected_shadow")
        reranked = list(selection.get("reranked") or [])

        if selected_main:
            selected_agent_ids.append(str(selected_main))

        selections[role] = {
            "main": selected_main,
            "shadows": selected_shadows,
        }

        extra_context: Dict[str, Any] = {"available_roles": role_list}
        if candidate_final_answer_state.get("answer") and not _is_pseudo_final_answer(
            candidate_final_answer_state.get("answer")
        ):
            extra_context["candidate_final_answer"] = candidate_final_answer_state.get("answer")
            extra_context["candidate_final_answer_source"] = candidate_final_answer_state.get("source")
            extra_context["candidate_final_answer_round"] = candidate_final_answer_state.get("step")
            extra_context["candidate_final_answer_role"] = candidate_final_answer_state.get("role")
            if tool_only:
                extra_context["allow_tool_only_final_decision"] = True

        exec_result = _execute_role_with_selection(
            role=role,
            task_text=task_text,
            selection={
                "selected_main": selected_main,
                "selected_shadow": selected_shadow,
                "selected_shadows": selected_shadows,
                "reranked": reranked,
            },
            registry=registry,
            llm=llm,
            tool_executor=tool_executor,
            max_attempts=max_attempts,
            results=results,
            constraints=constraints,
            allow_unknown_roles=allow_unknown_roles,
            runs_dir=runs_dir,
            workflow_version=workflow_version,
            task_id=task_id,
            log_task_id=event_task_id,
            extra_context=extra_context,
            tool_max_rounds=tool_max_rounds,
            tool_only=tool_only,
            enable_postprocess_repair=enable_postprocess_repair,
            task_context=inference_task_context,
        )
        output = exec_result.get("output")
        if output is None:
            output = {"error": "no_valid_output"}
        results[role] = output
        if exec_result.get("executor_result") is not None:
            tool_exec[role] = exec_result.get("executor_result")
        if exec_result.get("log_path"):
            log_path = exec_result.get("log_path")

        role_artifact_bag = exec_result.get("artifact_bag") if isinstance(exec_result.get("artifact_bag"), dict) else {}
        role_candidate_answer = role_artifact_bag.get("candidate_final_answer")
        role_candidate_source = role_artifact_bag.get("candidate_final_answer_source")
        if _is_pseudo_final_answer(role_candidate_answer):
            role_candidate_answer = None
            role_candidate_source = None
        if not role_candidate_answer:
            detected_candidate = _extract_candidate_final_answer_from_tool_history(exec_result.get("executor_result"))
            if detected_candidate:
                role_candidate_answer, role_candidate_source = detected_candidate
        if role_candidate_answer:
            candidate_source = str(role_candidate_source or f"{_normalize_role_name(role) or role}.tool_history")
            if _should_replace_candidate_final_answer(
                candidate_final_answer_state,
                role_candidate_answer,
                candidate_source,
            ):
                candidate_final_answer_state.clear()
                candidate_final_answer_state.update(
                    {
                        "answer": str(role_candidate_answer),
                        "source": candidate_source,
                        "role": role,
                        "step": len(execution_trace),
                    }
                )
        execution_trace.append(
            {
                "step": len(execution_trace),
                "role": role,
                "selected_main": exec_result.get("selected_main", selected_main),
                "tool_history": exec_result.get("executor_result"),
                "artifact_bag": exec_result.get("artifact_bag"),
                "output": results.get(role),
            }
        )

    checker_ok = True
    checker_output = results.get("checker")
    if isinstance(checker_output, dict) and checker_output.get("error"):
        checker_ok = False

    if effective_bandit_db_path and update_posteriors_across_instances and not defer_bandit_update:
        static_selections = [
            {
                "role": role,
                "selected_main": (selections.get(role) or {}).get("main"),
                "selected_shadows": list((selections.get(role) or {}).get("shadows") or []),
            }
            for role in role_list
            if (selections.get(role) or {}).get("main")
        ]
        apply_bandit_feedback(
            effective_bandit_db_path,
            workflow_version=workflow_version,
            selections=static_selections,
            success=bool(checker_ok),
        )

    answer_lines = []
    for role in role_list:
        answer_lines.append(f"[{role}] {results.get(role)}")
    answer = "\n".join(answer_lines)

    return {
        "task_id": task_id,
        "answer": answer,
        "log_path": log_path,
        "results": results,
        "tool_exec": tool_exec,
        "execution_trace": execution_trace,
        "selections": None,
    }
