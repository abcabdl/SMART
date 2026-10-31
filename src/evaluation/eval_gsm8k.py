from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import os
import re
import sqlite3
import sys
import time
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.smart_pdf_preset import apply_smart_pdf_preset

getcontext().prec = 50

_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_NUM_TOKEN_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?"
)
_PLACEHOLDER_NUMBER_RE = re.compile(
    r"^<\s*number\s*>(?:\s*\.)?(?:\s*without\b.*)?$",
    re.IGNORECASE,
)
_CURRENCY_CHARS = "$\u20ac\u00a3\u00a5\u20b9"
_ROLE_ALIASES = {
    "code-generation": "builder",
    "code-planner": "planner",
    "code-testing": "checker",
    "tester": "checker",
    "code-refactoring": "refactor",
    "formatter": "refactor",
    "refractor": "refactor",
}
_STRUCTURED_SKIP_KEYS = {
    "tool_id",
    "id",
    "family",
    "stage",
    "focus",
    "model",
    "temperature",
    "confidence",
    "notes",
    "checks",
    "llm_status",
    "payload_source",
}


def _resolve_repo_aware_path(raw: str, *, prefer_repo_if_missing: bool = False) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    cwd_candidate = Path.cwd() / path
    repo_candidate = REPO_ROOT / path
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


def _iter_progress(items: List[dict], desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items))


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


def _task_name(task: Dict[str, Any], idx: int) -> str:
    return str(task.get("name") or task.get("task_id") or task.get("id") or f"gsm8k-{idx}")


def _task_question(task: Dict[str, Any]) -> str:
    return str(task.get("question") or task.get("prompt") or task.get("task") or "").strip()


def _normalize_role_name(role: str) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIASES.get(key, key)


def _load_solutions(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    records = _load_jsonl(path, ignore_errors=True)
    solutions: Dict[str, str] = {}
    for record in records:
        name = record.get("name") or record.get("task_id") or record.get("id")
        completion = (
            record.get("completion")
            or record.get("prediction")
            or record.get("pred")
            or record.get("answer")
            or record.get("solution")
            or record.get("output")
        )
        if name and completion is not None:
            solutions[str(name)] = str(completion)
    return solutions


def _extract_gold_answer_text(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    matches = _HASH_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text


def _extract_number_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in _NUM_TOKEN_RE.finditer(text)]


def _clean_numeric_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    text = text.strip(" \t\r\n;:,!?)]}")
    text = text.lstrip("([{")
    for ch in _CURRENCY_CHARS:
        text = text.replace(ch, "")
    if text.endswith("%"):
        text = text[:-1]
    text = text.replace(",", "")
    text = text.strip()
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-."):
        text = "-0" + text[1:]
    elif text.startswith("."):
        text = "0" + text
    return text


def _parse_numeric_token(token: str) -> Optional[Decimal]:
    cleaned = _clean_numeric_token(token)
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            num, den = cleaned.split("/", 1)
            return Decimal(num) / Decimal(den)
        return Decimal(cleaned)
    except (InvalidOperation, DivisionByZero, ZeroDivisionError):
        return None


def _is_blocked_answer_text(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"true", "false"}:
        return True
    normalized = re.sub(r"\s+", " ", normalized)
    if _PLACEHOLDER_NUMBER_RE.fullmatch(normalized):
        return True
    return False


def _extract_numeric_from_text(text: str) -> Tuple[Optional[Decimal], str]:
    content = str(text or "")
    if not content.strip():
        return None, ""

    hash_matches = _HASH_ANSWER_RE.findall(content)
    if hash_matches:
        candidate_text = hash_matches[-1].strip()
        tokens = _extract_number_tokens(candidate_text)
        if tokens:
            value = _parse_numeric_token(tokens[-1])
            if value is not None:
                return value, tokens[-1]

    tokens = _extract_number_tokens(content)
    for token in reversed(tokens):
        value = _parse_numeric_token(token)
        if value is not None:
            return value, token
    return None, ""


def _extract_answer_candidate(obj: Any, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 10:
        return None

    if isinstance(obj, bool):
        return None

    if isinstance(obj, (int, float, Decimal)):
        return str(obj)

    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None
        hash_matches = _HASH_ANSWER_RE.findall(text)
        if hash_matches:
            candidate = hash_matches[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate
        if _is_blocked_answer_text(text):
            return None
        if re.search(r"\d", text):
            return text
        return None

    if isinstance(obj, dict):
        strong_keys = (
            "final_answer",
            "numeric_answer",
            "answer",
            "recomputed_answer",
        )
        medium_keys = (
            "formatted",
            "result",
            "response",
            "content",
            "message",
            "summary",
        )
        wrapper_keys = (
            "payload",
            "output",
            "data",
            "results",
            "tool_exec",
            "execution_trace",
        )
        skip_keys = {
            "tool_id",
            "id",
            "family",
            "stage",
            "focus",
            "model",
            "temperature",
            "confidence",
            "notes",
            "checks",
        }

        for key in strong_keys:
            if key in obj:
                candidate = _extract_answer_candidate(obj.get(key), depth + 1)
                if candidate:
                    return candidate

        for key in medium_keys:
            if key in obj:
                candidate = _extract_answer_candidate(obj.get(key), depth + 1)
                if candidate:
                    return candidate

        for key in wrapper_keys:
            if key in obj:
                candidate = _extract_answer_candidate(obj.get(key), depth + 1)
                if candidate:
                    return candidate

        for key, value in obj.items():
            if str(key).strip().lower() in skip_keys:
                continue
            candidate = _extract_answer_candidate(value, depth + 1)
            if candidate:
                return candidate
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            candidate = _extract_answer_candidate(item, depth + 1)
            if candidate:
                return candidate
        return None

    return None


def _append_role_ordered_candidates(
    sink: List[Tuple[str, Any]],
    container: Any,
    *,
    prefix: str,
    role_order: Tuple[str, ...],
) -> None:
    if not isinstance(container, dict):
        return
    for role in role_order:
        if role in container:
            sink.append((f"{prefix}.{role}", container.get(role)))
    for role, value in container.items():
        if role not in role_order:
            sink.append((f"{prefix}.{role}", value))


def _extract_structured_answer(
    obj: Any,
    depth: int = 0,
    *,
    allow_numeric_scalar: bool = False,
) -> Optional[Tuple[str, str]]:
    if obj is None or depth > 14:
        return None

    if isinstance(obj, bool):
        return None

    if isinstance(obj, (int, float, Decimal)):
        if allow_numeric_scalar:
            return str(obj), "numeric_value"
        return None

    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None
        if _HASH_ANSWER_RE.search(text):
            candidate = _HASH_ANSWER_RE.findall(text)[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate, "hash_answer"
        return None

    if isinstance(obj, dict):
        for key in ("final_answer", "numeric_answer"):
            if key in obj:
                value = obj.get(key)
                if value is None:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float, Decimal)):
                    return str(value), key
                text = str(value).strip()
                if text and not _is_blocked_answer_text(text):
                    return text, key
                found = _extract_structured_answer(
                    value,
                    depth + 1,
                    allow_numeric_scalar=True,
                )
                if found:
                    return found[0], key

        # Prefer typical nested wrappers first.
        for key in ("payload", "output", "result", "data"):
            if key in obj:
                found = _extract_structured_answer(
                    obj.get(key),
                    depth + 1,
                    allow_numeric_scalar=allow_numeric_scalar,
                )
                if found:
                    return found

        for key, value in obj.items():
            if str(key).strip().lower() in _STRUCTURED_SKIP_KEYS:
                continue
            found = _extract_structured_answer(
                value,
                depth + 1,
                allow_numeric_scalar=False,
            )
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            found = _extract_structured_answer(
                item,
                depth + 1,
                allow_numeric_scalar=allow_numeric_scalar,
            )
            if found:
                return found
        return None

    return None


def _extract_hash_answer_only(obj: Any, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 14:
        return None

    if isinstance(obj, str):
        matches = _HASH_ANSWER_RE.findall(obj.strip())
        if matches:
            candidate = matches[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate
        return None

    if isinstance(obj, dict):
        for key in ("formatted", "result", "response", "content", "message", "summary"):
            if key in obj:
                found = _extract_hash_answer_only(obj.get(key), depth + 1)
                if found:
                    return found
        for key in ("payload", "output", "result", "data"):
            if key in obj:
                found = _extract_hash_answer_only(obj.get(key), depth + 1)
                if found:
                    return found
        for value in obj.values():
            found = _extract_hash_answer_only(value, depth + 1)
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            found = _extract_hash_answer_only(item, depth + 1)
            if found:
                return found
        return None

    return None


def _extract_free_text_numeric_with_convergence(obj: Any, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 14:
        return None

    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None
        lowered = text.lower()
        convergence_signals = (
            "final answer",
            "answer is",
            "therefore",
            "thus",
            "so the answer",
            "最终答案",
            "答案是",
        )
        if not any(signal in lowered for signal in convergence_signals):
            return None
        _, token = _extract_numeric_from_text(text)
        if token:
            return token
        return None

    if isinstance(obj, dict):
        for key in ("formatted", "result", "response", "content", "message", "summary"):
            if key in obj:
                found = _extract_free_text_numeric_with_convergence(obj.get(key), depth + 1)
                if found:
                    return found
        for key in ("payload", "output", "result", "data"):
            if key in obj:
                found = _extract_free_text_numeric_with_convergence(obj.get(key), depth + 1)
                if found:
                    return found
        for value in obj.values():
            found = _extract_free_text_numeric_with_convergence(value, depth + 1)
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            found = _extract_free_text_numeric_with_convergence(item, depth + 1)
            if found:
                return found
        return None

    return None


def _extract_prediction_from_workflow_result(result: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(result, dict):
        return "", ""

    candidates: List[Tuple[str, Any]] = []

    # 1) Prefer structured tool outputs first.
    role_priority = ("refactor", "refractor", "formatter", "builder", "checker")
    _append_role_ordered_candidates(
        candidates,
        result.get("tool_exec"),
        prefix="tool_exec",
        role_order=role_priority,
    )
    _append_role_ordered_candidates(
        candidates,
        result.get("tool_trace"),
        prefix="tool_trace",
        role_order=role_priority,
    )
    _append_role_ordered_candidates(
        candidates,
        result.get("results"),
        prefix="results",
        role_order=role_priority,
    )

    execution_trace = result.get("execution_trace")
    if isinstance(execution_trace, list):
        for idx, entry in enumerate(reversed(execution_trace), start=1):
            if not isinstance(entry, dict):
                continue
            candidates.append((f"execution_trace[-{idx}].output", entry.get("output")))
            candidates.append((f"execution_trace[-{idx}].tool_history", entry.get("tool_history")))

    candidates.append(("answer", result.get("answer")))

    # Pass 1: strongly-typed structured fields (final_answer > numeric_answer).
    for source, obj in candidates:
        structured = _extract_structured_answer(obj)
        if structured:
            answer_text, answer_kind = structured
            return str(answer_text).strip(), f"{source}.{answer_kind}"

    # Pass 2: explicit #### answer format.
    for source, obj in candidates:
        hash_answer = _extract_hash_answer_only(obj)
        if hash_answer:
            return str(hash_answer).strip(), f"{source}.hash_answer"

    # Pass 3: free text numeric extraction, but only when convergence signals exist.
    for source, obj in candidates:
        free_text = _extract_free_text_numeric_with_convergence(obj)
        if free_text:
            return str(free_text).strip(), f"{source}.free_text_numeric"

    # Format completeness gate: no structured answer and no #### => unresolved.
    return "UNRESOLVED_NO_FINAL_ANSWER", "unresolved_no_final_answer"


def _extract_prediction_from_completion(completion: Any) -> Tuple[str, Optional[Decimal]]:
    if isinstance(completion, bool):
        return "", None

    raw = str(completion or "").strip()
    if not raw:
        return "", None
    if _is_blocked_answer_text(raw):
        return "", None
    hash_matches = _HASH_ANSWER_RE.findall(raw)
    if hash_matches and _is_blocked_answer_text(hash_matches[-1].strip()):
        return "", None
    candidate = _extract_answer_candidate(raw) or raw
    if _is_blocked_answer_text(candidate):
        return "", None
    value, _ = _extract_numeric_from_text(candidate)
    if value is None and candidate != raw:
        value, _ = _extract_numeric_from_text(raw)
    return candidate.strip(), value


def _normalize_text_answer(text: str) -> str:
    normalized = str(text or "").strip().lower()
    return re.sub(r"\s+", "", normalized)


def _answers_match(
    pred_value: Optional[Decimal],
    gold_value: Optional[Decimal],
    pred_text: str,
    gold_text: str,
    tolerance: Decimal = Decimal("1e-6"),
) -> Tuple[bool, str]:
    if pred_value is not None and gold_value is not None:
        diff = abs(pred_value - gold_value)
        if diff <= tolerance:
            return True, ""
        return False, f"numeric_mismatch(pred={pred_value}, gold={gold_value}, diff={diff})"

    if pred_value is None and gold_value is not None:
        return False, "missing_numeric_prediction"
    if pred_value is not None and gold_value is None:
        return False, "gold_not_numeric"

    if _normalize_text_answer(pred_text) == _normalize_text_answer(gold_text):
        return True, ""
    return False, "text_mismatch"


def _build_gsm_task_text(question: str, strict_prompt_text: Optional[str]) -> str:
    base = (
        "Solve this GSM8K math word problem.\n"
        "Return the final answer in this exact format: #### <number>\n"
        "Do not add extra content after the final answer line.\n\n"
        f"Problem: {question}\n"
    )
    if strict_prompt_text:
        return f"{base}\n{strict_prompt_text.strip()}\n"
    return base


def _ensure_gsm_default_constraints(
    constraints: Optional[Dict[str, Dict[str, Any]]],
    roles: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if constraints:
        for role, cfg in constraints.items():
            merged[str(role)] = dict(cfg or {})

    defaults_by_role = {
        "planner": ["arithmetic-word-problem"],
        "builder": ["arithmetic-word-problem"],
        "checker": ["arithmetic-word-problem"],
        "refactor": ["arithmetic-word-problem"],
    }
    role_tags_by_role = {
        "planner": ["planner", "code-planner"],
        "builder": ["builder", "code-generation"],
        "checker": ["checker", "tester", "code-testing"],
        "refactor": ["refactor", "refractor", "code-refactoring"],
    }

    for role in roles:
        role_name = str(role or "").strip()
        if not role_name:
            continue
        normalized = _normalize_role_name(role_name)
        base_domain_tags = defaults_by_role.get(normalized)
        base_role_tags = role_tags_by_role.get(normalized)
        if base_domain_tags is None:
            continue
        current = dict(merged.get(role_name) or {})
        if not current.get("domain_tags"):
            current["domain_tags"] = list(base_domain_tags)
        if not current.get("role_tags"):
            current["role_tags"] = list(base_role_tags or [])
        merged[role_name] = current
    return merged


def _auto_generate_solutions_local(
    tasks: List[dict],
    out_path: str,
    *,
    local_model_dir: str,
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt_text: Optional[str],
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
    pending = [(_task_name(task, i), task) for i, task in enumerate(tasks) if _task_name(task, i) not in solutions]
    if existing_solutions is not None:
        print(
            f"[INFO] Resume generate (local): existing={len(existing_solutions)} pending={len(pending)}",
            file=sys.stderr,
        )
    if not pending:
        return solutions

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_mode = "a" if existing_solutions is not None else "w"
    with open(out_path, write_mode, encoding="utf-8") as f:
        for row in _iter_progress([{"name": n, "task": t} for n, t in pending], desc="generate"):
            name = str(row.get("name") or "")
            item = row.get("task") if isinstance(row, dict) else {}
            question = _task_question(item)
            prompt = _build_gsm_task_text(question, strict_prompt_text)
            completion = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            completion = str(completion or "").strip()
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
    return solutions


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
    max_attempts: int,
    tool_max_rounds: int,
    local_model_dir: Optional[str],
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt_text: Optional[str],
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
    allow_unknown_roles: bool,
    no_reuse_role_selection: bool,
    reuse_same_role_agent_once: bool,
    force_role: Optional[str],
    prevent_early_finish: bool,
    existing_solutions: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    from core.registry import SQLiteRegistry
    from execution.orchestrator import run_workflow
    from generation.llm_client import LLMClient
    from generation.local_llm import LocalLLMClient
    from retrieval.embedder import build_embedder
    from retrieval.faiss_index import HNSWIndex
    supports_defer_bandit_update = "defer_bandit_update" in inspect.signature(run_workflow).parameters
    warned_defer_bandit_update_fallback = False

    local_client = None
    if local_model_dir:
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

    if code_base_url:
        os.environ["LLM_API_BASE"] = code_base_url
    if code_api_key:
        os.environ["LLM_API_KEY"] = code_api_key

    code_client = None
    if code_model or code_base_url or code_api_key:
        code_client = LLMClient(
            api_key=code_api_key or None,
            base_url=code_base_url or None,
            model=code_model or None,
            timeout_s=code_timeout,
            max_retries=code_retries,
        )

    next_role_client = None
    if next_role_model or next_role_base_url or next_role_api_key:
        next_role_client = LLMClient(
            api_key=next_role_api_key or None,
            base_url=next_role_base_url or None,
            model=next_role_model or None,
            timeout_s=next_role_timeout,
            max_retries=next_role_retries,
        )

    remote_primary_client = code_client or LLMClient(
        api_key=None,
        base_url=None,
        model=code_model or next_role_model or None,
        timeout_s=code_timeout,
        max_retries=code_retries,
    )
    remote_routing_client = next_role_client
    if remote_routing_client is None and getattr(remote_primary_client, "api_key", None):
        remote_routing_client = remote_primary_client

    if (local_client is None and code_client is None) and not tool_only:
        raise SystemExit(
            "Orchestrator generation requires either --local_model_dir or remote code client args, "
            "unless --tool_only is enabled."
        )

    index_path = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Set --index_dir to a directory containing faiss.index."
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
            f"Embedder dim {embedder.dim} != index dim {index.dim}. Rebuild index with matching embedder/model."
        )

    solutions: Dict[str, str] = dict(existing_solutions or {})
    meta: Dict[str, Dict[str, Any]] = {}
    role_exec_counter: Counter[str] = Counter()
    pending_tasks = [(_task_name(task, i), task, i) for i, task in enumerate(tasks) if _task_name(task, i) not in solutions]
    if existing_solutions is not None:
        print(
            f"[INFO] Resume generate (orchestrator): existing={len(existing_solutions)} pending={len(pending_tasks)}",
            file=sys.stderr,
        )
    if not pending_tasks:
        return solutions, meta

    effective_constraints = _ensure_gsm_default_constraints(constraints, roles)
    experiment_log_id = _build_experiment_log_id(out_path, out_path, "gsm8k")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_mode = "a" if existing_solutions is not None else "w"
    with SQLiteRegistry(db_path) as registry, open(out_path, write_mode, encoding="utf-8") as f:
        for row in _iter_progress(
            [{"name": n, "task": t, "idx": idx} for n, t, idx in pending_tasks],
            desc="generate",
        ):
            item = row.get("task") if isinstance(row, dict) else {}
            idx = int(row.get("idx") or 0) if isinstance(row, dict) else 0
            name = _task_name(item, idx)
            question = _task_question(item)
            task_text = _build_gsm_task_text(question, strict_prompt_text)

            workflow_started = time.perf_counter()
            workflow_kwargs: Dict[str, Any] = {
                "task_text": task_text,
                "roles": roles,
                "constraints_per_role": effective_constraints,
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
                "tool_max_rounds": tool_max_rounds,
                "llm_client": code_client or local_client,
                "router_llm_client": remote_routing_client or local_client,
                "router_top_m": router_top_m,
                "router_no_rerank": router_no_rerank,
                "paper_mode": True,
                "task_family": "gsm8k",
                "dataset": "gsm8k",
                "final_candidate_cap": 5,
                "artifact_contract_version": "v2",
                "dynamic_topology": dynamic_topology,
                "topology": topology,
                "topology_config": topology_config,
                "meta_router_llm_client": remote_routing_client or local_client,
                "next_role_llm_client": remote_routing_client or local_client,
                "soft_connection": soft_connection,
                "max_steps": max_steps,
                "tool_only": tool_only,
                "tool_timeout_s": tool_timeout_s,
                "mcts_dynamic_optimization": mcts_dynamic_optimization,
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
                "experimental_reranker_after_mmr": experimental_reranker_after_mmr,
                "allow_unknown_roles": allow_unknown_roles,
                "reuse_role_selection": not no_reuse_role_selection,
                "reuse_same_role_agent_once": reuse_same_role_agent_once,
                "force_role": force_role,
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

            tool_exec = result.get("tool_exec")
            if isinstance(tool_exec, dict):
                for role_name in tool_exec.keys():
                    role_exec_counter[str(role_name)] += 1

            prediction, source = _extract_prediction_from_workflow_result(result)
            completion = prediction or ""
            solutions[name] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")

            task_meta: Dict[str, Any] = {
                "prediction_source": source or "",
                "workflow_time_s": round(workflow_elapsed, 3),
                "topology": result.get("topology"),
            }
            if include_tool_trace and result.get("tool_exec") is not None:
                task_meta["tool_trace"] = json.loads(
                    json.dumps(result.get("tool_exec"), ensure_ascii=True, default=str)
                )
            raw_selections = result.get("selections")
            if isinstance(raw_selections, list):
                task_meta["_bandit_selections"] = json.loads(
                    json.dumps(raw_selections, ensure_ascii=True, default=str)
                )
            meta[name] = task_meta

    normalized_roles = {_normalize_role_name(role) for role in roles}
    if "refactor" in normalized_roles:
        refactor_exec = (
            int(role_exec_counter.get("refactor", 0))
            + int(role_exec_counter.get("refractor", 0))
            + int(role_exec_counter.get("formatter", 0))
        )
        if refactor_exec == 0:
            print(
                "[WARN] refactor/formatter role never executed in tool_exec; "
                "check routing constraints, role tags, and topology max_steps.",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] refactor/formatter execution count: {refactor_exec}",
                file=sys.stderr,
            )

    return solutions, meta


def _evaluate_tasks(
    tasks: List[dict],
    solutions: Dict[str, str],
    *,
    meta_by_task: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    results: List[dict] = []
    passed = 0
    for idx, task in enumerate(_iter_progress(tasks, desc="evaluate")):
        started = time.perf_counter()
        name = _task_name(task, idx)
        gold_text = _extract_gold_answer_text(str(task.get("answer") or ""))
        gold_value, gold_token = _extract_numeric_from_text(gold_text)

        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(name, "")
        pred_text, pred_value = _extract_prediction_from_completion(completion)

        ok, reason = _answers_match(pred_value, gold_value, pred_text, gold_text)
        if ok:
            passed += 1

        record: Dict[str, Any] = {
            "name": name,
            "ok": ok,
            "pred": pred_text,
            "gold": gold_text,
            "error": reason,
            "eval_time_s": round(time.perf_counter() - started, 3),
        }
        if pred_value is not None:
            record["pred_num"] = str(pred_value)
        if gold_value is not None:
            record["gold_num"] = str(gold_value)
        if gold_token:
            record["gold_token"] = gold_token
        if meta_by_task and name in meta_by_task:
            public_meta = {k: v for k, v in meta_by_task[name].items() if not str(k).startswith("_")}
            record.update(public_meta)
        results.append(record)

    total = len(results)
    pass_rate = (passed / total) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "accuracy": pass_rate,
        "pass_rate": pass_rate,
        "results": results,
    }


def _audit_gsm_tools(db_path: str, tools_dir: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "db_path": db_path,
        "tools_dir": tools_dir,
        "db_exists": os.path.exists(db_path),
        "tools_dir_exists": os.path.isdir(tools_dir),
        "warnings": [],
    }
    warnings: List[str] = []

    if os.path.isdir(tools_dir):
        files = [
            name
            for name in os.listdir(tools_dir)
            if name.endswith(".py") and ("gsm8k-" in name or "ratio-" in name or "age-" in name or "unit-" in name)
        ]
        summary["gsm8k_tool_files"] = len(files)
    else:
        summary["gsm8k_tool_files"] = 0
        warnings.append("generated_tools directory missing or inaccessible")

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        domain_like = '%arithmetic-word-problem%'
        cur.execute("select count(*) from cards where kind='tool' and domain_tags like ?", (domain_like,))
        summary["gsm8k_tool_cards"] = int(cur.fetchone()[0])
        cur.execute("select count(*) from cards where kind='agent' and domain_tags like ?", (domain_like,))
        summary["gsm8k_agent_cards"] = int(cur.fetchone()[0])
        cur.execute(
            "select count(*) from tool_code where id in "
            "(select id from cards where kind='tool' and domain_tags like ?)",
            (domain_like,),
        )
        summary["gsm8k_tool_code_rows"] = int(cur.fetchone()[0])
        cur.execute(
            "select stage, count(*) from cards where kind='tool' and domain_tags like ? "
            "group by stage order by stage",
            (domain_like,),
        )
        summary["stage_counts"] = {str(stage or ""): int(count or 0) for stage, count in cur.fetchall()}
        cur.execute(
            "select count(*) from tool_code "
            "where id in (select id from cards where kind='tool' and domain_tags like ? and stage='format') "
            "and code like '%keys: stage, focus, result, confidence, checks%' "
            "and code like '%payload.get(\"final_answer\")%'",
            (domain_like,),
        )
        summary["format_schema_mismatch_tools"] = int(cur.fetchone()[0])
        cur.execute(
            "select count(*) from tool_code where id in "
            "(select id from cards where kind='tool' and domain_tags like ?) "
            "and code like '%numbers[-1]%'",
            (domain_like,),
        )
        summary["fallback_last_number_tools"] = int(cur.fetchone()[0])
        conn.close()
    else:
        warnings.append("registry db missing; skipped SQL-level GSM tool audit")

    if int(summary.get("format_schema_mismatch_tools") or 0) > 0:
        warnings.append(
            "format-stage GSM tools request generic result schema but fallback formatting expects final_answer; "
            "this can force output to 0 when parsing fails"
        )
    if int(summary.get("fallback_last_number_tools") or 0) > 0:
        warnings.append(
            "many GSM tools fallback to taking the last number in question text when API call fails; "
            "accuracy can collapse if LLM_API_* env vars are unset"
        )
    if int(summary.get("gsm8k_tool_cards") or 0) == 0:
        warnings.append("no gsm8k tools registered in registry")
    if int(summary.get("gsm8k_agent_cards") or 0) == 0:
        warnings.append("no gsm8k agents registered in registry")

    summary["warnings"] = warnings
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K with standalone generation/evaluation pipeline")
    parser.add_argument("--tasks", required=True, type=str, help="GSM8K JSONL tasks path")
    parser.add_argument("--solutions", default="", type=str, help="optional JSONL solutions path")
    parser.add_argument("--out", default="", type=str, help="optional output report JSON")
    parser.add_argument("--max_samples", type=int, default=0, help="evaluate/generate first N samples")

    parser.add_argument("--auto_generate", action="store_true", help="generate predictions when missing")
    parser.add_argument("--use_orchestrator", action="store_true", help="use orchestrator-based generation")
    parser.add_argument("--resume_generate", action="store_true", help="resume from existing solutions file")
    parser.add_argument("--strict_prompt_text", type=str, default="", help="custom prompt suffix")

    parser.add_argument("--local_model_dir", type=str, default=None)
    parser.add_argument("--local_lora_dir", type=str, default=None)
    parser.add_argument("--local_device", type=str, default=None)
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_use_4bit", action="store_true")
    parser.add_argument("--local_use_8bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--code_model", type=str, default="", help="remote code model (OpenAI-compatible)")
    parser.add_argument("--code_base_url", type=str, default="", help="remote code API base URL")
    parser.add_argument("--code_api_key", type=str, default="", help="remote code API key")
    parser.add_argument("--code_timeout", type=float, default=120.0, help="remote code API timeout seconds")
    parser.add_argument("--code_retries", type=int, default=5, help="remote code API retry count")
    parser.add_argument("--next_role_model", type=str, default="", help="remote next-role model")
    parser.add_argument("--next_role_base_url", type=str, default="", help="remote next-role API base URL")
    parser.add_argument("--next_role_api_key", type=str, default="", help="remote next-role API key")
    parser.add_argument("--next_role_timeout", type=float, default=60.0)
    parser.add_argument("--next_role_retries", type=int, default=5)

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

    parser.add_argument("--roles", default="builder,checker,refactor", type=str)
    parser.add_argument("--constraints", default="", type=str, help="JSON string per role")
    parser.add_argument("--workflow_version", default="gsm8k-v1", type=str)
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
    parser.add_argument("--max_attempts", default=3, type=int)
    parser.add_argument(
        "--tool_max_rounds",
        default=3,
        type=int,
        help="max tool planning/execution rounds per role",
    )
    parser.add_argument("--allow_unknown_roles", action="store_true")
    parser.add_argument("--no_reuse_role_selection", action="store_true")
    parser.add_argument("--reuse_same_role_agent_once", action="store_true")
    parser.add_argument("--force_role", default="", type=str, help="force all queries to use this role")
    parser.add_argument("--allow_early_finish", action="store_true")
    parser.add_argument("--tool_only", action="store_true")
    parser.add_argument("--tool_timeout", type=float, default=60.0)
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
    parser.add_argument("--update_posteriors_across_instances", action="store_true")
    parser.add_argument("--use_parallel_frontier", action="store_true")
    parser.add_argument("--parallel_mode", default="frontier", type=str)
    parser.add_argument("--inference_visibility", default="tool_visible", choices=["tool_visible", "public_only"])
    parser.add_argument("--include_tool_trace", action="store_true")

    parser.add_argument("--audit_tools", action="store_true", help="audit GSM tools/cards and report issues")
    parser.add_argument("--tools_dir", type=str, default="generated_tools_real", help="directory for gsm tool files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_smart_pdf_preset(args, benchmark="gsm8k")
    if args.mcts_dynamic_optimization and not args.dynamic_topology:
        args.dynamic_topology = True
        print(
            "[INFO] --mcts_dynamic_optimization enabled: auto-enable --dynamic_topology",
            file=sys.stderr,
        )

    tasks_path = _resolve_repo_aware_path(args.tasks)
    solutions_path = _resolve_repo_aware_path(args.solutions) if args.solutions else ""
    out_path = _resolve_repo_aware_path(args.out) if args.out else ""

    args.db = _resolve_repo_aware_path(args.db, prefer_repo_if_missing=True)
    args.index_dir = _resolve_repo_aware_path(args.index_dir, prefer_repo_if_missing=True)
    args.reranker_model = _resolve_repo_aware_path(args.reranker_model, prefer_repo_if_missing=True)
    args.bandit_db = _resolve_repo_aware_path(args.bandit_db, prefer_repo_if_missing=True)
    tools_dir = _resolve_repo_aware_path(args.tools_dir, prefer_repo_if_missing=True)

    audit = _audit_gsm_tools(args.db, tools_dir) if args.audit_tools else None
    if audit:
        print(json.dumps({"gsm_tool_audit": audit}, ensure_ascii=True), file=sys.stderr)

    tasks = _load_jsonl(tasks_path)
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]

    if not tasks:
        raise SystemExit(f"No tasks loaded from: {tasks_path}")

    should_generate = args.auto_generate or bool(args.local_model_dir) or args.use_orchestrator
    if not solutions_path and out_path:
        solutions_path = str(Path(out_path).with_suffix(".solutions.jsonl"))

    solutions: Dict[str, str] = {}
    meta_by_task: Dict[str, Dict[str, Any]] = {}
    existing_solutions: Dict[str, str] = {}
    if solutions_path and os.path.exists(solutions_path):
        existing_solutions = _load_solutions(solutions_path)
        solutions = dict(existing_solutions)

    topology_config = None
    if args.topology_config:
        topology_config_path = _resolve_repo_aware_path(args.topology_config)
        if os.path.exists(topology_config_path):
            with open(topology_config_path, "r", encoding="utf-8") as f:
                topology_config = json.load(f)
        else:
            topology_config = json.loads(args.topology_config)

    if should_generate:
        if not solutions_path:
            solutions_path = tasks_path + ".solutions.jsonl"

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
            resume_seed = existing_solutions if (args.resume_generate and os.path.exists(solutions_path)) else None
            if args.use_orchestrator:
                roles = [item.strip() for item in args.roles.split(",") if item.strip()]
                normalized_roles = {_normalize_role_name(role) for role in roles}
                effective_max_steps = args.max_steps
                if (
                    "refactor" in normalized_roles
                    and effective_max_steps < 5
                    and not args.allow_early_finish
                ):
                    effective_max_steps = 5
                    print(
                        "[INFO] refactor role is enabled: auto-raise --max_steps to 5 for better closing-stage routing",
                        file=sys.stderr,
                    )
                constraints = json.loads(args.constraints) if args.constraints else {}
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
                    max_steps=effective_max_steps,
                    max_attempts=args.max_attempts,
                    tool_max_rounds=args.tool_max_rounds,
                    local_model_dir=args.local_model_dir,
                    local_lora_dir=args.local_lora_dir,
                    local_device=args.local_device,
                    local_dtype=args.local_dtype,
                    local_use_4bit=args.local_use_4bit,
                    local_use_8bit=args.local_use_8bit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    strict_prompt_text=args.strict_prompt_text or None,
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
                    allow_unknown_roles=args.allow_unknown_roles,
                    no_reuse_role_selection=args.no_reuse_role_selection,
                    reuse_same_role_agent_once=args.reuse_same_role_agent_once,
                    force_role=args.force_role or None,
                    prevent_early_finish=not args.allow_early_finish,
                    existing_solutions=resume_seed,
                )
            else:
                if not args.local_model_dir:
                    raise SystemExit("Local generation requires --local_model_dir")
                solutions = _auto_generate_solutions_local(
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
                    strict_prompt_text=args.strict_prompt_text or None,
                    existing_solutions=resume_seed,
                )
    elif solutions_path and not os.path.exists(solutions_path):
        raise SystemExit(f"Solutions file not found: {solutions_path}")
    elif solutions_path and os.path.exists(solutions_path):
        solutions = _load_solutions(solutions_path)

    report = _evaluate_tasks(tasks, solutions, meta_by_task=meta_by_task)
    if args.use_orchestrator and meta_by_task:
        from execution.orchestrator import apply_bandit_feedback

        for item in report.get("results", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            meta = meta_by_task.get(name) or {}
            selections = meta.get("_bandit_selections")
            if isinstance(selections, list):
                apply_bandit_feedback(
                    args.bandit_db,
                    workflow_version=args.workflow_version,
                    selections=selections,
                    success=bool(item.get("ok")),
                )
    if audit:
        report["gsm_tool_audit"] = audit

    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=True))

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
