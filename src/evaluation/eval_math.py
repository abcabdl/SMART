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

getcontext().prec = 60

_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_NUM_TOKEN_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?"
)
_LATEX_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_PURE_NUMERIC_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))(?:/[+-]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+)))?$"
)
_PLACEHOLDER_ANSWER_RE = re.compile(
    r"^<\s*(?:answer|number)\s*>(?:\s*\.)?(?:\s*without\b.*)?$",
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
_CONVERGENCE_SIGNALS = (
    "final answer",
    "answer is",
    "therefore",
    "thus",
    "hence",
    "so the answer",
    "最终答案",
    "答案是",
)
_CONVERGENCE_MARKERS = (
    "final answer",
    "answer is",
    "so the answer",
    "therefore",
    "thus",
    "hence",
    "最终答案",
    "答案是",
)


def _extract_equation_rhs_candidate(text: str) -> Optional[str]:
    content = str(text or "").strip()
    if not content or "=" not in content:
        return None
    rhs = content.rsplit("=", 1)[-1].strip()
    if not rhs:
        return None

    hash_matches = _HASH_ANSWER_RE.findall(rhs)
    if hash_matches:
        candidate = hash_matches[-1].strip()
        if candidate and not _is_blocked_answer_text(candidate):
            return candidate

    boxed = _extract_boxed_candidates(rhs)
    if boxed:
        candidate = boxed[-1].strip()
        if candidate and not _is_blocked_answer_text(candidate):
            return candidate

    frac_value, frac_token = _extract_latex_fraction_value(rhs)
    if frac_value is not None and frac_token:
        return frac_token

    for token in reversed([m.group(0) for m in _NUM_TOKEN_RE.finditer(rhs)]):
        if _parse_numeric_token(token) is not None:
            return token
    return None


def _extract_checks_equation_rhs(obj: Any, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 10:
        return None

    if isinstance(obj, str):
        return _extract_equation_rhs_candidate(obj)

    if isinstance(obj, dict):
        for key in ("result", "formatted", "response", "content", "message", "summary", "output"):
            if key in obj:
                found = _extract_checks_equation_rhs(obj.get(key), depth + 1)
                if found:
                    return found
        for value in obj.values():
            found = _extract_checks_equation_rhs(value, depth + 1)
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            found = _extract_checks_equation_rhs(item, depth + 1)
            if found:
                return found
        return None

    return None


def _numeric_text_conflict(lhs: str, rhs: str, tolerance: Decimal = Decimal("1e-12")) -> bool:
    left_value, _ = _extract_numeric_from_text(lhs)
    right_value, _ = _extract_numeric_from_text(rhs)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) > tolerance


def _looks_like_tool_artifact_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    if not compact:
        return False
    if any(token in compact for token in ("tool_id", "payload_source", "llm_status", "isformattedappropriately")):
        return True
    if re.search(r"math-(?:builder|checker|planner|refactor|format)-[a-z0-9-]*-v\d{3}\b", compact):
        return True
    if "usingmath-builder-format-" in compact or "usingmath-checker-format-" in compact:
        return True
    if "math-builder-format-" in compact or "math-checker-format-" in compact:
        return True
    return False


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


def _parse_list_arg(raw: Optional[str]) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


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


def _task_name(task: Dict[str, Any], idx: int) -> str:
    return str(task.get("name") or task.get("task_id") or task.get("id") or f"math-{idx}")


def _task_question(task: Dict[str, Any]) -> str:
    return str(task.get("problem") or task.get("question") or task.get("prompt") or task.get("task") or "").strip()


def _normalize_role_name(role: str) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIASES.get(key, key)


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


def _is_pure_numeric_text(text: str) -> bool:
    normalized = _normalize_math_text(text)
    if not normalized:
        return False

    s = normalized.strip()
    if s.startswith("####"):
        s = s[4:].strip()

    for _ in range(8):
        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            s = s[1:-1].strip()
            continue
        if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
            s = s[1:-1].strip()
            continue
        break

    if not s:
        return False
    if _LATEX_FRAC_RE.fullmatch(s):
        return True

    s = s.replace(",", "")
    return bool(_PURE_NUMERIC_RE.fullmatch(s))


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


def _extract_latex_fraction_value(text: str) -> Tuple[Optional[Decimal], str]:
    content = str(text or "")
    matches = list(_LATEX_FRAC_RE.finditer(content))
    for match in reversed(matches):
        num = _parse_numeric_token(match.group(1))
        den = _parse_numeric_token(match.group(2))
        if num is None or den is None:
            continue
        try:
            return num / den, match.group(0)
        except (InvalidOperation, DivisionByZero, ZeroDivisionError):
            continue
    return None, ""


def _extract_numeric_from_text(text: str) -> Tuple[Optional[Decimal], str]:
    content = str(text or "")
    if not content.strip():
        return None, ""

    hash_matches = _HASH_ANSWER_RE.findall(content)
    if hash_matches:
        candidate_text = hash_matches[-1].strip()
        frac_value, frac_token = _extract_latex_fraction_value(candidate_text)
        if frac_value is not None:
            return frac_value, frac_token
        for token in reversed([m.group(0) for m in _NUM_TOKEN_RE.finditer(candidate_text)]):
            value = _parse_numeric_token(token)
            if value is not None:
                return value, token

    frac_value, frac_token = _extract_latex_fraction_value(content)
    if frac_value is not None:
        return frac_value, frac_token

    for token in reversed([m.group(0) for m in _NUM_TOKEN_RE.finditer(content)]):
        value = _parse_numeric_token(token)
        if value is not None:
            return value, token
    return None, ""


def _is_blocked_answer_text(text: str) -> bool:
    raw = str(text or "").strip()
    normalized = raw.lower()
    if not normalized:
        return False
    if normalized in {
        "true",
        "false",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "unresolved_no_final_answer",
    }:
        return True
    if _looks_like_tool_artifact_text(raw):
        return True
    normalized = re.sub(r"\s+", " ", normalized)
    if _PLACEHOLDER_ANSWER_RE.fullmatch(normalized):
        return True
    return False


def _extract_boxed_candidates(text: str) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    out: List[str] = []
    for marker in ("\\boxed{", "\\fbox{"):
        i = 0
        while i < len(raw):
            start = raw.find(marker, i)
            if start < 0:
                break
            pos = start + len(marker)
            depth = 1
            while pos < len(raw) and depth > 0:
                ch = raw[pos]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                pos += 1
            if depth == 0:
                content = raw[start + len(marker) : pos - 1].strip()
                if content:
                    out.append(content)
            i = pos
    return out


def _normalize_math_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = re.sub(r"\{\s*,\s*\}", ",", s)
    s = s.replace("−", "-")
    s = re.sub(r"\s+", "", s)
    if s.startswith("####"):
        s = s[4:].strip()
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        if inner.count("{") == inner.count("}"):
            s = inner
    return s


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        normalized = _normalize_math_text(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _extract_gold_answer_candidates(task: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    answer_text = str(task.get("answer") or "").strip()
    if answer_text:
        boxed = _extract_boxed_candidates(answer_text)
        if boxed:
            candidates.extend(boxed)
        hash_matches = _HASH_ANSWER_RE.findall(answer_text)
        if hash_matches:
            candidates.append(hash_matches[-1].strip())
        if not boxed and not hash_matches:
            candidates.append(answer_text)

    solution_text = str(task.get("solution") or "").strip()
    if solution_text:
        boxed = _extract_boxed_candidates(solution_text)
        if boxed:
            candidates.extend(boxed)
        hash_matches = _HASH_ANSWER_RE.findall(solution_text)
        if hash_matches:
            candidates.append(hash_matches[-1].strip())
        if not boxed and not hash_matches:
            _, token = _extract_numeric_from_text(solution_text)
            if token:
                candidates.append(token)

    return _dedupe_keep_order(candidates)


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
        boxed = _extract_boxed_candidates(text)
        if boxed:
            candidate = boxed[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate
        if _is_blocked_answer_text(text):
            return None
        if re.search(r"\d|\\sqrt|\\pi|pi", text):
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
            if str(key).strip().lower() in _STRUCTURED_SKIP_KEYS:
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
        hash_matches = _HASH_ANSWER_RE.findall(text)
        if hash_matches:
            candidate = hash_matches[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate, "hash_answer"
        boxed = _extract_boxed_candidates(text)
        if boxed:
            candidate = boxed[-1].strip()
            if _is_blocked_answer_text(candidate):
                return None
            return candidate, "boxed_answer"
        if not _is_blocked_answer_text(text):
            stripped = str(text).strip().strip(" \t\r\n;,.。!?！？")
            if stripped and _is_pure_numeric_text(stripped) and _parse_numeric_token(stripped) is not None:
                return _clean_numeric_token(stripped), "numeric_text"
        return None

    if isinstance(obj, dict):
        checks_equation_rhs = _extract_checks_equation_rhs(obj.get("checks"), depth + 1)

        for key in ("final_answer", "numeric_answer", "recomputed_answer"):
            if key in obj:
                value = obj.get(key)
                if value is None or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float, Decimal)):
                    return str(value), key
                text = str(value).strip()
                if text and not _is_blocked_answer_text(text):
                    if (
                        key == "recomputed_answer"
                        and checks_equation_rhs
                        and _numeric_text_conflict(text, checks_equation_rhs)
                    ):
                        return checks_equation_rhs, "checks_equation_rhs"
                    return text, key
                found = _extract_structured_answer(
                    value,
                    depth + 1,
                    allow_numeric_scalar=True,
                )
                if found:
                    if (
                        key == "recomputed_answer"
                        and checks_equation_rhs
                        and _numeric_text_conflict(found[0], checks_equation_rhs)
                    ):
                        return checks_equation_rhs, "checks_equation_rhs"
                    return found[0], key

        for key in ("payload", "output", "result", "data"):
            if key in obj:
                found = _extract_structured_answer(
                    obj.get(key),
                    depth + 1,
                    allow_numeric_scalar=allow_numeric_scalar,
                )
                if found:
                    return found

        if checks_equation_rhs:
            return checks_equation_rhs, "checks_equation_rhs"

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
        boxed = _extract_boxed_candidates(obj)
        if boxed:
            candidate = boxed[-1].strip()
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


def _extract_free_text_answer_with_convergence(obj: Any, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 14:
        return None

    if isinstance(obj, str):
        def _strip_answer_punct(s: str) -> str:
            return str(s or "").strip().strip(" \t\r\n;,.。!?！？")

        text = obj.strip()
        if not text:
            return None
        lowered = text.lower()
        if not any(signal in lowered for signal in _CONVERGENCE_SIGNALS):
            return None

        hash_matches = _HASH_ANSWER_RE.findall(text)
        if hash_matches:
            candidate = hash_matches[-1].strip()
            if not _is_blocked_answer_text(candidate):
                return candidate
        boxed = _extract_boxed_candidates(text)
        if boxed:
            candidate = boxed[-1].strip()
            if not _is_blocked_answer_text(candidate):
                return candidate

        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            line_lowered = line.lower()
            for marker in _CONVERGENCE_MARKERS:
                marker_idx = line_lowered.find(marker)
                if marker_idx >= 0:
                    candidate = line[marker_idx + len(marker) :].strip()
                    candidate = candidate.lstrip(":：= \t").strip()
                    candidate = re.sub(
                        r"^(?:the\s+)?answer(?:\s+is)?\s*[:：=]?\s*",
                        "",
                        candidate,
                        flags=re.IGNORECASE,
                    ).strip()
                    candidate = re.sub(r"^is\s+", "", candidate, flags=re.IGNORECASE).strip()
                    if not candidate:
                        continue
                    parsed_candidate = _extract_answer_candidate(candidate) or candidate
                    parsed_candidate = _strip_answer_punct(str(parsed_candidate))
                    if parsed_candidate and not _is_blocked_answer_text(parsed_candidate):
                        numeric = _parse_numeric_token(parsed_candidate)
                        if numeric is not None:
                            return _clean_numeric_token(parsed_candidate)
                        return parsed_candidate
        return None

    if isinstance(obj, dict):
        for key in ("formatted", "result", "response", "content", "message", "summary"):
            if key in obj:
                found = _extract_free_text_answer_with_convergence(obj.get(key), depth + 1)
                if found:
                    return found
        for key in ("payload", "output", "result", "data"):
            if key in obj:
                found = _extract_free_text_answer_with_convergence(obj.get(key), depth + 1)
                if found:
                    return found
        for value in obj.values():
            found = _extract_free_text_answer_with_convergence(value, depth + 1)
            if found:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            found = _extract_free_text_answer_with_convergence(item, depth + 1)
            if found:
                return found
        return None

    return None


_MATH_STAGE_PREFERENCE = ("solve_support", "solve", "verify", "format", "model", "analyze")
_ANSWER_KIND_PREFERENCE = (
    "final_answer",
    "numeric_answer",
    "hash_answer",
    "boxed_answer",
    "free_text_answer",
    "recomputed_answer",
    "numeric_text",
    "formatted",
    "checks_equation_rhs",
)
_ANSWER_KIND_ORDER = {kind: idx for idx, kind in enumerate(_ANSWER_KIND_PREFERENCE)}


def _math_answers_equivalent(
    lhs: str,
    rhs: str,
    tolerance: Decimal = Decimal("1e-8"),
) -> bool:
    left_norm = _normalize_math_text(lhs)
    right_norm = _normalize_math_text(rhs)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    if _is_pure_numeric_text(left_norm) and _is_pure_numeric_text(right_norm):
        left_value, _ = _extract_numeric_from_text(left_norm)
        right_value, _ = _extract_numeric_from_text(right_norm)
        if left_value is not None and right_value is not None:
            return abs(left_value - right_value) <= tolerance

    return _try_symbolic_equivalence(left_norm, right_norm)


def _select_consensus_math_candidate(
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    groups: List[Dict[str, Any]] = []
    for record in records:
        placed = False
        for group in groups:
            if _math_answers_equivalent(record["answer_text"], group["canonical_answer"]):
                group["items"].append(record)
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "canonical_answer": record["answer_text"],
                    "items": [record],
                }
            )

    def _group_key(group: Dict[str, Any]) -> Tuple[int, int, int]:
        items = list(group.get("items") or [])
        first_index = min(int(item.get("index", 10**9)) for item in items) if items else 10**9
        best_kind = min(
            _ANSWER_KIND_ORDER.get(str(item.get("answer_kind") or ""), len(_ANSWER_KIND_ORDER) + 1)
            for item in items
        ) if items else len(_ANSWER_KIND_ORDER) + 1
        return (-len(items), first_index, best_kind)

    best_group = min(groups, key=_group_key)
    best_items = list(best_group.get("items") or [])
    if not best_items:
        return None
    return min(
        best_items,
        key=lambda item: (
            _ANSWER_KIND_ORDER.get(str(item.get("answer_kind") or ""), len(_ANSWER_KIND_ORDER) + 1),
            int(item.get("index", 10**9)),
        ),
    )


def _collect_math_tool_answer_records(entries: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seq = 0

    def _visit(container: Any, role_name: str = "") -> None:
        nonlocal seq
        if isinstance(container, dict):
            for nested_role, nested_entries in container.items():
                if isinstance(nested_entries, list):
                    _visit(nested_entries, str(nested_role or role_name or ""))
            return
        if not isinstance(container, list):
            return

        for entry in container:
            if not isinstance(entry, dict):
                continue
            tool_id = str(entry.get("tool_id") or "").strip()
            stage = _math_tool_stage(tool_id, entry.get("stage"))
            if not stage:
                continue
            payload = (((entry.get("result") or {}).get("output") or {}).get("output") or {})
            if not payload:
                payload = (entry.get("result") or {}).get("output") or entry.get("result") or entry

            candidate: Optional[Tuple[str, str]] = _extract_structured_answer(payload)
            if not candidate:
                hash_answer = _extract_hash_answer_only(payload)
                if hash_answer:
                    candidate = (hash_answer, "hash_answer")
            if not candidate:
                free_text = _extract_free_text_answer_with_convergence(payload)
                if free_text:
                    candidate = (free_text, "free_text_answer")
            if not candidate:
                continue

            answer_text, answer_kind = candidate
            if not answer_text or _is_blocked_answer_text(answer_text):
                continue
            records.append(
                {
                    "role": role_name,
                    "stage": stage,
                    "answer_kind": str(answer_kind),
                    "answer_text": str(answer_text).strip(),
                    "index": seq,
                }
            )
            seq += 1

    _visit(entries)
    return records


def _normalize_math_tool_stage(stage: Any) -> str:
    normalized = str(stage or "").strip().lower().replace("-", "_")
    if normalized == "solve":
        return "solve"
    if normalized in {"solve_support", "solve_supporting", "support"}:
        return "solve_support"
    if normalized in {"verify", "format", "model", "analyze"}:
        return normalized
    return ""


def _math_tool_stage(tool_id: str, stage: Any = None) -> str:
    stage_from_metadata = _normalize_math_tool_stage(stage)
    if stage_from_metadata:
        return stage_from_metadata
    parts = str(tool_id or "").strip().split("-")
    if len(parts) >= 3 and parts[0] == "math":
        return _normalize_math_tool_stage(parts[2])
    return ""


def _extract_prediction_from_math_tool_history(
    entries: Any,
) -> Optional[Tuple[str, str]]:
    records = _collect_math_tool_answer_records(entries)
    if not records:
        return None

    for stage in _MATH_STAGE_PREFERENCE:
        stage_records = [record for record in records if record.get("stage") == stage]
        best = _select_consensus_math_candidate(stage_records)
        if not best:
            continue
        source_parts = [str(best.get("role") or "").strip(), stage, str(best.get("answer_kind") or "").strip()]
        source = ".".join(part for part in source_parts if part)
        return _normalize_math_text(str(best.get("answer_text") or "").strip()), source
    return None


def _extract_prediction_from_workflow_result(result: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(result, dict):
        return "", "invalid_result"

    for source in ("tool_exec", "tool_trace"):
        tool_history_answer = _extract_prediction_from_math_tool_history(result.get(source))
        if tool_history_answer:
            answer_text, answer_kind = tool_history_answer
            return answer_text, f"{source}.{answer_kind}"

    candidates: List[Tuple[str, Any]] = []
    role_order = ("checker", "builder", "refactor", "refractor", "formatter", "planner")

    _append_role_ordered_candidates(
        candidates,
        result.get("tool_exec"),
        prefix="tool_exec",
        role_order=role_order,
    )
    _append_role_ordered_candidates(
        candidates,
        result.get("tool_trace"),
        prefix="tool_trace",
        role_order=role_order,
    )
    _append_role_ordered_candidates(
        candidates,
        result.get("results"),
        prefix="results",
        role_order=role_order,
    )

    trace = result.get("execution_trace")
    if isinstance(trace, list):
        for i, item in enumerate(reversed(trace), start=1):
            if isinstance(item, dict):
                candidates.append((f"execution_trace[-{i}].output", item.get("output")))
                candidates.append((f"execution_trace[-{i}].tool_history", item.get("tool_history")))

    candidates.append(("answer", result.get("answer")))

    for source, obj in candidates:
        tool_history_answer = _extract_prediction_from_math_tool_history(obj)
        if tool_history_answer:
            answer_text, answer_kind = tool_history_answer
            return answer_text, f"{source}.{answer_kind}"

    for source, obj in candidates:
        structured = _extract_structured_answer(obj)
        if structured:
            answer_text, answer_kind = structured
            return _normalize_math_text(str(answer_text).strip()), f"{source}.{answer_kind}"

    for source, obj in candidates:
        hash_answer = _extract_hash_answer_only(obj)
        if hash_answer:
            return _normalize_math_text(str(hash_answer).strip()), f"{source}.hash_answer"

    for source, obj in candidates:
        free_text = _extract_free_text_answer_with_convergence(obj)
        if free_text:
            return _normalize_math_text(str(free_text).strip()), f"{source}.free_text_answer"

    return "UNRESOLVED_NO_FINAL_ANSWER", "unresolved_no_final_answer"


def _extract_prediction_from_completion(completion: Any) -> Tuple[str, Optional[Decimal]]:
    if isinstance(completion, bool):
        return "", None
    if completion is None:
        return "", None
    if isinstance(completion, str):
        raw = completion.strip()
        if raw and _is_blocked_answer_text(raw):
            return _normalize_math_text(raw), None

    candidate = _extract_answer_candidate(completion)
    if not candidate:
        if isinstance(completion, str):
            candidate = _extract_free_text_answer_with_convergence(completion) or ""
        else:
            return "", None
    elif isinstance(completion, str) and str(candidate).strip() == completion.strip():
        rhs_candidate = _extract_equation_rhs_candidate(completion)
        if rhs_candidate:
            candidate = rhs_candidate
        else:
            free_text = _extract_free_text_answer_with_convergence(completion)
            if free_text:
                candidate = free_text

    candidate = str(candidate or "").strip()
    if not candidate:
        return "", None
    if _is_blocked_answer_text(candidate):
        return _normalize_math_text(candidate), None

    normalized = _normalize_math_text(candidate)
    value, _ = _extract_numeric_from_text(candidate)
    if value is None:
        value, _ = _extract_numeric_from_text(normalized)
    return normalized, value


def _prediction_is_resolved(prediction: str, source: str) -> bool:
    text = str(prediction or "").strip()
    if not text:
        return False
    if text == "UNRESOLVED_NO_FINAL_ANSWER":
        return False
    source_norm = str(source or "").strip().lower()
    if source_norm == "unresolved_no_final_answer":
        return False
    if _is_blocked_answer_text(text):
        return False
    extracted_text, _ = _extract_prediction_from_completion(text)
    if not extracted_text:
        return False
    if _is_blocked_answer_text(extracted_text):
        return False
    return True


def _latex_to_sympy(expr: str) -> str:
    text = str(expr or "")
    if not text:
        return ""
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    text = text.replace("\\pi", "pi")

    def _replace_frac_once(s: str) -> Tuple[str, bool]:
        key = "\\frac{"
        idx = s.find(key)
        if idx < 0:
            return s, False
        num_start = idx + len("\\frac")
        if num_start >= len(s) or s[num_start] != "{":
            return s, False
        n_content, n_end = _extract_braced_content(s, num_start)
        if n_end >= len(s) or s[n_end] != "{":
            return s, False
        d_content, d_end = _extract_braced_content(s, n_end)
        replaced = s[:idx] + f"(({n_content})/({d_content}))" + s[d_end:]
        return replaced, True

    def _replace_sqrt_once(s: str) -> Tuple[str, bool]:
        key = "\\sqrt{"
        idx = s.find(key)
        if idx < 0:
            return s, False
        arg_start = idx + len("\\sqrt")
        if arg_start >= len(s) or s[arg_start] != "{":
            return s, False
        arg_content, arg_end = _extract_braced_content(s, arg_start)
        replaced = s[:idx] + f"sqrt({arg_content})" + s[arg_end:]
        return replaced, True

    for _ in range(24):
        text, changed = _replace_frac_once(text)
        if not changed:
            break
    for _ in range(24):
        text, changed = _replace_sqrt_once(text)
        if not changed:
            break

    text = text.replace("^", "**")
    text = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", text)
    text = re.sub(r"([a-zA-Z)])(\d)", r"\1*\2", text)
    text = re.sub(r"\s+", "", text)
    return text


def _extract_braced_content(text: str, start_brace_idx: int) -> Tuple[str, int]:
    if start_brace_idx >= len(text) or text[start_brace_idx] != "{":
        return "", start_brace_idx
    depth = 1
    i = start_brace_idx + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return text[start_brace_idx + 1 :], len(text)
    return text[start_brace_idx + 1 : i - 1], i


def _try_symbolic_equivalence(pred_text: str, gold_text: str) -> bool:
    try:
        from sympy import Symbol, simplify  # type: ignore
        from sympy.parsing.sympy_parser import (  # type: ignore
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return False

    p = _latex_to_sympy(pred_text)
    g = _latex_to_sympy(gold_text)
    if not p or not g:
        return False

    transformations = standard_transformations + (implicit_multiplication_application,)
    local_dict = {
        "x": Symbol("x"),
        "y": Symbol("y"),
        "z": Symbol("z"),
    }
    try:
        p_expr = parse_expr(p, transformations=transformations, local_dict=local_dict)
        g_expr = parse_expr(g, transformations=transformations, local_dict=local_dict)
        return bool(simplify(p_expr - g_expr) == 0)
    except Exception:
        return False


def _answers_match(
    pred_text: str,
    pred_value: Optional[Decimal],
    gold_candidates: List[str],
    tolerance: Decimal = Decimal("1e-8"),
) -> Tuple[bool, str]:
    if not pred_text:
        return False, "empty_prediction"
    if not gold_candidates:
        return False, "missing_gold_answer"

    pred_norm = _normalize_math_text(pred_text)
    if _is_blocked_answer_text(pred_norm):
        return False, "blocked_prediction"

    pred_is_numeric = _is_pure_numeric_text(pred_norm)
    pred_scalar_value: Optional[Decimal] = None
    if pred_is_numeric:
        pred_scalar_value, _ = _extract_numeric_from_text(pred_norm)
        if pred_scalar_value is None:
            pred_scalar_value = pred_value

    for gold in gold_candidates:
        gold_norm = _normalize_math_text(gold)
        if pred_norm == gold_norm:
            return True, "text_exact"

        if pred_is_numeric and pred_scalar_value is not None and _is_pure_numeric_text(gold_norm):
            gold_value, _ = _extract_numeric_from_text(gold_norm)
            if gold_value is not None and abs(pred_scalar_value - gold_value) <= tolerance:
                return True, "numeric_close"

        if _try_symbolic_equivalence(pred_norm, gold_norm):
            return True, "symbolic_equivalent"

    return False, "no_candidate_matched"


def _build_math_task_text(question: str, strict_prompt_text: Optional[str]) -> str:
    base = (
        "Solve this MATH benchmark problem.\n"
        "Return the final answer in this exact format: #### <answer>\n"
        "The answer may be an integer, fraction, decimal, or symbolic expression.\n"
        "Do not add extra content after the final answer line.\n\n"
        f"Problem: {question}\n"
    )
    if strict_prompt_text:
        return f"{base}\n{strict_prompt_text.strip()}\n"
    return base


def _ensure_math_default_constraints(
    constraints: Optional[Dict[str, Dict[str, Any]]],
    roles: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if constraints:
        for role, cfg in constraints.items():
            merged[str(role)] = dict(cfg or {})

    defaults_by_role = {
        "planner": ["exact-competition"],
        "builder": ["exact-competition"],
        "checker": ["exact-competition"],
        "refactor": ["exact-competition"],
    }
    role_tags_by_role = {
        "planner": ["planner", "code-planner"],
        "builder": ["builder", "code-generation"],
        "checker": ["checker", "tester", "code-testing"],
        "refactor": ["refactor", "refractor", "formatter", "code-refactoring"],
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
    local_use: Optional[str],
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
            prompt = _build_math_task_text(question, strict_prompt_text)
            completion = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            completion = str(completion or "").strip()
            solutions[name] = completion
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
    local_use: Optional[str],
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
    workflow_max_runs: int,
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

    local_use_set = {item.lower() for item in _parse_list_arg(local_use)}
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

    remote_primary_client = code_client
    if remote_primary_client is None:
        remote_api_key = code_api_key or next_role_api_key or None
        remote_base_url = code_base_url or next_role_base_url or None
        remote_model = code_model or next_role_model or None
        if remote_model or remote_base_url or remote_api_key:
            remote_primary_client = LLMClient(
                api_key=remote_api_key,
                base_url=remote_base_url,
                model=remote_model,
                timeout_s=code_timeout,
                max_retries=code_retries,
            )
    remote_routing_client = next_role_client
    if remote_routing_client is None and remote_primary_client is not None and getattr(remote_primary_client, "api_key", None):
        remote_routing_client = remote_primary_client

    if local_use_set and local_client is None:
        raise SystemExit("local_use specified but --local_model_dir is missing.")

    if local_use_set and remote_primary_client is None and "all" not in local_use_set and "main" not in local_use_set and not tool_only:
        raise SystemExit(
            "local_use excludes main, but no remote primary client is configured. "
            "Please set --code_model/--code_base_url/--code_api_key or reuse "
            "--next_role_model/--next_role_base_url/--next_role_api_key."
        )

    if (local_client is None and remote_primary_client is None) and not tool_only:
        raise SystemExit(
            "Orchestrator generation requires either --local_model_dir or a remote client "
            "(--code_* or --next_role_*), unless --tool_only is enabled."
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
        raise ValueError(f"Embedder dim {embedder.dim} != index dim {index.dim}. Rebuild index with matching dim.")

    def _completion_is_resolved(completion: Any) -> bool:
        pred_text, _ = _extract_prediction_from_completion(completion)
        if not pred_text:
            return False
        if _is_blocked_answer_text(pred_text):
            return False
        return True

    solutions: Dict[str, str] = dict(existing_solutions or {})
    meta: Dict[str, Dict[str, Any]] = {}
    role_exec_counter: Counter[str] = Counter()
    pending_tasks = [
        (_task_name(task, i), task, i)
        for i, task in enumerate(tasks)
        if (_task_name(task, i) not in solutions) or (not _completion_is_resolved(solutions.get(_task_name(task, i), "")))
    ]
    if existing_solutions is not None:
        usable = sum(1 for v in existing_solutions.values() if _completion_is_resolved(v))
        print(
            f"[INFO] Resume generate (orchestrator): existing={len(existing_solutions)} usable={usable} pending={len(pending_tasks)}",
            file=sys.stderr,
        )
    if not pending_tasks:
        return solutions, meta

    effective_constraints = _ensure_math_default_constraints(constraints, roles)
    experiment_log_id = _build_experiment_log_id(out_path, out_path, "math")

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
            task_text = _build_math_task_text(question, strict_prompt_text)

            best_prediction = "UNRESOLVED_NO_FINAL_ANSWER"
            best_source = "unresolved_no_final_answer"
            best_result: Dict[str, Any] = {}
            total_elapsed = 0.0
            run_count = 0
            max_runs = max(1, int(workflow_max_runs))

            for run_idx in range(1, max_runs + 1):
                run_count = run_idx
                started = time.perf_counter()
                if local_use_set:
                    effective_llm_client = local_client if ("all" in local_use_set or "main" in local_use_set) else remote_primary_client
                    effective_router_llm_client = local_client if ("all" in local_use_set or "router" in local_use_set) else (remote_routing_client or local_client)
                    effective_next_role_llm_client = local_client if ("all" in local_use_set or "next_role" in local_use_set) else (remote_routing_client or local_client)
                    effective_meta_router_llm_client = local_client if ("all" in local_use_set or "meta_router" in local_use_set) else (remote_routing_client or local_client)
                else:
                    effective_llm_client = code_client or local_client
                    effective_router_llm_client = remote_routing_client or local_client
                    effective_next_role_llm_client = remote_routing_client or local_client
                    effective_meta_router_llm_client = remote_routing_client or local_client
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
                    "llm_client": effective_llm_client,
                    "router_llm_client": effective_router_llm_client,
                    "router_top_m": router_top_m,
                    "router_no_rerank": router_no_rerank,
                    "paper_mode": True,
                    "task_family": "math",
                    "dataset": "math",
                    "final_candidate_cap": 5,
                    "artifact_contract_version": "v2",
                    "dynamic_topology": dynamic_topology,
                    "topology": topology,
                    "topology_config": topology_config,
                    "meta_router_llm_client": effective_meta_router_llm_client,
                    "next_role_llm_client": effective_next_role_llm_client,
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
                elapsed = time.perf_counter() - started
                total_elapsed += elapsed

                tool_exec = result.get("tool_exec")
                if isinstance(tool_exec, dict):
                    for role_name in tool_exec.keys():
                        role_exec_counter[str(role_name)] += 1

                prediction, source = _extract_prediction_from_workflow_result(result)
                resolved = _prediction_is_resolved(prediction, source)

                if resolved or not _prediction_is_resolved(best_prediction, best_source):
                    best_prediction = prediction or ""
                    best_source = source or ""
                    best_result = result

                if resolved:
                    break

            best_resolved = _prediction_is_resolved(best_prediction, best_source)
            if not best_resolved:
                preview = str(best_prediction or "").strip().replace("\n", " ")
                if len(preview) > 180:
                    preview = preview[:177] + "..."
                print(
                    f"[WARN] Unresolved final answer: task={name} runs={run_count} time_s={round(total_elapsed, 3)} "
                    f"best_source={best_source!s} best_prediction={preview!r}",
                    file=sys.stderr,
                )
            completion = (best_prediction or "") if best_resolved else "UNRESOLVED_NO_FINAL_ANSWER"
            solutions[name] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")

            task_meta: Dict[str, Any] = {
                "prediction_source": best_source or "",
                "prediction_resolved": best_resolved,
                "workflow_runs": run_count,
                "workflow_time_s": round(total_elapsed, 3),
                "topology": best_result.get("topology"),
            }
            if not best_resolved and best_prediction and best_prediction != "UNRESOLVED_NO_FINAL_ANSWER":
                task_meta["_best_unresolved_prediction"] = best_prediction
            if include_tool_trace and best_result.get("tool_exec") is not None:
                task_meta["tool_trace"] = json.loads(
                    json.dumps(best_result.get("tool_exec"), ensure_ascii=True, default=str)
                )
            raw_selections = best_result.get("selections")
            if isinstance(raw_selections, list):
                task_meta["_bandit_selections"] = json.loads(
                    json.dumps(raw_selections, ensure_ascii=True, default=str)
                )
            meta[name] = task_meta

    if "refactor" in {_normalize_role_name(role) for role in roles}:
        refactor_exec = (
            int(role_exec_counter.get("refactor", 0))
            + int(role_exec_counter.get("refractor", 0))
            + int(role_exec_counter.get("formatter", 0))
        )
        if refactor_exec == 0:
            print(
                "[WARN] refactor/formatter role never executed in tool_exec; check role constraints and topology.",
                file=sys.stderr,
            )
        else:
            print(f"[INFO] refactor/formatter execution count: {refactor_exec}", file=sys.stderr)

    return solutions, meta


def _evaluate_tasks(
    tasks: List[dict],
    solutions: Dict[str, str],
    *,
    meta_by_task: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    results: List[dict] = []
    passed = 0
    reason_counter: Counter[str] = Counter()

    for idx, task in enumerate(_iter_progress(tasks, desc="evaluate")):
        started = time.perf_counter()
        name = _task_name(task, idx)
        gold_candidates = _extract_gold_answer_candidates(task)

        completion: Any = task.get("completion")
        if completion is None:
            completion = solutions.get(name, "")
        pred_text, pred_value = _extract_prediction_from_completion(completion)
        ok, reason = _answers_match(pred_text, pred_value, gold_candidates)
        reason_counter[reason] += 1
        if ok:
            passed += 1

        record: Dict[str, Any] = {
            "name": name,
            "ok": ok,
            "pred": pred_text,
            "gold_candidates": gold_candidates,
            "error": reason if not ok else "",
            "eval_time_s": round(time.perf_counter() - started, 3),
        }
        if pred_value is not None:
            record["pred_num"] = str(pred_value)
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
        "reason_stats": dict(reason_counter),
        "results": results,
    }


def _audit_math_tools(db_path: str, tools_dir: str) -> Dict[str, Any]:
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
            if name.endswith(".py")
            and (
                name.startswith("math-")
                or "counting-" in name
                or "equation-" in name
                or "geometry-" in name
                or "sequence-" in name
            )
        ]
        summary["math_tool_files"] = len(files)
    else:
        summary["math_tool_files"] = 0
        warnings.append("generated_tools directory missing or inaccessible")

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        domain_like = '%exact-competition%'
        cur.execute("select count(*) from cards where kind='tool' and domain_tags like ?", (domain_like,))
        summary["math_tool_cards"] = int(cur.fetchone()[0])
        cur.execute("select count(*) from cards where kind='agent' and domain_tags like ?", (domain_like,))
        summary["math_agent_cards"] = int(cur.fetchone()[0])
        cur.execute(
            "select count(*) from tool_code where id in "
            "(select id from cards where kind='tool' and domain_tags like ?)",
            (domain_like,),
        )
        summary["math_tool_code_rows"] = int(cur.fetchone()[0])
        cur.execute(
            "select stage, count(*) from cards where kind='tool' and domain_tags like ? "
            "group by stage order by stage",
            (domain_like,),
        )
        summary["stage_counts"] = {str(stage or ""): int(count or 0) for stage, count in cur.fetchall()}
        conn.close()
    else:
        warnings.append("registry db missing; skipped SQL-level MATH tool audit")

    if int(summary.get("math_tool_cards") or 0) == 0:
        warnings.append("no math tools registered in registry")
    if int(summary.get("math_agent_cards") or 0) == 0:
        warnings.append("no math agents registered in registry")
    summary["warnings"] = warnings
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MATH dataset with standalone generation/evaluation pipeline")
    parser.add_argument("--tasks", required=True, type=str, help="MATH JSONL tasks path")
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
    parser.add_argument(
        "--local_use",
        type=str,
        default="",
        help="comma-separated local binding targets: main,router,meta_router,next_role,all",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--code_model", type=str, default="", help="remote code model (OpenAI-compatible)")
    parser.add_argument("--code_base_url", type=str, default="", help="remote code API base URL")
    parser.add_argument("--code_api_key", type=str, default="", help="remote code API key")
    parser.add_argument("--code_timeout", type=float, default=120.0)
    parser.add_argument("--code_retries", type=int, default=5)
    parser.add_argument("--next_role_model", type=str, default="", help="remote next-role model")
    parser.add_argument("--next_role_base_url", type=str, default="", help="remote next-role API base URL")
    parser.add_argument("--next_role_api_key", type=str, default="", help="remote next-role API key")
    parser.add_argument("--next_role_timeout", type=float, default=60.0)
    parser.add_argument("--next_role_retries", type=int, default=5)

    parser.add_argument("--db", default="demo_registry.sqlite", type=str)
    parser.add_argument("--index_dir", default="index", type=str)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--embedder", default="sentence-transformer", type=str)
    parser.add_argument("--embedder_model", default="sentence-transformers/all-MiniLM-L6-v2", type=str)
    parser.add_argument("--embedder_device", default=None, type=str)
    parser.add_argument("--embedder_normalize", action="store_true")

    parser.add_argument("--roles", default="builder,checker", type=str)
    parser.add_argument("--constraints", default="", type=str, help="JSON string per role")
    parser.add_argument("--workflow_version", default="math-v1", type=str)
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
    parser.add_argument("--tool_max_rounds", default=3, type=int)
    parser.add_argument("--allow_unknown_roles", action="store_true")
    parser.add_argument("--no_reuse_role_selection", action="store_true")
    parser.add_argument("--reuse_same_role_agent_once", action="store_true")
    parser.add_argument("--force_role", default="", type=str)
    parser.add_argument("--allow_early_finish", action="store_true")
    parser.add_argument(
        "--strict_workflow_completion",
        action="store_true",
        help="require workflow success signals before finishing; slower and mainly useful for debugging",
    )
    parser.add_argument("--tool_only", action="store_true")
    parser.add_argument("--tool_timeout", type=float, default=60.0)
    parser.add_argument(
        "--workflow_max_runs",
        type=int,
        default=1,
        help="max workflow reruns per task; stop early once a resolved final answer is extracted",
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
    parser.add_argument("--update_posteriors_across_instances", action="store_true")
    parser.add_argument("--use_parallel_frontier", action="store_true")
    parser.add_argument("--parallel_mode", default="frontier", type=str)
    parser.add_argument("--inference_visibility", default="tool_visible", choices=["tool_visible", "public_only"])
    parser.add_argument("--include_tool_trace", action="store_true")

    parser.add_argument("--audit_tools", action="store_true", help="audit MATH tools/cards and report issues")
    parser.add_argument("--tools_dir", type=str, default="generated_tools_real", help="directory for math tool files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_smart_pdf_preset(args, benchmark="math")

    if args.mcts_dynamic_optimization and not args.dynamic_topology:
        args.dynamic_topology = True
        print(
            "[INFO] --mcts_dynamic_optimization enabled: auto-enable --dynamic_topology",
            file=sys.stderr,
        )

    if args.use_orchestrator and not args.strict_workflow_completion:
        if not args.allow_early_finish:
            print(
                "[INFO] eval_math defaults to early finish once a candidate final answer is found; "
                "use --strict_workflow_completion to disable this.",
                file=sys.stderr,
            )
        args.allow_early_finish = True

    tasks_path = _resolve_repo_aware_path(args.tasks)
    solutions_path = _resolve_repo_aware_path(args.solutions) if args.solutions else ""
    out_path = _resolve_repo_aware_path(args.out) if args.out else ""

    args.db = _resolve_repo_aware_path(args.db, prefer_repo_if_missing=True)
    args.index_dir = _resolve_repo_aware_path(args.index_dir, prefer_repo_if_missing=True)
    args.reranker_model = _resolve_repo_aware_path(args.reranker_model, prefer_repo_if_missing=True)
    args.bandit_db = _resolve_repo_aware_path(args.bandit_db, prefer_repo_if_missing=True)
    tools_dir = _resolve_repo_aware_path(args.tools_dir, prefer_repo_if_missing=True)

    audit = _audit_math_tools(args.db, tools_dir) if args.audit_tools else None
    if audit:
        print(json.dumps({"math_tool_audit": audit}, ensure_ascii=True), file=sys.stderr)

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
                    and args.strict_workflow_completion
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
                    local_use=args.local_use or None,
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
                    prevent_early_finish=args.strict_workflow_completion,
                    workflow_max_runs=args.workflow_max_runs,
                    existing_solutions=resume_seed,
                )
            else:
                if not args.local_model_dir:
                    raise SystemExit("--auto_generate local mode requires --local_model_dir")
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

    report = _evaluate_tasks(tasks, solutions, meta_by_task=meta_by_task or None)

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

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        print(f"[INFO] Wrote report: {out_path}", file=sys.stderr)

    print(json.dumps({k: report[k] for k in ("total", "passed", "accuracy", "pass_rate")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
