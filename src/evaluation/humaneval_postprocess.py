from __future__ import annotations

import ast
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_POSTPROCESS_AGENT_NAME = "postprocess_tool_agent"
_POSTPROCESS_AGENT_ID = "special-postprocess-agent"
_POSTPROCESS_TOOL_IDS = [
    "postprocess-fallback-guard",
    "postprocess-extract-cleanup",
    "postprocess-stop-token-trim",
    "postprocess-normalize-body",
    "postprocess-param-consistency",
    "postprocess-builtin-shadowing",
    "postprocess-syntax-indent-repair",
    "postprocess-name-scope-repair",
    "postprocess-timeout-guard",
    "postprocess-fallback-guard",
]
_POSTPROCESS_REGISTRY = None
_POSTPROCESS_EXECUTOR = None
_POSTPROCESS_DB_PATH: Optional[str] = None
_POSTPROCESS_TIMEOUT_S: Optional[float] = None
_CHAT_ROLE_LINE_RE = re.compile(r"^\s*(?:assistant|user|system|tool)\s*:?\s*$", re.IGNORECASE)
_CHAT_ROLE_TAG_RE = re.compile(r"^\s*</?(?:assistant|user|system|tool)\s*>\s*$", re.IGNORECASE)
_CHAT_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|user|system|tool)\s*:\s*(.*)$", re.IGNORECASE)
_BODY_BLOCK_OPENERS_RE = re.compile(
    r"^(?:if|elif|else|for|while|try|except|finally|with|def|class|match|case)\b.*:\s*$"
)
_BODY_CLAUSE_PREFIXES = ("elif ", "else:", "except", "finally:", "case ")


def _looks_like_name_error(message: Optional[str]) -> bool:
    text = str(message or "")
    return "NameError" in text


def _looks_like_unbound_local_error(message: Optional[str]) -> bool:
    text = str(message or "")
    return "UnboundLocalError" in text


def _looks_like_syntax_error(message: Optional[str]) -> bool:
    text = str(message or "")
    return any(token in text for token in ("SyntaxError", "IndentationError", "TabError"))


def _extract_missing_name_from_error(message: Optional[str]) -> Optional[str]:
    text = str(message or "")
    match = re.search(r"NameError:\s*name\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+is\s+not\s+defined", text)
    if match:
        return match.group(1)
    return None


def _normalize_identifier(name: str) -> str:
    return re.sub(r"_+", "", str(name or "").strip().lower())


def _identifier_relation_score(missing_name: str, param_name: str) -> float:
    missing_raw = str(missing_name or "").strip()
    param_raw = str(param_name or "").strip()
    if not missing_raw or not param_raw:
        return 0.0

    missing = _normalize_identifier(missing_raw)
    param = _normalize_identifier(param_raw)
    if not missing or not param:
        return 0.0

    if missing == param:
        return 1.0
    if param.startswith(missing) and len(missing) >= 3:
        return 0.95
    if missing.startswith(param) and len(param) >= 2:
        return 0.90

    alias_groups = [
        {"n", "num", "number", "val", "value", "x"},
        {"s", "str", "string", "text", "txt", "sentence", "word"},
        {"arr", "array", "list", "lst", "nums", "values", "items", "seq", "sequence", "grid"},
        {"i", "idx", "index", "pos"},
    ]
    for group in alias_groups:
        if missing in group and param in group:
            return 0.90

    return difflib.SequenceMatcher(a=missing, b=param).ratio()


def _best_param_match(missing_name: str, param_names: List[str]) -> Optional[str]:
    candidates = [str(p).strip() for p in param_names if str(p).strip()]
    if not missing_name or not candidates:
        return None

    scored = sorted(
        ((name, _identifier_relation_score(missing_name, name)) for name in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    best_name, best_score = scored[0]
    if best_score < 0.90:
        return None
    if len(scored) > 1 and scored[1][1] >= 0.90 and (best_score - scored[1][1]) < 0.05:
        # Ambiguous mapping (e.g. lst -> lst1/lst2), avoid risky replacement.
        return None
    return best_name


def _should_use_param_consistency(error_message: Optional[str], param_names: List[str]) -> bool:
    if not _looks_like_name_error(error_message):
        return False
    missing_name = _extract_missing_name_from_error(error_message)
    if not missing_name:
        return False
    return _best_param_match(missing_name, param_names) is not None


def _move_tool_before(tools: List[str], tool_id: str, before_tool_id: str) -> None:
    if tool_id not in tools or before_tool_id not in tools:
        return
    tools.remove(tool_id)
    idx = tools.index(before_tool_id)
    tools.insert(idx, tool_id)


def _select_postprocess_tools(error_message: Optional[str], param_names: List[str]) -> List[str]:
    tools = list(_POSTPROCESS_TOOL_IDS)
    is_name_resolution_error = _looks_like_name_error(error_message) or _looks_like_unbound_local_error(error_message)
    if _looks_like_assertion_error(str(error_message or "")):
        tools = [
            tool
            for tool in tools
            if tool not in {"postprocess-param-consistency", "postprocess-name-scope-repair"}
        ]
    if _looks_like_syntax_error(error_message):
        _move_tool_before(tools, "postprocess-syntax-indent-repair", "postprocess-param-consistency")
    if is_name_resolution_error:
        _move_tool_before(tools, "postprocess-name-scope-repair", "postprocess-param-consistency")
    else:
        tools = [tool for tool in tools if tool != "postprocess-name-scope-repair"]
    if not _should_use_param_consistency(error_message, param_names):
        tools = [tool for tool in tools if tool != "postprocess-param-consistency"]
    return tools

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
    """Unwrap code from JSON string format like '{"code_or_commands": "..."}'."""
    if not text or "{" not in text:
        return text
    try:
        # Try to find and parse JSON structure
        start_idx = text.find("{")
        if start_idx == -1:
            return text
        # Extract potential JSON
        json_text = text[start_idx:]
        # Try to parse as JSON
        data = json.loads(json_text)
        if isinstance(data, dict):
            # Look for code in common keys
            for key in ["code_or_commands", "code", "solution", "output"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and value.strip():
                        return value
                    elif isinstance(value, dict):
                        # Nested structure
                        for nested_key in ["code_or_commands", "code", "solution"]:
                            if nested_key in value and isinstance(value[nested_key], str):
                                return value[nested_key]
    except (json.JSONDecodeError, ValueError):
        pass
    return text


def _extract_code_block(text: str) -> str:
    if not text:
        return text
    
    # First try to unwrap JSON-formatted code
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
    target_pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    lines = text.splitlines()

    def _is_safe_prefix_line(raw_line: str) -> bool:
        stripped = raw_line.strip()
        if not stripped:
            return False
        if stripped.startswith("import ") or stripped.startswith("from "):
            return True
        # Keep simple top-level constants used by function body.
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped))

    for idx, line in enumerate(lines):
        if target_pattern.match(line):
            body_lines = lines[idx + 1 :]
            if not body_lines:
                return text
            prefix_lines = [ln.lstrip() for ln in lines[:idx] if _is_safe_prefix_line(ln)]
            merged_lines: List[str] = []
            if prefix_lines:
                merged_lines.extend(prefix_lines)
                merged_lines.append("")
            merged_lines.extend(body_lines)
            return "\n".join(merged_lines).lstrip("\n")
    return text


def _indent_level(raw_line: str) -> int:
    stripped = raw_line.lstrip()
    lead = len(raw_line) - len(stripped)
    return max(0, lead // 4)


def _is_clause_line(stripped: str) -> bool:
    return any(stripped.startswith(prefix) for prefix in _BODY_CLAUSE_PREFIXES)


def _clause_family(stripped: str) -> Optional[str]:
    if stripped.startswith(("if ", "elif ", "else:")):
        return "if"
    if stripped.startswith(("try:", "except", "finally:")):
        return "try"
    if stripped.startswith(("match ", "case ")):
        return "match"
    return None


def _is_block_opener(stripped: str) -> bool:
    return bool(_BODY_BLOCK_OPENERS_RE.match(stripped))


def _is_exit_line_global(stripped: str) -> bool:
    return any(stripped.startswith(prefix) for prefix in ("return", "raise", "break", "continue", "pass"))


def _seed_flattened_control_bodies(lines: List[str]) -> List[str]:
    adjusted = list(lines)
    idx = 0
    while idx < len(adjusted):
        stripped = adjusted[idx].lstrip()
        if not stripped or not _is_block_opener(stripped):
            idx += 1
            continue
        cur_level = _indent_level(adjusted[idx])
        next_idx = idx + 1
        while next_idx < len(adjusted) and not adjusted[next_idx].strip():
            next_idx += 1
        if next_idx >= len(adjusted):
            break
        next_line = adjusted[next_idx]
        next_level = _indent_level(next_line)
        if next_level <= cur_level:
            adjusted[next_idx] = (" " * 4 * (cur_level + 1)) + next_line.lstrip()
        idx += 1
    return adjusted


def _align_clause_indentation(lines: List[str]) -> List[str]:
    adjusted = list(lines)
    openers: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(adjusted):
        raw = adjusted[idx]
        stripped = raw.lstrip()
        if not stripped:
            idx += 1
            continue

        level = _indent_level(raw)
        is_clause = _is_clause_line(stripped)
        family = _clause_family(stripped)
        if is_clause and family:
            target_level = None
            for opener in reversed(openers):
                if opener.get("family") == family:
                    target_level = int(opener["level"])
                    break
            if target_level is not None and target_level != level:
                delta = target_level - level
                suite_idx = idx
                while suite_idx < len(adjusted):
                    suite_raw = adjusted[suite_idx]
                    suite_stripped = suite_raw.lstrip()
                    if not suite_stripped:
                        suite_idx += 1
                        continue
                    suite_level = _indent_level(suite_raw)
                    if suite_idx > idx:
                        if suite_level < level:
                            break
                        if suite_level == level and not _is_clause_line(suite_stripped):
                            break
                    new_level = max(0, suite_level + delta)
                    adjusted[suite_idx] = (" " * 4 * new_level) + suite_stripped
                    suite_idx += 1
                raw = adjusted[idx]
                stripped = raw.lstrip()
                level = _indent_level(raw)

        if not is_clause:
            while openers and int(openers[-1]["level"]) >= level:
                openers.pop()

        if _is_block_opener(stripped) and not is_clause:
            opener_family = _clause_family(stripped)
            if opener_family:
                openers.append({"family": opener_family, "level": level})

        idx += 1
    return adjusted


def _dedent_trailing_loop_exits(lines: List[str]) -> List[str]:
    adjusted = list(lines)
    for idx, raw in enumerate(adjusted):
        stripped = raw.lstrip()
        if not stripped.startswith(("for ", "while ")):
            continue
        level = _indent_level(raw)
        suite_indices: List[int] = []
        for suite_idx in range(idx + 1, len(adjusted)):
            suite_raw = adjusted[suite_idx]
            suite_stripped = suite_raw.lstrip()
            if not suite_stripped:
                continue
            suite_level = _indent_level(suite_raw)
            if suite_level <= level:
                break
            suite_indices.append(suite_idx)
        if not suite_indices:
            continue
        last_idx = suite_indices[-1]
        last_stripped = adjusted[last_idx].lstrip()
        last_level = _indent_level(adjusted[last_idx])
        if not _is_exit_line_global(last_stripped) or last_level != (level + 1):
            continue
        prior_non_exit_same_level = any(
            _indent_level(adjusted[suite_idx]) == (level + 1)
            and not _is_exit_line_global(adjusted[suite_idx].lstrip())
            for suite_idx in suite_indices[:-1]
            if adjusted[suite_idx].strip()
        )
        if prior_non_exit_same_level:
            adjusted[last_idx] = (" " * 4 * level) + last_stripped
    return adjusted


def _normalize_body_indentation(lines: List[str]) -> List[str]:
    """
    Normalize body indentation levels for extracted snippets.
    This repairs cases like:
        # comment
            if ...
    where an `unexpected indent` would occur in a function body.
    """
    exit_prefixes = ("return", "raise", "break", "continue", "pass")

    def _raw_level(raw_line: str) -> int:
        return _indent_level(raw_line)

    def _is_exit_line(stripped: str) -> bool:
        return any(stripped.startswith(prefix) for prefix in exit_prefixes)

    def _normalize_with_search() -> Optional[List[str]]:
        significant = []
        for idx, raw in enumerate(lines):
            stripped = raw.lstrip()
            if not stripped:
                significant.append(
                    {
                        "idx": idx,
                        "raw": raw,
                        "stripped": "",
                        "raw_level": 0,
                        "blank": True,
                        "is_block": False,
                        "is_clause": False,
                        "is_exit": False,
                    }
                )
                continue
            significant.append(
                {
                    "idx": idx,
                    "raw": raw,
                    "stripped": stripped,
                    "raw_level": _raw_level(raw),
                    "blank": False,
                    "is_block": bool(_BODY_BLOCK_OPENERS_RE.match(stripped)),
                    "is_clause": _is_clause_line(stripped),
                    "is_exit": _is_exit_line(stripped),
                }
            )

        nonblank = [item for item in significant if not item["blank"]]
        if not nonblank:
            return []

        states: Dict[Tuple[Tuple[int, ...], int, bool, bool], Tuple[float, List[Optional[int]]]] = {
            (tuple(), 0, False, False): (0.0, [None] * len(significant))
        }

        for pos, item in enumerate(significant):
            if item["blank"]:
                next_states: Dict[Tuple[Tuple[int, ...], int, bool, bool], Tuple[float, List[Optional[int]]]] = {}
                for key, (cost, path) in states.items():
                    next_path = list(path)
                    next_path[pos] = None
                    prev = next_states.get(key)
                    if prev is None or cost < prev[0]:
                        next_states[key] = (cost, next_path)
                states = next_states
                continue

            next_states = {}
            for (header_stack, prev_level, prev_was_block, prev_is_exit), (cost, path) in states.items():
                body_levels = [level + 1 for level in header_stack]
                if item["is_clause"]:
                    candidates = sorted(set(header_stack or (0,)))
                elif prev_was_block:
                    max_depth = max(body_levels or [prev_level + 1, item["raw_level"] + 2, 1]) + 1
                    candidates = list(range(prev_level + 1, max_depth + 1))
                else:
                    candidates = sorted(set(body_levels + [0]))
                    if prev_level not in candidates:
                        candidates.append(prev_level)
                        candidates = sorted(set(candidates))

                for level in candidates:
                    penalty = abs(level - int(item["raw_level"]))
                    if body_levels and level == 0 and not item["is_clause"] and not _is_exit_line(str(item["stripped"])):
                        penalty += 2
                    if not prev_was_block and level < prev_level:
                        if prev_is_exit:
                            penalty += max(0, (prev_level - level) - 1)
                        else:
                            penalty += max(0, prev_level - level) * 4
                            if not item["is_clause"] and not _is_exit_line(str(item["stripped"])):
                                penalty += 4
                    if prev_is_exit and level >= prev_level and not item["is_clause"]:
                        penalty += 3
                    if prev_is_exit and len(header_stack) >= 2 and not item["is_clause"]:
                        outer_body_level = header_stack[-2] + 1
                        if level < outer_body_level:
                            penalty += (outer_body_level - level) * 4
                    next_stack = list(header_stack)
                    if item["is_clause"]:
                        while next_stack and next_stack[-1] > level:
                            next_stack.pop()
                        if not next_stack or next_stack[-1] != level:
                            continue
                    else:
                        while next_stack and (next_stack[-1] + 1) > level:
                            next_stack.pop()
                    if item["is_block"]:
                        next_stack.append(level)
                    state_key = (tuple(next_stack), level, bool(item["is_block"]), bool(item["is_exit"]))
                    total_cost = cost + penalty
                    next_path = list(path)
                    next_path[pos] = level
                    prev_best = next_states.get(state_key)
                    if prev_best is None or total_cost < prev_best[0]:
                        next_states[state_key] = (total_cost, next_path)
            if not next_states:
                return None
            states = next_states

        ordered = sorted(states.items(), key=lambda item: item[1][0])
        for _, (_, path) in ordered:
            rebuilt: List[str] = []
            for entry, level in zip(significant, path):
                if entry["blank"] or level is None:
                    rebuilt.append("")
                    continue
                rebuilt.append((" " * 4 * int(level)) + str(entry["stripped"]))
            try:
                ast.parse("def _tmp():\n" + "\n".join(("    " + line) if line else "" for line in rebuilt) + "\n")
                return rebuilt
            except SyntaxError:
                continue
        return None

    lines = _seed_flattened_control_bodies(lines)
    searched = _normalize_with_search()
    if searched is not None:
        aligned = _dedent_trailing_loop_exits(_align_clause_indentation(searched))
        try:
            ast.parse("def _tmp():\n" + "\n".join(("    " + line) if line else "" for line in aligned) + "\n")
            return aligned
        except SyntaxError:
            return searched

    normalized: List[str] = []
    indent_stack: List[int] = []

    for raw in lines:
        if not raw.strip():
            normalized.append("")
            continue

        stripped = raw.lstrip()
        lead = len(raw) - len(stripped)

        while indent_stack and lead < indent_stack[-1]:
            indent_stack.pop()

        max_allowed = (indent_stack[-1] + 4) if indent_stack else 0
        if lead > max_allowed:
            lead = max_allowed

        rebuilt = (" " * lead) + stripped
        normalized.append(rebuilt)

        if _BODY_BLOCK_OPENERS_RE.match(stripped):
            indent_stack.append(lead + 4)

    return _dedent_trailing_loop_exits(_align_clause_indentation(normalized))


def _normalize_completion(text: str, entry_point: Optional[str]) -> str:
    if text is None:
        return ""
    # Keep escaped sequences inside string literals intact (e.g., r'[\\n*]').
    # JSON-wrapped code should already be decoded by _unwrap_json_code.
    text = str(text)
    if not entry_point:
        return text
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return text
    candidate_module = "\n".join(lines)
    # Keep as full code only if it explicitly defines the target entry point at top-level.
    # Otherwise treat as function-body snippet to avoid indentation breakage when helper defs appear.
    try:
        tree = ast.parse(candidate_module)
        top_level_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if entry_point in top_level_defs:
            return candidate_module
    except SyntaxError:
        pass

    # Compute smallest positive indent (ignoring entirely blank lines); use it to dedent uneven bodies.
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

    lines = _normalize_body_indentation(lines)
    lines = [("    " + line) if line.strip() else line for line in lines]
    return "\n".join(lines)


def _fix_missing_indents(text: str, entry_point: Optional[str] = None) -> str:
    """Compatibility helper for callers expecting indentation repair API."""
    return _normalize_completion(str(text or ""), entry_point)


def _extract_function_signature(prompt: str, entry_point: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract function signature (name and parameters) from prompt."""
    if not prompt or not entry_point:
        return None
    
    import re
    # Match: def function_name(param1, param2, ...): including multiline signatures
    # Use DOTALL to handle signatures that span multiple lines
    pattern = rf"def\s+{re.escape(entry_point)}\s*\((.*?)\)\s*(?:->.*?)?:"
    match = re.search(pattern, prompt, re.DOTALL)
    if not match:
        return None
    
    params_str = match.group(1).strip()
    if not params_str:
        return {"function_name": entry_point, "parameters": []}
    
    # Parse parameters (simple parsing, doesn't handle complex annotations)
    params = []
    # Remove newlines and extra spaces
    params_str = " ".join(params_str.split())
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Remove type annotations like ": List[int]" or "= default_value"
        param_name = param.split(":")[0].split("=")[0].strip()
        if param_name and param_name not in ("*", "**"):
            params.append(param_name)
    
    return {"function_name": entry_point, "parameters": params}


def _resolve_postprocess_db_path(db_path: Optional[str]) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if db_path:
        candidate = Path(db_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    else:
        candidate = repo_root / "postprocess_registry.sqlite"  # SMART release: 9 post-process tools extracted from the dev registry
    return str(candidate)


def _resolve_postprocess_tool_timeout_s() -> float:
    raw = str(os.getenv("POSTPROCESS_TOOL_TIMEOUT_S") or "").strip()
    if not raw:
        return 8.0
    try:
        value = float(raw)
    except Exception:
        return 8.0
    if value <= 0:
        return 8.0
    return value


def _get_postprocess_executor(db_path: Optional[str] = None):
    global _POSTPROCESS_REGISTRY, _POSTPROCESS_EXECUTOR, _POSTPROCESS_DB_PATH, _POSTPROCESS_TIMEOUT_S

    resolved = _resolve_postprocess_db_path(db_path)
    timeout_s = _resolve_postprocess_tool_timeout_s()
    if (
        _POSTPROCESS_EXECUTOR is not None
        and _POSTPROCESS_REGISTRY is not None
        and _POSTPROCESS_DB_PATH == resolved
        and _POSTPROCESS_TIMEOUT_S == timeout_s
    ):
        return _POSTPROCESS_EXECUTOR

    from core.registry import SQLiteRegistry
    from execution.tool_executor import ToolExecutor

    if _POSTPROCESS_REGISTRY is not None:
        try:
            _POSTPROCESS_REGISTRY.close()
        except Exception:
            pass

    _POSTPROCESS_REGISTRY = SQLiteRegistry(resolved)
    _POSTPROCESS_EXECUTOR = ToolExecutor(_POSTPROCESS_REGISTRY, timeout_s=timeout_s)
    _POSTPROCESS_DB_PATH = resolved
    _POSTPROCESS_TIMEOUT_S = timeout_s
    return _POSTPROCESS_EXECUTOR


def _extract_postprocess_payload(raw_output: Any) -> Dict[str, Any]:
    if not isinstance(raw_output, dict):
        return {}
    nested = raw_output.get("output")
    if isinstance(nested, dict) and ("completion" in nested or "changed" in nested or "is_fallback" in nested):
        return nested
    return raw_output


def _run_postprocess_tool_direct(tool_id: str, inputs: Dict[str, Any], error_hint: Optional[str] = None) -> Dict[str, Any]:
    if _POSTPROCESS_REGISTRY is None:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "registry_missing", "message": "postprocess registry not initialized"},
        }
    code = _POSTPROCESS_REGISTRY.get_tool_code(tool_id)
    if not code:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "tool_code_missing", "message": f"missing tool code: {tool_id}"},
        }
    try:
        scope: Dict[str, Any] = {}
        exec(code, scope)
        run_fn = scope.get("run")
        if not callable(run_fn):
            return {
                "ok": False,
                "output": None,
                "error": {"code": "invalid_tool", "message": f"{tool_id} has no callable run(inputs)"},
            }
        output = run_fn(inputs)
        if not isinstance(output, dict):
            return {
                "ok": False,
                "output": None,
                "error": {"code": "invalid_output", "message": f"{tool_id} run(inputs) must return dict"},
            }
        return {"ok": True, "output": output, "error": None}
    except Exception as exc:
        message = str(exc)
        if error_hint:
            message = f"{error_hint}; direct_exec_error={message}"
        return {"ok": False, "output": None, "error": {"code": type(exc).__name__, "message": message}}


def _run_postprocess_tool_agent(
    completion: str,
    *,
    prompt: str,
    entry_point: Optional[str],
    stop_tokens: Optional[List[str]] = None,
    use_stop_tokens: bool = False,
    param_names: Optional[List[str]] = None,
    enable_syntax_repair: bool = True,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    current = str(completion or "")
    resolved_params = [p for p in (param_names or _extract_param_names_from_prompt(prompt, entry_point)) if p]
    meta: Dict[str, Any] = {
        "agent": _POSTPROCESS_AGENT_NAME,
        "agent_id": _POSTPROCESS_AGENT_ID,
        "applied_tools": [],
        "attempted_tools": [],
        "fallback_detected": False,
        "tool_errors": {},
    }

    if not current.strip():
        meta["fallback_detected"] = True
        return "", meta

    try:
        executor = _get_postprocess_executor(db_path)
    except Exception as exc:
        meta["tool_errors"]["bootstrap"] = str(exc)
        return current.strip("\n"), meta

    base_inputs = {
        "prompt": prompt or "",
        "entry_point": entry_point,
        "stop_tokens": [str(t) for t in (stop_tokens or []) if str(t)],
        "use_stop_tokens": bool(use_stop_tokens),
        "param_names": resolved_params,
        "enable_syntax_repair": bool(enable_syntax_repair),
        "error_message": str(error_message or ""),
    }

    selected_tool_ids = _select_postprocess_tools(error_message, resolved_params)
    for tool_id in selected_tool_ids:
        meta["attempted_tools"].append(tool_id)
        tool_inputs = dict(base_inputs)
        tool_inputs["completion"] = current
        try:
            result = executor.run_tool(tool_id, tool_inputs)
        except Exception as exc:
            result = _run_postprocess_tool_direct(tool_id, tool_inputs, error_hint=f"sandbox_exec_error={exc}")
        if not result.get("ok"):
            meta["tool_errors"][tool_id] = result.get("error") or {"code": "unknown", "message": "tool failed"}
            continue
        payload = _extract_postprocess_payload(result.get("output"))
        if payload.get("is_fallback"):
            meta["fallback_detected"] = True
            meta["applied_tools"].append(tool_id)
            return "", meta
        next_completion = payload.get("completion")
        if not isinstance(next_completion, str):
            continue
        changed = bool(payload.get("changed")) or (next_completion != current)
        if changed:
            current = next_completion
            meta["applied_tools"].append(tool_id)
        if not current.strip():
            meta["fallback_detected"] = True
            return "", meta

    current = current.strip("\n")
    if not current.strip():
        meta["fallback_detected"] = True
        return "", meta
    if not meta["tool_errors"]:
        meta.pop("tool_errors", None)
    return current, meta


def _is_tool_error(error_obj: Any) -> bool:
    """Check if error indicates tool execution failure (not code logic error)."""
    if not error_obj:
        return False
    if isinstance(error_obj, dict):
        error_code = error_obj.get("code", "")
        error_msg = error_obj.get("message", "")
        # Tool-level errors that should disqualify the result
        fatal_errors = [
            "ImportError",
            "SyntaxError", 
            "IndentationError",
            "TabError",
            "NameError",
        ]
        if error_code in fatal_errors:
            return True
        # Also check message content
        if isinstance(error_msg, str):
            if "Import blocked" in error_msg:
                return True
            if "unterminated string literal" in error_msg:
                return True
    return False


def _is_fallback(text: str) -> bool:
    """Check if text is a fallback/placeholder code (like 'return None')."""
    if not text:
        return True
    
    # Check raw text for common indented patterns first (before stripping)
    # Handle "return None" with various indentation levels
    raw_stripped_lines = [line for line in text.split('\n') if line.strip()]
    if len(raw_stripped_lines) == 1:
        line_content = raw_stripped_lines[0].strip().lower()
        if line_content in ("none", "return none", "return null", "null"):
            return True
    
    # Strip and check for empty
    stripped = text.strip()
    if not stripped:
        return True
    
    # Normalize for case-insensitive comparison
    normalized = stripped.lower()

    # Tool timeout/error text accidentally extracted as code.
    if "error calling llm" in normalized:
        return True
    
    # Check if it's exactly "return None" or just "None"
    if normalized in ("none", "return none", "return null", "null"):
        return True
    
    # Check if the ENTIRE text is just a single return statement with None
    # This handles multi-line strings that only contain "return None"
    lines = [line.strip() for line in stripped.split('\n') if line.strip()]
    if len(lines) == 1 and lines[0].lower() in ("return none", "return null", "none", "null"):
        return True
    
    # Also check for lines with only whitespace + return None
    # e.g., "    return None" or "\treturn None"
    if all(line.strip().lower() in ("return none", "return null", "none", "null", "") for line in text.split('\n')):
        # Make sure at least one line has the return statement
        if any(line.strip().lower() in ("return none", "return null") for line in text.split('\n')):
            return True
    
    # Check for AI refusal patterns (these appear at START of text)
    refusal_patterns = [
        "i'm sorry",
        "i need more",
        "could you please",
        "please provide",
        "more specific",
        "more information",
        "i apologize",
        "i cannot",
    ]
    for pattern in refusal_patterns:
        if normalized.startswith(pattern):
            return True
    
    # Check for stdin template code (these are very specific patterns)
    if "sys.stdin.read" in text or "input(" in text:
        # Only flag if it's clearly a template (has both import and read)
        if "import sys" in text and "sys.stdin" in text:
            return True
    
    # Check for completely unrelated helper functions
    # These are signs of wrong code extraction
    unrelated_patterns = [
        "def is_sorted_recursive",  # Wrong function extracted
        "def helper",
        "def check_",
    ]
    for pattern in unrelated_patterns:
        if pattern in text:
            return True
    
    return False


def _extract_param_names_from_prompt(prompt: str, entry_point: Optional[str]) -> List[str]:
    sig = _extract_function_signature(prompt or "", entry_point)
    if not sig:
        return []
    params = sig.get("parameters")
    if not isinstance(params, list):
        return []
    return [str(p).strip() for p in params if str(p).strip()]


def _has_irrelevant_top_level_def(text: str, entry_point: Optional[str]) -> bool:
    """
    Check only module top-level function definitions via AST.
    Nested defs in function bodies are allowed.
    """
    if not text or not entry_point:
        return False

    candidate = _extract_code_block(str(text))
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        # Most function-body snippets are not parseable as a standalone module.
        # In body mode we should not enforce entry_point naming.
        return False

    top_level_defs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not top_level_defs:
        return False
    return entry_point not in top_level_defs


def _extract_code_from_results(
    results: Dict[str, Any], entry_point: Optional[str], param_names: Optional[List[str]] = None
) -> Optional[str]:
    role_priority = [
        "builder",
        "code-generation",
        "refactor",
        "refractor",
        "code-refactoring",
        "planner",
        "code-planner",
        "evidence_finder",
        "researcher",
        "checker",
        "tester",
        "code-testing",
    ]

    def _is_irrelevant(text: str) -> bool:
        return _has_irrelevant_top_level_def(text, entry_point)

    expected_params = [str(p).strip() for p in (param_names or []) if str(p).strip()]
    _builtins_obj = __builtins__
    builtin_names = set(_builtins_obj.keys()) if isinstance(_builtins_obj, dict) else set(dir(_builtins_obj))

    def _uses_expected_params(text: str) -> bool:
        if not expected_params:
            return True
        if not text:
            return False
        lowered = str(text)
        for param in expected_params:
            if re.search(rf"\b{re.escape(param)}\b", lowered):
                return True
        return False

    def _has_suspicious_free_names(text: str) -> bool:
        """
        Detect snippets that rely on undeclared names (common when extractor picks
        an intermediate helper fragment instead of final function body).
        """
        if not text:
            return False
        snippet = _extract_code_block(str(text))
        if not snippet.strip():
            return False

        # Skip strict check for full module outputs that define entry_point explicitly.
        try:
            module_tree = ast.parse(snippet)
            if entry_point:
                for node in module_tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
                        return False
        except SyntaxError:
            pass

        if not expected_params:
            return False

        probe_header = f"def __candidate({', '.join(expected_params)}):\n"
        probe_body = "\n".join(
            ("    " + line) if line.strip() else "    "
            for line in snippet.splitlines()
        )
        probe_src = probe_header + (probe_body or "    pass\n")
        try:
            tree = ast.parse(probe_src)
        except SyntaxError:
            return False

        fn = tree.body[0] if tree.body else None
        if not isinstance(fn, ast.FunctionDef):
            return False

        allowed_names = set(expected_params) | set(builtin_names)
        if entry_point:
            allowed_names.add(str(entry_point))

        for stmt in fn.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                allowed_names.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    alias_name = alias.asname or alias.name.split(".")[0]
                    if alias_name:
                        allowed_names.add(alias_name)

        load_names: set[str] = set()
        store_names: set[str] = set()
        call_names: set[str] = set()

        def _walk_no_nested(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Load):
                        load_names.add(child.id)
                    elif isinstance(child.ctx, ast.Store):
                        store_names.add(child.id)
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    call_names.add(child.func.id)
                _walk_no_nested(child)

        for stmt in fn.body:
            _walk_no_nested(stmt)

        allowed_names.update(store_names)
        unresolved_calls = {name for name in call_names if name not in allowed_names}
        unresolved_loads = {name for name in load_names if name not in allowed_names}
        return bool(unresolved_calls or unresolved_loads)

    def _normalize(value: Any, *, allow_string: bool = False) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            if not allow_string:
                return None
            unwrapped = _unwrap_json_code(value)
            result = unwrapped if unwrapped != value else value
            if (
                result
                and result.strip()
                and not _is_fallback(result)
                and not _is_irrelevant(result)
                and _uses_expected_params(result)
                and not _has_suspicious_free_names(result)
            ):
                return result
            return None
        if isinstance(value, dict):
            if value.get("ok") is False:
                return None
            if "error" in value and _is_tool_error(value["error"]):
                return None
            if "error" in value and value["error"]:
                return None

            for key in ["code_or_commands", "code", "solution"]:
                if key in value:
                    nested = _normalize(value.get(key), allow_string=True)
                    if nested:
                        return nested

            if "output" in value:
                nested = _normalize(value.get("output"), allow_string=False)
                if nested:
                    return nested
            if "result" in value:
                nested = _normalize(value.get("result"), allow_string=False)
                if nested:
                    return nested
        return None

    def _first_candidate(obj: Any) -> Optional[str]:
        text = _normalize(obj, allow_string=False)
        if text:
            return text

        if isinstance(obj, list):
            for item in obj:
                hit = _first_candidate(item)
                if hit:
                    return hit
            return None

        if isinstance(obj, dict):
            for key in ("code_or_commands", "code", "solution"):
                if key in obj:
                    hit = _normalize(obj.get(key), allow_string=True)
                    if hit:
                        return hit
            for key in ("output", "result"):
                if key in obj:
                    hit = _first_candidate(obj.get(key))
                    if hit:
                        return hit
        return None

    def _last_candidate(obj: Any) -> Optional[str]:
        text = _normalize(obj, allow_string=False)
        if text:
            return text

        if isinstance(obj, list):
            for item in reversed(obj):
                hit = _last_candidate(item)
                if hit:
                    return hit
            return None

        if isinstance(obj, dict):
            for key in ("code_or_commands", "code", "solution"):
                if key in obj:
                    hit = _normalize(obj.get(key), allow_string=True)
                    if hit:
                        return hit
            for key in ("output", "result"):
                if key in obj:
                    hit = _last_candidate(obj.get(key))
                    if hit:
                        return hit
        return None

    code: Optional[str] = None
    forced_final_builder = bool(results.get("forced_final_builder_executed"))
    execution_trace = results.get("execution_trace")

    # Prefer strict reverse-call-order extraction when execution trace exists.
    # This preserves real invocation order even when roles repeat (e.g., builder->checker->builder).
    if isinstance(execution_trace, list) and execution_trace:
        for call in reversed(execution_trace):
            if not isinstance(call, dict):
                continue
            # Prefer finalized role output first; tool histories often contain intermediate snippets.
            for key in ("output", "executor_result", "tool_history"):
                if key not in call:
                    continue
                hit = _last_candidate(call.get(key))
                if hit:
                    code = hit
                    break
            if code:
                break

    trace_key = "tool_trace" if "tool_trace" in results else ("tool_exec" if "tool_exec" in results else None)
    reversed_roles = list(reversed(role_priority))

    if not code and forced_final_builder:
        if trace_key:
            tool_trace = results.get(trace_key)
            if isinstance(tool_trace, dict):
                for role in ("builder", "code-generation"):
                    traces = tool_trace.get(role)
                    if isinstance(traces, list):
                        for trace in reversed(traces):
                            result_obj = trace.get("result") if isinstance(trace, dict) else trace
                            hit = _first_candidate(result_obj)
                            if hit:
                                code = hit
                                break
                    if code:
                        break
        if not code:
            for role in ("builder", "code-generation"):
                hit = _last_candidate(results.get(role))
                if hit:
                    code = hit
                    break

    if not code and trace_key:
        tool_trace = results.get(trace_key)
        if isinstance(tool_trace, dict):
            for role in reversed_roles:
                traces = tool_trace.get(role)
                if isinstance(traces, list):
                    for trace in reversed(traces):
                        result_obj = trace.get("result") if isinstance(trace, dict) else trace
                        hit = _first_candidate(result_obj)
                        if hit:
                            code = hit
                            break
                if code:
                    break

    if not code:
        for role in reversed_roles:
            hit = _first_candidate(results.get(role))
            if hit:
                code = hit
                break

    if not code or _is_fallback(code):
        return None

    if not _uses_expected_params(code) or _has_suspicious_free_names(code):
        return None

    code = _normalize_completion(_extract_code_block(code), entry_point)
    return code.strip("\n") if code.strip() else None


def _build_program(
    prompt: str, completion: str, test_code: str, entry_point: Optional[str] = None
) -> str:
    prompt_text = prompt or ""
    completion_text = completion or ""
    completion_text = _extract_code_block(completion_text)
    completion_text = _normalize_completion(completion_text, entry_point)
    if not completion_text.endswith("\n"):
        completion_text += "\n"
    test_imports: List[str] = []
    if re.search(r"\bmath\.", test_code or "") and not re.search(r"^\s*import\s+math\b", test_code or "", re.MULTILINE):
        test_imports.append("import math")
    if re.search(r"\bsys\.", test_code or "") and not re.search(r"^\s*import\s+sys\b", test_code or "", re.MULTILINE):
        test_imports.append("import sys")
    import_prefix = ("\n".join(test_imports) + "\n") if test_imports else ""
    return f"{prompt_text}{completion_text}\n{import_prefix}{test_code}\n"


def _looks_like_assertion_error(message: str) -> bool:
    if not message:
        return False
    return "AssertionError" in message


def _error_family(message: str) -> str:
    text = str(message or "")
    if not text:
        return "ok"
    if "AssertionError" in text:
        return "assertion"
    if any(token in text for token in ("SyntaxError", "IndentationError", "TabError")):
        return "syntax"
    if any(token in text for token in ("NameError", "UnboundLocalError")):
        return "name"
    if "timeout" in text.lower():
        return "timeout"
    if "Traceback" in text:
        return "runtime"
    return "other"


def _error_severity(message: str) -> int:
    family = _error_family(message)
    ranking = {
        "ok": 0,
        "assertion": 1,
        "name": 2,
        "runtime": 3,
        "other": 3,
        "syntax": 4,
        "timeout": 5,
    }
    return ranking.get(family, 3)


def _is_error_class_regressed(before_error: str, after_error: str) -> bool:
    if not after_error:
        return False
    return _error_severity(after_error) > _error_severity(before_error)


def _extract_assertion_hint(message: str) -> str:
    if not message:
        return ""
    for line in message.splitlines():
        if "assert candidate" in line:
            return line.strip()
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _repair_assertion_completion_with_llm(
    *,
    llm_client: Any,
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    error_message: str,
    failure_context: str = "",
    max_tokens: int = 700,
) -> Optional[str]:
    if llm_client is None:
        return None

    assertion_hint = _extract_assertion_hint(error_message)
    tests_preview = "\n".join((test_code or "").splitlines()[:80])
    messages = [
        {
            "role": "user",
            "content": (
                f"{prompt or ''}\n\n"
                f"Entry point: {entry_point or ''}\n\n"
                "You are fixing a failing implementation.\n"
                "Return ONLY raw function body code (no markdown, no explanation, no function signature).\n"
                "Keep correct Python indentation for function body lines.\n"
                "Do not output import/main/test code.\n\n"
                "Pass ALL provided assertions exactly.\n"
                "Match output container types exactly (list vs tuple) and preserve order exactly.\n"
                "For accumulator/filter/count/list-building code, return after the loop finishes; "
                "only return inside a loop for genuine first-match/search predicates.\n"
                "Avoid unrelated defensive branches unless tests/prompt require them.\n"
                "Ensure termination and avoid loops that can hang.\n\n"
                f"Current implementation body:\n{completion or ''}\n\n"
                f"Failing assertion/error hint:\n{assertion_hint}\n\n"
                f"Detailed failure context:\n{failure_context or ''}\n\n"
                f"Reference tests (truncated):\n{tests_preview}\n"
            ),
        }
    ]
    try:
        repaired = llm_client.chat(messages, temperature=0.1, max_tokens=max_tokens)
    except Exception:
        return None

    repaired = _extract_code_block(str(repaired))
    repaired = _normalize_completion(repaired, entry_point)
    repaired = repaired.strip("\n")
    if not repaired or _is_fallback(repaired):
        return None
    return repaired


def _evaluate_with_postprocess_check(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
    max_postprocess_rounds: int = 1,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    completion_text = str(completion or "")
    code = _build_program(prompt, completion_text, test_code, entry_point)
    ok, message = _run_python(code, timeout_s=timeout_s)
    if ok:
        return True, "", None

    rounds = max(0, int(max_postprocess_rounds))
    original_error = message
    attempts: List[Dict[str, Any]] = []
    current = completion_text

    for round_idx in range(rounds):
        # Invoke postprocess agent after initial failure.
        param_names = _extract_param_names_from_prompt(prompt, entry_point)
        repaired_text, post_meta = _run_postprocess_tool_agent(
            completion=current,
            prompt=prompt,
            entry_point=entry_point,
            stop_tokens=[],
            use_stop_tokens=False,
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=message,
        )

        attempt: Dict[str, Any] = {
            "round": round_idx + 1,
            "tools": post_meta.get("applied_tools", []),
            "attempted_tools": post_meta.get("attempted_tools", []),
            "original_error": message,
        }
        if isinstance(post_meta.get("tool_errors"), dict) and post_meta.get("tool_errors"):
            attempt["tool_errors"] = post_meta.get("tool_errors")
        if not repaired_text or repaired_text == current:
            attempt["no_change"] = True
            attempts.append(attempt)
            break

        repaired_code = _build_program(prompt, repaired_text, test_code, entry_point)
        ok2, message2 = _run_python(repaired_code, timeout_s=timeout_s)
        attempt["success"] = ok2
        attempt["final_error"] = "" if ok2 else message2

        if ok2:
            attempts.append(attempt)
            postprocess_info: Dict[str, Any] = {
                "attempted": True,
                "success": True,
                "check_type": "postprocess_agent_check",
                "original_error": original_error,
                "attempts": attempts,
            }
            return True, "", postprocess_info

        # Roll back degraded repairs (e.g. NameError -> TypeError, AssertionError -> runtime error).
        if _is_error_class_regressed(message, message2):
            attempt["rejected"] = "error_class_regressed"
            attempt["rollback"] = True
            attempts.append(attempt)
            continue

        attempts.append(attempt)
        current = repaired_text
        message = message2

    postprocess_info = {
        "attempted": True,
        "success": False,
        "check_type": "postprocess_agent_check",
        "original_error": original_error,
        "attempts": attempts,
    }
    if attempts and attempts[-1].get("no_change"):
        postprocess_info["no_change"] = True
    return False, message, postprocess_info


def _run_python(code: str, timeout_s: float) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "eval_task.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"exit_code={result.returncode}"
    return False, message
