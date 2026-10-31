from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.convert_mbpp import convert_mbpp_file
from evaluation import eval_humaneval
from evaluation import humaneval_postprocess


_MBPP_SCORE_MARKER = "__MBPP_ASSERT_SCORE__"
_MBPP_ASSERT_DIAG_MARKER = "__MBPP_ASSERT_DIAG__"
_MBPP_POSTPROCESS_ROUNDS = 3
_MBPP_PATCHED = False
_MBPP_REPAIR_STYLE_RULES = (
    "Repair style rules: keep logic minimal and test-driven. "
    "Do not add new input validation, isinstance checks, early guards, or raise statements "
    "unless explicitly required by the provided asserts."
)
_MBPP_REPAIR_STYLE_RULES_STRICT = (
    "Strict round rule: previous repair was ineffective. "
    "Modify core computation to satisfy the failing assert exactly; "
    "avoid defensive programming and avoid introducing new exceptions."
)
_MBPP_REWRITE_AFTER_NO_CHANGE_RULE = (
    "Rewrite mode: prior repair attempts made no useful change. "
    "Ignore the previous algorithm if needed and produce a fresh minimal function body "
    "that satisfies every provided assert."
)
_MBPP_INITIAL_GEN_RULES = (
    "MBPP generation rules: prioritize passing shown asserts exactly. "
    "Preserve required output order/type/shape. "
    "Output only the function body. "
    "Keep top-level body statements indented by one level and indent nested control blocks one additional level. "
    "Do not place a control header and its body at the same indentation. "
    "For accumulator, filter, counting, sorting, and list-building code, return after the loop finishes, "
    "not from inside the first iteration, unless the task is explicitly a search/early-exit predicate. "
    "Do not add broad input validation or defensive exception branches unless asserts require them."
)
_MBPP_STRICT_PROMPT_TEXT = (
    "Output only the function body code with correct indentation. "
    "Do not repeat function definition, explanation, or markdown fences.\n"
    "Important constraints:\n"
    "1. MUST use exact parameter names from function signature.\n"
    "2. Prioritize satisfying provided asserts exactly.\n"
    "3. Keep logic minimal and test-driven; avoid broad defensive input validation "
    "or new exceptions unless asserts require them.\n"
    "4. Top-level function body statements must be indented by one level.\n"
    "5. For if/for/while/try/with/match blocks, indent the block body one additional level.\n"
    "6. Do not place a control header and its body at the same indentation level.\n"
    "7. Generate only direct solution code; no unrelated parsing/template boilerplate.\n"
    "8. Do not return from inside accumulator/filter/list-building loops unless the problem explicitly asks for the first match.\n"
)
_MBPP_AUTO_IMPORT_MODULES = {
    "cmath",
    "math",
    "re",
    "json",
    "itertools",
    "collections",
    "heapq",
    "functools",
    "statistics",
    "random",
    "bisect",
    "string",
    "operator",
    "datetime",
    "sys",
    "array",
    "copy",
    "calendar",
    "time",
    "fractions",
    "decimal",
    "typing",
}
_MBPP_AUTO_IMPORT_SYMBOLS = {
    "Counter": ("collections", "Counter"),
    "defaultdict": ("collections", "defaultdict"),
    "OrderedDict": ("collections", "OrderedDict"),
    "deque": ("collections", "deque"),
    "groupby": ("itertools", "groupby"),
    "chain": ("itertools", "chain"),
    "tee": ("itertools", "tee"),
    "zip_longest": ("itertools", "zip_longest"),
    "combinations": ("itertools", "combinations"),
    "combinations_with_replacement": ("itertools", "combinations_with_replacement"),
    "itemgetter": ("operator", "itemgetter"),
    "eq": ("operator", "eq"),
    "heappush": ("heapq", "heappush"),
    "heappop": ("heapq", "heappop"),
    "merge": ("heapq", "merge"),
    "pi": ("math", "pi"),
    "sin": ("math", "sin"),
    "cos": ("math", "cos"),
    "tan": ("math", "tan"),
    "acos": ("math", "acos"),
    "radians": ("math", "radians"),
    "sqrt": ("math", "sqrt"),
    "factorial": ("math", "factorial"),
    "gcd": ("math", "gcd"),
    "lcm": ("math", "lcm"),
    "floor": ("math", "floor"),
    "ceil": ("math", "ceil"),
    "log": ("math", "log"),
    "exp": ("math", "exp"),
    "permutations": ("itertools", "permutations"),
    "product": ("itertools", "product"),
    "accumulate": ("itertools", "accumulate"),
    "reduce": ("functools", "reduce"),
    "cmp_to_key": ("functools", "cmp_to_key"),
    "lru_cache": ("functools", "lru_cache"),
    "choice": ("random", "choice"),
    "shuffle": ("random", "shuffle"),
    "randint": ("random", "randint"),
    "sample": ("random", "sample"),
    "mean": ("statistics", "mean"),
    "median": ("statistics", "median"),
    "mode": ("statistics", "mode"),
    "stdev": ("statistics", "stdev"),
    "ascii_lowercase": ("string", "ascii_lowercase"),
    "ascii_uppercase": ("string", "ascii_uppercase"),
    "bisect_left": ("bisect", "bisect_left"),
    "bisect_right": ("bisect", "bisect_right"),
    "insort": ("bisect", "insort"),
    "List": ("typing", "List"),
    "Tuple": ("typing", "Tuple"),
    "Dict": ("typing", "Dict"),
    "Set": ("typing", "Set"),
    "Optional": ("typing", "Optional"),
    "maxsize": ("sys", "maxsize"),
    "deepcopy": ("copy", "deepcopy"),
}
_MBPP_CONTROL_HEADER_RE = re.compile(
    r"^(?:if|elif|else|for|while|try|except|finally|with|match|case)\b.*:\s*$"
)


def _strip_tasks_arg(argv: List[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--tasks":
            if i + 1 < len(argv):
                skip_next = True
            continue
        cleaned.append(token)
    return cleaned


def _ensure_include_tool_trace(argv: List[str]) -> List[str]:
    if "--include_tool_trace" not in argv:
        argv = list(argv) + ["--include_tool_trace"]
    return argv


def _ensure_postprocess_rounds(argv: List[str], rounds: int) -> List[str]:
    if "--postprocess_rounds" not in argv:
        argv = list(argv) + ["--postprocess_rounds", str(rounds)]
    return argv


def _extract_assert_lines(test_code: str) -> List[str]:
    lines: List[str] = []
    for line in (test_code or "").splitlines():
        s = line.strip()
        if s.startswith("assert "):
            lines.append(s)
    return lines


def _compact_text(text: str, max_chars: int = 180) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _compact_value(value: Any, max_chars: int = 240) -> str:
    if value is None:
        return ""
    return _compact_text(str(value), max_chars=max_chars)


def _extract_assert_examples(test_code: str, max_examples: int = 2) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for line in _extract_assert_lines(test_code):
        if len(examples) >= max_examples:
            break
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        call, _ = _unwrap_candidate_call(left)
        if call is None:
            continue
        arg_src = ", ".join(ast.unparse(arg) if hasattr(ast, "unparse") else "arg" for arg in call.args)
        expected_src = ast.unparse(right) if hasattr(ast, "unparse") else "expected"
        examples.append({
            "input": _compact_text(arg_src, max_chars=180),
            "expected": _compact_text(expected_src, max_chars=180),
            "assert": _compact_text(line, max_chars=240),
        })
    return examples


def _build_mbpp_assertion_context(test_code: str, max_examples: int = 2) -> str:
    examples = _extract_assert_examples(test_code, max_examples=max_examples)
    if not examples:
        return ""
    lines = ["MBPP assertion focus (must satisfy exactly):"]
    for idx, example in enumerate(examples, start=1):
        lines.append(f"{idx}. input: {example['input']}")
        lines.append(f"   expected: {example['expected']}")
    return "\n".join(lines)


def _comment_block(text: str) -> str:
    if not text:
        return ""
    commented: List[str] = []
    for line in text.splitlines():
        if line.strip():
            commented.append(f"# {line}")
        else:
            commented.append("#")
    return "\n".join(commented)


def _append_mbpp_assertion_context(prompt: str, test_code: str) -> str:
    base = str(prompt or "")
    ctx = _build_mbpp_assertion_context(test_code)
    if not ctx:
        return base
    if "MBPP assertion focus" in base:
        return base
    return f"{base}\n\n{_comment_block(ctx)}\n"


def _append_mbpp_initial_generation_rules(prompt: str, test_code: str) -> str:
    base = _append_mbpp_assertion_context(prompt, test_code)
    if _MBPP_INITIAL_GEN_RULES in base:
        return base
    return f"{base}\n{_comment_block(_MBPP_INITIAL_GEN_RULES)}\n"


def _prepare_mbpp_tasks_for_generation(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            prepared.append(task)
            continue
        if str(task.get("source") or "").lower() != "mbpp":
            prepared.append(task)
            continue
        copied = dict(task)
        copied["prompt"] = _append_mbpp_initial_generation_rules(
            str(task.get("prompt") or ""),
            str(task.get("test") or ""),
        )
        prepared.append(copied)
    return prepared


def _format_param_types(contract: Dict[str, Any]) -> str:
    raw = contract.get("param_types") if isinstance(contract, dict) else None
    if not isinstance(raw, dict) or not raw:
        return ""
    parts: List[str] = []
    for idx, types in sorted(raw.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        if isinstance(types, list):
            type_desc = "|".join(str(t) for t in types if t)
        else:
            type_desc = str(types)
        if type_desc:
            parts.append(f"{idx}={type_desc}")
    return "; ".join(parts)


def _looks_like_error_text(code: str) -> bool:
    text = str(code or "").strip()
    if not text:
        return False
    lower = text.lower()
    if text.startswith("# Error") or text.startswith("Error:"):
        return True
    if "traceback (most recent call last)" in lower:
        return True
    if "error calling llm" in lower:
        return True
    if "timed out" in lower or "timeout" in lower:
        return True
    return False


def _extract_completion_text(completion: str) -> str:
    return humaneval_postprocess._extract_code_block(str(completion or "")).strip("\n")


def _mbpp_is_full_def_completion(completion: str, entry_point: Optional[str]) -> bool:
    text = _extract_completion_text(completion)
    if not text.strip():
        return False
    if entry_point and re.match(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", text):
        return True
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if entry_point is None or node.name == entry_point:
                return True
    return False


def _mbpp_completion_shape(completion: str, entry_point: Optional[str]) -> str:
    text = _extract_completion_text(completion)
    if _mbpp_is_full_def_completion(text, entry_point):
        return "full_def"
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped and _MBPP_CONTROL_HEADER_RE.match(stripped):
            return "body_with_control"
    return "body_no_control"


def _mbpp_has_flattened_control_flow(completion: str, entry_point: Optional[str]) -> bool:
    text = _extract_completion_text(completion)
    if not text.strip():
        return False
    lines = text.splitlines()
    start_idx = 0
    if _mbpp_is_full_def_completion(text, entry_point):
        for idx, raw_line in enumerate(lines):
            if re.match(r"^\s*def\s+", raw_line):
                start_idx = idx + 1
                break
    for idx in range(start_idx, len(lines)):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped or not _MBPP_CONTROL_HEADER_RE.match(stripped):
            continue
        cur_indent = len(raw) - len(raw.lstrip())
        next_idx = idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx >= len(lines):
            continue
        next_indent = len(lines[next_idx]) - len(lines[next_idx].lstrip())
        if next_indent <= cur_indent:
            return True
    return False


def _completion_function_node(completion: str, entry_point: Optional[str]) -> Optional[ast.FunctionDef]:
    text = _extract_completion_text(completion)
    if not text.strip():
        return None
    if _mbpp_is_full_def_completion(text, entry_point):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and (entry_point is None or node.name == entry_point):
                return node
        return None

    lines = text.splitlines()
    indented = "\n".join(("    " + line) if line.strip() and not line.startswith("    ") else line for line in lines)
    wrapped = f"def __mbpp_candidate__():\n{indented or '    pass'}\n"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        fixed = humaneval_postprocess._fix_missing_indents(text, entry_point)
        fixed_lines = fixed.splitlines()
        fixed_indented = "\n".join(
            ("    " + line) if line.strip() and not line.startswith("    ") else line
            for line in fixed_lines
        )
        try:
            tree = ast.parse(f"def __mbpp_candidate__():\n{fixed_indented or '    pass'}\n")
        except SyntaxError:
            return None
    node = tree.body[0] if tree.body else None
    return node if isinstance(node, ast.FunctionDef) else None


def _mbpp_return_risk(completion: str, entry_point: Optional[str]) -> Dict[str, Any]:
    fn = _completion_function_node(completion, entry_point)
    if fn is None:
        return {"return_in_loop": False, "return_in_conditional": False, "return_lines": []}

    return_lines: List[int] = []
    conditional_return_lines: List[int] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loop_depth = 0
            self.conditional_depth = 0

        def visit_For(self, node: ast.For) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node: ast.While) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_If(self, node: ast.If) -> None:
            self.conditional_depth += 1
            self.generic_visit(node)
            self.conditional_depth -= 1

        def visit_Return(self, node: ast.Return) -> None:
            lineno = int(getattr(node, "lineno", 0) or 0)
            if self.loop_depth:
                return_lines.append(lineno)
            if self.conditional_depth:
                conditional_return_lines.append(lineno)

    Visitor().visit(fn)
    return {
        "return_in_loop": bool(return_lines),
        "return_in_conditional": bool(conditional_return_lines),
        "return_lines": return_lines[:8],
        "conditional_return_lines": conditional_return_lines[:8],
    }


def _mbpp_return_risk_guidance(completion: str, entry_point: Optional[str]) -> str:
    risk = _mbpp_return_risk(completion, entry_point)
    if not risk.get("return_in_loop"):
        return ""
    lines = [
        "Static code lint: return statement appears inside a loop.",
        "If this code accumulates/counts/filters/sorts/builds a result, move the final return after the loop completes.",
        "Keep loop-internal returns only for true first-match/search predicates required by the asserts.",
    ]
    return_lines = risk.get("return_lines") or []
    if return_lines:
        lines.append(f"loop_return_lines: {return_lines}")
    return "\n".join(lines)


def _mbpp_hard_gate_mode(completion: str, entry_point: Optional[str]) -> str:
    return "raw_full_def" if _mbpp_is_full_def_completion(completion, entry_point) else "wrapped_body"


def _looks_like_repairable_runtime_failure(error_message: str) -> bool:
    text = str(error_message or "")
    if humaneval_postprocess._looks_like_assertion_error(text):
        return True
    return bool(re.search(r"\b(TypeError|ValueError)\b", text))


def _classify_failure_kind(error_message: str, diag: Optional[Dict[str, Any]] = None) -> str:
    text = str(error_message or "")
    lower = text.lower()
    if humaneval_postprocess._looks_like_assertion_error(text):
        return "assertion"
    if re.search(r"\bTypeError\b", text):
        return "type"
    if re.search(r"\bValueError\b", text):
        return "value"
    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", text):
        return "syntax"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    diag_error = str((diag or {}).get("error_type") or "")
    if diag_error in {"TypeError", "ValueError", "SyntaxError", "IndentationError", "TabError"}:
        return diag_error.replace("Error", "").lower()
    return "other"


def _failure_kind_guidance_lines(
    failure_kind: str,
    diag: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if failure_kind == "assertion":
        lines = [
            "Error-class guidance (assertion): focus on output logic for the shown failing assert.",
            "Do not add blanket input validation branches or new exceptions.",
            "Keep output type/order exactly as tests require.",
        ]
        if diag and diag.get("input_src"):
            lines.append(f"Failing input focus: {diag.get('input_src')}")
        return lines
    if failure_kind == "type":
        return [
            "Error-class guidance (type): fix operand/container type mismatch near failing expression.",
            "Do not add global isinstance gates that reject valid test inputs.",
            "Prefer local conversion/selection consistent with asserts, and keep return type contract.",
        ]
    if failure_kind == "value":
        return [
            "Error-class guidance (value): adjust computation domain/edge handling to match asserts.",
            "Avoid introducing new raise paths unless tests explicitly expect exceptions.",
            "Preserve existing structure and only patch the branch causing wrong value behavior.",
        ]
    if failure_kind == "syntax":
        return [
            "Error-class guidance (syntax): repair only syntax/indentation.",
            "Do not alter algorithmic structure unless strictly necessary for syntax validity.",
        ]
    if failure_kind == "timeout":
        return [
            "Error-class guidance (timeout): ensure finite loops and reduce unnecessary nested work.",
            "Keep algorithm shape and patch only the non-terminating/slow branch.",
        ]
    return [
        "Error-class guidance (generic): keep edits minimal and tied to the observed failing behavior.",
    ]


def _code_structure_metrics(code: str) -> Dict[str, int]:
    text = str(code or "")
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    non_comment = [ln for ln in non_empty if not ln.strip().startswith("#")]
    indented = "\n".join(("    " + ln) if ln.strip() else "    " for ln in lines) or "    pass"
    wrapped = f"def __mbpp_tmp__():\n{indented}\n"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return {
            "line_count": len(non_comment),
            "stmt_count": 0,
            "flow_count": 0,
            "raise_count": 0,
            "guard_count": 0,
        }
    fn = tree.body[0] if tree.body else None
    body = fn.body if isinstance(fn, ast.FunctionDef) else []
    flow_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    flow_count = 0
    raise_count = 0
    guard_count = 0
    for node in ast.walk(fn) if fn is not None else []:
        if isinstance(node, flow_types):
            flow_count += 1
        if isinstance(node, ast.Raise):
            raise_count += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            guard_count += 1
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Is, ast.IsNot)):
                    guard_count += 1
    return {
        "line_count": len(non_comment),
        "stmt_count": len(body),
        "flow_count": flow_count,
        "raise_count": raise_count,
        "guard_count": guard_count,
    }


def _is_non_minimal_change(
    base_code: str,
    new_code: str,
    base_quality: Dict[str, Any],
    new_quality: Dict[str, Any],
) -> bool:
    if not str(base_code or "").strip() or not str(new_code or "").strip():
        return False
    if new_quality.get("full_ok"):
        return False
    # Allow broad edits only when score genuinely improves.
    if int(new_quality.get("passed") or 0) > int(base_quality.get("passed") or 0):
        return False

    base = _code_structure_metrics(base_code)
    new = _code_structure_metrics(new_code)
    line_limit = max(base["line_count"] + 12, int(base["line_count"] * 1.8) + 2)
    if new["line_count"] > line_limit:
        return True
    if abs(new["stmt_count"] - base["stmt_count"]) > 6:
        return True
    if abs(new["flow_count"] - base["flow_count"]) > 3:
        return True
    if new["raise_count"] > base["raise_count"]:
        return True
    if new["guard_count"] > base["guard_count"] + 2:
        return True
    return False


def _failure_context_message(
    error_message: str,
    test_code: str,
    max_examples: int = 8,
    diag: Optional[Dict[str, Any]] = None,
    contract: Optional[Dict[str, Any]] = None,
    failure_kind: str = "other",
    strict_round: bool = False,
    force_logic_change: bool = False,
    current_completion: str = "",
    entry_point: Optional[str] = None,
    rewrite_mode: bool = False,
) -> str:
    base = _compact_text(error_message or "", max_chars=700)
    lines: List[str] = []
    if base:
        lines.append(base)

    if diag and not diag.get("ok"):
        lines.append("Failed assert details:")
        idx = diag.get("assert_index")
        if idx is not None:
            lines.append(f"assert_index: {idx}")
        assert_src = _compact_value(diag.get("assert_src"), max_chars=260)
        if assert_src:
            lines.append(f"assert: {assert_src}")
        input_src = _compact_value(diag.get("input_src"), max_chars=220)
        if input_src:
            lines.append(f"input: {input_src}")
        expected_src = _compact_value(diag.get("expected_src"), max_chars=220)
        if expected_src:
            lines.append(f"expected: {expected_src}")
        expected_type = _compact_value(diag.get("expected_type"), max_chars=80)
        actual_type = _compact_value(diag.get("actual_type"), max_chars=80)
        if expected_type or actual_type:
            lines.append(f"expected_type: {expected_type} actual_type: {actual_type}")
        expected_val = _compact_value(diag.get("expected"), max_chars=240)
        actual_val = _compact_value(diag.get("actual"), max_chars=240)
        if expected_val:
            lines.append(f"expected_value: {expected_val}")
        if actual_val:
            lines.append(f"actual_value: {actual_val}")
        error_type = _compact_value(diag.get("error_type"), max_chars=80)
        error_msg = _compact_value(diag.get("error_message"), max_chars=220)
        if error_type:
            lines.append(f"exception: {error_type} {error_msg}")

    if contract:
        param_types = _format_param_types(contract)
        if param_types:
            lines.append(f"param_types: {param_types}")
        if contract.get("list_of_lists"):
            lines.append("Output must be list of lists (not tuples).")

    return_risk = _mbpp_return_risk_guidance(current_completion, entry_point)
    if return_risk:
        lines.append("")
        lines.append(return_risk)

    ctx = _build_mbpp_assertion_context(test_code, max_examples=max_examples)
    if ctx:
        lines.append("")
        lines.append(ctx)
    lines.append("")
    lines.extend(_failure_kind_guidance_lines(failure_kind, diag=diag))
    lines.append("")
    lines.append(_MBPP_REPAIR_STYLE_RULES)
    if strict_round:
        lines.append(_MBPP_REPAIR_STYLE_RULES_STRICT)
    if force_logic_change:
        lines.append("Previous attempt made no functional change; update computation logic directly.")
    if rewrite_mode:
        lines.append(_MBPP_REWRITE_AFTER_NO_CHANGE_RULE)
    return "\n".join(line for line in lines if line is not None)


def _split_setup_and_asserts(test_code: str) -> Tuple[str, List[str]]:
    lines = (test_code or "").splitlines()
    setup_lines: List[str] = []
    asserts: List[str] = []
    in_check = False
    check_indent = 0
    seen_check = False

    for line in lines:
        stripped = line.strip()
        if not seen_check:
            if stripped.startswith("def check("):
                seen_check = True
                in_check = True
                check_indent = len(line) - len(line.lstrip())
                continue
            # Keep only real setup before check(); this avoids pulling lines
            # from test_check(), e.g. check(entry_point), into setup.
            if stripped:
                setup_lines.append(line)
            continue

        if in_check:
            cur_indent = len(line) - len(line.lstrip())
            if stripped and cur_indent <= check_indent and not stripped.startswith("#"):
                in_check = False
            else:
                if stripped.startswith("assert "):
                    asserts.append(stripped)
                continue

        # We do not consume anything after check() definition.

    return "\n".join(setup_lines).strip(), asserts


def _infer_node_type(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "none"
        return type(node.value).__name__
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return type(node.operand.value).__name__
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"list", "tuple", "set", "dict", "str", "int", "float", "bool"}:
            return node.func.id
    return "unknown"


def _is_list_of_lists(node: ast.AST) -> bool:
    return isinstance(node, ast.List) and any(isinstance(elt, ast.List) for elt in node.elts)


def _unwrap_candidate_call(node: ast.AST) -> Tuple[Optional[ast.Call], bool]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "candidate":
            return node, True
        if node.func.id in {"sorted", "set", "list", "tuple"} and node.args:
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "candidate":
                # wrapped candidate call, typically order-insensitive for sorted/set wrappers
                return inner, node.func.id not in {"sorted", "set"}
    return None, True


def _extract_io_contract(test_code: str) -> Dict[str, Any]:
    assert_lines = _extract_assert_lines(test_code)
    param_types: Dict[int, set[str]] = {}
    return_types: set[str] = set()
    order_required = False
    list_of_lists = False

    for line in assert_lines:
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        call, left_order_sensitive = _unwrap_candidate_call(left)
        if call is None:
            continue
        for idx, arg in enumerate(call.args):
            param_types.setdefault(idx, set()).add(_infer_node_type(arg))
        ret_type = _infer_node_type(right)
        return_types.add(ret_type)
        if ret_type in {"list", "tuple"} and left_order_sensitive:
            order_required = True
        if isinstance(right, ast.List) and _is_list_of_lists(right):
            list_of_lists = True

    return {
        "assert_count": len(assert_lines),
        "param_types": {idx: sorted(types) for idx, types in sorted(param_types.items())},
        "return_types": sorted(return_types),
        "order_required": order_required,
        "list_of_lists": list_of_lists,
    }


def _collect_applied_tools(attempts: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for attempt in attempts:
        tools = attempt.get("tools") if isinstance(attempt, dict) else None
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, str) or not tool:
                continue
            if tool in seen:
                continue
            seen.add(tool)
            ordered.append(tool)
    return ordered


def _compare_small_scores(
    current_passed: int,
    current_total: int,
    candidate_passed: int,
    candidate_total: int,
    *,
    label: str,
) -> Tuple[bool, str]:
    if current_total != candidate_total:
        return (
            False,
            f"{label}_incomparable_total_mismatch: "
            f"current={current_passed}/{current_total}, "
            f"{label}={candidate_passed}/{candidate_total}",
        )
    if candidate_passed > current_passed:
        return (
            True,
            f"{label}_improved: "
            f"current={current_passed}/{current_total}, "
            f"{label}={candidate_passed}/{candidate_total}",
        )
    if candidate_passed == current_passed:
        return (
            False,
            f"{label}_tied_no_gain: "
            f"current={current_passed}/{current_total}, "
            f"{label}={candidate_passed}/{candidate_total}",
        )
    return (
        False,
        f"{label}_regressed: "
        f"current={current_passed}/{current_total}, "
        f"{label}={candidate_passed}/{candidate_total}",
    )


def _normalize_mbpp_postprocess_meta(
    post_meta: Dict[str, Any],
    *,
    attempted: bool,
    small_score_before: str,
) -> Dict[str, Any]:
    attempts = post_meta.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    normalized: Dict[str, Any] = {
        "attempted": bool(post_meta.get("attempted", attempted or bool(attempts))),
        "success": bool(post_meta.get("success", False)),
        "small_score_before": str(post_meta.get("small_score_before", small_score_before)),
        "attempts": attempts,
    }
    no_change_flag = post_meta.get("no_change")
    if isinstance(no_change_flag, bool):
        normalized["no_change"] = no_change_flag
    elif attempts:
        normalized["no_change"] = all(
            isinstance(item, dict) and bool(item.get("no_change")) for item in attempts
        )
    for key in ("assert_score_passed", "assert_score_total", "first_failure_kind", "first_failing_assert"):
        if key in post_meta and post_meta.get(key) not in (None, ""):
            normalized[key] = post_meta.get(key)
    return normalized


def _parse_assert_line(assert_line: str) -> Optional[Dict[str, str]]:
    try:
        node = ast.parse(assert_line).body[0]
    except SyntaxError:
        return None
    if not isinstance(node, ast.Assert):
        return None
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    left = test.left
    right = test.comparators[0]
    call, _ = _unwrap_candidate_call(left)
    input_src = ""
    if call is not None:
        input_src = ", ".join(ast.unparse(arg) if hasattr(ast, "unparse") else "arg" for arg in call.args)
    left_src = ast.unparse(left) if hasattr(ast, "unparse") else "candidate(...)"
    right_src = ast.unparse(right) if hasattr(ast, "unparse") else "expected"
    return {
        "assert_src": assert_line.strip(),
        "left_src": left_src,
        "right_src": right_src,
        "input_src": input_src,
        "expected_src": right_src,
    }


def _build_assert_diagnostic_test(test_code: str, entry_point: str) -> str:
    setup, asserts = _split_setup_and_asserts(test_code)
    parsed_asserts: List[Dict[str, str]] = []
    for line in asserts:
        parsed = _parse_assert_line(line)
        if parsed:
            parsed_asserts.append(parsed)

    lines: List[str] = []
    if setup:
        lines.append(setup)
        lines.append("")
    lines.append("def _mbpp_diag(candidate):")
    lines.append("    import json")
    lines.append("    def _emit(payload):")
    lines.append(f"        print('{_MBPP_ASSERT_DIAG_MARKER}' + json.dumps(payload, ensure_ascii=True))")
    if not parsed_asserts:
        lines.append("    _emit({'ok': True})")
    else:
        for idx, parsed in enumerate(parsed_asserts, start=1):
            left_src = parsed["left_src"]
            right_src = parsed["right_src"]
            input_src = parsed.get("input_src", "")
            assert_src = parsed.get("assert_src", "")
            expected_src = parsed.get("expected_src", "")
            lines.append("    try:")
            lines.append(f"        _actual = {left_src}")
            lines.append(f"        _expected = {right_src}")
            lines.append("        if _actual != _expected:")
            lines.append("            _emit({")
            lines.append("                'ok': False,")
            lines.append(f"                'assert_index': {idx},")
            lines.append(f"                'assert_src': {json.dumps(assert_src)},")
            lines.append(f"                'input_src': {json.dumps(input_src)},")
            lines.append(f"                'expected_src': {json.dumps(expected_src)},")
            lines.append("                'actual': repr(_actual),")
            lines.append("                'expected': repr(_expected),")
            lines.append("                'actual_type': type(_actual).__name__,")
            lines.append("                'expected_type': type(_expected).__name__,")
            lines.append("            })")
            lines.append("            return")
            lines.append("    except Exception as e:")
            lines.append("        _emit({")
            lines.append("            'ok': False,")
            lines.append(f"            'assert_index': {idx},")
            lines.append(f"            'assert_src': {json.dumps(assert_src)},")
            lines.append(f"            'input_src': {json.dumps(input_src)},")
            lines.append(f"            'expected_src': {json.dumps(expected_src)},")
            lines.append("            'error_type': type(e).__name__,")
            lines.append("            'error_message': str(e),")
            lines.append("        })")
            lines.append("        return")
        lines.append("    _emit({'ok': True})")
    lines.append("")
    lines.append("def test_check():")
    lines.append(f"    _mbpp_diag({entry_point})")
    lines.append("")
    lines.append("test_check()")
    return "\n".join(lines) + "\n"


def _parse_assert_diag_output(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    for line in text.splitlines():
        if _MBPP_ASSERT_DIAG_MARKER not in line:
            continue
        payload = line.split(_MBPP_ASSERT_DIAG_MARKER, 1)[-1].strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _diagnose_first_assert_failure(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Optional[Dict[str, Any]]:
    if not entry_point:
        return None
    diag_test = _build_assert_diagnostic_test(test_code, entry_point)
    program = eval_humaneval._build_program(prompt, candidate, diag_test, entry_point)
    ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
    text = (stdout or "") + "\n" + (stderr or "")
    diag = _parse_assert_diag_output(text)
    if diag is None:
        return {"ok": ok}
    return diag


def _build_mbpp_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    raw_code_model = str(config.get("code_model") or "").strip()
    raw_base_url = str(config.get("code_base_url") or "").strip()
    raw_api_key = config.get("code_api_key") or None
    env_base_url = str(os.getenv("LLM_API_BASE") or "").strip()
    env_api_key = os.getenv("LLM_API_KEY") or None
    remote_requested = bool(raw_code_model or raw_base_url or raw_api_key or env_base_url or env_api_key)
    code_model = raw_code_model
    code_base_url = raw_base_url
    code_api_key = raw_api_key
    code_timeout = float(config.get("code_timeout") or 60.0)
    code_retries = int(config.get("code_retries") or 2)

    local_model_dir = config.get("local_model_dir") or None
    local_lora_dir = config.get("local_lora_dir") or None
    local_device = config.get("local_device") or None
    local_dtype = config.get("local_dtype") or "auto"
    local_use_4bit = bool(config.get("local_use_4bit") or False)
    local_use_8bit = bool(config.get("local_use_8bit") or False)
    max_new_tokens = int(config.get("max_new_tokens") or 256)
    temperature = float(config.get("temperature") or 0.2)

    if remote_requested:
        if not code_model:
            code_model = "gpt-4o"
        if not code_base_url:
            code_base_url = env_base_url
        if not code_api_key:
            code_api_key = env_api_key
        try:
            from generation.llm_client import LLMClient
        except Exception:
            return None
        return LLMClient(
            api_key=code_api_key or None,
            base_url=code_base_url or None,
            model=code_model or None,
            timeout_s=code_timeout,
            max_retries=code_retries,
        )

    if local_model_dir:
        try:
            from generation.local_llm import LocalLLMClient
        except Exception:
            return None
        return LocalLLMClient(
            model_path=local_model_dir,
            lora_path=local_lora_dir,
            device=local_device,
            dtype=local_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_4bit=local_use_4bit,
            use_8bit=local_use_8bit,
        )

    return None


def _run_python_capture(code: str, timeout_s: float) -> Tuple[bool, str, str]:
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
            return False, "", "timeout"
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def _syntax_precheck(
    prompt: str,
    completion: str,
    entry_point: Optional[str],
) -> Optional[str]:
    if not completion or not completion.strip():
        return None
    program = eval_humaneval._build_program(prompt, completion, "", entry_point)
    try:
        compile(program, "<mbpp_precheck>", "exec")
    except SyntaxError as exc:
        lineno = f" line {exc.lineno}" if exc.lineno else ""
        return f"{type(exc).__name__}: {exc.msg}{lineno}"
    return None


def _extract_nameerror_identifier(error_text: str) -> Optional[str]:
    text = str(error_text or "")
    m = re.search(r"NameError:\s+name\s+'([^']+)'\s+is not defined", text)
    if not m:
        return None
    name = str(m.group(1) or "").strip()
    return name or None


def _resolve_auto_import_spec(identifier: Optional[str]) -> Optional[Tuple[str, str, Optional[str]]]:
    name = str(identifier or "").strip()
    if not name:
        return None
    symbol_spec = _MBPP_AUTO_IMPORT_SYMBOLS.get(name)
    if symbol_spec:
        module, symbol = symbol_spec
        return ("from", module, symbol)
    if name in _MBPP_AUTO_IMPORT_MODULES:
        return ("import", name, None)
    return None


def _inject_import_into_completion(
    completion: str,
    module: str,
    symbol: Optional[str] = None,
) -> str:
    text = str(completion or "")
    mod = str(module or "").strip()
    sym = str(symbol or "").strip()
    if not text.strip() or not mod:
        return text
    if sym:
        if re.search(rf"(?m)^\s*from\s+{re.escape(mod)}\s+import\s+.*\b{re.escape(sym)}\b", text):
            return text
        import_line_body = f"from {mod} import {sym}"
    else:
        if re.search(rf"(?m)^\s*import\s+{re.escape(mod)}\b", text):
            return text
        if re.search(rf"(?m)^\s*from\s+{re.escape(mod)}\s+import\b", text):
            return text
        import_line_body = f"import {mod}"

    lines = text.splitlines()
    indent = "    "
    for line in lines:
        if not line.strip():
            continue
        leading = line[: len(line) - len(line.lstrip())]
        indent = leading or "    "
        break
    import_line = f"{indent}{import_line_body}"
    insert_idx = 0
    while insert_idx < len(lines):
        stripped = lines[insert_idx].strip()
        if stripped and not stripped.startswith("#"):
            break
        insert_idx += 1
    patched = lines[:insert_idx] + [import_line] + lines[insert_idx:]
    return "\n".join(patched)


def _run_mbpp_precheck_auto_repairs(
    *,
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[str, Dict[str, Any]]:
    current = str(completion or "")
    meta: Dict[str, Any] = {
        "attempted": bool(entry_point),
        "auto_repairs": [],
    }
    if not entry_point or not current.strip():
        return current, meta

    normalized = humaneval_postprocess._fix_missing_indents(current, entry_point)
    if normalized and normalized.strip() and normalized != current:
        current = normalized
        meta["auto_repairs"].append({"kind": "indent_normalize", "applied": True})

    syntax_error = _syntax_precheck(prompt, current, entry_point)
    if syntax_error and re.search(r"\b(IndentationError|TabError)\b", syntax_error):
        retry = humaneval_postprocess._fix_missing_indents(current, entry_point)
        if retry and retry.strip() and retry != current:
            current = retry
            meta["auto_repairs"].append(
                {"kind": "indent_repair_after_error", "applied": True, "error": syntax_error}
            )
        syntax_error = _syntax_precheck(prompt, current, entry_point)

    meta["syntax_error_after_auto_repair"] = syntax_error or ""

    runtime_precheck_ran = bool(test_code)
    meta["runtime_precheck_ran"] = runtime_precheck_ran
    if not syntax_error and runtime_precheck_ran:
        program = eval_humaneval._build_program(prompt, current, test_code, entry_point)
        ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
        runtime_error = ((stderr or "") + "\n" + (stdout or "")).strip()
        meta["runtime_precheck_ok"] = bool(ok)
        if not ok and "NameError" in runtime_error:
            missing = _extract_nameerror_identifier(runtime_error)
            import_spec = _resolve_auto_import_spec(missing)
            if import_spec:
                _, module, symbol = import_spec
                patched = _inject_import_into_completion(current, module, symbol=symbol)
                patched = humaneval_postprocess._fix_missing_indents(patched, entry_point)
                patched_syntax_error = _syntax_precheck(prompt, patched, entry_point)
                if not patched_syntax_error:
                    program2 = eval_humaneval._build_program(prompt, patched, test_code, entry_point)
                    ok2, stdout2, stderr2 = _run_python_capture(program2, timeout_s=timeout_s)
                    current = patched
                    runtime_error2 = ((stderr2 or "") + "\n" + (stdout2 or "")).strip()
                    meta["runtime_precheck_ok"] = bool(ok2)
                    meta["auto_repairs"].append(
                        {
                            "kind": "nameerror_auto_import",
                            "applied": True,
                            "identifier": missing,
                            "module": module,
                            "symbol": symbol,
                            "runtime_ok_after": bool(ok2),
                        }
                    )
                    if runtime_error2:
                        meta["runtime_error_after_auto_repair"] = _compact_text(runtime_error2, max_chars=500)
                else:
                    meta["auto_repairs"].append(
                        {
                            "kind": "nameerror_auto_import",
                            "applied": False,
                            "identifier": missing,
                            "module": module,
                            "symbol": symbol,
                            "reason": patched_syntax_error,
                        }
                    )
        if runtime_error:
            meta["runtime_error"] = _compact_text(runtime_error, max_chars=500)

    return current, meta


def _repair_hard_gate_error(
    prompt: str,
    completion: str,
    entry_point: Optional[str],
    *,
    test_code: str = "",
    timeout_s: float = 0.0,
) -> Optional[str]:
    code = str(completion or "")
    if not code.strip():
        return "empty_after_normalize"
    syntax_error = _syntax_precheck(prompt, code, entry_point)
    if syntax_error:
        return str(syntax_error)
    gate_mode = _mbpp_hard_gate_mode(code, entry_point)
    if gate_mode == "raw_full_def" and entry_point:
        expected_params = humaneval_postprocess._extract_param_names_from_prompt(prompt, entry_point)
        if expected_params:
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                lineno = f" line {exc.lineno}" if exc.lineno else ""
                return f"{type(exc).__name__}: {exc.msg}{lineno}"
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                    got_params = [arg.arg for arg in node.args.args]
                    if got_params and got_params != expected_params:
                        return (
                            "entry_point_signature_mismatch: "
                            f"expected ({', '.join(expected_params)}) got ({', '.join(got_params)})"
                        )
    if test_code and timeout_s > 0:
        program = eval_humaneval._build_program(prompt, code, test_code, entry_point)
        ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
        if not ok:
            runtime_error = (stderr or stdout or "execution failed").strip()
            if runtime_error and not humaneval_postprocess._looks_like_assertion_error(runtime_error):
                return f"runtime_precheck_failed: {_compact_text(runtime_error, max_chars=320)}"
    return None


def _postprocess_completion(
    prompt: str,
    completion: str,
    entry_point: Optional[str],
    *,
    rounds: int,
    error_message: str = "",
) -> Tuple[str, Dict[str, Any]]:
    current = str(completion or "")
    if not current.strip():
        return current, {"attempted": False}

    param_names = humaneval_postprocess._extract_param_names_from_prompt(prompt, entry_point)
    repaired_text, post_meta = humaneval_postprocess._run_postprocess_tool_agent(
        completion=current,
        prompt=prompt,
        entry_point=entry_point,
        stop_tokens=[],
        use_stop_tokens=False,
        param_names=param_names,
        enable_syntax_repair=True,
        error_message=error_message,
    )
    next_text = repaired_text if repaired_text and repaired_text.strip() else current
    normalized = humaneval_postprocess._normalize_completion(
        humaneval_postprocess._strip_redundant_def(
            humaneval_postprocess._extract_code_block(str(next_text)),
            entry_point,
        ),
        entry_point,
    )
    if normalized and normalized.strip():
        gate_error = _repair_hard_gate_error(prompt, normalized, entry_point)
        if gate_error:
            post_meta["hard_gate_error"] = gate_error
            post_meta["hard_gate_mode"] = _mbpp_hard_gate_mode(normalized, entry_point)
        else:
            current = normalized
    post_meta["rounds"] = int(rounds)
    return current, post_meta


def _build_assert_scoring_test(test_code: str, entry_point: str) -> Tuple[str, int]:
    setup, asserts = _split_setup_and_asserts(test_code)
    total = len(asserts)
    lines: List[str] = []
    if setup:
        lines.append(setup)
        lines.append("")
    lines.append("def check(candidate):")
    lines.append("    passed = 0")
    lines.append(f"    total = {total}")
    if asserts:
        for assert_line in asserts:
            lines.append("    try:")
            lines.append(f"        {assert_line}")
            lines.append("        passed += 1")
            lines.append("    except AssertionError:")
            lines.append("        pass")
    lines.append("    return passed, total")
    lines.append("")
    lines.append("def test_check():")
    lines.append(f"    p, t = check({entry_point})")
    lines.append(f"    print('{_MBPP_SCORE_MARKER}' + str(p) + '/' + str(t))")
    lines.append("")
    lines.append("test_check()")
    return "\n".join(lines) + "\n", total


def _score_candidate_against_asserts(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[int, int, Optional[str]]:
    if not entry_point:
        return 0, 0, None
    score_test, total = _build_assert_scoring_test(test_code, entry_point)
    program = eval_humaneval._build_program(prompt, candidate, score_test, entry_point)
    ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
    text = (stdout or "") + "\n" + (stderr or "")
    m = re.search(rf"{re.escape(_MBPP_SCORE_MARKER)}(\d+)/(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    if ok and total > 0:
        # If marker is missing but run succeeded, conservatively treat as all passed.
        return total, total, None
    runtime_error = (stderr or stdout or "").strip()
    compact_runtime_error = _compact_text(runtime_error, max_chars=500) if runtime_error else ""
    if compact_runtime_error and not humaneval_postprocess._looks_like_assertion_error(compact_runtime_error):
        return 0, total, compact_runtime_error
    return 0, total, None


def _full_test_pass(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> bool:
    program = eval_humaneval._build_program(prompt, candidate, test_code, entry_point)
    ok, _, _ = _run_python_capture(program, timeout_s=timeout_s)
    return ok


def _evaluate_candidate_quality(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
    small_tests_only: bool = False,
) -> Dict[str, Any]:
    syntax_error = _syntax_precheck(prompt, candidate, entry_point) if entry_point else None
    full_ok = False
    full_error = ""
    if syntax_error:
        full_error = str(syntax_error)
    elif not small_tests_only:
        program = eval_humaneval._build_program(prompt, candidate, test_code, entry_point)
        ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
        full_ok = bool(ok)
        if not ok:
            full_error = (stderr or stdout or "execution failed").strip()
    passed, total, score_runtime_error = _score_candidate_against_asserts(
        prompt,
        candidate,
        test_code,
        entry_point,
        timeout_s=timeout_s,
    )
    if small_tests_only and not syntax_error:
        if score_runtime_error:
            full_ok = False
            full_error = score_runtime_error
        else:
            full_ok = bool(total > 0 and passed == total)
            if not full_ok:
                full_error = f"AssertionError: assert score {passed}/{total}"
    elif not full_ok and not full_error and score_runtime_error:
        full_error = score_runtime_error
    return {
        "syntax_error": syntax_error,
        "passed": int(passed),
        "total": int(total),
        "full_ok": bool(full_ok),
        "full_error": full_error,
        "score_runtime_error": score_runtime_error or "",
    }


def _candidate_quality_regressed(new_quality: Dict[str, Any], base_quality: Dict[str, Any]) -> bool:
    if new_quality.get("syntax_error"):
        return True
    base_total = int(base_quality.get("total") or 0)
    base_passed = int(base_quality.get("passed") or 0)
    new_total = int(new_quality.get("total") or 0)
    new_passed = int(new_quality.get("passed") or 0)
    if base_total > 0 and new_total == base_total and new_passed < base_passed:
        return True
    if bool(base_quality.get("full_ok")) and not bool(new_quality.get("full_ok")):
        return True
    return False


def _postprocess_after_failure(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
    rounds: int,
    llm_client: Optional[Any] = None,
    small_tests_only: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    current = candidate
    attempts: List[Dict[str, Any]] = []
    execution_prompt = str(prompt or "")
    llm_repair_prompt = _append_mbpp_assertion_context(execution_prompt, test_code)
    contract = _extract_io_contract(test_code)
    rounds = max(1, int(rounds))
    prev_round_no_change = False
    quality_cache: Dict[str, Dict[str, Any]] = {}
    no_change_streak = 0
    aggressive_relax_used = False
    last_assert_score_passed = 0
    last_assert_score_total = 0
    last_failure_kind = ""
    last_first_failing_assert = ""

    def _get_quality(candidate_text: str) -> Dict[str, Any]:
        key = str(candidate_text or "")
        cached = quality_cache.get(key)
        if cached is not None:
            return cached
        quality = _evaluate_candidate_quality(
            execution_prompt,
            key,
            test_code,
            entry_point,
            timeout_s=timeout_s,
            small_tests_only=small_tests_only,
        )
        quality_cache[key] = quality
        return quality

    for round_idx in range(rounds):
        round_changed = False
        if current:
            current = humaneval_postprocess._fix_missing_indents(str(current), entry_point)
        current_quality = _get_quality(current)
        if current_quality.get("full_ok"):
            return current, {
                "success": True,
                "attempts": attempts,
                "assert_score_passed": int(current_quality.get("passed") or 0),
                "assert_score_total": int(current_quality.get("total") or 0),
                "first_failure_kind": "",
                "first_failing_assert": "",
            }
        passed_now = int(current_quality.get("passed") or 0)
        total_now = int(current_quality.get("total") or 0)
        near_full_stuck = total_now > 0 and passed_now == max(0, total_now - 1)
        aggressive_mode = (no_change_streak >= 1 or near_full_stuck) and not aggressive_relax_used
        error_message = str(current_quality.get("full_error") or "execution failed").strip()
        diag = _diagnose_first_assert_failure(
            execution_prompt,
            current,
            test_code,
            entry_point,
            timeout_s=timeout_s,
        )
        failure_kind = _classify_failure_kind(error_message, diag=diag)
        last_assert_score_passed = int(passed_now)
        last_assert_score_total = int(total_now)
        last_failure_kind = str(failure_kind or "")
        last_first_failing_assert = str((diag or {}).get("assert_src") or "")
        rewrite_mode = no_change_streak >= 1
        failure_context = _failure_context_message(
            error_message,
            test_code,
            diag=diag,
            contract=contract,
            failure_kind=failure_kind,
            strict_round=round_idx > 0,
            force_logic_change=prev_round_no_change,
            current_completion=current,
            entry_point=entry_point,
            rewrite_mode=rewrite_mode,
        )
        if aggressive_mode:
            failure_context = (
                f"{failure_context}\n"
                "Aggressive targeted repair mode: patch exactly the first failing assert behavior. "
                "Prefer one focused logic edit over broad refactor."
            )
        if rewrite_mode:
            failure_context = (
                f"{failure_context}\n"
                "Fresh rewrite instruction: produce a new minimal body from the problem and asserts; "
                "do not preserve wrong control flow from the current implementation."
            )

        if llm_client is not None and _looks_like_repairable_runtime_failure(error_message):
            llm_attempt: Dict[str, Any] = {
                "round": round_idx + 1,
                "repair_type": "assertion_llm",
                "original_error": error_message,
                "failure_kind": failure_kind,
                "assert_score_passed": int(passed_now),
                "assert_score_total": int(total_now),
                "first_failing_assert": last_first_failing_assert,
            }
            repaired_llm = humaneval_postprocess._repair_assertion_completion_with_llm(
                llm_client=llm_client,
                prompt=llm_repair_prompt,
                completion=current,
                test_code=test_code,
                entry_point=entry_point,
                error_message=error_message,
                failure_context=failure_context,
            )
            if repaired_llm and repaired_llm != current:
                normalized_llm = eval_humaneval._normalize_completion(
                    humaneval_postprocess._strip_redundant_def(
                        eval_humaneval._extract_code_block(str(repaired_llm)),
                        entry_point,
                    ),
                    entry_point,
                )
                repaired_llm = humaneval_postprocess._fix_missing_indents(normalized_llm, entry_point)
                if not repaired_llm.strip():
                    llm_attempt["success"] = False
                    llm_attempt["rejected"] = "empty_after_normalize"
                    llm_attempt["final_error"] = "empty_after_normalize"
                    attempts.append(llm_attempt)
                    continue
                gate_error = _repair_hard_gate_error(
                    execution_prompt,
                    repaired_llm,
                    entry_point,
                    test_code=test_code,
                    timeout_s=timeout_s,
                )
                llm_attempt["hard_gate_mode"] = _mbpp_hard_gate_mode(repaired_llm, entry_point)
                if gate_error:
                    llm_attempt["success"] = False
                    llm_attempt["rejected"] = "hard_gate_failed"
                    llm_attempt["final_error"] = gate_error
                    attempts.append(llm_attempt)
                else:
                    llm_quality = _get_quality(repaired_llm)
                    llm_attempt["candidate_score"] = f"{llm_quality['passed']}/{llm_quality['total']}"
                    if llm_quality.get("syntax_error"):
                        llm_attempt["success"] = False
                        llm_attempt["rejected"] = "syntax_error"
                        llm_attempt["final_error"] = str(llm_quality.get("syntax_error") or "")
                        attempts.append(llm_attempt)
                    elif _candidate_quality_regressed(llm_quality, current_quality):
                        llm_attempt["success"] = False
                        llm_attempt["rejected"] = "score_dropped"
                        llm_attempt["final_error"] = "Rejected candidate because assert score regressed."
                        attempts.append(llm_attempt)
                    elif _is_non_minimal_change(current, repaired_llm, current_quality, llm_quality) and not aggressive_mode:
                        llm_attempt["success"] = False
                        llm_attempt["rejected"] = "non_minimal_edit"
                        llm_attempt["final_error"] = "Rejected candidate because structural edit is too large."
                        attempts.append(llm_attempt)
                    else:
                        if _is_non_minimal_change(current, repaired_llm, current_quality, llm_quality) and aggressive_mode:
                            llm_attempt["gate_relaxed"] = "non_minimal_edit_once"
                            aggressive_relax_used = True
                        ok_llm = bool(llm_quality.get("full_ok"))
                        llm_attempt["success"] = ok_llm
                        llm_attempt["final_error"] = "" if ok_llm else str(llm_quality.get("full_error") or "execution failed")
                        attempts.append(llm_attempt)
                        if ok_llm:
                            return repaired_llm, {
                                "success": True,
                                "attempts": attempts,
                                "assert_score_passed": int(llm_quality.get("passed") or 0),
                                "assert_score_total": int(llm_quality.get("total") or 0),
                                "first_failure_kind": str(failure_kind or ""),
                                "first_failing_assert": last_first_failing_assert,
                            }
                        current = repaired_llm
                        current_quality = llm_quality
                        error_message = str(llm_quality.get("full_error") or "execution failed").strip()
                        round_changed = True
            else:
                llm_attempt["success"] = False
                llm_attempt["no_change"] = True
                llm_attempt["final_error"] = error_message
                attempts.append(llm_attempt)

        param_names = humaneval_postprocess._extract_param_names_from_prompt(execution_prompt, entry_point)
        repaired_text, post_meta = humaneval_postprocess._run_postprocess_tool_agent(
            completion=current,
            prompt=execution_prompt,
            entry_point=entry_point,
            stop_tokens=[],
            use_stop_tokens=False,
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=failure_context,
        )
        attempt = {
            "round": round_idx + 1,
            "tools": post_meta.get("applied_tools", []),
            "attempted_tools": post_meta.get("attempted_tools", []),
            "original_error": error_message,
            "failure_kind": failure_kind,
            "assert_score_passed": int(passed_now),
            "assert_score_total": int(total_now),
            "first_failing_assert": last_first_failing_assert,
        }
        if isinstance(post_meta.get("tool_errors"), dict) and post_meta.get("tool_errors"):
            attempt["tool_errors"] = post_meta.get("tool_errors")
        if not repaired_text or repaired_text == current:
            attempt["no_change"] = True
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue

        normalized = eval_humaneval._normalize_completion(
            humaneval_postprocess._strip_redundant_def(
                eval_humaneval._extract_code_block(str(repaired_text)),
                entry_point,
            ),
            entry_point,
        )
        repaired_candidate = humaneval_postprocess._fix_missing_indents(normalized, entry_point)
        if not repaired_candidate.strip():
            attempt["rejected"] = "empty_after_normalize"
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue
        gate_error = _repair_hard_gate_error(
            execution_prompt,
            repaired_candidate,
            entry_point,
            test_code=test_code,
            timeout_s=timeout_s,
        )
        attempt["hard_gate_mode"] = _mbpp_hard_gate_mode(repaired_candidate, entry_point)
        if gate_error:
            attempt["rejected"] = "hard_gate_failed"
            attempt["final_error"] = gate_error
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue

        repaired_quality = _get_quality(repaired_candidate)
        attempt["candidate_score"] = f"{repaired_quality['passed']}/{repaired_quality['total']}"
        if repaired_quality.get("syntax_error"):
            attempt["rejected"] = "syntax_error"
            attempt["final_error"] = str(repaired_quality.get("syntax_error") or "")
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue
        if _candidate_quality_regressed(repaired_quality, current_quality):
            attempt["rejected"] = "score_dropped"
            attempt["final_error"] = "Rejected candidate because assert score regressed."
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue
        if _is_non_minimal_change(current, repaired_candidate, current_quality, repaired_quality) and not aggressive_mode:
            attempt["rejected"] = "non_minimal_edit"
            attempt["final_error"] = "Rejected candidate because structural edit is too large."
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            no_change_streak = no_change_streak + 1 if prev_round_no_change else 0
            continue
        if _is_non_minimal_change(current, repaired_candidate, current_quality, repaired_quality) and aggressive_mode:
            attempt["gate_relaxed"] = "non_minimal_edit_once"
            aggressive_relax_used = True

        current = repaired_candidate
        attempts.append(attempt)
        round_changed = True
        if repaired_quality.get("full_ok"):
            return current, {
                "success": True,
                "attempts": attempts,
                "assert_score_passed": int(repaired_quality.get("passed") or 0),
                "assert_score_total": int(repaired_quality.get("total") or 0),
                "first_failure_kind": str(failure_kind or ""),
                "first_failing_assert": last_first_failing_assert,
            }
        prev_round_no_change = not round_changed
        no_change_streak = no_change_streak + 1 if prev_round_no_change else 0

    return current, {
        "success": False,
        "attempts": attempts,
        "assert_score_passed": int(last_assert_score_passed),
        "assert_score_total": int(last_assert_score_total),
        "first_failure_kind": str(last_failure_kind or ""),
        "first_failing_assert": str(last_first_failing_assert or ""),
    }


def _postprocess_mbpp_solutions(
    tasks: List[Dict[str, Any]],
    solutions: Dict[str, str],
    meta: Dict[str, Dict[str, Any]],
    rounds: int,
    timeout_s: float,
    llm_client: Optional[Any] = None,
) -> None:
    task_by_name = {str(t.get("name") or t.get("task_id") or ""): t for t in tasks}
    for name, task in task_by_name.items():
        if str(task.get("source") or "").lower() != "mbpp":
            continue
        task_meta = meta.get(name, {})
        completion = str(solutions.get(name, ""))
        if _looks_like_error_text(completion):
            completion = ""
            solutions[name] = ""
        original_completion = completion
        if not completion.strip():
            meta[name] = task_meta
            continue
        prompt = str(task.get("prompt") or "")
        test_code = str(task.get("test") or "")
        entry_point = task.get("entry_point")
        precheck_before = completion
        precheck_meta: Dict[str, Any] = {"attempted": bool(entry_point), "auto_repairs": []}
        precheck_meta["completion_shape"] = _mbpp_completion_shape(precheck_before, entry_point)
        precheck_meta["flattened_control_detected"] = _mbpp_has_flattened_control_flow(precheck_before, entry_point)
        precheck_meta["return_risk"] = _mbpp_return_risk(precheck_before, entry_point)
        precheck_error = None
        if entry_point:
            auto_repaired, auto_meta = _run_mbpp_precheck_auto_repairs(
                prompt=prompt,
                completion=completion,
                test_code=test_code,
                entry_point=entry_point,
                timeout_s=timeout_s,
            )
            if auto_repaired and auto_repaired.strip() and auto_repaired != completion:
                completion = auto_repaired
                solutions[name] = completion
            if isinstance(auto_meta, dict):
                precheck_meta.update(auto_meta)
            precheck_error = _syntax_precheck(prompt, completion, entry_point)
        if precheck_error:
            repaired, precheck_post_meta = _postprocess_completion(
                prompt,
                completion,
                entry_point,
                rounds=1,
                error_message=precheck_error,
            )
            precheck_meta["attempted"] = True
            precheck_meta["error"] = precheck_error
            precheck_meta["postprocess"] = precheck_post_meta
            precheck_meta["changed"] = repaired.strip() != completion.strip()
            if repaired and repaired.strip():
                completion = repaired
                solutions[name] = completion
        else:
            precheck_meta["changed"] = completion.strip() != precheck_before.strip()
        indent_auto_repair_applied = any(
            isinstance(item, dict)
            and bool(item.get("applied"))
            and str(item.get("kind") or "").startswith("indent_")
            for item in precheck_meta.get("auto_repairs", [])
        )
        precheck_postprocess_meta = precheck_meta.get("postprocess")
        indent_postprocess_applied = bool(
            isinstance(precheck_postprocess_meta, dict)
            and "postprocess-syntax-indent-repair" in list(precheck_postprocess_meta.get("applied_tools", []) or [])
            and bool(precheck_meta.get("changed"))
        )
        precheck_meta["indent_repair_applied"] = bool(
            indent_auto_repair_applied or indent_postprocess_applied
        )
        task_meta["mbpp_precheck"] = precheck_meta
        small_passed = 0
        small_total = 0
        small_runtime_error = ""
        small_ok_before = True
        if entry_point and test_code:
            small_passed, small_total, small_runtime_error = _score_candidate_against_asserts(
                prompt,
                completion,
                test_code,
                entry_point,
                timeout_s=timeout_s,
            )
            small_ok_before = bool(
                not small_runtime_error and (small_total == 0 or small_passed == small_total)
            )

        post_meta: Dict[str, Any] = {
            "attempted": False,
            "success": small_ok_before,
            "small_score_before": f"{small_passed}/{small_total}",
        }
        if small_runtime_error:
            post_meta["small_runtime_error_before"] = small_runtime_error
        post_attempted = False
        if not small_ok_before:
            post_attempted = True
            repaired, post_meta = _postprocess_after_failure(
                prompt,
                completion,
                test_code,
                entry_point,
                timeout_s=timeout_s,
                rounds=rounds,
                llm_client=llm_client,
                small_tests_only=False,
            )
            if repaired and repaired.strip():
                completion = repaired
                solutions[name] = completion

        post_meta = _normalize_mbpp_postprocess_meta(
            post_meta,
            attempted=post_attempted,
            small_score_before=f"{small_passed}/{small_total}",
        )
        task_meta["postprocess"] = post_meta
        task_meta["mbpp_postprocess"] = {
            "rounds": int(rounds),
            "applied_tools": _collect_applied_tools(post_meta.get("attempts", [])),
            "fallback_detected": False,
            "changed": completion.strip() != original_completion.strip(),
            "assert_score_passed": int(post_meta.get("assert_score_passed") or 0),
            "assert_score_total": int(post_meta.get("assert_score_total") or 0),
            "first_failure_kind": str(post_meta.get("first_failure_kind") or ""),
            "first_failing_assert": str(post_meta.get("first_failing_assert") or ""),
        }
        meta[name] = task_meta


def _patch_eval_humaneval_for_mbpp() -> None:
    global _MBPP_PATCHED
    if _MBPP_PATCHED:
        return

    original_orchestrator = eval_humaneval._auto_generate_solutions_orchestrator
    original_local_generator = eval_humaneval._auto_generate_solutions

    def wrapped_orchestrator(*args: Any, **kwargs: Any):
        tasks = kwargs.get("tasks")
        if tasks is None and args:
            tasks = args[0]
        out_path = kwargs.get("out_path")
        if out_path is None and len(args) > 1:
            out_path = args[1]
        timeout_s = float(kwargs.get("tool_timeout_s") or 5.0)
        llm_client = _build_mbpp_llm_client(kwargs)

        call_args = list(args)
        call_kwargs = dict(kwargs)
        call_kwargs["strict_prompt"] = True
        if not str(call_kwargs.get("strict_prompt_text") or "").strip():
            call_kwargs["strict_prompt_text"] = _MBPP_STRICT_PROMPT_TEXT
        prepared_tasks = _prepare_mbpp_tasks_for_generation(tasks) if isinstance(tasks, list) else tasks
        if isinstance(prepared_tasks, list):
            if "tasks" in call_kwargs:
                call_kwargs["tasks"] = prepared_tasks
            elif call_args:
                call_args[0] = prepared_tasks

        solutions, meta = original_orchestrator(*call_args, **call_kwargs)
        effective_tasks = prepared_tasks if isinstance(prepared_tasks, list) else tasks
        if isinstance(effective_tasks, list) and isinstance(solutions, dict) and isinstance(meta, dict):
            _postprocess_mbpp_solutions(
                effective_tasks,
                solutions,
                meta,
                rounds=_MBPP_POSTPROCESS_ROUNDS,
                timeout_s=timeout_s,
                llm_client=llm_client,
            )
            if out_path:
                with open(str(out_path), "w", encoding="utf-8") as f:
                    for task in effective_tasks:
                        name = str(task.get("name") or task.get("task_id") or "")
                        completion = str(solutions.get(name, ""))
                        f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
        return solutions, meta

    def wrapped_local_generator(*args: Any, **kwargs: Any):
        tasks = kwargs.get("tasks")
        if tasks is None and args:
            tasks = args[0]
        out_path = kwargs.get("out_path")
        if out_path is None and len(args) > 1:
            out_path = args[1]
        timeout_s = 5.0
        llm_client = _build_mbpp_llm_client(kwargs)

        call_args = list(args)
        call_kwargs = dict(kwargs)
        call_kwargs["strict_prompt"] = True
        if not str(call_kwargs.get("strict_prompt_text") or "").strip():
            call_kwargs["strict_prompt_text"] = _MBPP_STRICT_PROMPT_TEXT
        prepared_tasks = _prepare_mbpp_tasks_for_generation(tasks) if isinstance(tasks, list) else tasks
        if isinstance(prepared_tasks, list):
            if "tasks" in call_kwargs:
                call_kwargs["tasks"] = prepared_tasks
            elif call_args:
                call_args[0] = prepared_tasks

        solutions = original_local_generator(*call_args, **call_kwargs)
        effective_tasks = prepared_tasks if isinstance(prepared_tasks, list) else tasks
        if isinstance(effective_tasks, list) and isinstance(solutions, dict):
            meta: Dict[str, Dict[str, Any]] = {}
            _postprocess_mbpp_solutions(
                effective_tasks,
                solutions,
                meta,
                rounds=_MBPP_POSTPROCESS_ROUNDS,
                timeout_s=timeout_s,
                llm_client=llm_client,
            )
            if out_path:
                with open(str(out_path), "w", encoding="utf-8") as f:
                    for task in effective_tasks:
                        name = str(task.get("name") or task.get("task_id") or "")
                        completion = str(solutions.get(name, ""))
                        f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
        return solutions

    eval_humaneval._auto_generate_solutions_orchestrator = wrapped_orchestrator
    eval_humaneval._auto_generate_solutions = wrapped_local_generator
    _MBPP_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP by converting it to HumanEval-style tasks and reusing eval_humaneval"
    )
    parser.add_argument("--mbpp_tasks", required=True, type=str, help="MBPP jsonl path")
    parser.add_argument(
        "--converted_tasks",
        default="",
        type=str,
        help="optional converted tasks path; if empty, use a temp file",
    )
    parser.add_argument("--keep_converted", action="store_true", help="keep temp converted file")
    parser.add_argument("--include_challenge_tests", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--mbpp_postprocess_rounds",
        type=int,
        default=_MBPP_POSTPROCESS_ROUNDS,
        help="MBPP postprocess repair rounds after initial failure",
    )
    args, remainder = parser.parse_known_args()
    setattr(args, "remainder", remainder)
    return args


def main() -> None:
    args = parse_args()
    global _MBPP_POSTPROCESS_ROUNDS
    _MBPP_POSTPROCESS_ROUNDS = max(1, int(args.mbpp_postprocess_rounds))
    _patch_eval_humaneval_for_mbpp()

    use_temp = not bool(args.converted_tasks)
    temp_path = ""
    converted_path = args.converted_tasks
    if use_temp:
        fd, temp_path = tempfile.mkstemp(prefix="mbpp_eval_tasks_", suffix=".jsonl")
        os.close(fd)
        converted_path = temp_path

    summary = convert_mbpp_file(
        input_path=args.mbpp_tasks,
        output_path=converted_path,
        include_challenge_tests=args.include_challenge_tests,
        max_samples=args.max_samples,
    )
    print(
        f"[eval_mbpp] converted {summary['converted_total']}/{summary['input_total']} tasks "
        f"to {summary['output']} (skipped={summary['skipped_total']})"
    )

    forwarded = _strip_tasks_arg(list(args.remainder))
    forwarded = _ensure_include_tool_trace(forwarded)
    forwarded = _ensure_postprocess_rounds(forwarded, 0)
    he_args = ["eval_humaneval", "--tasks", converted_path] + forwarded
    old_argv = list(sys.argv)
    try:
        sys.argv = he_args
        eval_humaneval.main()
    finally:
        sys.argv = old_argv
        if use_temp and temp_path and (not args.keep_converted):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
