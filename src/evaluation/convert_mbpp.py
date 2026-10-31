from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STOP_TOKENS = ["\ndef", "\n#", "\nif", "\nclass"]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_first_function_from_code(code: str) -> Optional[ast.FunctionDef]:
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _extract_entry_point_from_tests(tests: List[str]) -> Optional[str]:
    for test in tests:
        m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(test))
        if m:
            return m.group(1)
    return None


def _extract_function_from_code_by_name(code: str, name: str) -> Optional[ast.FunctionDef]:
    if not code or not name:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _extract_entry_point(row: Dict[str, Any]) -> Optional[str]:
    explicit = str(row.get("entry_point") or "").strip()
    if explicit:
        return explicit

    tests: List[str] = list(row.get("test_list") or [])
    test_entry = _extract_entry_point_from_tests(tests)
    if test_entry:
        return test_entry

    code = str(row.get("code") or "")
    fn = _extract_first_function_from_code(code)
    if fn is not None:
        return fn.name

    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    if m:
        return m.group(1)
    return None


def _build_signature(entry_point: str, code: str) -> str:
    fn = _extract_function_from_code_by_name(code, entry_point) or _extract_first_function_from_code(code)
    if fn is None:
        return f"def {entry_point}(*args, **kwargs):"

    args_src = ast.unparse(fn.args) if hasattr(ast, "unparse") else "*args, **kwargs"
    ret_src = ""
    if fn.returns is not None and hasattr(ast, "unparse"):
        ret_src = f" -> {ast.unparse(fn.returns)}"
    return f"def {entry_point}({args_src}){ret_src}:"


def _build_prompt(signature: str, text: str, constraints: Optional[List[str]] = None) -> str:
    doc = (text or "Implement the function according to the problem statement.").strip()
    doc = doc.replace('"""', '\\"\\"\\"')
    lines = doc.splitlines() if doc else []
    if constraints:
        if lines:
            lines.append("")
        lines.append("Output constraints:")
        lines.extend(f"- {line}" for line in constraints if line)
    body = "\n".join(f"    {line}" for line in lines) if lines else "    "
    return f"{signature}\n    \"\"\"\n{body}\n    \"\"\"\n"


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


def _unwrap_candidate_call(node: ast.AST, entry_point: str) -> Tuple[Optional[ast.Call], bool]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"candidate", entry_point}:
            return node, True
        if node.func.id in {"sorted", "set", "list", "tuple"} and node.args:
            inner = node.args[0]
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in {"candidate", entry_point}
            ):
                return inner, node.func.id not in {"sorted", "set"}
    return None, True


def _literal_length(node: ast.AST) -> Optional[int]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)
    return None


def _is_list_of_tuples(node: ast.AST) -> bool:
    return isinstance(node, ast.List) and node.elts and all(isinstance(item, ast.Tuple) for item in node.elts)


def _extract_output_constraints(tests: List[str], entry_point: str) -> List[str]:
    return_types: set[str] = set()
    lengths: set[int] = set()
    order_required = False
    list_of_tuples = False

    for raw in tests:
        line = str(raw).strip()
        if not line.startswith("assert "):
            continue
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
        _, left_order_sensitive = _unwrap_candidate_call(left, entry_point)

        ret_type = _infer_node_type(right)
        if ret_type != "unknown":
            return_types.add(ret_type)
            if ret_type in {"list", "tuple"} and left_order_sensitive:
                order_required = True
            if ret_type in {"list", "tuple"}:
                length = _literal_length(right)
                if length is not None:
                    lengths.add(length)
            if ret_type == "list" and _is_list_of_tuples(right):
                list_of_tuples = True

    constraints: List[str] = []
    if return_types:
        if list_of_tuples:
            type_desc = "list of tuples"
        elif len(return_types) == 1:
            type_desc = next(iter(return_types))
        else:
            type_desc = "/".join(sorted(return_types))
        constraints.append(f"Return type: {type_desc}.")
    if len(lengths) == 1:
        constraints.append(f"Return length: {next(iter(lengths))}.")
    if order_required:
        constraints.append("Preserve order in the returned sequence (avoid set/sorted if it changes order).")
    return constraints


def _to_candidate_assert(assert_line: str, entry_point: str) -> str:
    line = str(assert_line).strip()
    if not line.startswith("assert "):
        return f"# {line}"
    # Prefer replacing the declared entry point call.
    pattern = rf"\b{re.escape(entry_point)}\s*\("
    replaced = re.sub(pattern, "candidate(", line, count=1)
    if replaced != line:
        return replaced

    # Fallback: replace the first direct function call after `assert`.
    # This guards against noisy MBPP samples whose assert target name
    # does not match the extracted entry point.
    return re.sub(
        r"^(\s*assert\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*\()",
        r"\1candidate\3",
        line,
        count=1,
    )


def _build_test_code(row: Dict[str, Any], entry_point: str, include_challenge: bool) -> str:
    setup = str(row.get("test_setup_code") or "").strip()
    tests: List[str] = list(row.get("test_list") or [])
    if include_challenge:
        tests.extend(list(row.get("challenge_test_list") or []))
    candidate_asserts = [_to_candidate_assert(x, entry_point) for x in tests]
    check_lines = ["def check(candidate):"]
    if candidate_asserts:
        check_lines.extend([f"    {line}" for line in candidate_asserts])
    else:
        check_lines.append("    pass")
    check_block = "\n".join(check_lines)
    parts = []
    if setup:
        parts.append(setup)
    parts.append(check_block)
    parts.append("def test_check():")
    parts.append(f"    check({entry_point})")
    parts.append("")
    parts.append("test_check()")
    return "\n".join(parts) + "\n"


def convert_mbpp_rows(
    rows: List[Dict[str, Any]],
    *,
    include_challenge_tests: bool = False,
    name_prefix: str = "MBPP",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    converted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for row in rows:
        task_id = row.get("task_id")
        text = str(row.get("text") or "")
        code = str(row.get("code") or "")
        entry_point = _extract_entry_point(row)
        if not entry_point:
            skipped.append({"task_id": task_id, "reason": "missing_entry_point"})
            continue

        signature = _build_signature(entry_point, code)
        tests: List[str] = list(row.get("test_list") or [])
        if include_challenge_tests:
            tests.extend(list(row.get("challenge_test_list") or []))
        constraints = _extract_output_constraints(tests, entry_point)
        prompt = _build_prompt(signature, text, constraints)
        test_code = _build_test_code(row, entry_point, include_challenge_tests)

        mbpp_id = f"{name_prefix}_{task_id}"
        name = f"{mbpp_id}_{entry_point}"
        converted.append(
            {
                "name": name,
                "task_id": mbpp_id,
                "language": "py",
                "prompt": prompt,
                "entry_point": entry_point,
                "stop_tokens": STOP_TOKENS,
                "test": test_code,
                "original_task_id": task_id,
                "source": "mbpp",
            }
        )
    return converted, skipped


def convert_mbpp_file(
    input_path: str,
    output_path: str,
    *,
    include_challenge_tests: bool = False,
    max_samples: int = 0,
) -> Dict[str, Any]:
    rows = _load_jsonl(input_path)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    converted, skipped = convert_mbpp_rows(
        rows,
        include_challenge_tests=include_challenge_tests,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "input_total": len(rows),
        "converted_total": len(converted),
        "skipped_total": len(skipped),
        "skipped": skipped,
        "output": str(out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MBPP jsonl to HumanEval-style tasks jsonl")
    parser.add_argument("--input", required=True, type=str, help="path to MBPP jsonl")
    parser.add_argument("--output", required=True, type=str, help="output jsonl path")
    parser.add_argument("--include_challenge_tests", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_mbpp_file(
        input_path=args.input,
        output_path=args.output,
        include_challenge_tests=args.include_challenge_tests,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
