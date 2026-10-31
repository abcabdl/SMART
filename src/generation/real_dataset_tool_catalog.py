from __future__ import annotations

from typing import Any, Dict, List


CODE_AGENT_ROLE_TAGS = [
    "code-generation",
    "code-planner",
    "code-testing",
    "code-refactoring",
    "builder",
    "planner",
    "checker",
    "refactor",
]

MATH_AGENT_ROLE_TAGS = ["builder", "planner", "checker", "refactor"]


def _tool_record(
    *,
    tool_id: str,
    name: str,
    dataset: str,
    description: str,
    code: str,
    domain_tags: List[str],
    role_tags: List[str],
    tool_tags: List[str],
    inputs: List[str],
    outputs: List[str] | None = None,
    dataset_family: str = "",
    focus_slug: str = "",
    primary_role: str = "",
    capability_kind: str = "",
    stage: str = "",
    consumes: List[str] | None = None,
    produces: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "id": tool_id,
        "dataset": dataset,
        "spec": {
            "name": name,
            "description": description,
            "requirements": "standard-library",
            "domain_tags": list(domain_tags),
            "role_tags": list(role_tags),
            "tool_tags": list(tool_tags),
            "inputs": list(inputs),
            "outputs": list(outputs or ["output"]),
            "requires_api": False,
            "api_notes": "",
            "modalities": ["text"],
            "output_formats": ["json"],
            "permissions": ["read", "compute"],
            "cost_tier": "low",
            "latency_tier": "low",
            "reliability_prior": 0.9,
            "examples": [description],
            "embedding_text": f"{name}. {description}. Tags: {' '.join(tool_tags)}",
            "dataset_family": dataset_family or dataset,
            "focus_slug": focus_slug,
            "primary_role": primary_role,
            "capability_kind": capability_kind,
            "stage": stage,
            "deterministic": True,
            "consumes": list(consumes or []),
            "produces": list(produces or []),
        },
        "code": code,
    }


def _capability_metadata(dataset: str, capability_kind: str) -> Dict[str, Any]:
    if dataset in {"humaneval", "mbpp"}:
        mapping = {
            "signature-extractor": ("analyze", ["task", "query", "prompt"], ["entry_point", "diagnostics"]),
            "code-normalizer": ("prepare", ["draft_code", "entry_point"], ["normalized_code", "diagnostics"]),
            "structure-checker": ("verify", ["normalized_code", "draft_code"], ["verification", "diagnostics"]),
            "public-test-runner": ("execute", ["normalized_code", "draft_code", "tests_public"], ["verification", "diagnostics"]),
            "traceback-localizer": ("diagnose", ["draft_code", "diagnostics"], ["diagnostics"]),
            "assert-parser": ("analyze", ["task", "query", "tests_public"], ["diagnostics"]),
            "import-injector": ("prepare", ["draft_code"], ["normalized_code", "diagnostics"]),
            "contract-gate": ("verify", ["draft_code", "entry_point"], ["verification", "diagnostics"]),
            "assert-scorer": ("execute", ["normalized_code", "draft_code", "tests_public"], ["verification", "diagnostics"]),
            "failure-diagnoser": ("diagnose", ["draft_code", "diagnostics"], ["diagnostics"]),
        }
    else:
        mapping = {
            "expression-engine": ("solve_support", ["task", "query", "reasoning_trace"], ["reasoning_trace", "diagnostics"]),
            "equation-solver": ("solve_support", ["task", "query", "reasoning_trace"], ["reasoning_trace", "diagnostics"]),
            "number-theory-inspector": ("analyze", ["task", "query"], ["reasoning_trace", "diagnostics"]),
            "geometry-measurement-engine": ("solve_support", ["task", "query", "reasoning_trace"], ["reasoning_trace", "diagnostics"]),
            "answer-canonicalizer-verifier": ("format", ["candidate_answer"], ["formatted_answer", "verification", "final_answer"]),
        }
    stage, io = mapping.get(capability_kind, ("analyze", ["task"], ["diagnostics"]))
    consumes, produces = io
    return {
        "capability_kind": capability_kind,
        "stage": stage,
        "consumes": consumes,
        "produces": produces,
    }


def _infer_capability_kind(tool_id: str) -> str:
    parts = str(tool_id).split("-")
    if len(parts) <= 1:
        return str(tool_id)
    return "-".join(parts[1:])


HUMANEVAL_SIGNATURE_CONTRACT_EXTRACTOR = """import ast
import re


_DEF_RE = re.compile(r"def\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*?)\\):", re.DOTALL)


def _extract_parameters(signature_text):
    params_raw = signature_text.strip()
    if not params_raw:
        return []
    params = []
    for item in params_raw.split(","):
        name = item.strip()
        if not name:
            continue
        name = name.split("=", 1)[0].strip()
        name = name.lstrip("*").strip()
        if name:
            params.append(name)
    return params


def _infer_arity_from_tests(test_list, entry_point):
    arity = None
    samples = []
    for raw in test_list:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"candidate", entry_point}:
                samples.append(ast.unparse(node) if hasattr(ast, "unparse") else node.func.id)
                arity = max(arity or 0, len(node.args) + len(node.keywords))
    return arity, samples[:10]


def run(inputs):
    prompt = str(inputs.get("prompt") or inputs.get("query") or inputs.get("task") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    tests = test_list if test_list else ([test] if test else [])
    match = _DEF_RE.search(prompt)
    parameters = []
    signature_text = ""
    inferred = False
    if match:
        entry_point = entry_point or match.group(1)
        signature_text = match.group(0)
        parameters = _extract_parameters(match.group(2))
    inferred_arity, test_calls = _infer_arity_from_tests(tests, entry_point or "candidate")
    if not parameters and inferred_arity is not None:
        parameters = [f"arg{i}" for i in range(inferred_arity)]
        inferred = True
    return {
        "output": {
            "entry_point": entry_point or None,
            "parameters": parameters,
            "arity": len(parameters),
            "signature_text": signature_text,
            "inferred_from_tests": inferred,
            "observed_test_calls": test_calls,
        }
    }
"""


HUMANEVAL_CODE_NORMALIZER = """import ast
import re
import textwrap


def run(inputs):
    code = str(inputs.get("code") or inputs.get("query") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    if not code.strip():
        raise Exception("Missing code")
    text = re.sub(r"^```(?:python)?\\s*", "", code.strip())
    text = re.sub(r"```\\s*$", "", text)
    normalized = textwrap.dedent(text).strip()
    if normalized:
        normalized += "\\n"
    extracted = normalized
    if entry_point:
        pattern = re.compile(
            r"(^def\\s+" + re.escape(entry_point) + r"\\s*\\(.*?)(?=^def\\s+|\\Z)",
            re.DOTALL | re.MULTILINE,
        )
        match = pattern.search(normalized)
        if match:
            extracted = match.group(1).rstrip() + "\\n"
    syntax_ok = True
    syntax_error = ""
    target = extracted or normalized
    try:
        ast.parse(target)
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = str(exc)
    return {
        "output": {
            "normalized_code": normalized,
            "extracted_function": extracted,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_error,
        }
    }
"""


HUMANEVAL_AST_STRUCTURE_CHECKER = """import ast


def run(inputs):
    code = str(inputs.get("code") or "")
    if not code.strip():
        raise Exception("Missing code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"output": {"valid": False, "syntax_error": str(exc)}}
    functions = []
    imports = []
    loop_count = 0
    branch_count = 0
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        elif isinstance(node, ast.If):
            branch_count += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    recursive_functions = [name for name in functions if name in calls]
    return {
        "output": {
            "valid": True,
            "functions": functions,
            "imports": imports,
            "loop_count": loop_count,
            "branch_count": branch_count,
            "recursive_functions": recursive_functions,
        }
    }
"""


HUMANEVAL_PUBLIC_TEST_RUNNER = """import json
import subprocess
import sys
import tempfile


def _build_test_body(entry_point, test, test_list):
    body = "\\n".join(str(item) for item in test_list) if test_list else str(test)
    if entry_point:
        body = body.replace("candidate(", f"{entry_point}(")
        body = body.replace("check(candidate)", f"check({entry_point})")
        alias = f"candidate = {entry_point}\\n"
    else:
        alias = ""
    return alias + body


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    if not code.strip():
        raise Exception("Missing code")
    if not test.strip() and not test_list:
        raise Exception("Missing test or test_list")
    body = _build_test_body(entry_point, test, test_list)
    harness = (
        "import json\\n"
        + code
        + "\\n"
        + body
        + "\\n"
        + "print(json.dumps({'passed': True}))\\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    completed = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    payload = {"passed": False}
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and stdout_lines:
        try:
            payload = json.loads(stdout_lines[-1])
        except Exception:
            payload = {"passed": False, "error": "non_json_success_output"}
    elif completed.stderr.strip():
        payload = {"passed": False, "error": completed.stderr.splitlines()[-1]}
    payload["returncode"] = completed.returncode
    payload["stdout"] = completed.stdout[-400:]
    payload["stderr"] = completed.stderr[-400:]
    return {"output": payload}
"""


HUMANEVAL_TRACEBACK_LOCALIZER = """def run(inputs):
    code = str(inputs.get("code") or "")
    error = str(inputs.get("traceback") or inputs.get("error") or "")
    if not error.strip():
        raise Exception("Missing traceback or error")
    lines = code.splitlines()
    focus = []
    for raw in error.splitlines():
        if ", line " not in raw:
            continue
        try:
            line_no = int(raw.split(", line ", 1)[1].split(",", 1)[0].strip())
        except Exception:
            continue
        context = []
        for pos in range(max(0, line_no - 3), min(len(lines), line_no + 2)):
            context.append({"line": pos + 1, "code": lines[pos]})
        focus.append({"line": line_no, "context": context})
    return {
        "output": {
            "error_summary": error.splitlines()[-1] if error.splitlines() else error,
            "focus": focus,
        }
    }
"""


MBPP_ASSERT_EXAMPLE_PARSER = """import ast


def _parse_assert(line):
    try:
        node = ast.parse(line).body[0]
    except SyntaxError:
        return None
    if not isinstance(node, ast.Assert):
        return None
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    left = test.left
    if not isinstance(left, ast.Call) or not isinstance(left.func, ast.Name):
        return None
    if left.func.id != "candidate":
        return None
    expected = ast.unparse(test.comparators[0]) if hasattr(ast, "unparse") else ""
    call = ast.unparse(left) if hasattr(ast, "unparse") else "candidate(...)"
    return {"call": call, "arg_count": len(left.args) + len(left.keywords), "expected": expected}


def run(inputs):
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    items = test_list if test_list else test.splitlines()
    examples = []
    for item in items:
        parsed = _parse_assert(str(item).strip())
        if parsed is not None:
            examples.append(parsed)
    return {"output": {"example_count": len(examples), "examples": examples[:20]}}
"""


MBPP_IMPORT_SYMBOL_INJECTOR = """import ast


_SYMBOL_IMPORTS = {
    "factorial": ("math", "factorial"),
    "sqrt": ("math", "sqrt"),
    "gcd": ("math", "gcd"),
    "ceil": ("math", "ceil"),
    "floor": ("math", "floor"),
    "Counter": ("collections", "Counter"),
    "defaultdict": ("collections", "defaultdict"),
    "deque": ("collections", "deque"),
    "combinations": ("itertools", "combinations"),
    "permutations": ("itertools", "permutations"),
}

_MODULE_IMPORTS = {"math", "collections", "itertools", "heapq", "functools", "statistics", "re"}


def run(inputs):
    code = str(inputs.get("code") or "")
    if not code.strip():
        raise Exception("Missing code")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.parse("")
    imported_modules = set()
    imported_symbols = set()
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".", 1)[0])
            for alias in node.names:
                imported_symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
    injections = []
    for name in sorted(used_names):
        if name in _SYMBOL_IMPORTS and name not in imported_symbols:
            module, symbol = _SYMBOL_IMPORTS[name]
            injections.append(f"from {module} import {symbol}")
        elif name in _MODULE_IMPORTS and name not in imported_modules:
            injections.append(f"import {name}")
    injected_imports = []
    for line in injections:
        if line not in injected_imports:
            injected_imports.append(line)
    patched_code = code
    if injected_imports:
        patched_code = "\\n".join(injected_imports) + "\\n" + code
    return {
        "output": {
            "patched_code": patched_code,
            "injected_imports": injected_imports,
            "missing_symbols": [line.split()[-1] for line in injected_imports],
        }
    }
"""


MBPP_CONTRACT_GATE = """import ast


def _expected_arities(test_list):
    arities = []
    for line in test_list:
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.Assert):
            continue
        compare = node.test
        if not isinstance(compare, ast.Compare):
            continue
        call = compare.left
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "candidate":
            arities.append(len(call.args) + len(call.keywords))
    return arities


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    tests = test_list if test_list else ([line.strip() for line in test.splitlines() if line.strip()])
    if not code.strip():
        raise Exception("Missing code")
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    expected_arities = _expected_arities(tests)
    function_found = entry_point in functions
    parameter_names = []
    parameter_count = 0
    if function_found:
        parameter_names = [arg.arg for arg in functions[entry_point].args.args]
        parameter_count = len(parameter_names)
    arity_match = (not expected_arities) or all(item == parameter_count for item in expected_arities)
    return {
        "output": {
            "entry_point": entry_point,
            "function_found": function_found,
            "parameter_names": parameter_names,
            "parameter_count": parameter_count,
            "expected_arities": expected_arities,
            "contract_ok": bool(function_found and arity_match),
        }
    }
"""


MBPP_ASSERT_SCORER = """import subprocess
import sys
import tempfile


def _run_assert(code, entry_point, assert_line, timeout_s):
    body = assert_line.replace("candidate(", f"{entry_point}(")
    harness = (
        code
        + "\\n"
        + f"candidate = {entry_point}\\n"
        + body
        + "\\nprint('MBPP_ASSERT_OK')\\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    completed = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    ok = completed.returncode == 0 and "MBPP_ASSERT_OK" in completed.stdout
    return ok, completed.stderr[-300:] or completed.stdout[-300:]


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    if not code.strip() or not entry_point:
        raise Exception("Missing code or entry_point")
    asserts = test_list if test_list else [line.strip() for line in test.splitlines() if line.strip().startswith("assert ")]
    passed = 0
    failures = []
    for index, assert_line in enumerate(asserts):
        ok, details = _run_assert(code, entry_point, assert_line, timeout_s)
        if ok:
            passed += 1
        else:
            failures.append({"index": index, "assert": assert_line, "details": details})
    return {"output": {"passed": passed, "total": len(asserts), "failures": failures[:10]}}
"""


MBPP_FAILURE_DIAGNOSER = """import subprocess
import sys
import tempfile


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    if not code.strip() or not entry_point:
        raise Exception("Missing code or entry_point")
    asserts = test_list if test_list else [line.strip() for line in test.splitlines() if line.strip().startswith("assert ")]
    first_failure = None
    for index, assert_line in enumerate(asserts):
        body = assert_line.replace("candidate(", f"{entry_point}(")
        harness = code + "\\n" + f"candidate = {entry_point}\\n" + body + "\\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(harness)
            path = fh.name
        completed = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            first_failure = {
                "index": index,
                "assert": assert_line,
                "error_summary": (completed.stderr.splitlines()[-1] if completed.stderr.splitlines() else completed.stdout[-240:]),
                "traceback": completed.stderr[-500:] or completed.stdout[-500:],
            }
            break
    return {"output": {"failure_count": 0 if first_failure is None else 1, "first_failure": first_failure}}
"""


GSM8K_QUANTITY_UNIT_EXTRACTOR = """import re


_UNITS = [
    "dollar", "dollars", "cent", "cents", "nickel", "nickels", "dime", "dimes", "quarter", "quarters",
    "mile", "miles", "hour", "hours", "minute", "minutes", "day", "days", "week", "weeks",
    "cup", "cups", "plate", "plates", "gumball", "gumballs", "month", "months", "percent", "%"
]


def run(inputs):
    question = str(inputs.get("question") or inputs.get("query") or inputs.get("task") or "")
    if not question.strip():
        raise Exception("Missing question")
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", question)
    currencies = re.findall(r"\\$\\s*-?\\d+(?:\\.\\d+)?", question)
    percents = re.findall(r"-?\\d+(?:\\.\\d+)?\\s*%", question)
    words = question.lower()
    units = [unit for unit in _UNITS if unit in words]
    return {
        "output": {
            "numbers": numbers,
            "currencies": currencies,
            "percents": percents,
            "units": units,
            "number_count": len(numbers),
        }
    }
"""


GSM8K_EXPRESSION_ENGINE = """import ast
import math
from fractions import Fraction


_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "Fraction": Fraction,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.func.id](*[_eval(arg) for arg in node.args])
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    raise ValueError("Unsupported expression")


def run(inputs):
    expression = str(inputs.get("expression") or inputs.get("query") or "").strip()
    if not expression:
        raise Exception("Missing expression")
    tree = ast.parse(expression, mode="eval")
    value = _eval(tree)
    return {"output": {"expression": expression, "value": value}}
"""


GSM8K_RATIO_RATE_CALCULATOR = """import math


def run(inputs):
    mode = str(inputs.get("mode") or "").strip().lower()
    if mode == "simplify_ratio":
        a = int(inputs["a"])
        b = int(inputs["b"])
        g = math.gcd(a, b)
        return {"output": {"mode": mode, "ratio": [a // g, b // g]}}
    if mode == "unit_count_from_total_and_unit_price":
        total = float(inputs["total"])
        unit_price = float(inputs["unit_price"])
        return {"output": {"mode": mode, "count": total / unit_price}}
    if mode == "distance":
        speed = float(inputs["speed"])
        time = float(inputs["time"])
        return {"output": {"mode": mode, "distance": speed * time}}
    if mode == "time":
        distance = float(inputs["distance"])
        speed = float(inputs["speed"])
        return {"output": {"mode": mode, "time": distance / speed}}
    if mode == "unit_price":
        total = float(inputs["total"])
        quantity = float(inputs["quantity"])
        return {"output": {"mode": mode, "unit_price": total / quantity}}
    if mode == "percent_of":
        percent = float(inputs["percent"])
        whole = float(inputs["whole"])
        return {"output": {"mode": mode, "value": percent * whole / 100.0}}
    raise Exception("Unsupported mode")
"""


GSM8K_UNIT_SANITY_CHECKER = """def run(inputs):
    candidate = inputs.get("candidate")
    answer_type = str(inputs.get("answer_type") or "generic").strip().lower()
    unit = str(inputs.get("unit") or "").strip().lower()
    if candidate is None:
        raise Exception("Missing candidate")
    value = float(candidate)
    is_integer = abs(value - round(value)) <= float(inputs.get("integer_tolerance") or 1e-9)
    nonnegative = value >= 0
    unit_ok = True
    if answer_type in {"count", "integer"}:
        unit_ok = is_integer and nonnegative
    elif answer_type in {"money", "rate", "distance", "time", "generic"}:
        unit_ok = nonnegative
    return {
        "output": {
            "candidate": value,
            "answer_type": answer_type,
            "unit": unit,
            "is_integer_like": is_integer,
            "nonnegative": nonnegative,
            "unit_sanity_ok": unit_ok,
        }
    }
"""


GSM8K_ANSWER_VERIFIER_FORMATTER = """def _parse_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.startswith("####"):
        text = text[4:].strip()
    return float(text)


def _format_number(value):
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text


def run(inputs):
    candidate = inputs.get("candidate")
    expected = inputs.get("expected")
    if candidate is None:
        raise Exception("Missing candidate")
    candidate_value = _parse_number(candidate)
    matches = None
    expected_value = None
    if expected is not None:
        expected_value = _parse_number(expected)
        matches = abs(candidate_value - expected_value) <= float(inputs.get("tolerance") or 1e-6)
    formatted = "#### " + _format_number(candidate_value)
    return {
        "output": {
            "candidate_value": candidate_value,
            "expected_value": expected_value,
            "matches_expected": matches,
            "formatted_answer": formatted,
        }
    }
"""


MATH_EXPRESSION_ENGINE = """import ast
import math
import re
from fractions import Fraction


def _normalize(text):
    value = str(text or "").strip()
    value = re.sub(r"\\\\(?:d|t)?frac\\s*\\{([^{}]+)\\}\\s*\\{([^{}]+)\\}", r"(\\1)/(\\2)", value)
    value = value.replace("^", "**")
    return value


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError("Unsupported symbol")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        funcs = {
            "sqrt": math.sqrt,
            "factorial": math.factorial,
            "comb": math.comb,
            "perm": math.perm,
            "Fraction": Fraction,
            "abs": abs,
        }
        if node.func.id not in funcs:
            raise ValueError("Unsupported function")
        return funcs[node.func.id](*[_eval(arg) for arg in node.args])
    raise ValueError("Unsupported expression")


def run(inputs):
    expression = str(inputs.get("expression") or inputs.get("query") or "").strip()
    if not expression:
        raise Exception("Missing expression")
    normalized = _normalize(expression)
    value = _eval(ast.parse(normalized, mode="eval"))
    return {"output": {"expression": expression, "normalized_expression": normalized, "value": value}}
"""


MATH_EQUATION_SOLVER = """import math


def _parse_side(expr, variable):
    expr = expr.replace(" ", "").replace("-", "+-")
    coeff = 0.0
    constant = 0.0
    quad = 0.0
    for token in [part for part in expr.split("+") if part]:
        if variable + "^2" in token or variable + "**2" in token:
            raw = token.replace(variable + "^2", "").replace(variable + "**2", "")
            if raw in {"", "+"}:
                quad += 1.0
            elif raw == "-":
                quad -= 1.0
            else:
                quad += float(raw)
        elif variable in token:
            raw = token.replace(variable, "")
            if raw in {"", "+"}:
                coeff += 1.0
            elif raw == "-":
                coeff -= 1.0
            else:
                coeff += float(raw)
        else:
            constant += float(token)
    return quad, coeff, constant


def run(inputs):
    equation = str(inputs.get("equation") or inputs.get("query") or "").strip()
    variable = str(inputs.get("variable") or "x").strip() or "x"
    if "=" not in equation:
        raise Exception("Expected equation with '='")
    normalized = equation.replace("^", "**")
    try:
        import sympy as sp
        symbol = sp.Symbol(variable)
        left, right = normalized.split("=", 1)
        solutions = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)), symbol)
        return {
            "output": {
                "equation": equation,
                "variable": variable,
                "solutions": [str(item) for item in solutions],
                "numeric_solutions": [float(item) if getattr(item, "is_real", False) else str(item) for item in solutions],
                "solver": "sympy",
            }
        }
    except Exception:
        left, right = normalized.split("=", 1)
        ql, cl, kl = _parse_side(left, variable)
        qr, cr, kr = _parse_side(right, variable)
        a = ql - qr
        b = cl - cr
        c = kl - kr
        solutions = []
        if abs(a) <= 1e-12:
            if abs(b) > 1e-12:
                solutions = [(-c) / b]
        else:
            disc = b * b - 4.0 * a * c
            if disc >= -1e-12:
                disc = max(disc, 0.0)
                root = math.sqrt(disc)
                solutions = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        deduped = []
        for item in solutions:
            if all(abs(item - seen) > 1e-9 for seen in deduped):
                deduped.append(item)
        return {
            "output": {
                "equation": equation,
                "variable": variable,
                "solutions": [str(item) for item in deduped],
                "numeric_solutions": deduped,
                "solver": "fallback",
            }
        }
"""


MATH_NUMBER_THEORY_INSPECTOR = """import math


def _factorize(n):
    value = abs(int(n))
    factors = {}
    d = 2
    while d * d <= value:
        while value % d == 0:
            factors[d] = factors.get(d, 0) + 1
            value //= d
        d += 1 if d == 2 else 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return factors


def _divisors(factors):
    divisors = [1]
    for prime, power in factors.items():
        base = list(divisors)
        for exponent in range(1, power + 1):
            base.extend(item * (prime ** exponent) for item in divisors)
        divisors = base
    return sorted(set(divisors))


def run(inputs):
    if "n" not in inputs:
        raise Exception("Missing n")
    n = int(inputs["n"])
    m = inputs.get("m")
    factors = _factorize(n)
    divisors = _divisors(factors)
    result = {
        "n": n,
        "prime_factors": factors,
        "divisors": divisors,
        "divisor_count": len(divisors),
        "divisor_sum": sum(divisors),
        "is_prime": n > 1 and len(factors) == 1 and next(iter(factors.values())) == 1,
    }
    if m is not None:
        m_int = int(m)
        result["gcd_with_m"] = math.gcd(n, m_int)
        result["lcm_with_m"] = abs(n * m_int) // math.gcd(n, m_int) if n and m_int else 0
    return {"output": result}
"""


MATH_GEOMETRY_MEASUREMENT_ENGINE = """import math


def run(inputs):
    shape = str(inputs.get("shape") or "").strip().lower()
    if shape == "circle":
        radius = float(inputs["radius"])
        return {"output": {"shape": shape, "area": math.pi * radius * radius, "circumference": 2 * math.pi * radius}}
    if shape == "rectangle":
        width = float(inputs["width"])
        height = float(inputs["height"])
        return {"output": {"shape": shape, "area": width * height, "perimeter": 2 * (width + height)}}
    if shape == "triangle":
        a = float(inputs["a"])
        b = float(inputs["b"])
        c = float(inputs["c"])
        s = (a + b + c) / 2.0
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        return {"output": {"shape": shape, "area": area, "perimeter": a + b + c}}
    if shape == "right_triangle":
        width = float(inputs["width"])
        height = float(inputs["height"])
        return {"output": {"shape": shape, "area": width * height / 2.0, "hypotenuse": math.hypot(width, height)}}
    if shape == "sphere":
        radius = float(inputs["radius"])
        return {"output": {"shape": shape, "surface_area": 4 * math.pi * radius * radius, "volume": 4 * math.pi * radius ** 3 / 3.0}}
    raise Exception("Unsupported shape")
"""


MATH_ANSWER_CANONICALIZER_VERIFIER = """import re


_HASH_RE = re.compile(r"####\\s*([^\\n\\r]+)")
_BOX_RE = re.compile(r"\\\\boxed\\{([^{}]+)\\}")
_FRAC_RE = re.compile(r"\\\\(?:d|t)?frac\\s*\\{([^{}]+)\\}\\s*\\{([^{}]+)\\}")
_NUM_RE = re.compile(r"[-+]?(?:(?:\\d+(?:\\.\\d+)?)|(?:\\.\\d+))(?:/[-+]?(?:(?:\\d+(?:\\.\\d+)?)|(?:\\.\\d+)))?")


def _normalize_candidate(value):
    text = str(value or "").strip()
    hash_matches = _HASH_RE.findall(text)
    if hash_matches:
        text = hash_matches[-1].strip()
    box_matches = _BOX_RE.findall(text)
    if box_matches:
        text = box_matches[-1].strip()
    frac_match = _FRAC_RE.search(text)
    if frac_match:
        text = f"{frac_match.group(1)}/{frac_match.group(2)}"
    return text.strip().replace("$", "")


def _parse_numeric(text):
    cleaned = _normalize_candidate(text).replace(",", "")
    if not cleaned:
        return None
    if "/" in cleaned and _NUM_RE.fullmatch(cleaned):
        left, right = cleaned.split("/", 1)
        return float(left) / float(right)
    if _NUM_RE.fullmatch(cleaned):
        return float(cleaned)
    tokens = _NUM_RE.findall(cleaned)
    if tokens:
        return _parse_numeric(tokens[-1])
    return None


def run(inputs):
    candidate = inputs.get("candidate")
    expected = inputs.get("expected")
    if candidate is None:
        raise Exception("Missing candidate")
    canonical = _normalize_candidate(candidate)
    candidate_value = _parse_numeric(canonical)
    expected_canonical = None
    expected_value = None
    matches = None
    if expected is not None:
        expected_canonical = _normalize_candidate(expected)
        expected_value = _parse_numeric(expected_canonical)
        if candidate_value is not None and expected_value is not None:
            matches = abs(candidate_value - expected_value) <= float(inputs.get("tolerance") or 1e-6)
        else:
            matches = canonical == expected_canonical
    return {
        "output": {
            "canonical_answer": canonical,
            "candidate_value": candidate_value,
            "expected_canonical": expected_canonical,
            "expected_value": expected_value,
            "matches_expected": matches,
        }
    }
"""


def build_real_dataset_tool_records() -> List[Dict[str, Any]]:
    records = [
        _tool_record(
            tool_id="humaneval-signature-contract-extractor",
            name="HumanEvalSignatureContractExtractor",
            dataset="humaneval",
            description="Extract entry point, parameters, and signature hints from HumanEval-style prompts and tests.",
            code=HUMANEVAL_SIGNATURE_CONTRACT_EXTRACTOR,
            domain_tags=["humaneval", "python-function-synthesis", "real-tools"],
            role_tags=["planner", "builder"],
            tool_tags=["humaneval", "signature", "contract", "entry_point", "real-tool"],
            inputs=["prompt", "query", "task", "entry_point", "test", "test_list"],
        ),
        _tool_record(
            tool_id="humaneval-code-normalizer",
            name="HumanEvalCodeNormalizer",
            dataset="humaneval",
            description="Normalize fenced code and extract the target HumanEval function deterministically.",
            code=HUMANEVAL_CODE_NORMALIZER,
            domain_tags=["humaneval", "python-function-synthesis", "real-tools"],
            role_tags=["builder", "refactor"],
            tool_tags=["humaneval", "normalize", "extract", "function", "real-tool"],
            inputs=["code", "query", "entry_point"],
        ),
        _tool_record(
            tool_id="humaneval-ast-structure-checker",
            name="HumanEvalAstStructureChecker",
            dataset="humaneval",
            description="Analyze HumanEval candidate code structure using Python AST without remote help.",
            code=HUMANEVAL_AST_STRUCTURE_CHECKER,
            domain_tags=["humaneval", "python-function-synthesis", "real-tools"],
            role_tags=["checker", "planner"],
            tool_tags=["humaneval", "ast", "structure", "syntax", "real-tool"],
            inputs=["code"],
        ),
        _tool_record(
            tool_id="humaneval-public-test-runner",
            name="HumanEvalPublicTestRunner",
            dataset="humaneval",
            description="Run HumanEval public tests locally against candidate code and return pass or fail with tracebacks.",
            code=HUMANEVAL_PUBLIC_TEST_RUNNER,
            domain_tags=["humaneval", "python-function-synthesis", "real-tools"],
            role_tags=["builder", "checker"],
            tool_tags=["humaneval", "execute", "public_test", "verification", "real-tool"],
            inputs=["code", "entry_point", "test", "test_list", "timeout_s"],
        ),
        _tool_record(
            tool_id="humaneval-traceback-localizer",
            name="HumanEvalTracebackLocalizer",
            dataset="humaneval",
            description="Map HumanEval traceback lines back to local source context for deterministic debugging.",
            code=HUMANEVAL_TRACEBACK_LOCALIZER,
            domain_tags=["humaneval", "python-function-synthesis", "real-tools"],
            role_tags=["checker", "refactor"],
            tool_tags=["humaneval", "traceback", "debug", "localize", "real-tool"],
            inputs=["code", "traceback", "error"],
        ),
        _tool_record(
            tool_id="mbpp-assert-example-parser",
            name="MBPPAssertExampleParser",
            dataset="mbpp",
            description="Parse MBPP assertion examples into structured call and expected-value pairs.",
            code=MBPP_ASSERT_EXAMPLE_PARSER,
            domain_tags=["mbpp", "assert-driven-programming", "real-tools"],
            role_tags=["planner", "checker"],
            tool_tags=["mbpp", "assert", "examples", "parser", "real-tool"],
            inputs=["test", "test_list"],
        ),
        _tool_record(
            tool_id="mbpp-import-symbol-injector",
            name="MBPPImportSymbolInjector",
            dataset="mbpp",
            description="Inject deterministic missing imports for common MBPP math and collection helpers.",
            code=MBPP_IMPORT_SYMBOL_INJECTOR,
            domain_tags=["mbpp", "assert-driven-programming", "real-tools"],
            role_tags=["builder", "refactor"],
            tool_tags=["mbpp", "imports", "symbols", "repair", "real-tool"],
            inputs=["code"],
        ),
        _tool_record(
            tool_id="mbpp-contract-gate",
            name="MBPPContractGate",
            dataset="mbpp",
            description="Verify MBPP candidate entry point and arity against candidate(...) assertions.",
            code=MBPP_CONTRACT_GATE,
            domain_tags=["mbpp", "assert-driven-programming", "real-tools"],
            role_tags=["checker"],
            tool_tags=["mbpp", "contract", "arity", "gate", "real-tool"],
            inputs=["code", "entry_point", "test", "test_list"],
        ),
        _tool_record(
            tool_id="mbpp-assert-scorer",
            name="MBPPAssertScorer",
            dataset="mbpp",
            description="Score MBPP candidate code by running each public assert locally and counting passes.",
            code=MBPP_ASSERT_SCORER,
            domain_tags=["mbpp", "assert-driven-programming", "real-tools"],
            role_tags=["checker", "builder"],
            tool_tags=["mbpp", "assert", "score", "execute", "real-tool"],
            inputs=["code", "entry_point", "test", "test_list", "timeout_s"],
        ),
        _tool_record(
            tool_id="mbpp-failure-diagnoser",
            name="MBPPFailureDiagnoser",
            dataset="mbpp",
            description="Run MBPP public asserts until first failure and return deterministic failure diagnostics.",
            code=MBPP_FAILURE_DIAGNOSER,
            domain_tags=["mbpp", "assert-driven-programming", "real-tools"],
            role_tags=["checker", "refactor"],
            tool_tags=["mbpp", "failure", "diagnostic", "debug", "real-tool"],
            inputs=["code", "entry_point", "test", "test_list", "timeout_s"],
        ),
        _tool_record(
            tool_id="gsm8k-quantity-unit-extractor",
            name="GSM8KQuantityUnitExtractor",
            dataset="gsm8k",
            description="Extract numbers, currency markers, percents, and unit tokens from GSM8K questions.",
            code=GSM8K_QUANTITY_UNIT_EXTRACTOR,
            domain_tags=["gsm8k", "arithmetic-word-problem", "real-tools"],
            role_tags=["planner", "builder"],
            tool_tags=["gsm8k", "quantities", "units", "extract", "real-tool"],
            inputs=["question", "query", "task"],
        ),
        _tool_record(
            tool_id="gsm8k-expression-engine",
            name="GSM8KExpressionEngine",
            dataset="gsm8k",
            description="Evaluate GSM8K arithmetic expressions locally using a restricted deterministic evaluator.",
            code=GSM8K_EXPRESSION_ENGINE,
            domain_tags=["gsm8k", "arithmetic-word-problem", "real-tools"],
            role_tags=["builder"],
            tool_tags=["gsm8k", "expression", "arithmetic", "compute", "real-tool"],
            inputs=["expression", "query"],
        ),
        _tool_record(
            tool_id="gsm8k-ratio-rate-calculator",
            name="GSM8KRatioRateCalculator",
            dataset="gsm8k",
            description="Compute ratio, rate, unit-price, and speed-distance-time relations for GSM8K tasks.",
            code=GSM8K_RATIO_RATE_CALCULATOR,
            domain_tags=["gsm8k", "arithmetic-word-problem", "real-tools"],
            role_tags=["builder", "planner"],
            tool_tags=["gsm8k", "ratio", "rate", "unit_price", "real-tool"],
            inputs=["mode", "a", "b", "total", "unit_price", "quantity", "speed", "time", "distance", "percent", "whole"],
        ),
        _tool_record(
            tool_id="gsm8k-unit-sanity-checker",
            name="GSM8KUnitSanityChecker",
            dataset="gsm8k",
            description="Apply simple nonnegative and integer-like sanity checks to GSM8K candidate answers.",
            code=GSM8K_UNIT_SANITY_CHECKER,
            domain_tags=["gsm8k", "arithmetic-word-problem", "real-tools"],
            role_tags=["checker"],
            tool_tags=["gsm8k", "sanity", "units", "check", "real-tool"],
            inputs=["candidate", "answer_type", "unit", "integer_tolerance"],
        ),
        _tool_record(
            tool_id="gsm8k-answer-verifier-formatter",
            name="GSM8KAnswerVerifierFormatter",
            dataset="gsm8k",
            description="Verify GSM8K numeric answers locally and format final output as #### <number>.",
            code=GSM8K_ANSWER_VERIFIER_FORMATTER,
            domain_tags=["gsm8k", "arithmetic-word-problem", "real-tools"],
            role_tags=["checker", "refactor"],
            tool_tags=["gsm8k", "answer", "verify", "format", "real-tool"],
            inputs=["candidate", "expected", "tolerance"],
        ),
        _tool_record(
            tool_id="math-expression-engine",
            name="MathExpressionEngine",
            dataset="math",
            description="Evaluate exact-competition-math expressions locally, including simple LaTeX fractions.",
            code=MATH_EXPRESSION_ENGINE,
            domain_tags=["math", "exact-competition-math", "real-tools"],
            role_tags=["builder", "planner"],
            tool_tags=["math", "expression", "exact", "latex_fraction", "real-tool"],
            inputs=["expression", "query"],
        ),
        _tool_record(
            tool_id="math-equation-solver",
            name="MathEquationSolver",
            dataset="math",
            description="Solve linear and quadratic equations locally with optional sympy acceleration.",
            code=MATH_EQUATION_SOLVER,
            domain_tags=["math", "exact-competition-math", "real-tools"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "equation", "solver", "algebra", "real-tool"],
            inputs=["equation", "variable", "query"],
        ),
        _tool_record(
            tool_id="math-number-theory-inspector",
            name="MathNumberTheoryInspector",
            dataset="math",
            description="Inspect factorization, divisors, primality, gcd, and lcm locally for MATH tasks.",
            code=MATH_NUMBER_THEORY_INSPECTOR,
            domain_tags=["math", "exact-competition-math", "real-tools"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "number_theory", "factor", "divisor", "real-tool"],
            inputs=["n", "m"],
        ),
        _tool_record(
            tool_id="math-geometry-measurement-engine",
            name="MathGeometryMeasurementEngine",
            dataset="math",
            description="Compute geometry measurements locally for common 2D and 3D shapes.",
            code=MATH_GEOMETRY_MEASUREMENT_ENGINE,
            domain_tags=["math", "exact-competition-math", "real-tools"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "geometry", "measurements", "shapes", "real-tool"],
            inputs=["shape", "radius", "width", "height", "a", "b", "c"],
        ),
        _tool_record(
            tool_id="math-answer-canonicalizer-verifier",
            name="MathAnswerCanonicalizerVerifier",
            dataset="math",
            description="Canonicalize boxed or hash-style math answers and compare them locally against expected results.",
            code=MATH_ANSWER_CANONICALIZER_VERIFIER,
            domain_tags=["math", "exact-competition-math", "real-tools"],
            role_tags=["checker", "refactor"],
            tool_tags=["math", "answer", "canonicalize", "verify", "real-tool"],
            inputs=["candidate", "expected", "tolerance"],
        ),
    ]
    for item in records:
        spec = item["spec"]
        dataset = item["dataset"]
        capability_kind = _infer_capability_kind(item["id"])
        meta = _capability_metadata(dataset, capability_kind)
        spec["dataset_family"] = dataset
        spec["focus_slug"] = ""
        spec["primary_role"] = ""
        spec["capability_kind"] = meta["capability_kind"]
        spec["stage"] = meta["stage"]
        spec["deterministic"] = True
        spec["consumes"] = list(meta["consumes"])
        spec["produces"] = list(meta["produces"])
    return records


def build_real_dataset_agent_records() -> List[Dict[str, Any]]:
    return [
        {
            "id": "agent-real-humaneval-function-synthesis",
            "name": "HumanEvalFunctionSynthesis-RealTools",
            "description": "HumanEval-focused real-tool agent for Python function synthesis, execution, and traceback diagnosis.",
            "domain_tags": ["humaneval", "python-function-synthesis", "real-tools"],
            "role_tags": list(CODE_AGENT_ROLE_TAGS),
            "tool_ids": [
                "humaneval-signature-contract-extractor",
                "humaneval-code-normalizer",
                "humaneval-ast-structure-checker",
                "humaneval-public-test-runner",
                "humaneval-traceback-localizer",
            ],
            "embedding_text": "HumanEval real-tools agent. Signature parsing, code normalization, AST checks, public tests, traceback localization.",
        },
        {
            "id": "agent-real-mbpp-assert-driven-repair",
            "name": "MBPPAssertDrivenRepair-RealTools",
            "description": "MBPP-focused real-tool agent for assert parsing, import repair, contract gating, scoring, and failure diagnosis.",
            "domain_tags": ["mbpp", "assert-driven-programming", "real-tools"],
            "role_tags": list(CODE_AGENT_ROLE_TAGS),
            "tool_ids": [
                "mbpp-assert-example-parser",
                "mbpp-import-symbol-injector",
                "mbpp-contract-gate",
                "mbpp-assert-scorer",
                "mbpp-failure-diagnoser",
            ],
            "embedding_text": "MBPP real-tools agent. Assert parsing, import injection, contract gate, assert scoring, failure diagnosis.",
        },
        {
            "id": "agent-real-gsm8k-arithmetic-word-problem",
            "name": "GSM8KArithmeticWordProblem-RealTools",
            "description": "GSM8K-focused real-tool agent for quantity extraction, arithmetic computation, rate reasoning, and final answer checks.",
            "domain_tags": ["gsm8k", "arithmetic-word-problem", "real-tools"],
            "role_tags": list(MATH_AGENT_ROLE_TAGS),
            "tool_ids": [
                "gsm8k-quantity-unit-extractor",
                "gsm8k-expression-engine",
                "gsm8k-ratio-rate-calculator",
                "gsm8k-unit-sanity-checker",
                "gsm8k-answer-verifier-formatter",
            ],
            "embedding_text": "GSM8K real-tools agent. Quantity extraction, arithmetic engine, ratio/rate calculator, sanity checking, answer formatting.",
        },
        {
            "id": "agent-real-math-exact-reasoning",
            "name": "MATHExactReasoning-RealTools",
            "description": "MATH-focused real-tool agent for exact expression evaluation, algebra, number theory, geometry, and answer canonicalization.",
            "domain_tags": ["math", "exact-competition-math", "real-tools"],
            "role_tags": list(MATH_AGENT_ROLE_TAGS),
            "tool_ids": [
                "math-expression-engine",
                "math-equation-solver",
                "math-number-theory-inspector",
                "math-geometry-measurement-engine",
                "math-answer-canonicalizer-verifier",
            ],
            "embedding_text": "MATH real-tools agent. Exact expression engine, equation solver, number theory inspector, geometry engine, answer canonicalizer.",
        },
    ]


def build_agent_design_manifest() -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    for record in build_real_dataset_agent_records():
        manifest[record["name"]] = {
            "description": record["description"],
            "role_tags": list(record["role_tags"]),
            "domain_tags": list(record["domain_tags"]),
            "tools": list(record["tool_ids"]),
        }
    return manifest


def build_manifest_payload() -> Dict[str, Any]:
    tools = build_real_dataset_tool_records()
    agents = build_real_dataset_agent_records()
    return {
        "tools": [
            {
                "id": item["id"],
                "name": item["spec"]["name"],
                "dataset": item["dataset"],
                "description": item["spec"]["description"],
                "tool_tags": list(item["spec"]["tool_tags"]),
                "dataset_family": item["spec"].get("dataset_family", item["dataset"]),
                "focus_slug": item["spec"].get("focus_slug", ""),
                "primary_role": item["spec"].get("primary_role", ""),
                "capability_kind": item["spec"].get("capability_kind", ""),
                "stage": item["spec"].get("stage", ""),
                "consumes": list(item["spec"].get("consumes") or []),
                "produces": list(item["spec"].get("produces") or []),
            }
            for item in tools
        ],
        "agents": [
            {
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "domain_tags": list(item["domain_tags"]),
                "role_tags": list(item["role_tags"]),
                "tool_ids": list(item["tool_ids"]),
            }
            for item in agents
        ],
    }
