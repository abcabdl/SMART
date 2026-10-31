from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Sequence


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(value: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((value or "").lower()) if token}


def _tool(
    *,
    tool_id: str,
    name: str,
    description: str,
    code: str,
    domain_tags: list[str],
    role_tags: list[str],
    tool_tags: list[str],
    inputs: list[str],
):
    return {
        "tool_id": tool_id,
        "spec": {
            "name": name,
            "description": description,
            "requirements": "standard-library",
            "domain_tags": domain_tags,
            "role_tags": role_tags,
            "tool_tags": tool_tags,
            "inputs": inputs,
            "outputs": ["output"],
            "requires_api": False,
            "api_notes": "",
            "modalities": ["text"],
            "output_formats": ["json"],
            "permissions": ["read", "compute"],
            "cost_tier": "low",
            "latency_tier": "low",
            "reliability_prior": 0.85,
            "examples": [description],
            "embedding_text": f"{name}. {description}. Tags: {' '.join(tool_tags)}",
        },
        "code": code,
    }


MATH_SAFE_EXPRESSION_EVALUATOR = """import ast
import math


_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "gcd": math.gcd,
    "comb": math.comb,
    "perm": math.perm,
    "factorial": math.factorial,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError("Unsupported symbol: " + node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.func.id](*[_eval(arg) for arg in node.args])
    raise ValueError("Unsupported expression")


def run(inputs):
    expression = inputs.get("expression") or inputs.get("query") or inputs.get("task")
    if not isinstance(expression, str) or not expression.strip():
        raise Exception("Missing expression")
    tree = ast.parse(expression.strip(), mode="eval")
    value = _eval(tree)
    return {"output": {"expression": expression, "value": value}}
"""


MATH_EQUATION_SOLVER = """def _linear_fallback(equation, variable):
    left, right = equation.split("=", 1)
    left = left.replace(" ", "").replace("-", "+-")
    right_value = float(eval(right, {"__builtins__": {}}, {}))
    coeff = 0.0
    bias = 0.0
    for term in [part for part in left.split("+") if part]:
        if variable in term:
            raw = term.replace(variable, "")
            if raw in ("", "+"):
                coeff += 1.0
            elif raw == "-":
                coeff -= 1.0
            else:
                coeff += float(raw)
        else:
            bias += float(term)
    if coeff == 0:
        return []
    return [(right_value - bias) / coeff]


def run(inputs):
    equation = str(inputs.get("equation") or inputs.get("query") or inputs.get("task") or "").strip().replace("^", "**")
    variable = str(inputs.get("variable") or "x").strip() or "x"
    if "=" not in equation:
        raise Exception("Expected an equation string like '2*x+3=7'")
    try:
        import sympy as sp

        symbol = sp.Symbol(variable)
        left, right = equation.split("=", 1)
        expr = sp.Eq(sp.sympify(left), sp.sympify(right))
        solutions = sp.solve(expr, symbol)
        return {
            "output": {
                "equation": equation,
                "variable": variable,
                "solutions": [str(item) for item in solutions],
                "numeric_solutions": [float(item) if getattr(item, "is_real", False) else str(item) for item in solutions],
            }
        }
    except Exception:
        solutions = _linear_fallback(equation, variable)
        return {"output": {"equation": equation, "variable": variable, "solutions": [str(item) for item in solutions], "numeric_solutions": solutions}}
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


def _divisors_from_factors(factors):
    divisors = [1]
    for prime, power in factors.items():
        current = list(divisors)
        for exponent in range(1, power + 1):
            current.extend([item * (prime ** exponent) for item in divisors])
        divisors = current
    return sorted(set(divisors))


def run(inputs):
    if "n" not in inputs:
        raise Exception("Missing n")
    n = int(inputs["n"])
    other = inputs.get("m")
    factors = _factorize(n)
    divisors = _divisors_from_factors(factors)
    result = {
        "n": n,
        "is_prime": n > 1 and len(factors) == 1 and next(iter(factors.values())) == 1,
        "prime_factors": factors,
        "divisors": divisors,
        "divisor_count": len(divisors),
        "divisor_sum": sum(divisors),
    }
    if other is not None:
        m = int(other)
        result["gcd_with_m"] = math.gcd(n, m)
        result["lcm_with_m"] = abs(n * m) // math.gcd(n, m) if n and m else 0
    return {"output": result}
"""


MATH_GEOMETRY_MEASUREMENTS = """import math


def run(inputs):
    shape = str(inputs.get("shape") or "").strip().lower()
    if shape == "circle":
        r = float(inputs["radius"])
        return {"output": {"shape": shape, "area": math.pi * r * r, "circumference": 2 * math.pi * r}}
    if shape == "rectangle":
        w = float(inputs["width"])
        h = float(inputs["height"])
        return {"output": {"shape": shape, "area": w * h, "perimeter": 2 * (w + h)}}
    if shape == "triangle":
        a = float(inputs["a"])
        b = float(inputs["b"])
        c = float(inputs["c"])
        s = (a + b + c) / 2
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        return {"output": {"shape": shape, "area": area, "perimeter": a + b + c}}
    if shape == "right_triangle":
        width = float(inputs["width"])
        height = float(inputs["height"])
        return {"output": {"shape": shape, "area": width * height / 2, "hypotenuse": math.hypot(width, height)}}
    if shape == "sphere":
        r = float(inputs["radius"])
        return {"output": {"shape": shape, "surface_area": 4 * math.pi * r * r, "volume": 4 * math.pi * (r ** 3) / 3}}
    raise Exception("Unsupported shape")
"""


MATH_RATIO_PERCENT_RATE = """import math


def run(inputs):
    mode = str(inputs.get("mode") or "").strip().lower()
    if mode == "simplify_ratio":
        a = int(inputs["a"])
        b = int(inputs["b"])
        g = math.gcd(a, b)
        return {"output": {"mode": mode, "ratio": [a // g, b // g]}}
    if mode == "percent_change":
        original = float(inputs["original"])
        new = float(inputs["new"])
        value = 0.0 if original == 0 else (new - original) / original * 100.0
        return {"output": {"mode": mode, "percent_change": value}}
    if mode == "distance":
        speed = float(inputs["speed"])
        time = float(inputs["time"])
        return {"output": {"mode": mode, "distance": speed * time}}
    if mode == "time":
        distance = float(inputs["distance"])
        speed = float(inputs["speed"])
        return {"output": {"mode": mode, "time": distance / speed}}
    if mode == "speed":
        distance = float(inputs["distance"])
        time = float(inputs["time"])
        return {"output": {"mode": mode, "speed": distance / time}}
    raise Exception("Unsupported mode")
"""


MATH_NUMERIC_VERIFIER = """import math


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    return float(eval(text.replace("^", "**"), {"__builtins__": {}}, {"pi": math.pi, "e": math.e}))


def run(inputs):
    if "candidate" not in inputs or "expected" not in inputs:
        raise Exception("Missing candidate or expected")
    tolerance = float(inputs.get("tolerance") or 1e-6)
    candidate_value = _to_float(inputs["candidate"])
    expected_value = _to_float(inputs["expected"])
    diff = abs(candidate_value - expected_value)
    return {"output": {"candidate": candidate_value, "expected": expected_value, "difference": diff, "within_tolerance": diff <= tolerance}}
"""


CODE_PROMPT_PARSER = """import re


_DEF_RE = re.compile(r"def\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*?)\\):", re.DOTALL)


def run(inputs):
    prompt = str(inputs.get("prompt") or inputs.get("query") or inputs.get("task") or "")
    match = _DEF_RE.search(prompt)
    if not match:
        raise Exception("No function definition found in prompt")
    entry_point = match.group(1)
    params_raw = match.group(2).strip()
    params = []
    if params_raw:
        for item in params_raw.split(","):
            name = item.strip().split("=")[0].strip()
            if name:
                params.append(name)
    examples = re.findall(r"assert\\s+(.+)", prompt)
    return {"output": {"entry_point": entry_point, "parameters": params, "examples": examples[:10]}}
"""


CODE_AST_ANALYZER = """import ast


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.imports = []
        self.loops = 0
        self.branches = 0
        self.calls = []

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.branches += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        self.generic_visit(node)


def run(inputs):
    code = str(inputs.get("code") or "")
    if not code.strip():
        raise Exception("Missing code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"output": {"valid": False, "syntax_error": str(exc)}}
    visitor = Visitor()
    visitor.visit(tree)
    recursion = [name for name in visitor.functions if name in visitor.calls]
    return {"output": {"valid": True, "functions": visitor.functions, "imports": visitor.imports, "loop_count": visitor.loops, "branch_count": visitor.branches, "recursive_functions": recursion}}
"""


CODE_CONTRACT_CHECKER = """import ast
import re


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    tests = str(inputs.get("test") or "")
    if not code.strip():
        raise Exception("Missing code")
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    result = {"entry_point": entry_point, "function_found": entry_point in functions}
    if entry_point in functions:
        node = functions[entry_point]
        params = [arg.arg for arg in node.args.args]
        result["parameters"] = params
        result["arity"] = len(params)
    else:
        result["parameters"] = []
        result["arity"] = None
    if tests:
        calls = re.findall(r"(?:candidate|" + re.escape(entry_point) + r")\\((.*?)\\)", tests)
        result["observed_test_calls"] = calls[:10]
    return {"output": result}
"""


CODE_TEST_RUNNER = """import json
import subprocess
import sys
import tempfile


def run(inputs):
    code = str(inputs.get("code") or "")
    test = inputs.get("test")
    test_list = list(inputs.get("test_list") or [])
    entry_point = str(inputs.get("entry_point") or "").strip()
    if not code.strip():
        raise Exception("Missing code")
    if not test and not test_list:
        raise Exception("Missing test or test_list")
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    test_body = "\\n".join(str(item) for item in test_list) if test_list else str(test)
    harness = code + "\\n\\n" + test_body + "\\n\\n" + (
        "import json\\n"
        "try:\\n"
        + (f"    _candidate = globals().get({entry_point!r})\\n" if entry_point else "    _candidate = globals().get('candidate')\\n")
        + "    if 'check' in globals():\\n"
        + "        if _candidate is not None:\\n"
        + "            check(_candidate)\\n"
        + "        else:\\n"
        + "            check()\\n"
        + "    print(json.dumps({'passed': True}))\\n"
        + "except Exception as exc:\\n"
        + "    import traceback\\n"
        + "    print(json.dumps({'passed': False, 'error': str(exc), 'traceback': traceback.format_exc()}))\\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    completed = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout_s, check=False)
    stdout = completed.stdout.strip().splitlines()
    payload = {}
    if stdout:
        try:
            payload = json.loads(stdout[-1])
        except Exception:
            payload = {"passed": False, "error": "non_json_output", "stdout": completed.stdout}
    else:
        payload = {"passed": False, "error": "empty_output"}
    payload["returncode"] = completed.returncode
    if completed.stderr.strip():
        payload["stderr"] = completed.stderr
    return {"output": payload}
"""


CODE_FUNCTION_NORMALIZER = """import re
import textwrap


def run(inputs):
    code = str(inputs.get("code") or inputs.get("query") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    if not code.strip():
        raise Exception("Missing code")
    text = re.sub(r"^```(?:python)?\\s*", "", code.strip())
    text = re.sub(r"```\\s*$", "", text)
    normalized = textwrap.dedent(text).strip() + "\\n"
    extracted = normalized
    if entry_point:
        pattern = re.compile(r"(^def\\s+" + re.escape(entry_point) + r"\\s*\\(.*?)(?=^def\\s+|\\Z)", re.DOTALL | re.MULTILINE)
        match = pattern.search(normalized)
        if match:
            extracted = match.group(1).rstrip() + "\\n"
    return {"output": {"normalized_code": normalized, "extracted_code": extracted}}
"""


CODE_EXCEPTION_INSPECTOR = """def run(inputs):
    code = str(inputs.get("code") or "")
    error = str(inputs.get("error") or inputs.get("traceback") or "")
    if not error.strip():
        raise Exception("Missing error")
    lines = code.splitlines()
    focus = []
    for line in error.splitlines():
        if ", line " not in line:
            continue
        try:
            number = int(line.split(", line ", 1)[1].split(",", 1)[0].strip())
        except Exception:
            continue
        start = max(0, number - 3)
        end = min(len(lines), number + 2)
        snippet = []
        for pos in range(start, end):
            snippet.append({"line": pos + 1, "code": lines[pos]})
        focus.append({"line": number, "context": snippet})
    return {"output": {"error_summary": error.splitlines()[-1] if error.splitlines() else error, "focus": focus}}
"""


CODE_COMPLEXITY_HEURISTICS = """import ast


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    if not code.strip():
        raise Exception("Missing code")
    tree = ast.parse(code)
    loop_count = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    branch_count = sum(isinstance(node, ast.If) for node in ast.walk(tree))
    call_count = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    recursive = False
    if entry_point:
        recursive = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry_point for node in ast.walk(tree))
    return {"output": {"loop_count": loop_count, "branch_count": branch_count, "call_count": call_count, "recursive": recursive}}
"""


@lru_cache(maxsize=1)
def load_functional_tools() -> list[dict]:
    return [
        _tool(
            tool_id="func-math-safe-expression-evaluator",
            name="SafeExpressionEvaluator",
            description="Safely evaluate arithmetic expressions and common math functions for GSM8K and MATH style reasoning.",
            code=MATH_SAFE_EXPRESSION_EVALUATOR,
            domain_tags=["math", "gsm8k", "competition-math"],
            role_tags=["planner", "builder", "checker"],
            tool_tags=["math", "arithmetic", "expression", "evaluate", "gsm8k", "math"],
            inputs=["expression", "query", "task"],
        ),
        _tool(
            tool_id="func-math-equation-solver",
            name="EquationSolver",
            description="Solve algebraic equations, especially linear and simple symbolic equations.",
            code=MATH_EQUATION_SOLVER,
            domain_tags=["math", "algebra", "competition-math"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "algebra", "equation", "solve", "symbolic"],
            inputs=["equation", "variable", "query", "task"],
        ),
        _tool(
            tool_id="func-math-number-theory-inspector",
            name="NumberTheoryInspector",
            description="Inspect primality, factorization, divisors, gcd, and lcm for integer reasoning tasks.",
            code=MATH_NUMBER_THEORY_INSPECTOR,
            domain_tags=["math", "number-theory", "gsm8k"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "number_theory", "prime", "factor", "divisor", "gcd", "lcm"],
            inputs=["n", "m"],
        ),
        _tool(
            tool_id="func-math-geometry-measurements",
            name="GeometryMeasurements",
            description="Compute areas, perimeters, volumes, and triangle or circle measurements.",
            code=MATH_GEOMETRY_MEASUREMENTS,
            domain_tags=["math", "geometry", "competition-math"],
            role_tags=["builder", "checker"],
            tool_tags=["math", "geometry", "circle", "triangle", "sphere", "measurement"],
            inputs=["shape", "radius", "width", "height", "a", "b", "c"],
        ),
        _tool(
            tool_id="func-math-ratio-percent-rate",
            name="RatioPercentRateAnalyzer",
            description="Handle ratio simplification, percentages, and speed-distance-time computations.",
            code=MATH_RATIO_PERCENT_RATE,
            domain_tags=["math", "gsm8k", "word-problem"],
            role_tags=["planner", "builder", "checker"],
            tool_tags=["math", "ratio", "percent", "rate", "distance", "speed", "time"],
            inputs=["mode", "a", "b", "original", "new", "distance", "speed", "time"],
        ),
        _tool(
            tool_id="func-math-numeric-verifier",
            name="NumericVerifier",
            description="Compare a candidate numeric answer against an expected value with tolerance.",
            code=MATH_NUMERIC_VERIFIER,
            domain_tags=["math", "verification", "gsm8k", "competition-math"],
            role_tags=["checker", "refactor"],
            tool_tags=["math", "verify", "numeric", "answer_check", "tolerance"],
            inputs=["candidate", "expected", "tolerance"],
        ),
        _tool(
            tool_id="func-code-prompt-parser",
            name="PythonPromptParser",
            description="Extract entry point, parameters, and example assertions from Humaneval or MBPP style prompts.",
            code=CODE_PROMPT_PARSER,
            domain_tags=["code", "humaneval", "mbpp"],
            role_tags=["planner", "builder"],
            tool_tags=["code", "prompt", "signature", "entry_point", "humaneval", "mbpp"],
            inputs=["prompt", "query", "task"],
        ),
        _tool(
            tool_id="func-code-ast-analyzer",
            name="PythonAstAnalyzer",
            description="Parse Python code and summarize functions, imports, loops, branches, and recursion signals.",
            code=CODE_AST_ANALYZER,
            domain_tags=["code", "analysis", "humaneval", "mbpp"],
            role_tags=["planner", "checker"],
            tool_tags=["code", "ast", "analysis", "syntax", "loop", "recursion"],
            inputs=["code"],
        ),
        _tool(
            tool_id="func-code-contract-checker",
            name="PythonContractChecker",
            description="Check whether the requested entry point exists and whether tests match the function contract.",
            code=CODE_CONTRACT_CHECKER,
            domain_tags=["code", "verification", "humaneval", "mbpp"],
            role_tags=["checker", "refactor"],
            tool_tags=["code", "contract", "entry_point", "signature", "test_match"],
            inputs=["code", "entry_point", "test"],
        ),
        _tool(
            tool_id="func-code-test-runner",
            name="PythonTestRunner",
            description="Execute candidate Python code against public tests or assert lists and return pass or fail with traceback.",
            code=CODE_TEST_RUNNER,
            domain_tags=["code", "verification", "humaneval", "mbpp"],
            role_tags=["builder", "checker"],
            tool_tags=["code", "test", "execute", "public_test", "humaneval", "mbpp"],
            inputs=["code", "test", "test_list", "entry_point", "timeout_s"],
        ),
        _tool(
            tool_id="func-code-function-normalizer",
            name="PythonFunctionNormalizer",
            description="Strip markdown fences, normalize indentation, and extract a target function definition from code text.",
            code=CODE_FUNCTION_NORMALIZER,
            domain_tags=["code", "refactor", "humaneval", "mbpp"],
            role_tags=["builder", "refactor"],
            tool_tags=["code", "normalize", "extract", "indentation", "function"],
            inputs=["code", "entry_point", "query"],
        ),
        _tool(
            tool_id="func-code-exception-inspector",
            name="PythonExceptionInspector",
            description="Map traceback lines back to source context for faster debugging and repair.",
            code=CODE_EXCEPTION_INSPECTOR,
            domain_tags=["code", "debugging", "humaneval", "mbpp"],
            role_tags=["checker", "refactor"],
            tool_tags=["code", "traceback", "debug", "exception", "repair"],
            inputs=["code", "error", "traceback"],
        ),
        _tool(
            tool_id="func-code-complexity-heuristics",
            name="PythonComplexityHeuristics",
            description="Estimate structural complexity from AST features such as loops, branches, and call count.",
            code=CODE_COMPLEXITY_HEURISTICS,
            domain_tags=["code", "analysis", "optimization"],
            role_tags=["planner", "checker"],
            tool_tags=["code", "complexity", "ast", "performance", "heuristic"],
            inputs=["code", "entry_point"],
        ),
    ]


def _score(record: dict, request_text: str, usage_count: int = 0) -> float:
    record_text = " ".join(
        [
            record["spec"]["name"],
            record["spec"]["description"],
            " ".join(record["spec"].get("domain_tags") or []),
            " ".join(record["spec"].get("role_tags") or []),
            " ".join(record["spec"].get("tool_tags") or []),
        ]
    )
    req_tokens = _tokens(request_text)
    rec_tokens = _tokens(record_text)
    score = float(len(req_tokens & rec_tokens))
    request_is_code = bool({"humaneval", "mbpp", "python", "code", "program", "function"} & req_tokens)
    request_is_math = bool({"gsm8k", "math", "algebra", "geometry", "number", "equation", "ratio"} & req_tokens)
    record_is_code = "code" in rec_tokens
    record_is_math = "math" in rec_tokens
    if request_is_code and not request_is_math:
        score += 12.0 if record_is_code else -6.0
    elif request_is_math and not request_is_code:
        score += 12.0 if record_is_math else -6.0
    else:
        if request_is_code and record_is_code:
            score += 4.0
        if request_is_math and record_is_math:
            score += 4.0
    score -= 0.45 * usage_count
    return score


def select_functional_tools(request_text: str, count: int, usage_counts: dict[str, int] | None = None, exclude_ids: Sequence[str] | None = None) -> list[dict]:
    usage_counts = usage_counts or {}
    exclude = set(exclude_ids or [])
    request_tokens = _tokens(request_text)
    request_is_code = bool({"humaneval", "mbpp", "python", "code", "program", "function"} & request_tokens)
    request_is_math = bool({"gsm8k", "math", "algebra", "geometry", "number", "equation", "ratio"} & request_tokens)
    records = [record for record in load_functional_tools() if record["tool_id"] not in exclude]
    if request_is_code and not request_is_math:
        same_family = [record for record in records if "code" in _tokens(" ".join(record["spec"].get("domain_tags") or []))]
        if len(same_family) >= min(count, 1):
            records = same_family
    elif request_is_math and not request_is_code:
        same_family = [record for record in records if "math" in _tokens(" ".join(record["spec"].get("domain_tags") or []))]
        if len(same_family) >= min(count, 1):
            records = same_family
    ranked = sorted(records, key=lambda record: (-_score(record, request_text, usage_counts.get(record["tool_id"], 0)), record["tool_id"]))
    return ranked[: max(0, count)]


def spec_from_record(record: dict) -> dict:
    return json.loads(json.dumps(record["spec"]))


def code_from_record(record: dict) -> str:
    return str(record["code"])
