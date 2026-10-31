from __future__ import annotations

import json
from functools import lru_cache


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned.strip("-") or "item"


def _spec(
    family: str,
    code: str,
    title: str,
    summary: str,
    keywords: list[str],
    answer_style: str,
    strategy_hints: list[str],
) -> dict:
    return {
        "family": family,
        "code": code,
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "answer_style": answer_style,
        "strategy_hints": strategy_hints,
    }


def _math_specializations() -> list[dict]:
    rows = [
        ("coins-small-change", "Coins And Small Change", "Reason about pennies, nickels, dimes, quarters, and mixed small-denomination money totals.", ["coins", "change", "nickel", "dime", "quarter", "pennies"], "integer"),
        ("unit-price-and-cost", "Unit Price And Cost", "Compute per-item price, total cost, and reverse unit-cost calculations.", ["unit price", "cost", "each", "per", "buy", "price"], "currency"),
        ("simple-interest", "Simple Interest", "Handle basic percentage-based interest and repayment amounts.", ["interest", "owed", "borrowed", "loan", "repay"], "currency"),
        ("installment-balance", "Installment Balance", "Track repeated payments and remaining balances after installments.", ["month", "remaining", "paid", "installment", "balance"], "currency"),
        ("discount-tax", "Discount Tax", "Apply sequential discounts, markups, tax, and coupon adjustments.", ["discount", "tax", "off", "sale", "coupon"], "currency"),
        ("profit-margin", "Profit And Margin", "Compute revenue, cost, profit, margin, and markup relations.", ["profit", "margin", "markup", "sell", "cost"], "currency"),
        ("percent-change", "Percent Change", "Analyze percent increase, decrease, and relative change.", ["percent", "increase", "decrease", "more than", "less than"], "percentage"),
        ("ratio-simplification", "Ratio Simplification", "Reduce ratios and scale equivalent proportional relationships.", ["ratio", "proportion", "simplify", "equivalent"], "fraction"),
        ("fraction-part-whole", "Fraction Part Whole", "Work with part-whole reasoning using fractions and shares.", ["fraction", "part", "whole", "share", "portion"], "fraction"),
        ("mixture-weighted-average", "Mixture Weighted Average", "Solve mixture and weighted-average quantity problems.", ["mixture", "average", "weighted", "blend", "concentration"], "decimal"),
        ("work-rate-single", "Single Work Rate", "Compute completion times and one-worker productivity.", ["work", "rate", "complete", "alone", "job"], "decimal"),
        ("work-rate-combined", "Combined Work Rate", "Combine worker rates and shared completion schedules.", ["together", "combined", "work", "rate", "finish"], "decimal"),
        ("speed-distance-time", "Speed Distance Time", "Use distance equals speed times time and its rearrangements.", ["speed", "distance", "time", "travel"], "decimal"),
        ("relative-motion", "Relative Motion", "Handle opposite-direction and same-direction motion comparisons.", ["relative", "toward", "away", "meet", "catch"], "decimal"),
        ("time-calendar-clock", "Time Calendar Clock", "Reason about elapsed time, schedules, and calendar transitions.", ["clock", "calendar", "hours", "minutes", "days"], "time"),
        ("age-equations", "Age Equations", "Translate age relationships into arithmetic or algebraic constraints.", ["age", "older", "younger", "years"], "integer"),
        ("arithmetic-sequence", "Arithmetic Sequence", "Find terms and sums of arithmetic progressions.", ["arithmetic sequence", "common difference", "sequence"], "integer"),
        ("geometric-sequence", "Geometric Sequence", "Find terms and sums of geometric progressions.", ["geometric", "ratio", "sequence", "powers"], "decimal"),
        ("divisibility-remainder", "Divisibility And Remainder", "Inspect divisibility, quotients, and modular leftovers.", ["divisible", "remainder", "mod", "left over"], "integer"),
        ("gcd-lcm", "Gcd And Lcm", "Use gcd and lcm for synchronization and grouping problems.", ["gcd", "lcm", "common multiple", "common divisor"], "integer"),
        ("digit-properties", "Digit Properties", "Reason about place value, digit sums, reversals, and decimal representation.", ["digit", "digits", "number", "place value"], "integer"),
        ("prime-factorization", "Prime Factorization", "Use prime factors, divisor structure, and relatively prime relations.", ["prime", "factor", "relatively prime", "divisor"], "integer"),
        ("counting-arrangements", "Counting Arrangements", "Count structured arrangements with simple case analysis.", ["arrangements", "ways", "count", "arrange"], "integer"),
        ("combinations-selection", "Combinations Selection", "Choose subsets and unordered selections.", ["choose", "combination", "select", "committee"], "integer"),
        ("permutations-ordering", "Permutations Ordering", "Count ordered arrangements and rank permutations.", ["permute", "order", "arrangement", "line up"], "integer"),
        ("probability-basic", "Probability Basic", "Compute elementary discrete probabilities and complements.", ["probability", "chance", "random", "likely"], "fraction"),
        ("expected-value", "Expected Value", "Estimate average outcome across repeated random processes.", ["expected", "average", "random", "outcome"], "decimal"),
        ("set-overlap", "Set Overlap", "Use inclusion-exclusion and overlap accounting for two or three sets.", ["both", "either", "neither", "overlap"], "integer"),
        ("linear-equation", "Linear Equation", "Solve one-variable linear equations and isolate unknowns.", ["equation", "solve for", "x", "linear"], "fraction"),
        ("system-two-variables", "Two Variable System", "Solve two-variable linear systems from word relationships.", ["system", "two variables", "simultaneous", "equations"], "fraction"),
        ("inequalities-intervals", "Inequalities And Intervals", "Interpret bounds, ranges, and interval constraints.", ["at least", "at most", "greater than", "less than"], "inequality"),
        ("exponents-roots", "Exponents And Roots", "Simplify powers, radicals, and root relationships.", ["square root", "cube root", "power", "exponent"], "decimal"),
        ("logarithm-exponential", "Logarithm And Exponential", "Manipulate logarithmic and exponential forms.", ["log", "exponential", "growth", "decay"], "decimal"),
        ("polynomial-evaluation", "Polynomial Evaluation", "Substitute values and simplify polynomial expressions.", ["polynomial", "evaluate", "expression"], "decimal"),
        ("circle-geometry", "Circle Geometry", "Work with radii, diameters, circumference, arcs, and sectors.", ["circle", "radius", "diameter", "arc", "sector"], "exact"),
        ("triangle-geometry", "Triangle Geometry", "Use triangle area, perimeter, and special triangle facts.", ["triangle", "base", "height", "hypotenuse"], "exact"),
        ("quadrilateral-geometry", "Quadrilateral Geometry", "Handle rectangles, squares, parallelograms, and trapezoids.", ["rectangle", "square", "parallelogram", "trapezoid"], "exact"),
        ("polygon-geometry", "Polygon Geometry", "Compute polygon perimeters, interior-angle totals, and regular polygon measures.", ["polygon", "regular", "interior angle", "perimeter"], "exact"),
        ("volume-3d", "Three Dimensional Volume", "Compute volumes of prisms, cylinders, cones, and spheres.", ["volume", "cylinder", "cone", "sphere", "prism"], "exact"),
        ("surface-area-3d", "Three Dimensional Surface Area", "Compute exposed surface area and total outer area.", ["surface area", "outer area", "3d"], "exact"),
        ("coordinate-distance", "Coordinate Distance", "Use coordinate geometry distance and midpoint calculations.", ["coordinate", "distance", "midpoint", "points"], "decimal"),
        ("angle-chasing", "Angle Chasing", "Track complementary, supplementary, and polygon angle relations.", ["angle", "supplementary", "complementary"], "integer"),
        ("right-trigonometry", "Right Triangle Trigonometry", "Use sine, cosine, tangent, and right-triangle ratios.", ["sin", "cos", "tan", "right triangle"], "decimal"),
        ("recursive-sequence", "Recursive Sequence", "Advance recursively defined sequences and pattern rules.", ["recursive", "defined by", "next term"], "integer"),
        ("modular-arithmetic", "Modular Arithmetic", "Work with modular classes, parity, and periodic residues.", ["modulo", "mod", "remainder", "parity"], "integer"),
        ("averages-medians", "Averages And Medians", "Compute mean and median under structured data changes.", ["average", "mean", "median"], "decimal"),
        ("mode-range", "Mode And Range", "Find mode, range, and frequency-based descriptive statistics.", ["mode", "range", "frequency"], "integer"),
        ("comparison-scaling", "Comparison Scaling", "Scale one scenario into another through multiplicative comparison.", ["times as many", "scale", "compare"], "decimal"),
        ("resource-allocation", "Resource Allocation", "Distribute items, capacity, or budget under simple constraints.", ["allocate", "distribute", "budget", "capacity"], "integer"),
        ("unit-conversion", "Unit Conversion", "Convert among common measurement, money, and count units.", ["convert", "units", "dozen", "feet", "meters"], "decimal"),
    ]
    specs = []
    for code, title, summary, keywords, answer_style in rows:
        specs.append(_spec("math", code, title, summary, keywords, answer_style, [f"Identify whether the problem is primarily about {title.lower()}.", "Extract the relevant quantities and align them to the correct relation.", "Apply the smallest valid computation path and sanity check the magnitude."]))
    return specs


def _code_specializations() -> list[dict]:
    rows = [
        ("string-split-join", "String Split And Join", "Transform strings through splitting, joining, token sequencing, and delimiter control.", ["string", "split", "join", "delimiter", "token"]),
        ("string-pattern-regex", "String Pattern Regex", "Match and validate string patterns with regular-expression style logic.", ["regex", "pattern", "match", "validate", "substring"]),
        ("string-case-normalization", "String Case Normalization", "Normalize case, trimming, spacing, and canonical formatting.", ["lowercase", "uppercase", "normalize", "strip", "format"]),
        ("string-palindrome-subsequence", "String Palindrome And Subsequence", "Detect mirrored patterns and subsequence relationships in strings.", ["palindrome", "subsequence", "reverse", "mirror"]),
        ("string-frequency-hashing", "String Frequency Hashing", "Count characters, words, and repeated tokens for frequency-driven logic.", ["count", "frequency", "character", "occurrence"]),
        ("list-filter-map", "List Filter And Map", "Build transformed lists through filtering, mapping, and element-wise rules.", ["list", "filter", "map", "elements"]),
        ("list-sorting-order", "List Sorting And Order", "Sort sequences and reason about ordered outputs or ranking.", ["sort", "sorted", "order", "rank"]),
        ("list-window-two-pointer", "List Window And Two Pointer", "Use moving windows and paired-pointer scans on arrays.", ["window", "two pointers", "subarray", "scan"]),
        ("tuple-dict-set-transform", "Tuple Dict Set Transform", "Manipulate tuples, dictionaries, and sets through structured conversion.", ["tuple", "dict", "set", "mapping"]),
        ("numeric-loop-accumulator", "Numeric Loop Accumulator", "Aggregate numeric values through iterative accumulation.", ["sum", "count", "accumulate", "loop"]),
        ("math-number-theory-code", "Math Number Theory Code", "Implement gcd, prime, factor, and divisibility logic in Python.", ["prime", "gcd", "factor", "divisible"]),
        ("recursion-tree", "Recursion Tree", "Implement recursive traversals and recurrence-style decomposition.", ["recursion", "recursive", "base case"]),
        ("dynamic-programming-1d", "Dynamic Programming 1D", "Use one-dimensional DP tables or rolling states.", ["dynamic programming", "dp", "table", "state"]),
        ("dynamic-programming-2d", "Dynamic Programming 2D", "Use grid or matrix DP with two-dimensional state updates.", ["grid", "matrix", "dp", "2d"]),
        ("stack-queue-simulation", "Stack Queue Simulation", "Use stack or queue discipline to simulate process ordering.", ["stack", "queue", "simulate", "push", "pop"]),
        ("heap-priority-queue", "Heap Priority Queue", "Use heap-like ordering for top-k, repeated extraction, or scheduling.", ["heap", "priority", "top k", "smallest", "largest"]),
        ("binary-search-space", "Binary Search Space", "Search sorted collections or monotonic answer spaces with binary search.", ["binary search", "sorted", "monotonic"]),
        ("graph-traversal", "Graph Traversal", "Traverse graphs with bfs, dfs, or connectivity logic.", ["graph", "bfs", "dfs", "path", "connected"]),
        ("matrix-grid-simulation", "Matrix Grid Simulation", "Simulate movement, updates, and neighborhood logic on grids.", ["grid", "matrix", "neighbors", "board"]),
        ("intervals-greedy", "Intervals Greedy", "Merge, schedule, or select intervals with greedy strategies.", ["interval", "merge", "schedule", "overlap"]),
        ("combinatorics-implementation", "Combinatorics Implementation", "Implement counting formulas and combinatorial generation.", ["combination", "permutation", "count ways"]),
        ("parsing-tokenization", "Parsing Tokenization", "Tokenize structured text and parse lightweight grammars.", ["parse", "tokenize", "tokens", "text"]),
        ("date-time-formatting", "Date Time Formatting", "Manipulate dates, times, and string formatting around calendars.", ["date", "time", "calendar", "format"]),
        ("bit-manipulation", "Bit Manipulation", "Use xor, shifts, masks, and binary representations.", ["bit", "binary", "xor", "shift", "mask"]),
        ("geometry-implementation", "Geometry Implementation", "Implement coordinate, distance, and geometry helper functions.", ["geometry", "coordinate", "distance", "angle"]),
        ("string-dp-editdistance", "String Dynamic Programming", "Use edit-distance or sequence-alignment style string DP.", ["edit distance", "lcs", "string dp"]),
        ("backtracking-search", "Backtracking Search", "Explore constrained search trees with pruning and undo logic.", ["backtracking", "search", "choose", "prune"]),
        ("memoization-caching", "Memoization And Caching", "Reuse repeated subproblem results through memoization.", ["memoization", "cache", "reuse"]),
        ("frequency-counter", "Frequency Counter", "Use counters and histograms to drive logic and comparisons.", ["counter", "frequency", "histogram"]),
        ("prefix-suffix-scan", "Prefix Suffix Scan", "Use prefix sums, suffix aggregates, and running scans.", ["prefix", "suffix", "running sum"]),
        ("linked-simulation", "Linked Structure Simulation", "Simulate pointer-like updates and next-prev structure behavior.", ["linked", "next", "prev", "pointer"]),
        ("signature-contract", "Signature Contract", "Respect function signature, argument order, and contract details.", ["signature", "entry point", "parameters", "contract"]),
        ("syntax-normalization", "Syntax Normalization", "Normalize indentation, fences, and syntax structure before execution.", ["indentation", "syntax", "normalize", "format"]),
        ("exception-debugging", "Exception Debugging", "Inspect exceptions, tracebacks, and likely failure lines.", ["exception", "traceback", "error", "debug"]),
        ("public-test-execution", "Public Test Execution", "Run candidate solutions against explicit public assertions.", ["tests", "assert", "check", "public"]),
        ("sample-based-debugging", "Sample Based Debugging", "Use concrete example calls to inspect runtime behavior.", ["example", "sample", "runtime", "call"]),
        ("complexity-analysis", "Complexity Analysis", "Estimate complexity from loops, branching, and recursive structure.", ["complexity", "loops", "branching", "performance"]),
        ("refactor-cleanup", "Refactor Cleanup", "Simplify code structure while preserving behavior and contract.", ["refactor", "cleanup", "simplify"]),
        ("input-output-formatting", "Input Output Formatting", "Handle literal formatting, string rendering, and display contracts.", ["format", "output", "render", "string"]),
        ("prompt-spec-parser", "Prompt Spec Parser", "Extract implementation constraints from natural-language prompt text.", ["prompt", "spec", "requirements", "constraints"]),
        ("list-search-membership", "List Search Membership", "Search arrays for membership, first-hit, and position logic.", ["contains", "find", "index", "membership"]),
        ("list-rotation-slicing", "List Rotation Slicing", "Rotate arrays, slice windows, and rebuild list orderings.", ["rotate", "slice", "shift"]),
        ("dict-aggregation", "Dictionary Aggregation", "Aggregate keyed values and fold records into mapping summaries.", ["dictionary", "mapping", "group", "aggregate"]),
        ("set-relations", "Set Relations", "Check unions, intersections, subset relations, and duplicate elimination.", ["set", "intersection", "union", "subset"]),
        ("numeric-conversion-rounding", "Numeric Conversion Rounding", "Convert numeric representations and control rounding behavior.", ["convert", "round", "integer", "float"]),
        ("comparison-custom-order", "Comparison Custom Order", "Implement custom comparison and ordering logic for structured data.", ["compare", "ordering", "custom sort"]),
        ("greedy-local-choice", "Greedy Local Choice", "Build solutions by repeated local-choice rules.", ["greedy", "choose", "best", "local"]),
        ("simulation-state-machine", "Simulation State Machine", "Advance discrete states under explicit transition rules.", ["simulate", "state", "transition"]),
        ("tree-nested-structure", "Tree Nested Structure", "Traverse nested lists, trees, or recursive containers.", ["tree", "nested", "children", "traverse"]),
        ("string-builder-template", "String Builder Template", "Assemble strings from fragments, templates, and placeholders.", ["template", "format string", "builder", "concatenate"]),
    ]
    specs = []
    for code, title, summary, keywords in rows:
        specs.append(_spec("code", code, title, summary, keywords, "python", [f"Focus on the {title.lower()} aspect of the prompt.", "Keep the contract exact and preserve the required entry point.", "Use the smallest implementation pattern that satisfies the observed examples."]))
    return specs


MATH_TOOL_KINDS = [
    ("signal", "SignalProfiler", "Profile whether the query matches this micro-domain and what cues are present.", "planner"),
    ("quantity", "QuantityExtractor", "Extract quantities, units, and numeric cues relevant to this micro-domain.", "planner"),
    ("plan", "FormulaPlanner", "Suggest the minimal solving plan and formulas for this micro-domain.", "builder"),
    ("compute", "NumericEngine", "Execute arithmetic or formula-driven computations for this micro-domain.", "builder"),
    ("verify", "ConstraintVerifier", "Check candidate answers and sanity constraints for this micro-domain.", "checker"),
    ("format", "AnswerFormatter", "Format the final answer in the style most natural for this micro-domain.", "refactor"),
]


CODE_TOOL_KINDS = [
    ("signature", "SignatureParser", "Parse the prompt into entry point, parameters, and signature obligations.", "planner"),
    ("examples", "ExampleExtractor", "Extract doctest-style and assert-style I/O examples from the prompt or tests.", "planner"),
    ("structure", "StructureAnalyzer", "Analyze code structure, imports, loops, recursion, and implementation shape.", "builder"),
    ("execute", "PublicTestRunner", "Execute candidate code against tests or sample calls.", "builder"),
    ("contract", "ContractVerifier", "Verify that the candidate respects the declared interface and test contract.", "checker"),
    ("debug", "DebugInspector", "Normalize code and inspect likely failure regions from traceback signals.", "refactor"),
]


def _math_signal_code(spec: dict) -> str:
    return f"""import re

SPECIALTY = {spec['title']!r}
SPECIALTY_ID = {spec['code']!r}
KEYWORDS = {json.dumps(spec['keywords'])}


def run(inputs):
    query = str(inputs.get("query") or inputs.get("task") or "").strip()
    text = query.lower()
    keyword_hits = [word for word in KEYWORDS if word.lower() in text]
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
    percents = re.findall(r"-?\\d+(?:\\.\\d+)?%", text)
    return {{"output": {{"specialty": SPECIALTY, "specialty_id": SPECIALTY_ID, "keyword_hits": keyword_hits, "match_score": len(keyword_hits), "number_count": len(numbers), "percent_count": len(percents)}}}}
"""


def _math_quantity_code(spec: dict) -> str:
    return f"""import re

SPECIALTY = {spec['title']!r}


def run(inputs):
    query = str(inputs.get("query") or inputs.get("task") or "").strip()
    text = query.lower()
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
    fractions = re.findall(r"\\b\\d+\\s*/\\s*\\d+\\b", text)
    percents = re.findall(r"-?\\d+(?:\\.\\d+)?%", text)
    currencies = re.findall(r"\\$\\s*-?\\d+(?:\\.\\d+)?", query)
    units = []
    for unit in ["hour", "hours", "minute", "minutes", "day", "days", "meter", "meters", "mile", "miles", "kg", "cm", "dozen"]:
        if unit in text:
            units.append(unit)
    return {{"output": {{"specialty": SPECIALTY, "numbers": numbers, "fractions": fractions, "percents": percents, "currencies": currencies, "units": units}}}}
"""


def _math_plan_code(spec: dict) -> str:
    return f"""SPECIALTY = {spec['title']!r}
STRATEGY_HINTS = {json.dumps(spec['strategy_hints'])}
KEYWORDS = {json.dumps(spec['keywords'])}


def run(inputs):
    query = str(inputs.get("query") or inputs.get("task") or "").lower()
    active = [word for word in KEYWORDS if word.lower() in query]
    plan = list(STRATEGY_HINTS)
    if active:
        plan.insert(0, "Matched cues: " + ", ".join(active))
    return {{"output": {{"specialty": SPECIALTY, "active_cues": active, "plan_steps": plan}}}}
"""


def _math_compute_code(spec: dict) -> str:
    return f"""import ast
import math

SPECIALTY = {spec['title']!r}


def _safe_eval(expression):
    allowed = {{"pi": math.pi, "e": math.e}}
    tree = ast.parse(str(expression), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.FloorDiv, ast.Load, ast.Name)):
            raise ValueError("Unsupported expression")
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError("Unsupported symbol")
    return eval(compile(tree, "<expr>", "eval"), {{"__builtins__": {{}}}}, allowed)


def run(inputs):
    if inputs.get("expression"):
        value = _safe_eval(inputs.get("expression"))
        return {{"output": {{"specialty": SPECIALTY, "mode": "expression", "value": value}}}}
    values = [float(item) for item in (inputs.get("values") or [])]
    if values:
        return {{"output": {{"specialty": SPECIALTY, "mode": "sum_values", "value": sum(values)}}}}
    if all(key in inputs for key in ("rate", "time")):
        return {{"output": {{"specialty": SPECIALTY, "mode": "rate_time", "value": float(inputs["rate"]) * float(inputs["time"])}}}}
    if all(key in inputs for key in ("quantity", "price")):
        return {{"output": {{"specialty": SPECIALTY, "mode": "quantity_price", "value": float(inputs["quantity"]) * float(inputs["price"])}}}}
    if all(key in inputs for key in ("part", "whole")):
        whole = float(inputs["whole"])
        value = 0.0 if whole == 0 else float(inputs["part"]) / whole
        return {{"output": {{"specialty": SPECIALTY, "mode": "part_whole_ratio", "value": value}}}}
    raise Exception("Provide expression or structured numeric inputs")
"""


def _math_verify_code(spec: dict) -> str:
    return f"""SPECIALTY = {spec['title']!r}


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def run(inputs):
    candidate = inputs.get("candidate")
    expected = inputs.get("expected")
    tolerance = float(inputs.get("tolerance") or 1e-6)
    if candidate is not None and expected is not None:
        c = _to_float(candidate)
        e = _to_float(expected)
        diff = abs(c - e)
        return {{"output": {{"specialty": SPECIALTY, "within_tolerance": diff <= tolerance, "difference": diff, "candidate": c, "expected": e}}}}
    values = [float(item) for item in (inputs.get("values") or [])]
    if values:
        return {{"output": {{"specialty": SPECIALTY, "nonnegative": all(item >= 0 for item in values), "sorted_values": sorted(values)}}}}
    raise Exception("Provide candidate/expected or values")
"""


def _math_format_code(spec: dict) -> str:
    return f"""SPECIALTY = {spec['title']!r}
ANSWER_STYLE = {spec['answer_style']!r}


def run(inputs):
    answer = inputs.get("answer")
    if answer is None:
        answer = inputs.get("value")
    if answer is None:
        raise Exception("Missing answer or value")
    if ANSWER_STYLE == "integer":
        formatted = str(int(round(float(answer))))
    elif ANSWER_STYLE == "currency":
        formatted = f"${{float(answer):.2f}}"
    elif ANSWER_STYLE == "percentage":
        formatted = f"{{float(answer):.2f}}%"
    else:
        formatted = str(answer)
    return {{"output": {{"specialty": SPECIALTY, "answer_style": ANSWER_STYLE, "formatted_answer": formatted}}}}
"""


def _code_signature_code(spec: dict) -> str:
    return f"""import re

SPECIALTY = {spec['title']!r}


def run(inputs):
    prompt = str(inputs.get("prompt") or inputs.get("query") or inputs.get("task") or "")
    match = re.search(r"def\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*?)\\):", prompt, re.DOTALL)
    if not match:
        raise Exception("No function definition found")
    params_raw = match.group(2).strip()
    params = []
    if params_raw:
        for item in params_raw.split(","):
            params.append(item.strip().split("=")[0].strip())
    return {{"output": {{"specialty": SPECIALTY, "entry_point": match.group(1), "parameters": [p for p in params if p], "signature_text": match.group(0)}}}}
"""


def _code_examples_code(spec: dict) -> str:
    return f"""import re

SPECIALTY = {spec['title']!r}


def run(inputs):
    prompt = str(inputs.get("prompt") or "")
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    doctests = re.findall(r">>>\\s*(.+)", prompt)
    asserts = re.findall(r"assert\\s+(.+)", test)
    for item in test_list:
        asserts.extend(re.findall(r"assert\\s+(.+)", item))
    return {{"output": {{"specialty": SPECIALTY, "doctests": doctests[:20], "asserts": asserts[:20], "example_count": len(doctests) + len(asserts)}}}}
"""


def _code_structure_code(spec: dict) -> str:
    return f"""import ast

SPECIALTY = {spec['title']!r}


def run(inputs):
    code = str(inputs.get("code") or "")
    if not code.strip():
        raise Exception("Missing code")
    tree = ast.parse(code)
    loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    branches = sum(isinstance(node, ast.If) for node in ast.walk(tree))
    calls = []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    recursion = [name for name in functions if name in calls]
    return {{"output": {{"specialty": SPECIALTY, "functions": functions, "loop_count": loops, "branch_count": branches, "recursive_functions": recursion, "top_calls": calls[:20]}}}}
"""


def _code_execute_code(spec: dict) -> str:
    return f"""import subprocess
import sys
import tempfile

SPECIALTY = {spec['title']!r}


def run(inputs):
    code = str(inputs.get("code") or "")
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    if not code.strip():
        raise Exception("Missing code")
    body = "\\n".join(test_list) if test_list else test
    if not body.strip() and inputs.get("call_expression"):
        body = "assert " + str(inputs["call_expression"])
    if not body.strip():
        raise Exception("Missing test, test_list, or call_expression")
    harness = code + "\\n\\n" + body + "\\n\\nprint('SPECIALTY_OK')\\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        path = fh.name
    completed = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout_s, check=False)
    passed = completed.returncode == 0 and "SPECIALTY_OK" in completed.stdout
    return {{"output": {{"specialty": SPECIALTY, "passed": passed, "returncode": completed.returncode, "stdout": completed.stdout[-400:], "stderr": completed.stderr[-400:]}}}}
"""


def _code_contract_code(spec: dict) -> str:
    return f"""import ast
import re

SPECIALTY = {spec['title']!r}


def run(inputs):
    code = str(inputs.get("code") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    test = str(inputs.get("test") or "")
    if not code.strip():
        raise Exception("Missing code")
    tree = ast.parse(code)
    funcs = {{node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}}
    result = {{"specialty": SPECIALTY, "entry_point": entry_point, "entry_found": entry_point in funcs}}
    if entry_point in funcs:
        result["parameters"] = [arg.arg for arg in funcs[entry_point].args.args]
        result["parameter_count"] = len(result["parameters"])
    else:
        result["parameters"] = []
        result["parameter_count"] = 0
    if test:
        result["assert_count"] = len(re.findall(r"assert\\s+", test))
    return {{"output": result}}
"""


def _code_debug_code(spec: dict) -> str:
    return f"""import re
import textwrap

SPECIALTY = {spec['title']!r}


def run(inputs):
    code = str(inputs.get("code") or "")
    error = str(inputs.get("error") or inputs.get("traceback") or "")
    cleaned = textwrap.dedent(re.sub(r"^```(?:python)?\\s*|```\\s*$", "", code.strip(), flags=re.MULTILINE)).strip()
    focus = []
    for line in error.splitlines():
        if ", line " not in line:
            continue
        try:
            number = int(line.split(", line ", 1)[1].split(",", 1)[0].strip())
        except Exception:
            continue
        snippet = cleaned.splitlines()[max(0, number - 3): number + 2]
        focus.append({{"line": number, "snippet": snippet}})
    return {{"output": {{"specialty": SPECIALTY, "normalized_code": cleaned + ("\\n" if cleaned else ""), "focus": focus, "error_summary": error.splitlines()[-1] if error.splitlines() else error}}}}
"""


def _tool_spec(specialty: dict, kind: tuple[str, str, str, str]) -> dict:
    kind_code, kind_title, kind_summary, role_tag = kind
    family = specialty["family"]
    tool_id = f"spec-{family}-{specialty['code']}-{kind_code}"
    name = f"{specialty['title']} {kind_title}"
    tool_tags = ["specialized", family, specialty["code"], kind_code, _slug(specialty["title"])] + [_slug(item) for item in specialty["keywords"][:4]]
    domain_tags = [family, specialty["code"], _slug(specialty["title"])]
    inputs_map = {
        "signal": ["query", "task"],
        "quantity": ["query", "task"],
        "plan": ["query", "task"],
        "compute": ["expression", "values", "rate", "time", "quantity", "price", "part", "whole"],
        "verify": ["candidate", "expected", "tolerance", "values"],
        "format": ["answer", "value"],
        "signature": ["prompt", "query", "task"],
        "examples": ["prompt", "test", "test_list"],
        "structure": ["code"],
        "execute": ["code", "test", "test_list", "call_expression", "timeout_s"],
        "contract": ["code", "entry_point", "test"],
        "debug": ["code", "error", "traceback"],
    }
    return {
        "id": tool_id,
        "spec": {
            "name": name,
            "description": f"{kind_summary} Specialty: {specialty['summary']}",
            "requirements": "standard-library",
            "domain_tags": domain_tags,
            "role_tags": [role_tag, family],
            "tool_tags": tool_tags,
            "inputs": inputs_map[kind_code],
            "outputs": ["output"],
            "requires_api": False,
            "api_notes": "",
            "modalities": ["text"],
            "output_formats": ["json"],
            "permissions": ["read", "compute"],
            "cost_tier": "low",
            "latency_tier": "low",
            "reliability_prior": 0.82,
            "examples": [specialty["summary"]],
            "embedding_text": f"{name}. {specialty['summary']}. Keywords: {' '.join(specialty['keywords'])}",
        },
        "kind_code": kind_code,
    }


def _tool_code(specialty: dict, kind_code: str) -> str:
    if specialty["family"] == "math":
        return {
            "signal": _math_signal_code,
            "quantity": _math_quantity_code,
            "plan": _math_plan_code,
            "compute": _math_compute_code,
            "verify": _math_verify_code,
            "format": _math_format_code,
        }[kind_code](specialty)
    return {
        "signature": _code_signature_code,
        "examples": _code_examples_code,
        "structure": _code_structure_code,
        "execute": _code_execute_code,
        "contract": _code_contract_code,
        "debug": _code_debug_code,
    }[kind_code](specialty)


@lru_cache(maxsize=1)
def load_specializations() -> list[dict]:
    return _math_specializations() + _code_specializations()


@lru_cache(maxsize=1)
def load_specialized_tool_records() -> list[dict]:
    records = []
    for specialty in load_specializations():
        kinds = MATH_TOOL_KINDS if specialty["family"] == "math" else CODE_TOOL_KINDS
        for kind in kinds:
            payload = _tool_spec(specialty, kind)
            payload["code"] = _tool_code(specialty, payload["kind_code"])
            payload["specialty_code"] = specialty["code"]
            payload["specialty_title"] = specialty["title"]
            records.append(payload)
    return records


def build_specialized_agents() -> list[dict]:
    variants = [("planner", [0, 1, 2, 4, 5]), ("builder", [0, 1, 2, 3, 5]), ("checker", [0, 2, 3, 4, 5])]
    agents = []
    for specialty in load_specializations():
        kinds = MATH_TOOL_KINDS if specialty["family"] == "math" else CODE_TOOL_KINDS
        tool_ids = [f"spec-{specialty['family']}-{specialty['code']}-{kind[0]}" for kind in kinds]
        for role, indices in variants:
            selected = [tool_ids[idx] for idx in indices]
            agents.append(
                {
                    "id": f"agent-spec-{specialty['family']}-{specialty['code']}-{role}",
                    "name": f"{specialty['title']} {role.title()}",
                    "description": f"Specialty: {specialty['summary']} Primary role: {role}.",
                    "domain_tags": [specialty["family"], specialty["code"]],
                    "role_tags": [role],
                    "tool_ids": selected,
                    "embedding_text": f"{specialty['title']} {role}. Keywords: {' '.join(specialty['keywords'])}.",
                }
            )
    return agents
