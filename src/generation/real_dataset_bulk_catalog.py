from __future__ import annotations

import json
from typing import Any, Dict, List


DATASET_ROLE_MAP = {
    "humaneval": ["planner", "builder", "checker"],
    "mbpp": ["planner", "builder", "checker"],
    "gsm8k": ["planner", "builder", "checker"],
    "math": ["planner", "builder", "checker"],
    "hotpotqa": ["planner", "builder", "checker"],
}


CODE_ROLE_TAGS = {
    "planner": ["planner", "code-planner"],
    "builder": ["builder", "code-generation"],
    "checker": ["checker", "code-testing", "refactor", "code-refactoring"],
}


MATH_ROLE_TAGS = {
    "planner": ["planner", "builder"],
    "builder": ["builder"],
    "checker": ["checker", "refactor"],
}


HOTPOTQA_ROLE_TAGS = {
    "planner": ["planner", "qa-planner", "evidence-planner", "decomposer", "retriever"],
    "builder": ["builder", "qa-answerer", "multi-hop-reasoner", "evidence-collector", "evidence-merger", "answer-synthesizer"],
    "checker": ["checker", "qa-verifier", "evidence-checker", "verifier"],
}


HUMANEVAL_FOCUSES: List[Dict[str, Any]] = [
    {"slug": "string-transform", "title": "String Transform", "keywords": ["string", "transform", "replace", "reverse"]},
    {"slug": "string-parsing", "title": "String Parsing", "keywords": ["string", "parse", "token", "split"]},
    {"slug": "list-aggregation", "title": "List Aggregation", "keywords": ["list", "sum", "aggregate", "fold"]},
    {"slug": "list-ordering", "title": "List Ordering", "keywords": ["list", "sort", "order", "rank"]},
    {"slug": "two-pointer-window", "title": "Two Pointer Window", "keywords": ["window", "two-pointer", "subarray", "scan"]},
    {"slug": "hashmap-frequency", "title": "Hashmap Frequency", "keywords": ["hashmap", "frequency", "count", "dictionary"]},
    {"slug": "set-relations", "title": "Set Relations", "keywords": ["set", "intersection", "union", "membership"]},
    {"slug": "stack-parentheses", "title": "Stack Parentheses", "keywords": ["stack", "parentheses", "balance", "brackets"]},
    {"slug": "queue-simulation", "title": "Queue Simulation", "keywords": ["queue", "simulate", "order", "process"]},
    {"slug": "recursion-backtracking", "title": "Recursion Backtracking", "keywords": ["recursion", "backtracking", "search", "choices"]},
    {"slug": "dynamic-programming", "title": "Dynamic Programming", "keywords": ["dp", "state", "recurrence", "memoization"]},
    {"slug": "graph-traversal", "title": "Graph Traversal", "keywords": ["graph", "bfs", "dfs", "path"]},
    {"slug": "tree-recursion", "title": "Tree Recursion", "keywords": ["tree", "recursion", "children", "traverse"]},
    {"slug": "heap-selection", "title": "Heap Selection", "keywords": ["heap", "priority", "top-k", "selection"]},
    {"slug": "binary-search", "title": "Binary Search", "keywords": ["binary search", "sorted", "monotonic", "mid"]},
    {"slug": "interval-processing", "title": "Interval Processing", "keywords": ["interval", "merge", "overlap", "schedule"]},
    {"slug": "matrix-grid", "title": "Matrix Grid", "keywords": ["matrix", "grid", "neighbors", "board"]},
    {"slug": "bit-manipulation", "title": "Bit Manipulation", "keywords": ["bit", "xor", "mask", "shift"]},
    {"slug": "number-theory", "title": "Number Theory", "keywords": ["prime", "gcd", "factor", "divisor"]},
    {"slug": "geometry-helper", "title": "Geometry Helper", "keywords": ["geometry", "distance", "coordinate", "angle"]},
    {"slug": "greedy-choice", "title": "Greedy Choice", "keywords": ["greedy", "choice", "local", "optimize"]},
    {"slug": "prefix-suffix-scan", "title": "Prefix Suffix Scan", "keywords": ["prefix", "suffix", "scan", "accumulate"]},
    {"slug": "linked-structure", "title": "Linked Structure", "keywords": ["linked", "pointer", "next", "node"]},
    {"slug": "simulation-state", "title": "Simulation State", "keywords": ["simulate", "state", "transition", "step"]},
    {"slug": "contract-edge-cases", "title": "Contract Edge Cases", "keywords": ["contract", "edge-case", "signature", "boundary"]},
]


MBPP_FOCUSES: List[Dict[str, Any]] = [
    {"slug": "tuple-transform", "title": "Tuple Transform", "keywords": ["tuple", "convert", "combine", "transform"]},
    {"slug": "tuple-number", "title": "Tuple Number", "keywords": ["tuple", "integer", "digits", "number"]},
    {"slug": "list-filter-map", "title": "List Filter Map", "keywords": ["list", "filter", "map", "elements"]},
    {"slug": "list-rotation", "title": "List Rotation", "keywords": ["list", "rotate", "shift", "slice"]},
    {"slug": "list-sorting", "title": "List Sorting", "keywords": ["list", "sort", "order", "arrange", "monotonic", "comb"]},
    {"slug": "dict-aggregation", "title": "Dict Aggregation", "keywords": ["dict", "group", "aggregate", "mapping"]},
    {"slug": "set-operations", "title": "Set Operations", "keywords": ["set", "union", "intersection", "dedupe"]},
    {"slug": "string-cleaning", "title": "String Cleaning", "keywords": ["string", "strip", "replace", "clean", "vowel", "character"]},
    {"slug": "regex-matching", "title": "Regex Matching", "keywords": ["regex", "match", "pattern", "string", "sequence"]},
    {"slug": "numeric-rounding", "title": "Numeric Rounding", "keywords": ["round", "ceil", "floor", "numeric", "integer", "digit"]},
    {"slug": "combinatorics-helper", "title": "Combinatorics Helper", "keywords": ["combination", "permutation", "ways", "count", "binomial", "coefficient", "product"]},
    {"slug": "divisor-factor", "title": "Divisor Factor", "keywords": ["divisor", "factor", "prime", "gcd", "lcm", "multiple"]},
    {"slug": "geometry-formula", "title": "Geometry Formula", "keywords": ["geometry", "area", "perimeter", "volume", "parabola", "directrix", "tetrahedron", "sector", "triangle", "circle"]},
    {"slug": "date-time-format", "title": "Date Time Format", "keywords": ["date", "time", "format", "calendar"]},
    {"slug": "heap-sort", "title": "Heap Sort", "keywords": ["heap", "sort", "priority", "order"]},
    {"slug": "stack-queue", "title": "Stack Queue", "keywords": ["stack", "queue", "push", "pop"]},
    {"slug": "recursion-pattern", "title": "Recursion Pattern", "keywords": ["recursion", "base-case", "recursive", "call"]},
    {"slug": "dp-basics", "title": "DP Basics", "keywords": ["dp", "tabulation", "memoization", "state"]},
    {"slug": "interval-utils", "title": "Interval Utils", "keywords": ["interval", "merge", "range", "overlap"]},
    {"slug": "matrix-iteration", "title": "Matrix Iteration", "keywords": ["matrix", "grid", "row", "column"]},
    {"slug": "bit-tricks", "title": "Bit Tricks", "keywords": ["bit", "xor", "mask", "binary"]},
    {"slug": "search-membership", "title": "Search Membership", "keywords": ["find", "contains", "search", "index"]},
    {"slug": "arithmetic-series", "title": "Arithmetic Series", "keywords": ["series", "sum", "sequence", "arithmetic"]},
    {"slug": "statistics-helper", "title": "Statistics Helper", "keywords": ["mean", "median", "mode", "range"]},
    {"slug": "import-heavy-utils", "title": "Import Heavy Utils", "keywords": ["math", "collections", "itertools", "helper"]},
]


GSM8K_FOCUSES: List[Dict[str, Any]] = [
    {"slug": "ratio-chain", "title": "Ratio Chain", "keywords": ["ratio", "times as many", "proportion", "chain"]},
    {"slug": "unit-rate", "title": "Unit Rate", "keywords": ["per", "each", "unit rate", "price"]},
    {"slug": "percent-change", "title": "Percent Change", "keywords": ["percent", "increase", "decrease", "discount"]},
    {"slug": "fraction-conversion", "title": "Fraction Conversion", "keywords": ["fraction", "part", "share", "convert"]},
    {"slug": "mixture-balance", "title": "Mixture Balance", "keywords": ["mixture", "average", "blend", "concentration"]},
    {"slug": "work-rate", "title": "Work Rate", "keywords": ["work", "rate", "together", "finish"]},
    {"slug": "time-distance", "title": "Time Distance", "keywords": ["time", "distance", "speed", "travel"]},
    {"slug": "profit-margin", "title": "Profit Margin", "keywords": ["profit", "cost", "revenue", "margin"]},
    {"slug": "discount-tax", "title": "Discount Tax", "keywords": ["discount", "tax", "sale", "coupon"]},
    {"slug": "sequence-count", "title": "Sequence Count", "keywords": ["sequence", "count", "repeat", "pattern"]},
    {"slug": "age-equation", "title": "Age Equation", "keywords": ["age", "older", "younger", "years"]},
    {"slug": "set-overlap", "title": "Set Overlap", "keywords": ["both", "either", "neither", "overlap"]},
    {"slug": "piecewise-pricing", "title": "Piecewise Pricing", "keywords": ["pricing", "tiers", "bands", "cost"]},
    {"slug": "geometry-area", "title": "Geometry Area", "keywords": ["area", "rectangle", "triangle", "shape"]},
    {"slug": "geometry-volume", "title": "Geometry Volume", "keywords": ["volume", "cube", "box", "cylinder"]},
    {"slug": "probability-expected", "title": "Probability Expected", "keywords": ["probability", "chance", "expected", "random"]},
    {"slug": "average-weighted", "title": "Average Weighted", "keywords": ["average", "weighted", "mean", "mix"]},
    {"slug": "clock-calendar", "title": "Clock Calendar", "keywords": ["clock", "calendar", "hours", "days"]},
    {"slug": "travel-schedule", "title": "Travel Schedule", "keywords": ["schedule", "trip", "depart", "arrive"]},
    {"slug": "inventory-flow", "title": "Inventory Flow", "keywords": ["inventory", "stock", "sold", "remaining"]},
    {"slug": "salary-overtime", "title": "Salary Overtime", "keywords": ["salary", "overtime", "hours", "wage"]},
    {"slug": "loan-interest", "title": "Loan Interest", "keywords": ["loan", "interest", "repay", "borrow"]},
    {"slug": "currency-exchange", "title": "Currency Exchange", "keywords": ["currency", "exchange", "convert", "money"]},
    {"slug": "measurement-conversion", "title": "Measurement Conversion", "keywords": ["convert", "units", "dozen", "feet"]},
    {"slug": "comparison-scaling", "title": "Comparison Scaling", "keywords": ["compare", "scale", "times", "relative"]},
]


MATH_FOCUSES: List[Dict[str, Any]] = [
    {"slug": "counting-arrangement", "title": "Counting Arrangement", "keywords": ["arrangement", "count", "ways", "permutation"]},
    {"slug": "probability-expected", "title": "Probability Expected", "keywords": ["probability", "expected", "random", "chance"]},
    {"slug": "fraction-decimal", "title": "Fraction Decimal", "keywords": ["fraction", "decimal", "ratio", "convert"]},
    {"slug": "geometry-area-perimeter", "title": "Geometry Area Perimeter", "keywords": ["area", "perimeter", "shape", "measure"]},
    {"slug": "geometry-angle", "title": "Geometry Angle", "keywords": ["angle", "triangle", "degree", "trisector"]},
    {"slug": "prime-divisibility", "title": "Prime Divisibility", "keywords": ["prime", "divisible", "factor", "divisor"]},
    {"slug": "unit-conversion", "title": "Unit Conversion", "keywords": ["unit", "convert", "measurement", "dozen"]},
    {"slug": "work-time-motion", "title": "Work Time Motion", "keywords": ["work", "time", "motion", "rate"]},
    {"slug": "expression-simplify", "title": "Expression Simplify", "keywords": ["expression", "simplify", "evaluate", "algebra"]},
    {"slug": "ratio-rate-percent", "title": "Ratio Rate Percent", "keywords": ["ratio", "rate", "percent", "proportion"]},
    {"slug": "sequence-statistics", "title": "Sequence Statistics", "keywords": ["sequence", "series", "mean", "median"]},
    {"slug": "linear-equation", "title": "Linear Equation", "keywords": ["equation", "linear", "solve", "x"]},
    {"slug": "quadratic-equation", "title": "Quadratic Equation", "keywords": ["quadratic", "parabola", "roots", "solve"]},
    {"slug": "polynomial-evaluation", "title": "Polynomial Evaluation", "keywords": ["polynomial", "evaluate", "coefficients", "value"]},
    {"slug": "modular-arithmetic", "title": "Modular Arithmetic", "keywords": ["mod", "remainder", "parity", "residue"]},
    {"slug": "gcd-lcm", "title": "GCD LCM", "keywords": ["gcd", "lcm", "common divisor", "multiple"]},
    {"slug": "divisor-counting", "title": "Divisor Counting", "keywords": ["divisor", "count", "factors", "factorization"]},
    {"slug": "logarithm-exponential", "title": "Logarithm Exponential", "keywords": ["log", "exponential", "power", "growth"]},
    {"slug": "radicals-surds", "title": "Radicals Surds", "keywords": ["sqrt", "radical", "surd", "simplify"]},
    {"slug": "coordinate-geometry", "title": "Coordinate Geometry", "keywords": ["coordinate", "distance", "midpoint", "slope"]},
    {"slug": "circle-geometry", "title": "Circle Geometry", "keywords": ["circle", "radius", "arc", "circumference"]},
    {"slug": "triangle-geometry", "title": "Triangle Geometry", "keywords": ["triangle", "side", "area", "angle"]},
    {"slug": "combinatorics-selection", "title": "Combinatorics Selection", "keywords": ["choose", "selection", "committee", "combinations"]},
    {"slug": "exact-simplification", "title": "Exact Simplification", "keywords": ["exact", "boxed", "simplify", "canonical"]},
    {"slug": "general-multistep", "title": "General Multistep", "keywords": ["multistep", "compose", "derive", "final"]},
]


HOTPOTQA_FOCUSES: List[Dict[str, Any]] = [
    {"slug": "bridge-entity-linking", "title": "Bridge Entity Linking", "keywords": ["bridge", "entity", "link", "mentioned", "related"]},
    {"slug": "bridge-headquarters-company", "title": "Bridge Headquarters Company", "keywords": ["headquarters", "company", "group", "located", "serves"]},
    {"slug": "bridge-creative-work-person", "title": "Bridge Creative Work Person", "keywords": ["album", "film", "book", "member", "creator"]},
    {"slug": "bridge-location-origin", "title": "Bridge Location Origin", "keywords": ["where", "from", "located", "born", "origin"]},
    {"slug": "bridge-acronym-expansion", "title": "Bridge Acronym Expansion", "keywords": ["acronym", "abbreviated", "stand for", "name", "changed"]},
    {"slug": "bridge-membership-affiliation", "title": "Bridge Membership Affiliation", "keywords": ["member", "team", "band", "affiliation", "joined"]},
    {"slug": "comparison-person-count", "title": "Comparison Person Count", "keywords": ["which", "more", "fewer", "member", "bands"]},
    {"slug": "comparison-organization-count", "title": "Comparison Organization Count", "keywords": ["organization", "more", "fewer", "count", "markets"]},
    {"slug": "comparison-date-temporal", "title": "Comparison Date Temporal", "keywords": ["date", "before", "after", "earlier", "later"]},
    {"slug": "comparison-location-scale", "title": "Comparison Location Scale", "keywords": ["larger", "smaller", "population", "area", "located"]},
    {"slug": "sports-team-owner", "title": "Sports Team Owner", "keywords": ["team", "owner", "played", "cup", "championship"]},
    {"slug": "music-band-member", "title": "Music Band Member", "keywords": ["band", "member", "album", "guitarist", "vocalist"]},
    {"slug": "film-tv-credit", "title": "Film TV Credit", "keywords": ["film", "television", "actor", "director", "played"]},
    {"slug": "corporate-subsidiary", "title": "Corporate Subsidiary", "keywords": ["company", "subsidiary", "owned", "parent", "group"]},
    {"slug": "education-institution", "title": "Education Institution", "keywords": ["school", "university", "college", "institution", "campus"]},
    {"slug": "geography-admin-region", "title": "Geography Admin Region", "keywords": ["city", "state", "country", "region", "province"]},
    {"slug": "award-recognition", "title": "Award Recognition", "keywords": ["award", "won", "nominated", "recognition", "prize"]},
    {"slug": "publication-album-work", "title": "Publication Album Work", "keywords": ["released", "published", "album", "book", "work"]},
    {"slug": "legal-political-office", "title": "Legal Political Office", "keywords": ["governor", "office", "election", "law", "political"]},
    {"slug": "architecture-building", "title": "Architecture Building", "keywords": ["building", "tower", "headquarters", "completed", "located"]},
    {"slug": "transportation-route", "title": "Transportation Route", "keywords": ["railway", "route", "line", "station", "transport"]},
    {"slug": "historical-event", "title": "Historical Event", "keywords": ["event", "history", "war", "compromise", "formed"]},
    {"slug": "biography-occupation", "title": "Biography Occupation", "keywords": ["born", "occupation", "known", "person", "career"]},
    {"slug": "science-technology-entity", "title": "Science Technology Entity", "keywords": ["technology", "science", "software", "system", "association"]},
    {"slug": "general-multihop", "title": "General Multihop", "keywords": ["multi-hop", "supporting facts", "bridge", "comparison", "answer"]},
]


DATASET_CONFIG = {
    "humaneval": {
        "display": "HumanEval",
        "domain_tags": ["humaneval", "python-function-synthesis", "real-tools", "code"],
        "role_tags": CODE_ROLE_TAGS,
        "focuses": HUMANEVAL_FOCUSES,
        "tool_kinds": [
            ("signature-extractor", "Signature Extractor"),
            ("code-normalizer", "Code Normalizer"),
            ("structure-checker", "Structure Checker"),
            ("public-test-runner", "Public Test Runner"),
            ("traceback-localizer", "Traceback Localizer"),
        ],
    },
    "mbpp": {
        "display": "MBPP",
        "domain_tags": ["mbpp", "assert-driven-programming", "real-tools", "code"],
        "role_tags": CODE_ROLE_TAGS,
        "focuses": MBPP_FOCUSES,
        "tool_kinds": [
            ("assert-parser", "Assert Parser"),
            ("import-injector", "Import Injector"),
            ("contract-gate", "Contract Gate"),
            ("assert-scorer", "Assert Scorer"),
            ("failure-diagnoser", "Failure Diagnoser"),
        ],
    },
    "gsm8k": {
        "display": "GSM8K",
        "domain_tags": ["gsm8k", "arithmetic-word-problem", "real-tools", "math"],
        "role_tags": MATH_ROLE_TAGS,
        "focuses": GSM8K_FOCUSES,
        "tool_kinds": [
            ("signal-profiler", "Signal Profiler"),
            ("quantity-extractor", "Quantity Extractor"),
            ("relation-calculator", "Relation Calculator"),
            ("answer-verifier", "Answer Verifier"),
            ("answer-formatter", "Answer Formatter"),
        ],
    },
    "math": {
        "display": "MATH",
        "domain_tags": ["math", "exact-competition-math", "real-tools"],
        "role_tags": MATH_ROLE_TAGS,
        "focuses": MATH_FOCUSES,
        "tool_kinds": [
            ("signal-profiler", "Signal Profiler"),
            ("quantity-extractor", "Quantity Extractor"),
            ("equation-engine", "Equation Engine"),
            ("answer-verifier", "Answer Verifier"),
            ("canonicalizer", "Canonicalizer"),
        ],
    },
    "hotpotqa": {
        "display": "HotpotQA",
        "domain_tags": ["hotpotqa", "multi-hop-qa", "evidence-grounded-qa", "real-tools"],
        "role_tags": HOTPOTQA_ROLE_TAGS,
        "focuses": HOTPOTQA_FOCUSES,
        "tool_kinds": [
            ("question-decomposer", "Question Decomposer"),
            ("context-indexer", "Context Indexer"),
            ("evidence-chain-builder", "Evidence Chain Builder"),
            ("answer-extractor", "Answer Extractor"),
            ("support-verifier", "Support Verifier"),
        ],
    },
}


def _title_case(slug: str) -> str:
    return "".join(part.title() for part in slug.split("-"))


def _dataset_keywords(dataset: str) -> List[str]:
    if dataset in {"humaneval", "mbpp"}:
        return ["python", "function", "code"]
    if dataset == "hotpotqa":
        return ["qa", "evidence", "multi-hop", "context"]
    return ["math", "reasoning", "numeric"]


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
    dataset_family: str,
    focus_slug: str,
    primary_role: str,
    capability_kind: str,
    stage: str,
    consumes: List[str],
    produces: List[str],
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
            "outputs": ["output"],
            "requires_api": False,
            "api_notes": "",
            "modalities": ["text"],
            "output_formats": ["json"],
            "permissions": ["read", "compute"],
            "cost_tier": "low",
            "latency_tier": "low",
            "reliability_prior": 0.88,
            "examples": [description],
            "embedding_text": f"{name}. {description}. Tags: {' '.join(tool_tags)}",
            "dataset_family": dataset_family,
            "focus_slug": focus_slug,
            "primary_role": primary_role,
            "capability_kind": capability_kind,
            "stage": stage,
            "deterministic": True,
            "consumes": list(consumes),
            "produces": list(produces),
        },
        "code": code,
    }


def _capability_metadata(dataset: str, kind_slug: str) -> Dict[str, Any]:
    if dataset in {"humaneval", "mbpp"}:
        mapping = {
            "signature-extractor": ("analyze", ["task", "query", "prompt"], ["entry_point", "diagnostics"]),
            "assert-parser": ("analyze", ["task", "query", "tests_public"], ["diagnostics"]),
            "import-injector": ("prepare", ["draft_code"], ["normalized_code", "diagnostics"]),
            "contract-gate": ("verify", ["draft_code", "entry_point"], ["verification", "diagnostics"]),
            "code-normalizer": ("prepare", ["draft_code", "entry_point"], ["normalized_code", "diagnostics"]),
            "structure-checker": ("verify", ["normalized_code", "draft_code"], ["verification", "diagnostics"]),
            "public-test-runner": ("execute", ["normalized_code", "draft_code", "tests_public"], ["verification", "diagnostics"]),
            "assert-scorer": ("execute", ["normalized_code", "draft_code", "tests_public"], ["verification", "diagnostics"]),
            "traceback-localizer": ("diagnose", ["draft_code", "diagnostics"], ["diagnostics"]),
            "failure-diagnoser": ("diagnose", ["draft_code", "diagnostics"], ["diagnostics"]),
        }
    elif dataset == "hotpotqa":
        mapping = {
            "question-decomposer": ("analyze", ["question", "query", "context"], ["reasoning_trace", "diagnostics"]),
            "context-indexer": ("retrieve", ["question", "context"], ["evidence_points", "diagnostics"]),
            "evidence-chain-builder": ("solve_support", ["question", "context", "evidence_points"], ["reasoning_trace", "candidate_answer", "evidence_points"]),
            "answer-extractor": ("answer", ["question", "evidence_points", "reasoning_trace"], ["candidate_answer", "final_answer"]),
            "support-verifier": ("verify", ["candidate_answer", "evidence_points"], ["verification", "formatted_answer", "final_answer"]),
        }
    else:
        mapping = {
            "signal-profiler": ("analyze", ["task", "query"], ["diagnostics"]),
            "quantity-extractor": ("analyze", ["task", "query"], ["reasoning_trace", "diagnostics"]),
            "relation-calculator": ("solve_support", ["task", "query", "reasoning_trace"], ["reasoning_trace", "diagnostics"]),
            "equation-engine": ("solve_support", ["task", "query", "reasoning_trace"], ["reasoning_trace", "diagnostics"]),
            "answer-verifier": ("verify", ["candidate_answer", "formatted_answer"], ["verification", "diagnostics"]),
            "answer-formatter": ("format", ["candidate_answer"], ["formatted_answer", "final_answer"]),
            "canonicalizer": ("format", ["candidate_answer"], ["formatted_answer", "final_answer"]),
        }
    stage, consumes, produces = mapping.get(kind_slug, ("analyze", ["task"], ["diagnostics"]))
    return {
        "capability_kind": kind_slug,
        "stage": stage,
        "consumes": consumes,
        "produces": produces,
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _code_tool_code(dataset: str, focus_title: str, keywords: List[str], primary_role: str, kind: str) -> str:
    keywords_json = _json_dumps(keywords)
    common = f"""import ast
import re
import subprocess
import sys
import tempfile
import textwrap

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
TOOL_KIND = {kind!r}
KEYWORDS = {keywords_json}
_DEF_RE = re.compile(r"def\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\((.*?)\\)\\s*:", re.DOTALL)


def _extract_parameters(signature_text):
    params = []
    for item in str(signature_text or "").split(","):
        name = item.strip()
        if not name:
            continue
        name = name.split("=", 1)[0].strip()
        if not name:
            continue
        params.append(name)
    return params


def _strip_fences(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:python)?\\s*", "", text)
    text = re.sub(r"```\\s*$", "", text)
    return text.strip()


def _extract_signature_from_prompt(prompt, entry_point):
    prompt_text = str(prompt or "")
    match = _DEF_RE.search(prompt_text)
    resolved_entry = str(entry_point or "").strip()
    params = []
    signature_text = ""
    if match:
        resolved_entry = resolved_entry or match.group(1)
        params = _extract_parameters(match.group(2))
        signature_text = match.group(0)
    return resolved_entry, params, signature_text


def _inspect_functions(code_text):
    functions = {{}}
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return functions, None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            if getattr(node.args, "vararg", None) is not None:
                params.append("*" + str(node.args.vararg.arg))
            if getattr(node.args, "kwarg", None) is not None:
                params.append("**" + str(node.args.kwarg.arg))
            functions[node.name] = params
    return functions, tree


def _resolve_signature(inputs):
    prompt = str(inputs.get("prompt") or inputs.get("query") or inputs.get("task") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    param_names = []
    if isinstance(inputs.get("param_names"), list):
        param_names = [str(item).strip() for item in inputs.get("param_names") if str(item).strip()]
    func_sig = inputs.get("function_signature")
    signature_text = ""
    if isinstance(func_sig, dict):
        if not param_names:
            param_names = [str(item).strip() for item in (func_sig.get("parameters") or []) if str(item).strip()]
        signature_text = str(func_sig.get("signature_text") or func_sig.get("signature") or "")
    prompt_entry, prompt_params, prompt_signature = _extract_signature_from_prompt(prompt, entry_point)
    entry_point = entry_point or prompt_entry
    if not param_names:
        param_names = prompt_params
    if not signature_text:
        signature_text = prompt_signature
    return prompt, entry_point, param_names, signature_text


def _indent_block(body):
    lines = str(body or "").splitlines()
    if not lines:
        return "    pass"
    return "\\n".join(("    " + line) if line.strip() else "    " for line in lines)


def _canonicalize_code(inputs):
    raw_code = (
        inputs.get("code")
        or inputs.get("draft_code")
        or inputs.get("normalized_code")
        or inputs.get("query")
        or ""
    )
    raw_candidate = str(raw_code or "")
    if not raw_candidate.strip():
        raise Exception("Missing code")
    prompt, entry_point, param_names, signature_text = _resolve_signature(inputs)
    cleaned = textwrap.dedent(_strip_fences(raw_candidate)).strip("\\n")
    functions, _ = _inspect_functions(cleaned)
    if functions and ((entry_point and entry_point in functions) or (not entry_point and functions)):
        if not entry_point:
            entry_point = next(iter(functions.keys()))
        if not param_names:
            param_names = list(functions.get(entry_point, []))
        wrapped_code = cleaned.rstrip() + "\\n"
        code_shape = "raw_full_def"
    else:
        if not entry_point:
            raise Exception("Missing entry_point for body-only code")
        signature = "def %s(%s):" % (entry_point, ", ".join(param_names))
        wrapped_code = signature + "\\n" + _indent_block(cleaned or "pass") + "\\n"
        code_shape = "wrapped_body"
    return {{
        "raw_candidate": raw_candidate,
        "cleaned_candidate": cleaned,
        "wrapped_code": wrapped_code,
        "code_shape": code_shape,
        "entry_point": entry_point or None,
        "param_names": list(param_names),
        "signature_text": signature_text,
        "prompt": prompt,
    }}


def _split_test_lines(test, test_list):
    raw_lines = []
    if isinstance(test_list, list) and test_list:
        for item in test_list:
            raw_lines.extend(str(item).splitlines())
    else:
        raw_lines = str(test or "").splitlines()
    setup = []
    asserts = []
    in_check = False
    check_indent = 0
    in_test_check = False
    test_check_indent = 0
    for line in raw_lines:
        stripped = str(line).strip()
        cur_indent = len(str(line)) - len(str(line).lstrip())
        if stripped.startswith("def check("):
            in_check = True
            check_indent = cur_indent
            continue
        if stripped.startswith("def test_check("):
            in_test_check = True
            test_check_indent = cur_indent
            continue
        if in_check and stripped and cur_indent <= check_indent and not stripped.startswith("#"):
            in_check = False
        if in_test_check and stripped and cur_indent <= test_check_indent and not stripped.startswith("#"):
            in_test_check = False
        if in_test_check:
            continue
        if stripped.startswith("check(") or stripped.startswith("test_check("):
            continue
        if stripped.startswith("assert "):
            asserts.append(stripped)
        elif not in_check:
            setup.append(str(line))
    return setup, asserts


def _rewrite_test_body(test, test_list, entry_point):
    body = "\\n".join(str(item) for item in (test_list or [])) if test_list else str(test or "")
    if entry_point:
        body = body.replace("candidate(", "%s(" % entry_point)
        body = body.replace("check(candidate)", "check(%s)" % entry_point)
        body = "candidate = %s\\n" % entry_point + body
    return body


def _run_python(program, timeout_s):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    completed = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout_s, check=False)
    return completed
"""
    if kind == "signature-extractor":
        return common + """

def run(inputs):
    prompt, entry_point, parameters, signature_text = _resolve_signature(inputs)
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    cue_hits = [word for word in KEYWORDS if word.lower() in prompt.lower()]
    observed_calls = []
    for raw in test_list or ([test] if test else []):
        text = str(raw).strip()
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {entry_point or "", "candidate"}:
                observed_calls.append(ast.unparse(node) if hasattr(ast, "unparse") else node.func.id)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "entry_point": entry_point or None, "parameters": parameters, "param_names": parameters, "signature_text": signature_text, "cue_hits": cue_hits, "observed_calls": observed_calls[:10], "function_signature": {"entry_point": entry_point or None, "parameters": parameters, "signature_text": signature_text}}}
"""
    if kind == "code-normalizer":
        return common + """

def run(inputs):
    canonical = _canonicalize_code(inputs)
    syntax_ok = True
    syntax_error = ""
    try:
        ast.parse(canonical["wrapped_code"])
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = str(exc)
    cue_hits = [word for word in KEYWORDS if word.lower() in canonical["wrapped_code"].lower()]
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "normalized_code": canonical["wrapped_code"], "extracted_function": canonical["wrapped_code"], "raw_candidate": canonical["raw_candidate"], "entry_point": canonical["entry_point"], "param_names": canonical["param_names"], "code_shape": canonical["code_shape"], "syntax_ok": syntax_ok, "syntax_error": syntax_error, "cue_hits": cue_hits}}
"""
    if kind == "example-extractor" or kind == "assert-parser":
        return f"""import ast

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}


def run(inputs):
    prompt = str(inputs.get("prompt") or "")
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    doctests = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith(">>>"):
            doctests.append(line[3:].strip())
    asserts = []
    for raw in test_list or test.splitlines():
        line = str(raw).strip()
        if not line.startswith("assert "):
            continue
        try:
            node = ast.parse(line).body[0]
            if isinstance(node, ast.Assert):
                asserts.append(ast.unparse(node.test) if hasattr(ast, "unparse") else line)
        except SyntaxError:
            asserts.append(line)
    cue_hits = [word for word in KEYWORDS if word.lower() in (prompt + "\\n" + test).lower()]
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "doctests": doctests[:20], "asserts": asserts[:20], "cue_hits": cue_hits, "example_count": len(doctests) + len(asserts)}}}}
"""
    if kind == "import-injector":
        return common + """
MODULES = ["math", "collections", "itertools", "heapq", "functools", "statistics", "re"]


def run(inputs):
    canonical = _canonicalize_code(inputs)
    try:
        tree = ast.parse(canonical["wrapped_code"])
    except SyntaxError as exc:
        return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "normalized_code": canonical["wrapped_code"], "injected_imports": [], "entry_point": canonical["entry_point"], "param_names": canonical["param_names"], "code_shape": canonical["code_shape"], "syntax_error": str(exc)}}
    imported = set()
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Name):
            used.add(node.id)
    injections = []
    for module in MODULES:
        if module in used and module not in imported:
            injections.append("import %s" % module)
    patched_code = ("\\n".join(injections) + "\\n" + canonical["wrapped_code"]) if injections else canonical["wrapped_code"]
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "normalized_code": patched_code, "injected_imports": injections, "entry_point": canonical["entry_point"], "param_names": canonical["param_names"], "code_shape": canonical["code_shape"]}}
"""
    if kind == "structure-checker" or kind == "contract-gate":
        return common + """

def run(inputs):
    canonical = _canonicalize_code(inputs)
    try:
        tree = ast.parse(canonical["wrapped_code"])
    except SyntaxError as exc:
        return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "entry_point": canonical["entry_point"] or None, "function_found": False, "parameters": canonical["param_names"], "param_names": canonical["param_names"], "functions": [], "loop_count": 0, "branch_count": 0, "recursive": False, "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"], "syntax_error": str(exc)}}
    functions = {}
    loops = 0
    branches = 0
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            if getattr(node.args, "vararg", None) is not None:
                params.append("*" + str(node.args.vararg.arg))
            if getattr(node.args, "kwarg", None) is not None:
                params.append("**" + str(node.args.kwarg.arg))
            functions[node.name] = params
        elif isinstance(node, (ast.For, ast.While)):
            loops += 1
        elif isinstance(node, ast.If):
            branches += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    entry_point = canonical["entry_point"]
    found = entry_point in functions if entry_point else False
    recursive = entry_point in calls if entry_point else False
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "entry_point": entry_point or None, "function_found": found, "parameters": functions.get(entry_point, canonical["param_names"]), "param_names": functions.get(entry_point, canonical["param_names"]), "functions": list(functions.keys()), "loop_count": loops, "branch_count": branches, "recursive": recursive, "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"]}}
"""
    if kind == "public-test-runner" or kind == "assert-scorer":
        return common + """

def run(inputs):
    canonical = _canonicalize_code(inputs)
    entry_point = canonical["entry_point"]
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    timeout_s = float(inputs.get("timeout_s") or 5.0)
    if TOOL_KIND == "assert-scorer":
        setup_lines, assert_lines = _split_test_lines(test, test_list)
        harness_lines = [canonical["wrapped_code"].rstrip(), ""]
        if setup_lines:
            harness_lines.extend(setup_lines)
            harness_lines.append("")
        if entry_point:
            harness_lines.append("candidate = %s" % entry_point)
        harness_lines.append("passed_count = 0")
        harness_lines.append("total_count = 0")
        harness_lines.append("first_failure_kind = ''")
        harness_lines.append("first_failure = ''")
        for assertion in assert_lines:
            harness_lines.append("total_count += 1")
            harness_lines.append("try:")
            harness_lines.append("    " + assertion)
            harness_lines.append("    passed_count += 1")
            harness_lines.append("except Exception as exc:")
            harness_lines.append("    if not first_failure_kind:")
            harness_lines.append("        first_failure_kind = type(exc).__name__")
            harness_lines.append("        first_failure = type(exc).__name__ + ': ' + str(exc)")
        harness_lines.append("print('ASSERT_SCORE:%d/%d' % (passed_count, total_count))")
        harness_lines.append("print('FIRST_FAILURE_KIND:' + first_failure_kind)")
        harness_lines.append("print('FIRST_FAILURE:' + first_failure)")
        completed = _run_python("\\n".join(harness_lines) + "\\n", timeout_s)
        passed = completed.returncode == 0 and ("ASSERT_SCORE:" in completed.stdout)
        score_match = re.search(r"ASSERT_SCORE:(\\d+)/(\\d+)", completed.stdout)
        first_failure_kind = ""
        first_failure = ""
        if score_match:
            assert_score_passed = int(score_match.group(1))
            assert_score_total = int(score_match.group(2))
        else:
            assert_score_passed = 0
            assert_score_total = len(assert_lines)
        for line in completed.stdout.splitlines():
            if line.startswith("FIRST_FAILURE_KIND:"):
                first_failure_kind = line.split(":", 1)[1].strip()
            elif line.startswith("FIRST_FAILURE:"):
                first_failure = line.split(":", 1)[1].strip()
        return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "passed": bool(assert_score_total >= 0 and assert_score_passed == assert_score_total and completed.returncode == 0), "assert_score_passed": int(assert_score_passed), "assert_score_total": int(assert_score_total), "first_failure_kind": first_failure_kind, "first_failure": first_failure, "returncode": completed.returncode, "stdout": completed.stdout[-400:], "stderr": completed.stderr[-400:], "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"]}}
    body = _rewrite_test_body(test, test_list, entry_point)
    harness = canonical["wrapped_code"] + "\\n\\n" + body + "\\n\\nprint('REAL_TOOL_OK')\\n"
    completed = _run_python(harness, timeout_s)
    passed = completed.returncode == 0 and "REAL_TOOL_OK" in completed.stdout
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "passed": passed, "returncode": completed.returncode, "stdout": completed.stdout[-400:], "stderr": completed.stderr[-400:], "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"]}}
"""
    if kind == "traceback-localizer" or kind == "failure-diagnoser":
        return f"""DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}


def run(inputs):
    code = str(inputs.get("code") or "")
    error = str(inputs.get("traceback") or inputs.get("error") or "")
    lines = code.splitlines()
    focus_lines = []
    for raw in error.splitlines():
        if ", line " not in raw:
            continue
        try:
            line_no = int(raw.split(", line ", 1)[1].split(",", 1)[0].strip())
        except Exception:
            continue
        context = []
        for pos in range(max(0, line_no - 3), min(len(lines), line_no + 2)):
            context.append({{"line": pos + 1, "code": lines[pos]}})
        focus_lines.append({{"line": line_no, "context": context}})
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "error_summary": error.splitlines()[-1] if error.splitlines() else error, "focus_lines": focus_lines}}}}
"""
    raise ValueError(f"Unsupported code tool kind: {kind}")


def _math_tool_code(dataset: str, focus_title: str, keywords: List[str], primary_role: str, kind: str) -> str:
    keywords_json = _json_dumps(keywords)
    if kind == "signal-profiler":
        return f"""import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}


def run(inputs):
    question = str(inputs.get("question") or inputs.get("problem") or inputs.get("query") or inputs.get("task") or "")
    lower = question.lower()
    cue_hits = [word for word in KEYWORDS if word.lower() in lower]
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", question)
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "cue_hits": cue_hits, "match_score": len(cue_hits), "number_count": len(numbers)}}}}
"""
    if kind == "quantity-extractor":
        return f"""import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}
UNITS = ["dollar", "dollars", "percent", "%", "hour", "hours", "day", "days", "mile", "miles", "degree", "degrees", "dozen", "feet", "meters"]


def run(inputs):
    text = str(inputs.get("question") or inputs.get("problem") or inputs.get("query") or inputs.get("task") or "")
    lower = text.lower()
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
    fractions = re.findall(r"\\b\\d+\\s*/\\s*\\d+\\b", text)
    units = [unit for unit in UNITS if unit in lower]
    cue_hits = [word for word in KEYWORDS if word.lower() in lower]
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "numbers": numbers, "fractions": fractions, "units": units, "cue_hits": cue_hits}}}}
"""
    if kind == "relation-calculator":
        return f"""import ast
import math
import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}


def _normalize(expr):
    text = str(expr or "").strip()
    text = re.sub(r"\\\\(?:d|t)?frac\\s*\\{{([^{{}}]+)\\}}\\s*\\{{([^{{}}]+)\\}}", r"(\\1)/(\\2)", text)
    return text.replace("^", "**")


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
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        funcs = {{"sqrt": math.sqrt, "abs": abs, "round": round, "factorial": math.factorial, "comb": math.comb, "perm": math.perm}}
        if node.func.id not in funcs:
            raise ValueError("Unsupported function")
        return funcs[node.func.id](*[_eval(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError("Unsupported symbol")
    raise ValueError("Unsupported expression")


def run(inputs):
    expression = inputs.get("expression")
    if expression is None:
        if all(key in inputs for key in ("total", "unit_price")):
            value = float(inputs["total"]) / float(inputs["unit_price"])
            return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "mode": "total_over_unit_price", "value": value}}}}
        if all(key in inputs for key in ("distance", "speed")):
            value = float(inputs["distance"]) / float(inputs["speed"])
            return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "mode": "distance_over_speed", "value": value}}}}
        raise Exception("Missing expression or structured numeric inputs")
    normalized = _normalize(expression)
    value = _eval(ast.parse(normalized, mode="eval"))
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "expression": expression, "normalized_expression": normalized, "value": value}}}}
"""
    if kind == "equation-engine":
        return f"""import math
import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}


def _parse_polynomial_side(text, variable):
    coeffs = {{}}
    normalized = text.replace("-", "+-").replace(" ", "")
    for term in [item for item in normalized.split("+") if item]:
        if variable not in term:
            coeffs[0] = coeffs.get(0, 0.0) + float(term)
            continue
        if term == variable:
            coeffs[1] = coeffs.get(1, 0.0) + 1.0
            continue
        if term == "-" + variable:
            coeffs[1] = coeffs.get(1, 0.0) - 1.0
            continue
        match = re.fullmatch(r"([+-]?\\d+(?:\\.\\d+)?)\\*?" + re.escape(variable) + r"(?:\\*\\*(\\d+))?", term)
        if not match:
            raise ValueError("Unsupported term")
        coefficient = float(match.group(1))
        power = int(match.group(2) or "1")
        coeffs[power] = coeffs.get(power, 0.0) + coefficient
    return coeffs


def _combine(lhs, rhs):
    powers = set(lhs) | set(rhs)
    return {{power: lhs.get(power, 0.0) - rhs.get(power, 0.0) for power in powers}}


def _solve_linear(coeffs):
    a = coeffs.get(1, 0.0)
    b = coeffs.get(0, 0.0)
    if abs(a) <= 1e-12:
        raise ValueError("Not a linear equation")
    return [(-b) / a]


def _solve_quadratic(coeffs):
    a = coeffs.get(2, 0.0)
    b = coeffs.get(1, 0.0)
    c = coeffs.get(0, 0.0)
    if abs(a) <= 1e-12:
        return _solve_linear(coeffs)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-12:
        return []
    discriminant = max(discriminant, 0.0)
    root = math.sqrt(discriminant)
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def run(inputs):
    equation = str(inputs.get("equation") or "").strip().replace("^", "**")
    variable = str(inputs.get("variable") or "x").strip() or "x"
    if not equation or "=" not in equation:
        raise Exception("Missing equation")
    left, right = equation.split("=", 1)
    coeffs = _combine(_parse_polynomial_side(left, variable), _parse_polynomial_side(right, variable))
    degree = max(coeffs) if coeffs else 0
    if degree <= 1:
        solutions = _solve_linear(coeffs)
    elif degree == 2:
        solutions = _solve_quadratic(coeffs)
    else:
        raise ValueError("Only linear and quadratic equations are supported")
    normalized = []
    seen = set()
    for item in solutions:
        rounded = round(float(item), 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        normalized.append(rounded)
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "equation": equation, "variable": variable, "solutions": normalized, "solution_count": len(normalized)}}}}
"""
    if kind == "answer-verifier":
        return f"""import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}
NUM_RE = re.compile(r"[-+]?(?:(?:\\d+(?:\\.\\d+)?)|(?:\\.\\d+))(?:/[-+]?(?:(?:\\d+(?:\\.\\d+)?)|(?:\\.\\d+)))?")


def _parse_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.startswith("####"):
        text = text[4:].strip()
    nums = NUM_RE.findall(text)
    if not nums:
        raise Exception("No numeric token found")
    token = nums[-1]
    if "/" in token:
        left, right = token.split("/", 1)
        return float(left) / float(right)
    return float(token)


def run(inputs):
    candidate = inputs.get("candidate")
    expected = inputs.get("expected")
    if candidate is None:
        raise Exception("Missing candidate")
    candidate_value = _parse_value(candidate)
    result = {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "candidate_value": candidate_value}}
    if expected is not None:
        expected_value = _parse_value(expected)
        result["expected_value"] = expected_value
        result["matches_expected"] = abs(candidate_value - expected_value) <= float(inputs.get("tolerance") or 1e-6)
    return {{"output": result}}
"""
    if kind == "answer-formatter" or kind == "canonicalizer":
        return f"""import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}
BOX_RE = re.compile(r"\\\\boxed\\{{([^{{}}]+)\\}}")


def _canonical(text):
    value = str(text).strip()
    boxes = BOX_RE.findall(value)
    if boxes:
        value = boxes[-1].strip()
    if value.startswith("####"):
        value = value[4:].strip()
    return value.replace("$", "").strip()


def run(inputs):
    answer = inputs.get("answer")
    if answer is None:
        answer = inputs.get("candidate")
    if answer is None:
        answer = inputs.get("value")
    if answer is None:
        raise Exception("Missing answer")
    canonical = _canonical(answer)
    formatted = "#### " + canonical if DATASET == "gsm8k" else canonical
    return {{"output": {{"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "canonical_answer": canonical, "formatted_answer": formatted}}}}
"""
    raise ValueError(f"Unsupported math tool kind: {kind}")


def _hotpotqa_tool_code(dataset: str, focus_title: str, keywords: List[str], primary_role: str, kind: str) -> str:
    keywords_json = _json_dumps(keywords)
    common = f"""import re

DATASET = {dataset!r}
FOCUS = {focus_title!r}
PRIMARY_ROLE = {primary_role!r}
KEYWORDS = {keywords_json}
ARTICLES_RE = re.compile(r"\\b(a|an|the)\\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.-]*")


def _question(inputs):
    return str(inputs.get("question") or inputs.get("query") or inputs.get("task") or "").strip()


def _normalize(text):
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\\s]", " ", value)
    value = ARTICLES_RE.sub(" ", value)
    return re.sub(r"\\s+", " ", value).strip()


def _tokens(text):
    stop = {{"what", "which", "who", "where", "when", "why", "how", "does", "did", "was", "were", "is", "are", "the", "a", "an", "of", "in", "on", "to", "for", "with", "and", "or", "this", "that", "whose", "has", "have", "had", "from", "kind", "type"}}
    return [token.lower() for token in TOKEN_RE.findall(str(text or "")) if token.lower() not in stop and len(token) > 2]


def _entities(question):
    entities = []
    for match in re.finditer(r'"([^"]+)"', question):
        entities.append(match.group(1).strip())
    for match in re.finditer(r"\\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\\s+(?:of|the|and|[A-Z][A-Za-z0-9&.'-]*))*", question):
        text = match.group(0).strip()
        if len(text) > 1 and text.lower() not in {{"What", "Which", "Who", "Where", "When", "How"}}:
            entities.append(text)
    deduped = []
    seen = set()
    for item in entities:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _iter_context(context):
    rows = []
    for item in context or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0])
        sentences = item[1] if isinstance(item[1], list) else []
        for idx, sentence in enumerate(sentences):
            rows.append({{"title": title, "sentence_index": idx, "sentence": str(sentence)}})
    return rows


def _score_sentence(question, row):
    q_tokens = set(_tokens(question))
    text = (row.get("title", "") + " " + row.get("sentence", "")).lower()
    score = sum(1 for token in q_tokens if token in text)
    for keyword in KEYWORDS:
        if keyword.lower() in text:
            score += 2
    return score


def _top_evidence(question, context, limit=6):
    rows = _iter_context(context)
    scored = []
    for row in rows:
        score = _score_sentence(question, row)
        scored.append({{**row, "score": score}})
    scored.sort(key=lambda item: (-item.get("score", 0), item.get("title", ""), item.get("sentence_index", 0)))
    return scored[:limit]


def _clean_candidate(text):
    value = str(text or "").strip()
    value = re.sub(r"^known as\\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^also known as\\s+", "", value, flags=re.IGNORECASE)
    value = value.strip(" .;,:")
    if len(value) > 160:
        value = value[:160].rsplit(" ", 1)[0].strip()
    return value


def _extract_answer(question, evidence_points):
    q = question.lower()
    sentences = [str(item.get("sentence") or "") for item in evidence_points if isinstance(item, dict)]
    titles = [str(item.get("title") or "") for item in evidence_points if isinstance(item, dict)]
    joined = " ".join(sentences)
    if "what does" in q and "stand for" in q:
        for sentence in sentences:
            match = re.search(r"\\b([A-Z][A-Za-z]*(?:\\s+mit\\s+[a-z]+|\\s+[a-zA-Z]+){{1,6}})\\s*\\([^)]*abbreviated\\s+[A-Z]{{2,}}", sentence)
            if match:
                return _clean_candidate(match.group(1))
            match = re.search(r"\\b([A-Z][A-Za-z]+(?:\\s+[a-zA-Z]+){{1,8}})\\s*\\(", sentence)
            if match:
                return _clean_candidate(match.group(1))
    if "which of" in q and ("more" in q or "fewer" in q):
        name_scores = []
        for title, sentence in zip(titles, sentences):
            count = len(re.findall(r",| and |\\bfor the bands?\\b|\\bprojects?\\b", sentence, re.IGNORECASE))
            name_scores.append((count, title))
        if name_scores:
            name_scores.sort(reverse=("more" in q))
            return _clean_candidate(name_scores[0][1])
    if "who" in q or q.startswith("which founding member") or "owner" in q:
        for title in titles:
            if title and _normalize(title) not in _normalize(question):
                return _clean_candidate(title)
    if "where" in q or "from" in q:
        for sentence in sentences:
            match = re.search(r"\\bfrom\\s+([A-Z][A-Za-z .'-]+(?:,\\s*[A-Z][A-Za-z .'-]+)?)", sentence)
            if match:
                return _clean_candidate(match.group(1))
            match = re.search(r"\\blocated in\\s+([A-Z][A-Za-z .'-]+(?:,\\s*[A-Z][A-Za-z .'-]+)?)", sentence)
            if match:
                return _clean_candidate(match.group(1))
    if "what kind" in q or "what type" in q:
        for sentence in sentences:
            match = re.search(r"\\bis\\s+(?:the\\s+)?(?:largest\\s+independent\\s+public\\s+listed\\s+)?([^.;]+?(?:company|group|association|entity|network|corporation))\\b", sentence, re.IGNORECASE)
            if match:
                return _clean_candidate(match.group(1))
    for sentence in sentences:
        match = re.search(r"\\bis\\s+(?:a|an|the)?\\s*([^.;]+)", sentence, re.IGNORECASE)
        if match:
            candidate = _clean_candidate(match.group(1))
            if candidate and len(candidate.split()) <= 12:
                return candidate
    return _clean_candidate(titles[-1] if titles else "")
"""
    if kind == "question-decomposer":
        return common + """

def run(inputs):
    question = _question(inputs)
    lower = question.lower()
    q_type = "comparison" if any(token in lower for token in ["which of", "more", "less", "fewer", "older", "larger"]) else "bridge"
    if lower.startswith("where") or " where " in lower or " from" in lower:
        answer_type = "location"
    elif lower.startswith("who") or " which founding member" in lower:
        answer_type = "person"
    elif "stand for" in lower or "acronym" in lower:
        answer_type = "expanded-name"
    elif "what kind" in lower or "what type" in lower:
        answer_type = "entity-type"
    elif "when" in lower or "date" in lower:
        answer_type = "date"
    else:
        answer_type = "entity"
    cue_hits = [word for word in KEYWORDS if word.lower() in lower]
    entities = _entities(question)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "question_type": q_type, "answer_type_hint": answer_type, "entities": entities, "cue_hits": cue_hits, "reasoning_trace": [f"{q_type} question with answer type hint {answer_type}"], "diagnostics": {"question_decomposition": True, "entity_count": len(entities)}}}
"""
    if kind == "context-indexer":
        return common + """

def run(inputs):
    question = _question(inputs)
    context = inputs.get("context") or []
    evidence = _top_evidence(question, context, limit=int(inputs.get("limit") or 8))
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "evidence_points": evidence, "diagnostics": {"context_title_count": len(context), "evidence_count": len(evidence)}}}
"""
    if kind == "evidence-chain-builder":
        return common + """

def run(inputs):
    question = _question(inputs)
    evidence = inputs.get("evidence_points")
    if not isinstance(evidence, list) or not evidence:
        evidence = _top_evidence(question, inputs.get("context") or [], limit=6)
    chain = evidence[:4]
    candidate = _extract_answer(question, chain)
    trace = []
    for item in chain:
        trace.append(f"{item.get('title')}[{item.get('sentence_index')}]: {item.get('sentence')}")
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "evidence_points": chain, "reasoning_trace": trace, "candidate_answer": candidate}}
"""
    if kind == "answer-extractor":
        return common + """

def run(inputs):
    question = _question(inputs)
    evidence = inputs.get("evidence_points") or []
    if not isinstance(evidence, list) or not evidence:
        evidence = _top_evidence(question, inputs.get("context") or [], limit=6)
    candidate = _extract_answer(question, evidence)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "candidate_answer": candidate, "final_answer": candidate, "normalized_answer": _normalize(candidate)}}
"""
    if kind == "support-verifier":
        return common + """

def run(inputs):
    candidate = inputs.get("candidate_answer")
    if candidate is None:
        candidate = inputs.get("candidate")
    if candidate is None:
        candidate = inputs.get("final_answer")
    candidate = str(candidate or "").strip()
    evidence = inputs.get("evidence_points") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence_text = " ".join(str(item.get("sentence") or "") for item in evidence if isinstance(item, dict))
    evidence_titles = set(_normalize(item.get("title")) for item in evidence if isinstance(item, dict))
    coverage = 1.0 if evidence_titles else 0.0
    normalized_candidate = _normalize(candidate)
    answer_in_evidence = bool(normalized_candidate and normalized_candidate in _normalize(evidence_text))
    verification = {"answer_in_evidence": answer_in_evidence, "supporting_fact_coverage": coverage}
    final_answer = candidate
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "candidate_answer": candidate, "formatted_answer": final_answer, "final_answer": final_answer, "normalized_answer": normalized_candidate, "verification": verification, "answer_in_evidence": answer_in_evidence, "supporting_fact_coverage": coverage}}
"""
    raise ValueError(f"Unsupported HotpotQA tool kind: {kind}")


def _kind_inputs(dataset: str, kind: str) -> List[str]:
    mapping = {
        "signature-extractor": ["prompt", "entry_point", "test", "test_list"],
        "code-normalizer": ["code", "query", "entry_point"],
        "assert-parser": ["prompt", "test", "test_list"],
        "import-injector": ["code"],
        "structure-checker": ["code", "entry_point"],
        "contract-gate": ["code", "entry_point", "test_list"],
        "public-test-runner": ["code", "entry_point", "test", "test_list", "timeout_s"],
        "assert-scorer": ["code", "entry_point", "test", "test_list", "timeout_s"],
        "traceback-localizer": ["code", "traceback", "error"],
        "failure-diagnoser": ["code", "traceback", "error"],
        "signal-profiler": ["question", "problem", "query", "task"],
        "quantity-extractor": ["question", "problem", "query", "task"],
        "relation-calculator": ["expression", "total", "unit_price", "distance", "speed"],
        "equation-engine": ["equation", "variable"],
        "answer-verifier": ["candidate", "expected", "tolerance"],
        "answer-formatter": ["answer", "candidate", "value"],
        "canonicalizer": ["answer", "candidate", "value"],
        "question-decomposer": ["question", "query", "context"],
        "context-indexer": ["question", "context"],
        "evidence-chain-builder": ["question", "context", "evidence_points"],
        "answer-extractor": ["question", "evidence_points", "reasoning_trace"],
        "support-verifier": ["candidate_answer", "evidence_points"],
    }
    return list(mapping[kind])


def _make_tool_code(dataset: str, focus_title: str, keywords: List[str], primary_role: str, kind: str) -> str:
    if dataset in {"humaneval", "mbpp"}:
        return _code_tool_code(dataset, focus_title, keywords, primary_role, kind)
    if dataset == "hotpotqa":
        return _hotpotqa_tool_code(dataset, focus_title, keywords, primary_role, kind)
    return _math_tool_code(dataset, focus_title, keywords, primary_role, kind)


def _focus_specific_domain_tags(dataset: str, focus_slug: str) -> List[str]:
    return [focus_slug, f"{dataset}-{focus_slug}", "specialized-agent", "real-tools"]


def _agent_tool_ids(dataset: str, focus_slug: str, primary_role: str, tool_kinds: List[Any]) -> List[str]:
    return [f"{dataset}-{focus_slug}-{primary_role}-{kind_slug}" for kind_slug, _ in tool_kinds]


def build_real_dataset_tool_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for dataset, config in DATASET_CONFIG.items():
        dataset_keywords = _dataset_keywords(dataset)
        for focus in config["focuses"]:
            focus_slug = str(focus["slug"])
            focus_title = str(focus["title"])
            keywords = list(dict.fromkeys(list(focus["keywords"]) + dataset_keywords))
            for primary_role in DATASET_ROLE_MAP[dataset]:
                role_tags = list(dict.fromkeys(config["role_tags"][primary_role] + [primary_role]))
                for kind_slug, kind_title in config["tool_kinds"]:
                    tool_id = f"{dataset}-{focus_slug}-{primary_role}-{kind_slug}"
                    if tool_id in seen_ids:
                        raise ValueError(f"Duplicate tool id: {tool_id}")
                    seen_ids.add(tool_id)
                    name = f"{config['display']}{_title_case(focus_slug)}{primary_role.title()}{_title_case(kind_slug)}RealTool"
                    description = (
                        f"Deterministic local {config['display']} tool for {focus_title.lower()} tasks. "
                        f"Primary role={primary_role}. Capability={kind_title.lower()}."
                    )
                    tool_tags = list(
                        dict.fromkeys(
                            [
                                dataset,
                                focus_slug,
                                f"{dataset}-{focus_slug}",
                                primary_role,
                                kind_slug,
                                "real-tool",
                            ]
                            + keywords[:4]
                        )
                    )
                    capability_meta = _capability_metadata(dataset, kind_slug)
                    records.append(
                        _tool_record(
                            tool_id=tool_id,
                            name=name,
                            dataset=dataset,
                            description=description,
                            code=_make_tool_code(dataset, focus_title, keywords, primary_role, kind_slug),
                            domain_tags=list(
                                dict.fromkeys(list(config["domain_tags"]) + _focus_specific_domain_tags(dataset, focus_slug))
                            ),
                            role_tags=role_tags,
                            tool_tags=tool_tags,
                            inputs=_kind_inputs(dataset, kind_slug),
                            dataset_family=dataset,
                            focus_slug=focus_slug,
                            primary_role=primary_role,
                            capability_kind=capability_meta["capability_kind"],
                            stage=capability_meta["stage"],
                            consumes=capability_meta["consumes"],
                            produces=capability_meta["produces"],
                        )
                    )
    return records


def build_real_dataset_agent_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for dataset, config in DATASET_CONFIG.items():
        for focus in config["focuses"]:
            focus_slug = str(focus["slug"])
            focus_title = str(focus["title"])
            for primary_role in DATASET_ROLE_MAP[dataset]:
                agent_id = f"agent-real-{dataset}-{focus_slug}-{primary_role}"
                if agent_id in seen_ids:
                    raise ValueError(f"Duplicate agent id: {agent_id}")
                seen_ids.add(agent_id)
                tool_ids = _agent_tool_ids(dataset, focus_slug, primary_role, config["tool_kinds"])
                name = f"{config['display']}{_title_case(focus_slug)}{primary_role.title()}-RealTools"
                description = (
                    f"{config['display']} specialized {primary_role} agent focused on {focus_title.lower()}. "
                    f"Uses 5 dedicated deterministic local tools."
                )
                domain_tags = list(dict.fromkeys(list(config["domain_tags"]) + _focus_specific_domain_tags(dataset, focus_slug)))
                role_tags = list(dict.fromkeys(config["role_tags"][primary_role] + [primary_role]))
                embedding_text = (
                    f"{config['display']} specialized agent for {focus_title.lower()}. "
                    f"Primary role {primary_role}. Tools: {' '.join(tool_ids)}. "
                    f"Keywords: {' '.join(focus['keywords'])}."
                )
                records.append(
                    {
                        "id": agent_id,
                        "name": name,
                        "description": description,
                        "domain_tags": domain_tags,
                        "role_tags": role_tags,
                        "tool_ids": tool_ids,
                        "embedding_text": embedding_text,
                        "dataset_family": dataset,
                        "focus_slug": focus_slug,
                        "primary_role": primary_role,
                    }
                )
    return records


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
                "dataset_family": item["spec"]["dataset_family"],
                "focus_slug": item["spec"]["focus_slug"],
                "primary_role": item["spec"]["primary_role"],
                "capability_kind": item["spec"]["capability_kind"],
                "stage": item["spec"]["stage"],
                "consumes": list(item["spec"]["consumes"]),
                "produces": list(item["spec"]["produces"]),
                "file_path": f"generated_tools_real/{item['id']}.py",
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
                "dataset_family": item.get("dataset_family", ""),
                "focus_slug": item.get("focus_slug", ""),
                "primary_role": item.get("primary_role", ""),
            }
            for item in agents
        ],
    }
