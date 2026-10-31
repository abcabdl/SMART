from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure `core`, `generation`, etc. are importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cards import AgentCard
from core.registry import SQLiteRegistry
from generation.functional_tool_catalog import code_from_record, select_functional_tools, spec_from_record
from generation.tool_generator import register_tool_from_spec
from generation.toolbench_catalog import endpoint_to_code, endpoint_to_spec, select_toolbench_endpoints
from retrieval.build_index import build_index


DEFAULT_DOMAINS = [
    "GSM8K Math Reasoning",
    "Competition Math",
    "HumanEval Code Generation",
    "MBPP Code Tasks",
]

DEFAULT_ROLES = ["planner", "builder", "checker", "refactor"]


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned or "item"


def _backup_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup = f"{path}.bak.{stamp}"
    os.replace(path, backup)
    return backup


def _assign_tool_ids_low_overlap(tool_ids_all: list[str], agents_per_theme: int, tools_per_agent: int) -> list[list[str]]:
    if len(tool_ids_all) < tools_per_agent:
        raise ValueError(f"Need at least {tools_per_agent} tools, have {len(tool_ids_all)}")
    assignments: list[list[str]] = []
    total = len(tool_ids_all)
    stride = max(1, tools_per_agent - 1)
    for idx in range(agents_per_theme):
        start = (idx * stride) % total
        chosen: list[str] = []
        offset = 0
        while len(chosen) < tools_per_agent and offset < total * 2:
            tool_id = tool_ids_all[(start + offset) % total]
            if tool_id not in chosen:
                chosen.append(tool_id)
            offset += 1
        assignments.append(chosen)
    return assignments


def _write_code_plan_tool() -> str:
    return (
        "def run(inputs):\n"
        "    task = inputs.get(\"task\") or inputs.get(\"query\") or \"\"\n"
        "    if not isinstance(task, str):\n"
        "        task = str(task)\n"
        "    task = task.strip()\n"
        "    steps = []\n"
        "    if task:\n"
        "        steps = [\n"
        "            f\"Clarify requirements for: {task}\",\n"
        "            \"Identify inputs/outputs and constraints\",\n"
        "            \"Break into milestones\",\n"
        "            \"Sequence tasks with dependencies\",\n"
        "        ]\n"
        "    return {\"output\": {\"steps\": steps}}\n"
    )


def _write_decompose_tool() -> str:
    return (
        "def run(inputs):\n"
        "    text = inputs.get(\"task\") or inputs.get(\"query\") or \"\"\n"
        "    if not isinstance(text, str):\n"
        "        text = str(text)\n"
        "    items = [seg.strip() for seg in text.replace(\";\", \".\").split(\".\") if seg.strip()]\n"
        "    subtasks = items[:8]\n"
        "    return {\"output\": {\"subtasks\": subtasks}}\n"
    )


def _write_estimate_tool() -> str:
    return (
        "def run(inputs):\n"
        "    steps = inputs.get(\"steps\") or []\n"
        "    if not isinstance(steps, list):\n"
        "        steps = [str(steps)]\n"
        "    estimate = max(1, min(8, len(steps)))\n"
        "    return {\"output\": {\"estimated_hours\": estimate}}\n"
    )


def _write_function_body_tool() -> str:
    return (
        "import json\n"
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments. \"\n"
        "        \"If asked for a function body, do NOT include def/class/import/main/docstrings. \"\n"
        "        \"Empty/whitespace output is invalid.\"\n"
        "    )\n"
        "    user = (\n"
        "        f\"Task: {prompt}. \"\n"
        "        \"Output format: code. \"\n"
        "        \"Return ONLY the function body using the original parameter names. \"\n"
        "        \"No labels, fences, or extra text.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": user}],\n"
        "        \"temperature\": 0.2,\n"
        "        \"max_tokens\": 800,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def _indent(lines):\n"
        "    return \"\\n\".join(\"    \" + line for line in lines)\n"
        "\n"
        "def run(inputs):\n"
        "    logic = inputs.get(\"logic\") or inputs.get(\"steps\") or []\n"
        "    if isinstance(logic, str):\n"
        "        logic_lines = [line.strip() for line in logic.splitlines() if line.strip()]\n"
        "    else:\n"
        "        logic_lines = [str(item).strip() for item in logic if str(item).strip()]\n"
        "    if not logic_lines:\n"
        "        logic_lines = [\"return None\"]\n"
        "    prompt = \"Generate a Python function body that implements: \" + \" \".join(logic_lines)\n"
        "    body = _call_llm(prompt)\n"
        "    if not body:\n"
        "        body = _indent(logic_lines)\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _write_module_skeleton_tool() -> str:
    return (
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments.\"\n"
        "    )\n"
        "    user = (\n"
        "        f\"Task: {prompt}. \"\n"
        "        \"Output format: code. \"\n"
        "        \"Return ONLY the code module text. No labels, fences, or extra text.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": user}],\n"
        "        \"temperature\": 0.2,\n"
        "        \"max_tokens\": 800,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def run(inputs):\n"
        "    name = inputs.get(\"function_name\") or inputs.get(\"name\") or \"handler\"\n"
        "    if not isinstance(name, str) or not name.strip():\n"
        "        name = \"handler\"\n"
        "    params = inputs.get(\"parameters\") or inputs.get(\"params\") or []\n"
        "    if isinstance(params, str):\n"
        "        params_list = [p.strip() for p in params.split(\",\") if p.strip()]\n"
        "    else:\n"
        "        params_list = [str(p).strip() for p in params if str(p).strip()]\n"
        "    sig = \", \".join(params_list) if params_list else \"\" \n"
        "    prompt = f\"Write a Python function def {name}({sig}): with a minimal body.\"\n"
        "    code = _call_llm(prompt)\n"
        "    if not code:\n"
        "        lines = [f\"def {name}({sig}):\", \"    return None\", \"\"]\n"
        "        code = \"\\n\".join(lines)\n"
        "    return {\"code_or_commands\": code}\n"
    )


def _write_glue_tool() -> str:
    return (
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments.\"\n"
        "    )\n"
        "    user = (\n"
        "        f\"Task: {prompt}. \"\n"
        "        \"Output format: code. \"\n"
        "        \"Return ONLY the combined code. No labels, fences, or extra text.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": user}],\n"
        "        \"temperature\": 0.2,\n"
        "        \"max_tokens\": 800,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def run(inputs):\n"
        "    parts = inputs.get(\"functions\") or inputs.get(\"snippets\") or []\n"
        "    if isinstance(parts, str):\n"
        "        prompt = \"Combine these snippets into a single module:\\n\" + parts\n"
        "    else:\n"
        "        cleaned = [str(p).rstrip() for p in parts if str(p).strip()]\n"
        "        prompt = \"Combine these snippets into a single module:\\n\" + \"\\n\\n\".join(cleaned)\n"
        "    code = _call_llm(prompt)\n"
        "    if not code:\n"
        "        if isinstance(parts, str):\n"
        "            code = parts.strip()\n"
        "        else:\n"
        "            code = \"\\n\\n\".join(cleaned)\n"
        "    return {\"code_or_commands\": code}\n"
    )


def _write_tests_tool() -> str:
    return (
        "def run(inputs):\n"
        "    func = inputs.get(\"function_name\") or \"solution\"\n"
        "    cases = inputs.get(\"cases\") or []\n"
        "    if not isinstance(cases, list):\n"
        "        cases = [cases]\n"
        "    lines = []\n"
        "    for idx, case in enumerate(cases, start=1):\n"
        "        lines.append(f\"def test_{func}_{idx}():\")\n"
        "        lines.append(\"    assert callable(\" + func + \")\")\n"
        "        if case:\n"
        "            lines.append(\"    # case: \" + str(case).replace(\"\\n\", \" \"))\n"
        "        lines.append(\"\")\n"
        "    if not lines:\n"
        "        lines = [f\"def test_{func}_smoke():\", \"    assert callable(\" + func + \")\"]\n"
        "    return {\"code_or_commands\": \"\\n\".join(lines)}\n"
    )


def _write_edge_tests_tool() -> str:
    return (
        "def run(inputs):\n"
        "    func = inputs.get(\"function_name\") or \"solution\"\n"
        "    lines = [\n"
        "        f\"def test_{func}_edge_empty():\",\n"
        "        \"    assert callable(\" + func + \")\",\n"
        "        \"\",\n"
        "        f\"def test_{func}_edge_null():\",\n"
        "        \"    assert callable(\" + func + \")\",\n"
        "    ]\n"
        "    return {\"code_or_commands\": \"\\n\".join(lines)}\n"
    )


def _write_coverage_tool() -> str:
    return (
        "def run(inputs):\n"
        "    notes = [\"Check branches\", \"Check error paths\", \"Check boundary values\"]\n"
        "    return {\"output\": {\"coverage_notes\": notes}}\n"
    )


def _write_rename_tool() -> str:
    return (
        "def run(inputs):\n"
        "    code = inputs.get(\"code\") or inputs.get(\"text\") or \"\"\n"
        "    old = inputs.get(\"old_name\") or \"\"\n"
        "    new = inputs.get(\"new_name\") or \"\"\n"
        "    if not isinstance(code, str):\n"
        "        code = str(code)\n"
        "    if old and new:\n"
        "        code = code.replace(old, new)\n"
        "    return {\"code_or_commands\": code}\n"
    )


def _write_extract_function_tool() -> str:
    return (
        "def run(inputs):\n"
        "    body = inputs.get(\"body\") or \"return None\"\n"
        "    name = inputs.get(\"function_name\") or \"helper\"\n"
        "    params = inputs.get(\"parameters\") or []\n"
        "    if isinstance(params, str):\n"
        "        params = [p.strip() for p in params.split(\",\") if p.strip()]\n"
        "    params_text = \", \".join(params)\n"
        "    lines = [f\"def {name}({params_text}):\"]\n"
        "    for line in str(body).splitlines():\n"
        "        lines.append(\"    \" + line)\n"
        "    return {\"code_or_commands\": \"\\n\".join(lines)}\n"
    )


def _write_simplify_tool() -> str:
    return (
        "def run(inputs):\n"
        "    code = inputs.get(\"code\") or \"\"\n"
        "    if not isinstance(code, str):\n"
        "        code = str(code)\n"
        "    simplified = code.replace(\"== True\", \"\").replace(\"== False\", \" is False\")\n"
        "    return {\"code_or_commands\": simplified}\n"
    )


def _planner_modes() -> list[str]:
    return [
        "milestone",
        "dependency",
        "risk",
        "timebox",
        "architecture",
        "testing",
        "refactor",
        "performance",
        "security",
        "documentation",
    ]


def _testing_modes() -> list[str]:
    return [
        "unit",
        "edge",
        "table",
        "property",
        "fuzz",
        "regression",
        "integration",
        "mutation",
        "boundary",
        "negative",
    ]


def _refactor_modes() -> list[str]:
    return [
        "rename",
        "extract",
        "simplify",
        "normalize",
        "dedupe",
        "cleanup",
        "format",
        "optimize",
        "decompose",
        "guard",
    ]


def _generation_modes() -> list[str]:
    return [
        "algorithm",
        "data-structure",
        "string",
        "graph",
        "dp",
        "math",
        "parsing",
        "greedy",
        "recursion",
        "io",
        "robustness",
        "edge-case",
    ]


def _write_planner_tool(mode: str) -> str:
    return (
        "def run(inputs):\n"
        "    task = inputs.get(\"task\") or inputs.get(\"query\") or \"\"\n"
        "    scope = inputs.get(\"scope\") or \"general\"\n"
        "    constraints = inputs.get(\"constraints\") or []\n"
        "    if not isinstance(task, str):\n"
        "        task = str(task)\n"
        "    if isinstance(constraints, str):\n"
        "        constraints = [c.strip() for c in constraints.split(\",\") if c.strip()]\n"
        "    steps = []\n"
        f"    mode = \"{mode}\"\n"
        "    if task.strip():\n"
        "        steps.append(f\"Define goal for {task} ({scope})\")\n"
        "        if mode == \"milestone\":\n"
        "            steps.extend([\"Draft milestones\", \"Assign owners\", \"Set review checkpoints\"])\n"
        "        elif mode == \"dependency\":\n"
        "            steps.extend([\"List dependencies\", \"Order by blocking risk\", \"Create dependency graph\"])\n"
        "        elif mode == \"risk\":\n"
        "            steps.extend([\"List risks\", \"Add mitigations\", \"Add fallback plan\"])\n"
        "        elif mode == \"timebox\":\n"
        "            steps.extend([\"Set timebox\", \"Prioritize essentials\", \"Defer stretch goals\"])\n"
        "        elif mode == \"architecture\":\n"
        "            steps.extend([\"Define modules\", \"Define interfaces\", \"Map data flow\"])\n"
        "        elif mode == \"testing\":\n"
        "            steps.extend([\"Define test plan\", \"Identify edge cases\", \"Set CI gates\"])\n"
        "        elif mode == \"refactor\":\n"
        "            steps.extend([\"Identify hotspots\", \"Plan safe refactors\", \"Add regression tests\"])\n"
        "        elif mode == \"performance\":\n"
        "            steps.extend([\"Set perf budget\", \"Add profiling points\", \"Plan optimizations\"])\n"
        "        elif mode == \"security\":\n"
        "            steps.extend([\"Threat model\", \"Add validation\", \"Audit sensitive paths\"])\n"
        "        else:\n"
        "            steps.extend([\"Draft plan\", \"Review constraints\", \"Finalize checklist\"])\n"
        "    return {\"output\": {\"mode\": mode, \"steps\": steps, \"constraints\": constraints}}\n"
    )


def _write_testing_tool(mode: str) -> str:
    return (
        "def run(inputs):\n"
        "    func = inputs.get(\"function_name\") or \"solution\"\n"
        "    cases = inputs.get(\"cases\") or []\n"
        "    if not isinstance(cases, list):\n"
        "        cases = [cases]\n"
        f"    mode = \"{mode}\"\n"
        "    lines = []\n"
        "    if mode in {\"unit\", \"table\", \"regression\"}:\n"
        "        for idx, case in enumerate(cases or [\"smoke\"], start=1):\n"
        "            lines.append(f\"def test_{func}_{idx}():\")\n"
        "            lines.append(\"    assert callable(\" + func + \")\")\n"
        "            lines.append(\"    # case: \" + str(case).replace(\"\\n\", \" \"))\n"
        "            lines.append(\"\")\n"
        "    elif mode == \"edge\":\n"
        "        lines.extend([\n"
        "            f\"def test_{func}_edge_empty():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "            \"\",\n"
        "            f\"def test_{func}_edge_null():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "        ])\n"
        "    elif mode == \"property\":\n"
        "        lines.extend([\n"
        "            f\"def test_{func}_property_idempotent():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "            \"\",\n"
        "            f\"def test_{func}_property_deterministic():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "        ])\n"
        "    elif mode == \"fuzz\":\n"
        "        lines.extend([\n"
        "            f\"def test_{func}_fuzz_inputs():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "            \"    # TODO: add randomized inputs\",\n"
        "        ])\n"
        "    elif mode == \"integration\":\n"
        "        lines.extend([\n"
        "            f\"def test_{func}_integration_smoke():\",\n"
        "            \"    assert callable(\" + func + \")\",\n"
        "        ])\n"
        "    else:\n"
        "        lines.append(f\"def test_{func}_basic():\")\n"
        "        lines.append(\"    assert callable(\" + func + \")\")\n"
        "    return {\"code_or_commands\": \"\\n\".join(lines)}\n"
    )


def _write_refactor_tool(mode: str) -> str:
    return (
        "def run(inputs):\n"
        "    code = inputs.get(\"code\") or inputs.get(\"text\") or \"\"\n"
        "    if not isinstance(code, str):\n"
        "        code = str(code)\n"
        f"    mode = \"{mode}\"\n"
        "    if mode == \"rename\":\n"
        "        old = inputs.get(\"old_name\") or \"\"\n"
        "        new = inputs.get(\"new_name\") or \"\"\n"
        "        if old and new:\n"
        "            code = code.replace(old, new)\n"
        "    elif mode == \"extract\":\n"
        "        body = inputs.get(\"body\") or \"return None\"\n"
        "        name = inputs.get(\"function_name\") or \"helper\"\n"
        "        params = inputs.get(\"parameters\") or []\n"
        "        if isinstance(params, str):\n"
        "            params = [p.strip() for p in params.split(\",\") if p.strip()]\n"
        "        params_text = \", \".join(params)\n"
        "        lines = [f\"def {name}({params_text}):\"]\n"
        "        for line in str(body).splitlines():\n"
        "            lines.append(\"    \" + line)\n"
        "        code = \"\\n\".join(lines)\n"
        "    elif mode == \"simplify\":\n"
        "        code = code.replace(\"== True\", \"\").replace(\"== False\", \" is False\")\n"
        "    elif mode == \"normalize\":\n"
        "        code = \"\\n\".join(line.rstrip() for line in code.splitlines())\n"
        "    elif mode == \"dedupe\":\n"
        "        seen = set()\n"
        "        lines = []\n"
        "        for line in code.splitlines():\n"
        "            if line not in seen:\n"
        "                seen.add(line)\n"
        "                lines.append(line)\n"
        "        code = \"\\n\".join(lines)\n"
        "    elif mode == \"cleanup\":\n"
        "        code = code.replace(\"\\t\", \"    \")\n"
        "    elif mode == \"format\":\n"
        "        code = \"\\n\".join(line.strip() for line in code.splitlines())\n"
        "    elif mode == \"optimize\":\n"
        "        code = code.replace(\"for i in range(len(\", \"for i, _ in enumerate(\")\n"
        "    elif mode == \"decompose\":\n"
        "        code = code.replace(\";\", \"\\n\")\n"
        "    elif mode == \"guard\":\n"
        "        guard = inputs.get(\"guard\") or \"if value is None: return None\"\n"
        "        code = guard + \"\\n\" + code\n"
        "    return {\"code_or_commands\": code}\n"
    )


def _write_codegen_tool(mode: str, variant: int) -> str:
    if variant % 5 == 0:
        return _write_codegen_chain_body(mode)
    if variant % 5 == 1:
        return _write_codegen_spec_to_body(mode)
    if variant % 5 == 2:
        return _write_codegen_io_first(mode)
    if variant % 5 == 3:
        return _write_codegen_validate_then_fix(mode)
    return _write_codegen_outline_then_body(mode)


def _write_codegen_chain_body(mode: str) -> str:
    return (
        "import json\n"
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt, response_format=None, max_tokens=800):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments. \"\n"
        "        \"If asked for a function body, do NOT include def/class/import/main/docstrings. \"\n"
        "        \"Empty/whitespace output is invalid.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": prompt}],\n"
        "        \"temperature\": 0.2,\n"
        "        \"max_tokens\": max_tokens,\n"
        "    }\n"
        "    if response_format:\n"
        "        payload[\"response_format\"] = response_format\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return text\n"
        "\n"
        "def _normalize_body(text):\n"
        "    body = _extract_code(text)\n"
        "    return body.strip()\n"
        "\n"
        "def run(inputs):\n"
        "    logic = inputs.get(\"logic\") or inputs.get(\"steps\") or []\n"
        "    constraints = inputs.get(\"constraints\") or []\n"
        "    if isinstance(logic, str):\n"
        "        logic_lines = [line.strip() for line in logic.splitlines() if line.strip()]\n"
        "    else:\n"
        "        logic_lines = [str(item).strip() for item in logic if str(item).strip()]\n"
        "    if isinstance(constraints, str):\n"
        "        constraints = [c.strip() for c in constraints.split(\",\") if c.strip()]\n"
        f"    mode = \"{mode}\"\n"
        "    outline_prompt = (\n"
        "        \"Task: Draft a 5-step outline for implementing a Python function body. \"\n"
        "        \"Return ONLY a JSON object with key steps (list of strings). \"\n"
        "        \"No extra keys.\"\n"
        "    )\n"
        "    outline_text = _call_llm(outline_prompt, response_format={\"type\": \"json_object\"}, max_tokens=300)\n"
        "    try:\n"
        "        outline = json.loads(outline_text).get(\"steps\") or []\n"
        "    except Exception:\n"
        "        outline = []\n"
        "    merged = \" \".join(logic_lines or [])\n"
        "    if outline:\n"
        "        merged = merged + \" | outline: \" + \"; \".join(str(s) for s in outline)\n"
        "    if constraints:\n"
        "        merged = merged + \" | constraints: \" + \", \".join(str(c) for c in constraints)\n"
        "    prompt = (\n"
        "        f\"Task: Generate a Python function body ({mode}) that implements: {merged}. \"\n"
        "        \"Output format: code. Return ONLY the function body using the original parameter names.\"\n"
        "    )\n"
        "    body_text = _call_llm(prompt)\n"
        "    body = _normalize_body(body_text)\n"
        "    if not body:\n"
        "        body = \"    return None\"\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _write_codegen_spec_to_body(mode: str) -> str:
    return (
        "import json\n"
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt, response_format=None, max_tokens=800):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments. \"\n"
        "        \"If asked for a function body, do NOT include def/class/import/main/docstrings. \"\n"
        "        \"Empty/whitespace output is invalid.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": prompt}],\n"
        "        \"temperature\": 0.15,\n"
        "        \"max_tokens\": max_tokens,\n"
        "    }\n"
        "    if response_format:\n"
        "        payload[\"response_format\"] = response_format\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return text\n"
        "\n"
        "def run(inputs):\n"
        "    task = inputs.get(\"task\") or inputs.get(\"query\") or \"\"\n"
        "    func = inputs.get(\"function_name\") or \"solution\"\n"
        "    params = inputs.get(\"parameters\") or []\n"
        "    if isinstance(params, str):\n"
        "        params = [p.strip() for p in params.split(\",\") if p.strip()]\n"
        f"    mode = \"{mode}\"\n"
        "    spec_prompt = (\n"
        "        f\"Task: Build a JSON spec for a Python function body ({mode}). \"\n"
        "        f\"Function name: {func}. Parameters: {params}. Task: {task}. \"\n"
        "        \"Return ONLY JSON with keys: approach, steps, invariants.\"\n"
        "    )\n"
        "    spec_text = _call_llm(spec_prompt, response_format={\"type\": \"json_object\"}, max_tokens=400)\n"
        "    try:\n"
        "        spec = json.loads(spec_text)\n"
        "    except Exception:\n"
        "        spec = {\"approach\": \"direct\", \"steps\": [], \"invariants\": []}\n"
        "    prompt = (\n"
        "        \"Task: Generate a Python function body based on this spec. \"\n"
        "        f\"Spec: {json.dumps(spec, ensure_ascii=True)}. \"\n"
        "        \"Output format: code. Return ONLY the function body.\"\n"
        "    )\n"
        "    body_text = _call_llm(prompt)\n"
        "    body = _extract_code(body_text)\n"
        "    if not body:\n"
        "        body = \"    return None\"\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _write_codegen_io_first(mode: str) -> str:
    return (
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": prompt}],\n"
        "        \"temperature\": 0.25,\n"
        "        \"max_tokens\": 900,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def run(inputs):\n"
        "    io_format = inputs.get(\"io_format\") or \"stdin/stdout\"\n"
        "    examples = inputs.get(\"examples\") or []\n"
        "    if isinstance(examples, str):\n"
        "        examples = [examples]\n"
        f"    mode = \"{mode}\"\n"
        "    prompt = (\n"
        "        f\"Task: Create a Python function body ({mode}) that adheres to IO format: {io_format}. \"\n"
        "        f\"Examples: {examples}. Output format: code. Return ONLY the function body.\"\n"
        "    )\n"
        "    body = _call_llm(prompt)\n"
        "    if not body:\n"
        "        body = \"    return None\"\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _write_codegen_validate_then_fix(mode: str) -> str:
    return (
        "import os\n"
        "import re\n"
        "import ast\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": prompt}],\n"
        "        \"temperature\": 0.2,\n"
        "        \"max_tokens\": 900,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def _is_valid(body):\n"
        "    if not body:\n"
        "        return False\n"
        "    try:\n"
        "        ast.parse(\"def _tmp():\\n\" + body)\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "def run(inputs):\n"
        "    logic = inputs.get(\"logic\") or inputs.get(\"steps\") or []\n"
        "    if isinstance(logic, str):\n"
        "        logic_lines = [line.strip() for line in logic.splitlines() if line.strip()]\n"
        "    else:\n"
        "        logic_lines = [str(item).strip() for item in logic if str(item).strip()]\n"
        f"    mode = \"{mode}\"\n"
        "    prompt = (\n"
        "        f\"Task: Generate a Python function body ({mode}) that implements: \"\n"
        "        + \" \".join(logic_lines)\n"
        "        + \". Output format: code. Return ONLY the function body.\"\n"
        "    )\n"
        "    body = _call_llm(prompt)\n"
        "    if not _is_valid(body):\n"
        "        fix_prompt = (\n"
        "            \"Task: Fix this function body to be valid Python. \"\n"
        "            \"Return ONLY the corrected function body. Body: \" + body\n"
        "        )\n"
        "        body = _call_llm(fix_prompt)\n"
        "    if not _is_valid(body):\n"
        "        body = \"    return None\"\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _write_codegen_outline_then_body(mode: str) -> str:
    return (
        "import os\n"
        "import re\n"
        "import httpx\n"
        "\n"
        "_CODE_BLOCK_RE = re.compile(r\"```(?:python|py)?\\s*(.*?)```\", re.DOTALL)\n"
        "\n"
        "def _extract_code(text):\n"
        "    if not text:\n"
        "        return \"\"\n"
        "    match = _CODE_BLOCK_RE.search(text)\n"
        "    if match:\n"
        "        return match.group(1).strip()\n"
        "    return str(text).strip()\n"
        "\n"
        "def _call_llm(prompt, max_tokens=800):\n"
        "    api_key = os.getenv(\"LLM_API_KEY\")\n"
        "    if not api_key:\n"
        "        raise Exception(\"LLM_API_KEY is not set\")\n"
        "    base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
        "    if not base:\n"
        "        raise Exception(\"LLM_API_BASE is not set\")\n"
        "    url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        "    system = (\n"
        "        \"You MUST follow these output rules exactly: \"\n"
        "        \"Output a single format only. If code is requested, return ONLY raw code text with 4-space indentation. \"\n"
        "        \"Do NOT include labels (python/json), markdown fences, titles, explanations, or comments.\"\n"
        "    )\n"
        "    payload = {\n"
        "        \"model\": \"gpt-4o\",\n"
        "        \"messages\": [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": prompt}],\n"
        "        \"temperature\": 0.1,\n"
        "        \"max_tokens\": max_tokens,\n"
        "    }\n"
        "    headers = {\"Authorization\": f\"Bearer {api_key}\", \"Content-Type\": \"application/json\"}\n"
        "    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)\n"
        "    resp.raise_for_status()\n"
        "    data = resp.json()\n"
        "    text = data.get(\"choices\", [{}])[0].get(\"message\", {}).get(\"content\", \"\")\n"
        "    return _extract_code(text)\n"
        "\n"
        "def run(inputs):\n"
        "    description = inputs.get(\"task\") or inputs.get(\"query\") or \"\"\n"
        "    edge_cases = inputs.get(\"edge_cases\") or []\n"
        "    if isinstance(edge_cases, str):\n"
        "        edge_cases = [edge_cases]\n"
        f"    mode = \"{mode}\"\n"
        "    outline_prompt = (\n"
        "        f\"Task: Create a 3-step outline for a Python function body ({mode}). \"\n"
        "        f\"Problem: {description}. Output format: code. Return ONLY bullet-like code comments removed.\"\n"
        "    )\n"
        "    outline = _call_llm(outline_prompt, max_tokens=200)\n"
        "    merged = description\n"
        "    if outline:\n"
        "        merged = merged + \" | outline: \" + outline.replace(\"\\n\", \" \")\n"
        "    if edge_cases:\n"
        "        merged = merged + \" | edge_cases: \" + \", \".join(edge_cases)\n"
        "    body_prompt = (\n"
        "        f\"Task: Generate a Python function body ({mode}) for: {merged}. \"\n"
        "        \"Output format: code. Return ONLY the function body.\"\n"
        "    )\n"
        "    body = _call_llm(body_prompt, max_tokens=900)\n"
        "    if not body:\n"
        "        body = \"    return None\"\n"
        "    return {\"code_or_commands\": body}\n"
    )


def _build_tools_for_domain(domain: str, count: int) -> list[tuple[str, str, str, bool]]:
    tools: list[tuple[str, str, str, bool]] = []
    if domain == "Code planner":
        modes = _planner_modes()
        for idx in range(count):
            mode = modes[idx % len(modes)]
            name = f"Plan{mode.title()}{idx + 1}"
            desc = f"Planner tool for {mode}-focused workflows in {domain.strip()}."
            tools.append((name, desc, _write_planner_tool(mode), False))
        return tools
    if domain == "Code Testing":
        modes = _testing_modes()
        for idx in range(count):
            mode = modes[idx % len(modes)]
            name = f"Test{mode.title()}{idx + 1}"
            desc = f"Testing tool for {mode} strategies in {domain.strip()}."
            tools.append((name, desc, _write_testing_tool(mode), False))
        return tools
    if domain == "Code Refactoring":
        modes = _refactor_modes()
        for idx in range(count):
            mode = modes[idx % len(modes)]
            name = f"Refactor{mode.title()}{idx + 1}"
            desc = f"Refactoring tool for {mode} operations in {domain.strip()}."
            tools.append((name, desc, _write_refactor_tool(mode), False))
        return tools
    if domain == "Code Generation ":
        modes = _generation_modes()
        for idx in range(count):
            mode = modes[idx % len(modes)]
            name = f"Generate{mode.title().replace('-', '')}{idx + 1}"
            desc = f"Code generation tool for {mode} tasks in {domain.strip()} with variant pipeline."
            tools.append((name, desc, _write_codegen_tool(mode, idx), True))
        return tools
    return tools


def _tool_spec(domain: str, name: str, description: str, code: str, requires_api: bool) -> tuple[dict, str]:
    domain_slug = _slug(domain)
    tool_slug = _slug(name)
    tool_id = f"{domain_slug}-{tool_slug}"
    requirements = "httpx, LLM API" if requires_api else ""
    spec = {
        "name": name,
        "description": description,
        "requirements": requirements,
        "domain_tags": [domain_slug],
        "role_tags": [domain_slug],
        "tool_tags": [tool_id, tool_slug, domain_slug],
        "inputs": ["task", "query", "code", "function_name", "parameters", "cases", "logic", "steps"],
        "outputs": ["output", "code_or_commands"],
        "requires_api": requires_api,
        "api_notes": "",
        "modalities": ["text"],
        "output_formats": ["json"],
        "permissions": ["read"],
        "cost_tier": "low",
        "latency_tier": "low",
        "reliability_prior": 0.8,
        "examples": [f"Use {name} for {domain} tasks."],
        "embedding_text": f"{name}: {description}",
    }
    return spec, code


def generate_assets(
    db_path: str,
    tools_dir: str,
    index_dir: str,
    agents_per_theme: int,
    tools_per_agent: int,
    domains: list[str] | None = None,
    roles: list[str] | None = None,
    reset_db: bool = True,
    tool_source: str = "functional",
    toolbench_dir: str = "tools",
    build_agent_index: bool = True,
) -> None:
    domains = domains or list(DEFAULT_DOMAINS)
    roles = roles or list(DEFAULT_ROLES)
    if reset_db:
        _backup_file(db_path)

    tools_dir = str(Path(tools_dir))
    os.makedirs(tools_dir, exist_ok=True)

    with SQLiteRegistry(db_path) as registry:
        tool_ids_by_domain: dict[str, list[str]] = {}
        tool_name_by_id: dict[str, str] = {}
        tools_per_domain = agents_per_theme * tools_per_agent
        used_tool_ids: set[str] = set()
        for domain in domains:
            if tool_source == "functional":
                selected = select_functional_tools(
                    request_text=domain,
                    count=max(tools_per_agent * 2, min(tools_per_domain, 12)),
                )
                for record in selected:
                    spec = spec_from_record(record)
                    tool_code = code_from_record(record)
                    tool_id = record["tool_id"]
                    card = register_tool_from_spec(
                        registry,
                        spec,
                        tool_code,
                        tool_id=tool_id,
                        save_dir=tools_dir,
                        skip_existing=True,
                    )
                    tool_ids_by_domain.setdefault(domain, []).append(card.id)
                    tool_name_by_id[card.id] = card.name
                continue
            if tool_source == "toolbench":
                selected = select_toolbench_endpoints(
                    toolbench_dir,
                    request_text=domain,
                    count=tools_per_domain,
                    exclude_ids=sorted(used_tool_ids),
                )
                if len(selected) < tools_per_domain:
                    raise ValueError(
                        f"Not enough ToolBench endpoints for {domain}: "
                        f"need {tools_per_domain}, have {len(selected)}"
                    )
                for endpoint in selected:
                    used_tool_ids.add(endpoint["tool_id"])
                    spec = endpoint_to_spec(endpoint)
                    tool_code = endpoint_to_code(endpoint)
                    tool_id = endpoint["tool_id"]
                    card = register_tool_from_spec(
                        registry,
                        spec,
                        tool_code,
                        tool_id=tool_id,
                        save_dir=tools_dir,
                        skip_existing=False,
                    )
                    tool_ids_by_domain.setdefault(domain, []).append(card.id)
                    tool_name_by_id[card.id] = card.name
                continue

            tools = _build_tools_for_domain(domain, tools_per_domain)
            for name, description, code, requires_api in tools:
                spec, tool_code = _tool_spec(domain, name, description, code, requires_api)
                tool_id = f"{_slug(domain)}-{_slug(name)}"
                card = register_tool_from_spec(
                    registry,
                    spec,
                    tool_code,
                    tool_id=tool_id,
                    save_dir=tools_dir,
                    skip_existing=False,
                )
                tool_ids_by_domain.setdefault(domain, []).append(card.id)
                tool_name_by_id[card.id] = card.name

        for domain in domains:
            tool_ids_all = tool_ids_by_domain.get(domain, [])
            domain_slug = _slug(domain)
            assignments = _assign_tool_ids_low_overlap(tool_ids_all, agents_per_theme, tools_per_agent)
            for i, tool_ids in enumerate(assignments, start=1):
                role = roles[(i - 1) % len(roles)]
                agent_id = f"agent-{domain_slug}-{role}-{i}"
                tool_names = [tool_name_by_id.get(tid, tid) for tid in tool_ids]
                tool_label = "/".join(name[:10] for name in tool_names)
                tool_tokens = []
                for tname in tool_names:
                    parts = _slug(tname).split("-")
                    core = parts[1:3] or parts[:1]
                    tool_tokens.append("-".join(core))
                tool_mode = "_".join(tool_tokens)
                name = f"{domain.strip()} {role.title()} {i} [{tool_label}]<{tool_mode}>"
                description = (
                    f"Role: {role}; Domain: {domain.strip()}; "
                    f"Preferred tools: {', '.join(tool_ids)}."
                )
                embedding_text = (
                    f"Role {role}. Domain {domain.strip()}. Tools {', '.join(tool_ids)}."
                )
                agent = AgentCard(
                    id=agent_id,
                    name=name,
                    kind="agent",
                    version="1.0",
                    updated_at=datetime.utcnow(),
                    domain_tags=[domain_slug],
                    role_tags=[role],
                    tool_tags=list(tool_ids),
                    modalities=["text"],
                    output_formats=["json"],
                    permissions=["read"],
                    cost_tier="low",
                    latency_tier="medium",
                    reliability_prior=0.75,
                    description=description,
                    examples=[f"Handle {domain.strip()} requests"],
                    embedding_text=embedding_text,
                    available_tool_ids=list(tool_ids),
                )
                registry.register(agent)

    if build_agent_index:
        build_index(
            db_path=db_path,
            kind="agent",
            out_dir=index_dir,
            dim=64,
            embedder_kind="sentence-transformer",
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            embedder_device=None,
            embedder_normalize=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic tools + agents")
    parser.add_argument("--db", default="demo_registry.sqlite")
    parser.add_argument("--tools_dir", default="generated_tools")
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--agents_per_theme", type=int, default=20)
    parser.add_argument("--tools_per_agent", type=int, default=5)
    parser.add_argument("--domains", default="")
    parser.add_argument("--roles", default="")
    parser.add_argument("--no_reset_db", action="store_true")
    parser.add_argument("--tool_source", choices=["functional", "toolbench", "generated"], default="functional")
    parser.add_argument("--toolbench_dir", default="tools")
    parser.add_argument("--skip_index", action="store_true")
    return parser.parse_args()


def _parse_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    generate_assets(
        db_path=args.db,
        tools_dir=args.tools_dir,
        index_dir=args.index_dir,
        agents_per_theme=args.agents_per_theme,
        tools_per_agent=args.tools_per_agent,
        domains=_parse_list(args.domains) or None,
        roles=_parse_list(args.roles) or None,
        reset_db=not args.no_reset_db,
        tool_source=args.tool_source,
        toolbench_dir=args.toolbench_dir,
        build_agent_index=not args.skip_index,
    )


if __name__ == "__main__":
    main()
