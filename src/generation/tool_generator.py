from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from core.cards import ToolCard
from core.registry import SQLiteRegistry
from generation.llm_client import LLMClient


_TEXT_CLEANER_CODE = """import re


def run(inputs):
    text = inputs.get(\"text\", \"\")
    if not isinstance(text, str):
        return {\"error\": \"text must be a string\", \"text\": \"\"}
    cleaned = text.strip().lower()
    cleaned = re.sub(r\"[^\\w\\s]\", \"\", cleaned)
    cleaned = re.sub(r\"\\s+\", \" \", cleaned).strip()
    return {\"text\": cleaned}
"""

_BASIC_STATS_CODE = """def run(inputs):
    values = inputs.get(\"values\", [])
    if not isinstance(values, list):
        return {\"error\": \"values must be a list\", \"count\": 0}
    cleaned = [value for value in values if isinstance(value, (int, float))]
    if not cleaned:
        return {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None}
    total = sum(cleaned)
    count = len(cleaned)
    return {\"count\": count, \"mean\": total / count, \"min\": min(cleaned), \"max\": max(cleaned)}
"""


class MockToolSynthesizer:
    """Rule-based mock tool synthesizer."""

    def __init__(self) -> None:
        self._templates = {
            "text_cleaner": {
                "code": _TEXT_CLEANER_CODE,
                "metadata": {
                    "description": "Normalize text by stripping, lowercasing, and removing punctuation.",
                    "domain_tags": ["text", "cleaning"],
                    "role_tags": ["utility"],
                    "tool_tags": ["text_cleaner"],
                    "modalities": ["text"],
                    "output_formats": ["json"],
                    "permissions": ["read"],
                    "cost_tier": "low",
                    "latency_tier": "low",
                    "reliability_prior": 0.8,
                    "examples": ["Clean a noisy string"],
                },
            },
            "basic_stats": {
                "code": _BASIC_STATS_CODE,
                "metadata": {
                    "description": "Compute basic statistics over a list of numbers.",
                    "domain_tags": ["analysis", "math"],
                    "role_tags": ["utility"],
                    "tool_tags": ["basic_stats"],
                    "modalities": ["text"],
                    "output_formats": ["json"],
                    "permissions": ["read"],
                    "cost_tier": "low",
                    "latency_tier": "low",
                    "reliability_prior": 0.85,
                    "examples": ["Compute mean/min/max"],
                },
            },
        }

    def synthesize(self, name: str) -> Tuple[str, Dict]:
        if name not in self._templates:
            raise ValueError(f"Unknown tool template: {name}")
        entry = self._templates[name]
        return entry["code"], entry["metadata"]


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_RUN_DEF_RE = re.compile(r"\bdef\s+run\s*\(", re.ASCII)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    matches = list(_CODE_BLOCK_RE.finditer(text))
    if matches:
        for match in matches:
            candidate = match.group(1).strip()
            if _RUN_DEF_RE.search(candidate):
                return candidate
        return matches[0].group(1).strip()
    return text.strip()


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return [str(values)]
    normalized: list[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue
        if isinstance(item, dict):
            name = item.get("name") or item.get("param") or item.get("field") or item.get("key")
            if name:
                normalized.append(str(name))
            else:
                normalized.append(str(item))
            continue
        normalized.append(str(item))
    return normalized


def generate_tool_code(spec: Dict[str, Any], llm: LLMClient) -> str:
    name = spec.get("name", "Tool")
    description = spec.get("description", "")
    requirements = spec.get("requirements", "")
    inputs = ", ".join(_normalize_list(spec.get("inputs")))
    outputs = ", ".join(_normalize_list(spec.get("outputs")))
    requires_api = bool(spec.get("requires_api"))
    api_notes = str(spec.get("api_notes") or "").strip()
    system_msg = (
        "You are an expert Python developer. "
        "Return ONLY executable Python code, no markdown."
    )
    api_guidance = ""
    if requires_api:
        api_guidance = (
            "\nAPI Requirements:\n"
            "- This tool MUST call an external HTTP API to fulfill its task.\n"
            "- Use httpx for HTTP requests.\n"
            "- Read API base URL and API key from environment variables.\n"
            "- REQUIRED env vars: LLM_API_BASE and LLM_API_KEY (or tool-specific overrides if noted).\n"
            "- Always call OpenAI-compatible Chat Completions at:\n"
            "  ${LLM_API_BASE}/chat/completions (append /chat/completions if missing).\n"
            "- Use model: gpt-4o.\n"
            "- Do NOT invent custom endpoints like /generate_algorithm or /analyze_scalability.\n"
            "- Use Authorization: Bearer <API_KEY>.\n"
            "- Handle non-200 responses and return {\"error\": \"...\", \"ok\": False}.\n"
            "- URL build example:\n"
            "  base = (os.getenv(\"LLM_API_BASE\") or \"\").rstrip(\"/\")\n"
            "  url = base if base.endswith(\"/chat/completions\") else f\"{base}/chat/completions\"\n"
        )
        if api_notes:
            api_guidance += f"- Notes: {api_notes}\n"
    user_msg = f"""
Write a single Python module for this tool.

Tool Name: {name}
Description: {description}
Requirements: {requirements}
Inputs: {inputs}
Outputs: {outputs}

Guidelines:
1) The module MUST define a top-level function: run(inputs: dict) -> dict
2) Read parameters from inputs.get(...); use inputs.get("query") as fallback.
3) If required inputs are missing, DO NOT fail immediately. Use the LLM API to infer missing parameters from
   the query text and the declared Inputs list, then proceed.
   - Build a JSON extraction prompt that returns ONLY a JSON object with keys matching Inputs.
   - Merge extracted values into the inputs dict before running the main logic.
   - Use this template (adapt field names):
     def _extract_params(query_text: str, fields: list[str]) -> dict:
         base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
         url = base if base.endswith("/chat/completions") else f"{{base}}/chat/completions"
         api_key = os.getenv("LLM_API_KEY")
         headers = {{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}}
         system = "Return ONLY JSON with keys: " + ", ".join(fields)
         user = f"Query: {{query_text}}"
         payload = {{
             "model": "gpt-4o",
             "messages": [{{"role": "system", "content": system}}, {{"role": "user", "content": user}}],
             "response_format": {{"type": "json_object"}},
             "max_tokens": 500,
         }}
         resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
         resp.raise_for_status()
         data = resp.json()
         text = data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "{{}}")
         return json.loads(text)
4) IMPLEMENT THE FULL TOOL LOGIC. Do NOT return placeholders, templates, "pass", or example-only code.
   Do NOT return prose like "Generated code..." or JSON snippets. The output MUST be executable Python.
5) CRITICAL: You are writing the tool implementation itself, so you MUST output a full Python module
   that includes def run(inputs: dict) -> dict. Do NOT omit def run. If any rule conflicts, keep def run.
6) When implementing the tool's BEHAVIOR, if the tool generates code for a single function:
   - The "function body only" rule applies ONLY to the tool's OUTPUT (code_or_commands), NOT to this module.
   - You MUST use the EXACT function_name and parameter names provided in inputs (no renaming).
   - The tool's OUTPUT (code_or_commands) MUST contain ONLY the function body code (properly indented).
   - Do NOT include a "def ..." line, imports, helper functions outside the target function, comments, or example usage.
   - If the LLM returns anything else, you MUST post-process it to extract only the valid function body.
   - If this tool is a code formatter, do NOT wrap output in JSON, YAML, or prose; return ONLY the formatted code body.
   - Do NOT prefix outputs with labels like "json", "python", or fenced code markers.
   - If this tool calls an LLM to produce the function body, the LLM prompt MUST explicitly demand:
     "Return ONLY the function body code. Do NOT include def/imports/markdown/labels."
7) If you call the LLM to generate code, ensure you strip markdown code fences (```...```).
   - Also verify the final code is syntactically valid Python (e.g., no unterminated strings).
   - You MUST post-process the LLM output locally: remove labels/fences, extract only the function body,
     and validate with ast.parse on a synthetic wrapper (e.g., "def _tmp(...):\\n" + body).
   - If syntax is invalid, fix it (e.g., remove stray fences/labels, close quotes) or re-prompt, then re-validate.
7b) When this tool calls an LLM, the prompt MUST also enforce output format rules:
   - Output must be a single format: if JSON is expected, return a pure JSON object only; if code is expected,
     return pure code text only (no JSON/prose/titles/markdown).
   - Do NOT include labels or fences like "python", "json", or ``` anywhere in the output.
   - Do NOT add explanations, comments, or natural-language prefaces like "Here is...".
   - If returning a function body, do NOT include def/class/import/main/docstrings/comments.
   - Function body indentation MUST be exactly 4 spaces; do not use tabs.
   - Empty or whitespace-only output is a failure and must trigger a retry.
   - If returning JSON, require response_format={{"type":"json_object"}} and ONLY the specified keys; extra keys are an error.
   - Do NOT wrap code in {{"code_or_commands": ...}} when returning from the LLM; the tool itself will wrap it.
   - Preserve original parameter names in the function body; never rename inputs.
7c) You MUST explicitly embed these output-format rules into the LLM prompt for every API call.
   - Include them verbatim in the system or user message so the LLM sees them directly.
   - Do NOT assume prior context or rely on this generator prompt; repeat the rules inside the tool's own LLM prompt.
   - Example (adapt as needed):
     system = "Return ONLY the function body. No def/imports/markdown/labels. No explanations."
     user = "Task: ... (and restate: no python/json/``` labels; no extra text; 4-space indent; empty output = error)."
7d) Strong template (copy/paste into any tool LLM call and fill placeholders):
   system = (
       "You MUST follow these output rules exactly: "
       "Output a single format only. If JSON is requested, return ONLY a pure JSON object with the specified keys. "
       "If code is requested, return ONLY raw code text with 4-space indentation. "
       "Do NOT include labels (python/json), markdown fences, titles, explanations, or comments. "
       "If asked for a function body, do NOT include def/class/import/main/docstrings. "
       "Empty/whitespace output is invalid."
   )
     user = (
         "Task: <describe task>. "
         "Output format: <JSON or code>. "
         "If code: return ONLY the function body using the original parameter names. "
         "If JSON: use response_format={{\"type\":\"json_object\"}} and ONLY the specified keys."
     )
8) For chat/completions requests that return plain code, do NOT set response_format to a text type; omit response_format.
9) Always set timeout=60.0 for httpx.post calls.
10) If LLM_API_BASE or LLM_API_KEY is missing, raise an Exception with a clear message.
11) The run(...) function must return a dict whose primary payload is in:
   - "code_or_commands" if you are producing executable code or shell commands, otherwise
   - "output" for general results.
12) Do NOT wrap the payload in {{"result": ...}}. Do NOT include "ok"/"error" fields on success.
13) On failure, raise an Exception with a clear message (do not return an error dict).
14) Use robust error handling. Keep code minimal and deterministic.
15) Only use standard library unless Requirements mention extra libs (e.g. httpx).
16) If calling external APIs, read base URL/key from LLM_API_BASE and LLM_API_KEY (os.getenv).
17) Ensure the code is executable as-is.
{api_guidance}
"""
    response = llm.chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=2000,
    )
    code = _extract_code(response)
    if not _RUN_DEF_RE.search(code):
        if _RUN_DEF_RE.search(response):
            code = response.strip()
        else:
            raise ValueError("Generated tool code missing run(inputs) function")
    return code


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned or "tool"


def register_tool_from_spec(
    registry: SQLiteRegistry,
    spec: Dict[str, Any],
    code: str,
    tool_id: Optional[str] = None,
    version: str = "1.0",
    save_dir: Optional[str] = None,
    skip_existing: bool = False,
) -> ToolCard:
    now = datetime.utcnow()
    name = str(spec.get("name") or tool_id or "tool")
    description = str(spec.get("description") or "")
    requirements = str(spec.get("requirements") or "")
    if requirements:
        description = f"{description} Requirements: {requirements}".strip()

    card = ToolCard(
        id=tool_id or _slug(name),
        name=name,
        kind="tool",
        version=version,
        updated_at=now,
        domain_tags=list(spec.get("domain_tags") or []),
        role_tags=list(spec.get("role_tags") or []),
        tool_tags=list(spec.get("tool_tags") or [name.lower()]),
        modalities=list(spec.get("modalities") or ["text"]),
        output_formats=list(spec.get("output_formats") or ["json"]),
        permissions=list(spec.get("permissions") or ["read"]),
        cost_tier=spec.get("cost_tier") or "medium",
        latency_tier=spec.get("latency_tier") or "medium",
        reliability_prior=spec.get("reliability_prior") or 0.7,
        description=description,
        examples=list(spec.get("examples") or []),
        embedding_text=spec.get("embedding_text") or description or name,
        dataset_family=str(spec.get("dataset_family") or ""),
        focus_slug=str(spec.get("focus_slug") or ""),
        primary_role=str(spec.get("primary_role") or ""),
        capability_kind=str(spec.get("capability_kind") or ""),
        stage=str(spec.get("stage") or ""),
        deterministic=bool(spec.get("deterministic", True)),
        consumes=list(spec.get("consumes") or []),
        produces=list(spec.get("produces") or []),
    )

    if skip_existing:
        existing = registry.get(card.id)
        if existing is not None:
            return existing
    registry.register(card)
    registry.register_tool_code(card.id, code, updated_at=now)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{card.id}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    return card


def register_tool(
    registry: SQLiteRegistry,
    tool_name: str,
    tool_id: Optional[str] = None,
    version: str = "1.0",
    skip_existing: bool = False,
) -> ToolCard:
    synthesizer = MockToolSynthesizer()
    code, metadata = synthesizer.synthesize(tool_name)
    now = datetime.utcnow()

    card = ToolCard(
        id=tool_id or tool_name,
        name=tool_name,
        kind="tool",
        version=version,
        updated_at=now,
        domain_tags=metadata.get("domain_tags", []),
        role_tags=metadata.get("role_tags", []),
        tool_tags=metadata.get("tool_tags", []),
        modalities=metadata.get("modalities", []),
        output_formats=metadata.get("output_formats", []),
        permissions=metadata.get("permissions", []),
        cost_tier=metadata.get("cost_tier"),
        latency_tier=metadata.get("latency_tier"),
        reliability_prior=metadata.get("reliability_prior"),
        description=metadata.get("description", ""),
        examples=metadata.get("examples", []),
        embedding_text=metadata.get("description", tool_name),
        dataset_family=str(metadata.get("dataset_family") or ""),
        focus_slug=str(metadata.get("focus_slug") or ""),
        primary_role=str(metadata.get("primary_role") or ""),
        capability_kind=str(metadata.get("capability_kind") or ""),
        stage=str(metadata.get("stage") or ""),
        deterministic=bool(metadata.get("deterministic", True)),
        consumes=list(metadata.get("consumes") or []),
        produces=list(metadata.get("produces") or []),
    )

    if skip_existing:
        existing = registry.get(card.id)
        if existing is not None:
            return existing
    registry.register(card)
    registry.register_tool_code(card.id, code, updated_at=now)
    return card
