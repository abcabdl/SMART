from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from core.registry import SQLiteRegistry
from execution.sandbox import execute_tool_code


class ToolExecutor:
    """Execute tool code retrieved from the registry."""

    _INPUT_GET_RE = re.compile(r"inputs\.get\(\s*['\"]([^'\"]+)['\"]")
    _INPUT_IDX_RE = re.compile(r"inputs\[\s*['\"]([^'\"]+)['\"]\s*\]")
    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    _COMMON_PACKAGES = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("sympy", "sympy"),
        ("sklearn", "scikit-learn"),
        ("seaborn", "seaborn"),
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("bs4", "beautifulsoup4"),
        ("lxml", "lxml"),
        ("yaml", "pyyaml"),
        ("toml", "toml"),
        ("openpyxl", "openpyxl"),
        ("openai", "openai"),
        ("PIL", "pillow"),
        ("networkx", "networkx"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib"),
        ("dateutil", "python-dateutil"),
        ("pytz", "pytz"),
        ("kiwisolver", "kiwisolver"),
        ("pyparsing", "pyparsing"),
        ("astor", "astor"),
        ("statsmodels", "statsmodels"),
        ("pygame", "pygame"),
        ("causalimpact", "causalimpact"),
    ]

    def __init__(
        self,
        registry: SQLiteRegistry,
        timeout_s: float = 1.0,
        auto_fill: bool = False,
        llm: Optional[Any] = None,
        auto_fill_max_tokens: int = 400,
        auto_install_common_libs: bool = False,
        auto_install_timeout_s: float = 300.0,
        auto_install_user: bool = False,
    ):
        self.registry = registry
        self.timeout_s = timeout_s
        self.auto_fill = auto_fill
        self.llm = llm
        self.auto_fill_max_tokens = auto_fill_max_tokens
        self.auto_install_common_libs = auto_install_common_libs
        self.auto_install_timeout_s = auto_install_timeout_s
        self.auto_install_user = auto_install_user
        if self.auto_install_common_libs:
            self._ensure_common_libs_installed()

    def _missing_common_packages(self) -> List[str]:
        missing: List[str] = []
        seen: set[str] = set()
        for import_name, pip_name in self._COMMON_PACKAGES:
            if import_name in seen:
                continue
            seen.add(import_name)
            if importlib.util.find_spec(import_name) is None:
                if pip_name not in missing:
                    missing.append(pip_name)
        return missing

    def _ensure_common_libs_installed(self) -> None:
        missing = self._missing_common_packages()
        if not missing:
            return
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
        if self.auto_install_user:
            cmd.append("--user")
        cmd.extend(missing)
        try:
            subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.auto_install_timeout_s,
            )
        except Exception:
            return

    def _infer_params(self, code: str) -> List[str]:
        if not code:
            return []
        params = set(self._INPUT_GET_RE.findall(code))
        params.update(self._INPUT_IDX_RE.findall(code))
        return sorted(params)

    def _extract_json_blob(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = self._JSON_BLOCK_RE.search(text)
        if match:
            return match.group(1)
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if end > start:
                return text[start : end + 1]
        return None

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        blob = self._extract_json_blob(text) or text
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _auto_fill_inputs(self, card, code: str, inputs: dict) -> dict:
        if not self.auto_fill or self.llm is None:
            return inputs
        params = self._infer_params(code)
        missing = [p for p in params if p not in inputs]
        if not missing:
            return inputs
        system_msg = (
            "You fill missing parameters for a tool call. "
            "Return ONLY a JSON object with values for the missing keys."
        )
        user_msg = (
            f"Tool name: {getattr(card, 'name', '')}\n"
            f"Tool description: {getattr(card, 'description', '')}\n"
            f"Missing params: {missing}\n"
            f"Query: {inputs.get('query','')}\n"
            f"Task: {inputs.get('task','')}\n"
            f"Role: {inputs.get('role','')}\n"
            "Provide concise, valid values. Use empty string if unknown."
        )
        try:
            response = self.llm.chat(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.1,
                max_tokens=self.auto_fill_max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            return inputs
        data = self._safe_json(str(response))
        if not data:
            return inputs
        merged = dict(data)
        merged.update(inputs)
        return merged

    def _prepare_tool_env(self, inputs: dict) -> Dict[str, str]:
        env_updates: Dict[str, str] = {}
        # Allow explicit overrides from tool inputs.
        explicit_base = inputs.get("API_BASE_URL") or inputs.get("api_base_url") or inputs.get("base_url")
        explicit_key = inputs.get("API_KEY") or inputs.get("api_key") or inputs.get("key")
        resolved_base: Optional[str] = str(explicit_base) if explicit_base else None
        resolved_key: Optional[str] = str(explicit_key) if explicit_key else None

        # Fallback to LLM/OpenAI env vars when tool-specific ones are missing.
        if resolved_base is None and not os.getenv("API_BASE_URL"):
            for name in ("LLM_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
                value = os.getenv(name)
                if value:
                    resolved_base = value
                    break
        if resolved_key is None and not os.getenv("API_KEY"):
            for name in ("LLM_API_KEY", "OPENAI_API_KEY"):
                value = os.getenv(name)
                if value:
                    resolved_key = value
                    break

        # Populate aliases so generated tools can read either API_* or LLM_* or OPENAI_*.
        if resolved_base:
            env_updates["API_BASE_URL"] = resolved_base
            env_updates["LLM_API_BASE"] = resolved_base
            env_updates["OPENAI_BASE_URL"] = resolved_base
            env_updates["OPENAI_API_BASE"] = resolved_base
        if resolved_key:
            env_updates["API_KEY"] = resolved_key
            env_updates["LLM_API_KEY"] = resolved_key
            env_updates["OPENAI_API_KEY"] = resolved_key
        return env_updates

    def run_tool(self, tool_id: str, inputs: dict) -> Dict:
        card = self.registry.get(tool_id)
        if card is None:
            return {
                "ok": False,
                "output": None,
                "error": {"code": "tool_not_found", "message": f"Missing tool: {tool_id}"},
            }
        if getattr(card, "kind", None) != "tool":
            return {
                "ok": False,
                "output": None,
                "error": {"code": "not_a_tool", "message": f"Card is not a tool: {tool_id}"},
            }

        code = getattr(card, "code", None)
        if not code:
            code = self.registry.get_tool_code(tool_id)
        if not code:
            return {
                "ok": False,
                "output": None,
                "error": {"code": "tool_code_missing", "message": f"No code for tool: {tool_id}"},
            }

        inputs = self._auto_fill_inputs(card, code, inputs)
        env_updates = self._prepare_tool_env(inputs)
        old_env: Dict[str, Optional[str]] = {key: os.getenv(key) for key in env_updates}
        for key, value in env_updates.items():
            os.environ[key] = value
        try:
            result = execute_tool_code(code, inputs, timeout_s=self.timeout_s)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if result.get("ok"):
            return {
                "ok": True,
                "output": result.get("output"),
                "error": None,
            }
        return {
            "ok": False,
            "output": None,
            "error": result.get("error") or {"code": "sandbox_error", "message": "Unknown error"},
        }
