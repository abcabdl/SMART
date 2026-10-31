from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation.llm_client import LLMClient


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    if "[" in text and "]" in text:
        start = text.find("[")
        end = text.rfind("]")
        if end > start:
            return text[start : end + 1]
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return text[start : end + 1]
    return None


def _safe_load_json(text: str, fallback: Any) -> Any:
    blob = _extract_json_blob(text) or text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return fallback


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned or "theme"


def _save_debug_response(
    debug_dir: str,
    label: str,
    system_msg: str,
    user_msg: str,
    response: str,
) -> str:
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    path = Path(debug_dir) / f"{stamp}_{safe_label}.json"
    payload = {
        "label": label,
        "system": system_msg,
        "user": user_msg,
        "response": response,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(path)


def _maybe_debug(
    debug: bool,
    debug_dir: Optional[str],
    label: str,
    system_msg: str,
    user_msg: str,
    response: str,
) -> None:
    if not debug:
        return
    target_dir = debug_dir or "runs/llm_debug"
    path = _save_debug_response(target_dir, label, system_msg, user_msg, response)
    preview = " ".join(str(response).split())[:400]
    print(f"[debug_llm] saved {label} -> {path}", flush=True)
    print(f"[debug_llm] {label} preview: {preview}", flush=True)


def plan_agent_themes(
    task_text: str,
    n_themes: int,
    llm: LLMClient,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    system_msg = (
        "You design agent themes for an autonomous agent system. "
        "Return ONLY JSON. Each theme must include: "
        "name (short), description (1-2 sentences), domain_tags (list), capability_hints (list)."
    )
    user_msg = (
        f"Task: {task_text}\n"
        f"Generate {n_themes} distinct agent themes to cover the task. "
        "Keep domain_tags short (e.g. finance, ops, legal, dev). "
        "Return a JSON list."
    )
    response = llm.chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.4,
        max_tokens=1200,
    )
    _maybe_debug(debug, debug_dir, "plan_agent_themes", system_msg, user_msg, response)
    data = _safe_load_json(response, [])
    if not isinstance(data, list):
        return []
    themes: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        theme = {
            "id": _slug(name),
            "name": name,
            "description": str(item.get("description") or "").strip(),
            "domain_tags": list(item.get("domain_tags") or []),
            "capability_hints": list(item.get("capability_hints") or []),
        }
        if not theme["domain_tags"]:
            theme["domain_tags"] = [_slug(name)]
        themes.append(theme)
    return themes


def plan_tools_for_theme(
    task_text: str,
    theme: Dict[str, Any],
    n_tools: int,
    llm: LLMClient,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    avoid_names: Optional[List[str]] = None,
    debug_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    theme_name = theme.get("name") or "Theme"
    theme_desc = theme.get("description") or ""
    hints = ", ".join(theme.get("capability_hints") or [])
    system_msg = (
        "You design executable Python tools for an agent system. "
        "Return ONLY JSON list. Each tool spec must include: "
        "name (CamelCase), description, requirements (comma-separated libs or empty), "
        "domain_tags (list), role_tags (list), tool_tags (list), inputs (list), outputs (list), "
        "requires_api (true/false), api_notes (short string or empty)."
    )
    task_line = f"Task: {task_text}\n" if task_text else ""
    avoid_names = avoid_names or []
    avoid_line = ""
    if avoid_names:
        avoid_preview = ", ".join(avoid_names[:50])
        avoid_line = f"Avoid these tool names: {avoid_preview}\n"
    user_msg = (
        f"{task_line}"
        f"Agent theme: {theme_name}\n"
        f"Theme description: {theme_desc}\n"
        f"Capability hints: {hints}\n"
        f"{avoid_line}"
        f"Generate {n_tools} tools needed for this theme. "
        "Use concise names with only letters and numbers. "
        "For each tool, decide if external API is needed and set requires_api accordingly. "
        "If requires_api is true, mention it in requirements (e.g. 'httpx') and add api_notes. "
        "Return JSON list."
    )
    response = llm.chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.4,
        max_tokens=1500,
    )
    label = debug_label or f"plan_tools_{_slug(theme_name)}"
    _maybe_debug(debug, debug_dir, label, system_msg, user_msg, response)
    data = _safe_load_json(response, [])
    if not isinstance(data, list):
        return []
    specs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        name = re.sub(r"[^A-Za-z0-9]", "", name)
        if not name:
            continue
        if name in seen:
            suffix = 2
            candidate = f"{name}{suffix}"
            while candidate in seen:
                suffix += 1
                candidate = f"{name}{suffix}"
            name = candidate
        seen.add(name)
        inputs = item.get("inputs") or []
        outputs = item.get("outputs") or []
        if isinstance(inputs, dict):
            inputs = [inputs]
        if isinstance(outputs, dict):
            outputs = [outputs]
        requires_api = bool(item.get("requires_api"))
        requirements = str(item.get("requirements") or "").strip()
        if requires_api and not requirements:
            requirements = "httpx"
        spec = {
            "name": name,
            "description": str(item.get("description") or "").strip(),
            "requirements": requirements,
            "domain_tags": list(item.get("domain_tags") or theme.get("domain_tags") or []),
            "role_tags": list(item.get("role_tags") or []),
            "tool_tags": list(item.get("tool_tags") or [name.lower()]),
            "inputs": list(inputs),
            "outputs": list(outputs),
            "requires_api": requires_api,
            "api_notes": str(item.get("api_notes") or "").strip(),
        }
        specs.append(spec)
    return specs


def themes_from_domains(domains: List[str]) -> List[Dict[str, Any]]:
    themes: List[Dict[str, Any]] = []
    for domain in domains:
        name = str(domain).strip()
        if not name:
            continue
        themes.append(
            {
                "id": _slug(name),
                "name": name,
                "description": f"Domain-focused agents for {name}.",
                "domain_tags": [_slug(name)],
                "capability_hints": [],
            }
        )
    return themes


def plan_agent_profiles_for_theme(
    theme: Dict[str, Any],
    n_agents: int,
    roles: List[str],
    llm: LLMClient,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    batch_size: int = 0,
) -> List[Dict[str, Any]]:
    theme_name = theme.get("name") or "Theme"
    theme_desc = theme.get("description") or ""
    role_list = ", ".join(roles) if roles else ""
    system_msg = (
        "You design distinct agent profiles for an autonomous agent system. "
        "Return ONLY JSON list. Each profile must include: "
        "name, description (1-2 sentences), focus_tags (list), preferred_tool_tags (list), "
        "role (optional), output_style (optional: concise or detailed)."
    )

    def _fetch_batch(target: int, label: str) -> List[Dict[str, Any]]:
        user_msg = (
            f"Theme: {theme_name}\n"
            f"Theme description: {theme_desc}\n"
            f"Roles available: {role_list}\n"
            f"Generate {target} distinct agent profiles with different focuses. "
            "If role is provided, use one of the available roles; otherwise omit it. "
            "Return JSON list."
        )
        response = llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.5,
            max_tokens=1200,
        )
        _maybe_debug(debug, debug_dir, label, system_msg, user_msg, response)
        data = _safe_load_json(response, [])
        return data if isinstance(data, list) else []

    profiles: List[Dict[str, Any]] = []
    seen: set[str] = set()
    remaining = n_agents
    batch_idx = 1
    while remaining > 0:
        current = remaining
        if batch_size and batch_size > 0:
            current = min(batch_size, remaining)
        label = f"plan_profiles_{_slug(theme_name)}_b{batch_idx}" if batch_size else f"plan_profiles_{_slug(theme_name)}"
        data = _fetch_batch(current, label)
        if not data:
            break
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if name in seen:
                suffix = 2
                candidate = f"{name}{suffix}"
                while candidate in seen:
                    suffix += 1
                    candidate = f"{name}{suffix}"
                name = candidate
            seen.add(name)
            focus_tags = item.get("focus_tags") or []
            if isinstance(focus_tags, dict):
                focus_tags = list(focus_tags.values())
            if isinstance(focus_tags, str):
                focus_tags = [focus_tags]
            preferred = item.get("preferred_tool_tags") or []
            if isinstance(preferred, dict):
                preferred = list(preferred.values())
            if isinstance(preferred, str):
                preferred = [preferred]
            role = str(item.get("role") or "").strip()
            if role and roles and role not in roles:
                role = ""
            profiles.append(
                {
                    "name": name,
                    "description": str(item.get("description") or "").strip(),
                    "focus_tags": [str(t).strip() for t in focus_tags if str(t).strip()],
                    "preferred_tool_tags": [str(t).strip() for t in preferred if str(t).strip()],
                    "role": role,
                    "output_style": str(item.get("output_style") or "").strip(),
                }
            )
            if len(profiles) >= n_agents:
                break
        remaining = n_agents - len(profiles)
        if not batch_size:
            break
        batch_idx += 1
    return profiles
