from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


_PATH_PARAM_RE = re.compile(r"{([^{}]+)}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    cleaned = cleaned.strip("-")
    return cleaned or "tool"


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.split())


def _normalize_param_names(params: Iterable[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for param in params:
        name = str((param or {}).get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _tokenize(value: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((value or "").lower()) if len(token) > 2}


def _make_tool_id(category: str, api_name: str, endpoint_name: str, url: str) -> str:
    digest = hashlib.sha1(f"{category}|{api_name}|{endpoint_name}|{url}".encode("utf-8")).hexdigest()[:8]
    return f"toolbench-{_slug(category)}-{_slug(api_name)}-{_slug(endpoint_name)}-{digest}"


@lru_cache(maxsize=4)
def load_toolbench_endpoints(root_dir: str) -> list[dict]:
    root = Path(root_dir).resolve()
    endpoints: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        api_list = payload.get("api_list")
        host = str(payload.get("host") or "").strip()
        if not isinstance(api_list, list) or not host:
            continue
        category = path.parent.name
        api_name = str(payload.get("tool_name") or payload.get("name") or path.stem).strip()
        api_title = str(payload.get("title") or api_name).strip()
        api_description = _clean_text(payload.get("tool_description") or payload.get("description") or "")
        home_url = str(payload.get("home_url") or "").strip()
        for item in api_list:
            if not isinstance(item, dict):
                continue
            endpoint_name = str(item.get("name") or "").strip()
            endpoint_url = str(item.get("url") or "").strip()
            method = str(item.get("method") or "GET").strip().upper()
            if not endpoint_name or not endpoint_url:
                continue
            required_parameters = list(item.get("required_parameters") or [])
            optional_parameters = list(item.get("optional_parameters") or [])
            path_params = _PATH_PARAM_RE.findall(endpoint_url)
            required_names = _normalize_param_names(required_parameters)
            optional_names = _normalize_param_names(optional_parameters)
            query_params = [name for name in required_names + optional_names if name not in path_params]
            tool_id = _make_tool_id(category, api_name, endpoint_name, endpoint_url)
            description = _clean_text(item.get("description") or api_description or endpoint_name)
            endpoint = {
                "tool_id": tool_id,
                "category": category,
                "category_slug": _slug(category),
                "api_name": api_name,
                "api_slug": _slug(api_name),
                "api_title": api_title,
                "api_description": api_description,
                "home_url": home_url,
                "host": host,
                "endpoint_name": endpoint_name,
                "endpoint_slug": _slug(endpoint_name),
                "endpoint_url": endpoint_url,
                "method": method,
                "description": description,
                "required_parameters": required_parameters,
                "optional_parameters": optional_parameters,
                "required_param_names": required_names,
                "optional_param_names": optional_names,
                "path_params": path_params,
                "query_params": query_params,
                "input_names": required_names + [name for name in optional_names if name not in required_names],
                "tool_tags": [
                    "toolbench",
                    _slug(category),
                    _slug(api_name),
                    _slug(endpoint_name),
                    _slug(host),
                ],
                "embedding_text": " ".join(
                    part
                    for part in [
                        category,
                        api_name,
                        api_title,
                        api_description,
                        endpoint_name,
                        description,
                        endpoint_url,
                    ]
                    if part
                ),
            }
            endpoints.append(endpoint)
    return endpoints


def _score_endpoint(endpoint: dict, request_text: str) -> tuple[float, str]:
    request_tokens = _tokenize(request_text)
    if not request_tokens:
        return 0.0, endpoint["tool_id"]
    score = 0.0
    weighted_fields = (
        ("category", 2.0),
        ("api_name", 1.5),
        ("api_title", 1.5),
        ("description", 2.5),
        ("endpoint_name", 2.0),
        ("endpoint_url", 0.5),
    )
    for field, weight in weighted_fields:
        overlap = request_tokens & _tokenize(str(endpoint.get(field) or ""))
        score += weight * len(overlap)
    if endpoint.get("category_slug") in request_text.lower():
        score += 3.0
    return score, endpoint["tool_id"]


def select_toolbench_endpoints(
    root_dir: str,
    request_text: str,
    count: int,
    exclude_ids: Sequence[str] | None = None,
) -> list[dict]:
    endpoints = load_toolbench_endpoints(root_dir)
    exclude = set(exclude_ids or [])
    available = [endpoint for endpoint in endpoints if endpoint["tool_id"] not in exclude]
    if count <= 0 or not available:
        return []
    scored = sorted(
        available,
        key=lambda endpoint: (-_score_endpoint(endpoint, request_text)[0], endpoint["tool_id"]),
    )
    return scored[:count]


def endpoint_to_spec(endpoint: dict) -> dict:
    name = f"{endpoint['api_name']} {endpoint['endpoint_name']}".strip()
    description = endpoint["description"]
    if endpoint.get("api_title") and endpoint["api_title"] != endpoint["api_name"]:
        description = f"{endpoint['api_title']}: {description}".strip()
    inputs = list(endpoint.get("input_names") or [])
    inputs.extend(["toolbench_rapidapi_key", "rapidapi_key", "timeout_s"])
    dedup_inputs: list[str] = []
    seen: set[str] = set()
    for item in inputs:
        if item in seen:
            continue
        seen.add(item)
        dedup_inputs.append(item)
    return {
        "name": name,
        "description": description,
        "requirements": "requests, RapidAPI",
        "domain_tags": [endpoint["category_slug"]],
        "role_tags": [endpoint["category_slug"]],
        "tool_tags": list(endpoint.get("tool_tags") or []),
        "inputs": dedup_inputs,
        "outputs": ["output"],
        "requires_api": False,
        "api_notes": endpoint.get("home_url") or endpoint.get("endpoint_url") or "",
        "modalities": ["text"],
        "output_formats": ["json"],
        "permissions": ["network"],
        "cost_tier": "medium",
        "latency_tier": "medium",
        "reliability_prior": 0.7,
        "examples": [f"Call {endpoint['endpoint_name']} from {endpoint['api_name']}"],
        "embedding_text": endpoint.get("embedding_text") or description or name,
    }


def endpoint_to_code(endpoint: dict) -> str:
    return f"""import os
from urllib.parse import quote

import requests


_TOOL_NAME = {endpoint['api_name']!r}
_ENDPOINT_NAME = {endpoint['endpoint_name']!r}
_METHOD = {endpoint['method']!r}
_URL_TEMPLATE = {endpoint['endpoint_url']!r}
_HOST = {endpoint['host']!r}
_REQUIRED_PARAMETERS = {endpoint['required_param_names']!r}
_OPTIONAL_PARAMETERS = {endpoint['optional_param_names']!r}
_PATH_PARAMETERS = {endpoint['path_params']!r}
_QUERY_PARAMETERS = {endpoint['query_params']!r}


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return value


def _resolve_api_key(inputs):
    for key in ("toolbench_rapidapi_key", "rapidapi_key", "x_rapidapi_key"):
        value = _clean_value(inputs.get(key))
        if value:
            return value
    for env_name in ("TOOLBENCH_RAPIDAPI_KEY", "RAPIDAPI_KEY", "X_RAPIDAPI_KEY"):
        value = _clean_value(os.getenv(env_name))
        if value:
            return value
    return None


def _resolve_timeout(inputs):
    value = inputs.get("timeout_s")
    if value is None:
        return 30.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 30.0


def _build_url(values):
    url = _URL_TEMPLATE
    for name in _PATH_PARAMETERS:
        url = url.replace("{{" + name + "}}", quote(str(values[name]), safe=""))
    return url


def run(inputs):
    if not isinstance(inputs, dict):
        raise Exception("inputs must be a dict")

    api_key = _resolve_api_key(inputs)
    if not api_key:
        raise Exception(
            "Missing RapidAPI key. Provide toolbench_rapidapi_key in inputs or set TOOLBENCH_RAPIDAPI_KEY."
        )

    values = {{}}
    missing = []
    for name in _REQUIRED_PARAMETERS:
        value = _clean_value(inputs.get(name))
        if value is None:
            missing.append(name)
            continue
        values[name] = value

    if missing:
        raise Exception("Missing required parameters for " + _ENDPOINT_NAME + ": " + ", ".join(missing))

    for name in _OPTIONAL_PARAMETERS:
        value = _clean_value(inputs.get(name))
        if value is not None:
            values[name] = value

    url = _build_url(values)
    querystring = {{}}
    for name in _QUERY_PARAMETERS:
        if name in values:
            querystring[name] = values[name]

    headers = {{
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": _HOST,
    }}

    extra_headers = inputs.get("headers")
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            if value is not None:
                headers[str(key)] = str(value)

    body = inputs.get("body")
    request_kwargs = {{
        "headers": headers,
        "params": querystring or None,
        "timeout": _resolve_timeout(inputs),
    }}
    if isinstance(body, dict):
        request_kwargs["json"] = body

    response = requests.request(_METHOD, url, **request_kwargs)
    try:
        payload = response.json()
    except Exception:
        payload = response.text

    return {{
        "output": {{
            "tool": _TOOL_NAME,
            "endpoint": _ENDPOINT_NAME,
            "method": _METHOD,
            "status_code": response.status_code,
            "data": payload,
        }}
    }}
"""
