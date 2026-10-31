from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import httpx


def _first_env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class LLMClient:
    """OpenAI-compatible chat client (prefer SDK, fallback to httpx)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        organization: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff_s: float = 2.0,
        trust_env: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key or _first_env("LLM_API_KEY", "OPENAI_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip("'\"")
        self.base_url = base_url or _first_env(
            "LLM_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            default="https://api.openai.com/v1",
        )
        if self.base_url:
            self.base_url = self.base_url.strip("'\"")
        self.model = model or _first_env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
        self.timeout_s = timeout_s
        self.organization = organization or _first_env("OPENAI_ORG")
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.trust_env = bool(trust_env) if trust_env is not None else _env_bool("LLM_TRUST_ENV", True)
        self.client_max_idle_s = max(0.0, _env_float("LLM_CLIENT_MAX_IDLE_S", 120.0))
        self._httpx_client: Optional[httpx.Client] = None
        self._httpx_last_used: float = 0.0
        self._sdk_client: Optional[Any] = None
        self._sdk_http_client: Optional[httpx.Client] = None
        self._sdk_last_used: float = 0.0
        self.raw_log_path = _first_env("LLM_RAW_LOG_PATH", default=os.path.join("runs", "llm_raw.jsonl"))

    def _append_raw_log(self, event: Dict[str, Any]) -> None:
        path = self.raw_log_path
        if not path:
            return
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = dict(event)
            payload.setdefault("timestamp", datetime.utcnow().isoformat())
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception as exc:
            sys.stderr.write(f"[LLMClient] Failed to write raw log: {exc}\n")

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY/OPENAI_API_KEY is not set")
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            url = base
        elif base.endswith("/responses"):
            url = base
        else:
            url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        sys.stderr.write(f"[LLMClient] Requesting {url} (model={self.model})\n")

        # Prefer official OpenAI SDK transport path first.
        sdk_text = self._chat_with_openai_sdk(url=url, payload=payload)
        if sdk_text is not None:
            return sdk_text
        sys.stderr.write("[LLMClient] SDK path unavailable or failed, fallback to httpx\n")
        return self._chat_with_httpx(url=url, headers=headers, payload=payload)

    def _chat_with_httpx(self, *, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._ensure_httpx_client()
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                self._httpx_last_used = time.monotonic()
                data = response.json()
                self._append_raw_log(
                    {
                        "transport": "httpx",
                        "url": url,
                        "model": self.model,
                        "request": _json_safe(payload),
                        "response": _json_safe(data),
                        "status_code": response.status_code,
                    }
                )
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}")
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                sys.stderr.write(f"[LLMClient] Error (attempt {attempt+1}): {exc}\n")
                error_event: Dict[str, Any] = {
                    "transport": "httpx",
                    "url": url,
                    "model": self.model,
                    "request": _json_safe(payload),
                    "attempt": attempt + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if isinstance(exc, httpx.HTTPStatusError):
                    error_event["status_code"] = exc.response.status_code
                    try:
                        error_event["response_text"] = exc.response.text
                    except Exception:
                        pass
                self._append_raw_log(error_event)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                    # Keep behavior-compatible path handling for non-standard gateways.
                    pass
                if not isinstance(exc, httpx.HTTPStatusError):
                    self._reset_httpx_client()
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed without response")

    def _chat_with_openai_sdk(self, *, url: str, payload: Dict[str, Any]) -> Optional[str]:
        """Primary transport path using official OpenAI SDK."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        for sdk_attempt in range(3):
            try:
                client = self._ensure_sdk_client(OpenAI)
                response = client.chat.completions.create(**payload)
                self._sdk_last_used = time.monotonic()
                raw_response: Any
                dump_fn = getattr(response, "model_dump", None)
                if callable(dump_fn):
                    try:
                        raw_response = dump_fn()
                    except Exception:
                        raw_response = str(response)
                else:
                    raw_response = str(response)
                message = response.choices[0].message
                content = getattr(message, "content", None)
                self._append_raw_log(
                    {
                        "transport": "openai_sdk",
                        "url": url,
                        "model": self.model,
                        "request": _json_safe(payload),
                        "response": _json_safe(raw_response),
                        "message_content": _json_safe(content),
                    }
                )
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    chunks: list[str] = []
                    for part in content:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            chunks.append(text)
                    if chunks:
                        return "".join(chunks)
                if content is None:
                    return None
                return str(content)
            except Exception as exc:
                self._append_raw_log(
                    {
                        "transport": "openai_sdk",
                        "url": url,
                        "model": self.model,
                        "request": _json_safe(payload),
                        "attempt": sdk_attempt + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                self._reset_sdk_client()
                if sdk_attempt == 0:
                    sys.stderr.write(f"[LLMClient] OpenAI SDK path failed (retrying fresh client): {exc}\n")
                    continue
                sys.stderr.write(f"[LLMClient] OpenAI SDK path failed: {exc}\n")
                return None
        return None

    def _is_stale(self, last_used: float) -> bool:
        if self.client_max_idle_s <= 0:
            return False
        if last_used <= 0:
            return False
        return (time.monotonic() - last_used) > self.client_max_idle_s

    def _ensure_httpx_client(self) -> httpx.Client:
        if self._httpx_client is None or self._is_stale(self._httpx_last_used):
            self._reset_httpx_client()
            self._httpx_client = httpx.Client(
                timeout=self.timeout_s,
                http2=False,
                trust_env=self.trust_env,
            )
        return self._httpx_client

    def _reset_httpx_client(self) -> None:
        if self._httpx_client is not None:
            try:
                self._httpx_client.close()
            except Exception:
                pass
        self._httpx_client = None
        self._httpx_last_used = 0.0

    def _ensure_sdk_client(self, openai_cls: Any) -> Any:
        if self._sdk_client is not None and not self._is_stale(self._sdk_last_used):
            return self._sdk_client

        self._reset_sdk_client()
        sdk_base = self._normalize_sdk_base_url(self.base_url)
        self._sdk_http_client = httpx.Client(
            timeout=self.timeout_s,
            http2=False,
            trust_env=self.trust_env,
        )
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_s,
            "max_retries": self.max_retries,
            "http_client": self._sdk_http_client,
        }
        if sdk_base:
            kwargs["base_url"] = sdk_base
        if self.organization:
            kwargs["organization"] = self.organization
        self._sdk_client = openai_cls(**kwargs)
        self._sdk_last_used = time.monotonic()
        return self._sdk_client

    def _reset_sdk_client(self) -> None:
        if self._sdk_client is not None:
            try:
                self._sdk_client.close()
            except Exception:
                pass
        if self._sdk_http_client is not None:
            try:
                self._sdk_http_client.close()
            except Exception:
                pass
        self._sdk_client = None
        self._sdk_http_client = None
        self._sdk_last_used = 0.0

    def close(self) -> None:
        self._reset_httpx_client()
        self._reset_sdk_client()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _normalize_sdk_base_url(self, base_url: Optional[str]) -> Optional[str]:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            return None
        if base.endswith("/chat/completions"):
            return base[: -len("/chat/completions")] or None
        if base.endswith("/responses"):
            return base[: -len("/responses")] or None
        return base
