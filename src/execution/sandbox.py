from __future__ import annotations

import builtins
import multiprocessing as mp
import os
import queue
from typing import Dict, Optional


_ALLOWED_MODULES = {
    "ast",
    "astor",
    "base64",
    "bisect",  # 添加：二分查找
    "bs4",
    "collections",
    "copy",  # 添加：深拷贝/浅拷贝
    "csv",
    "cycler",
    "datetime",
    "dateutil",
    "decimal",  # 添加：精确十进制运算
    "fractions",  # 添加：分数运算
    "functools",  # 添加：函数工具
    "hashlib",
    "heapq",  # 添加：堆队列
    "httpx",
    "itertools",  # 添加：迭代工具
    "joblib",
    "json",
    "kiwisolver",
    "logging",
    "math",
    "matplotlib",
    "networkx",
    "numpy",
    "operator",  # 添加：操作符函数
    "os",
    "pandas",
    "pathlib",
    "PIL",
    "pyparsing",
    "pytz",
    "queue",  # 添加：队列
    "random",  # 添加：随机数
    "re",
    "requests",
    "scipy",
    "seaborn",
    "statsmodels",
    "sklearn",
    "statistics",
    "string",  # 添加：字符串操作
    "subprocess",
    "sympy",
    "sys",  # 添加：系统模块（安全的只读操作）
    "tempfile",
    "textwrap",  # 添加：文本包装
    "time",
    "toml",
    "traceback",
    "tqdm",
    "typing",
    "urllib",
    "uuid",
    "xml",  # 添加：XML处理（包括xml.etree.ElementTree）
    "yaml",
    "openpyxl",
    "openai",
    "lxml",
    "pygame",
    "causalimpact",
    "cv2",
}
_extra_modules = {
    item.strip()
    for item in os.getenv("TOOL_ALLOWED_MODULES", "").split(",")
    if item.strip()
}
_ALLOWED_MODULES.update(_extra_modules)
_ORIGINAL_IMPORT = __import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if os.getenv("TOOL_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    root = name.split(".")[0]
    extra_modules = {
        item.strip()
        for item in os.getenv("TOOL_ALLOWED_MODULES", "").split(",")
        if item.strip()
    }
    if extra_modules:
        _ALLOWED_MODULES.update(extra_modules)
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"Import blocked: {name}")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


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


_SAFE_BUILTINS = {
    "__import__": _safe_import,
    "__build_class__": builtins.__build_class__,
    "__name__": "__tool__",
    "__file__": "<sandbox>",  # 添加：提供虚拟的 __file__ 路径
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,  # 添加：字符转换
    "compile": compile,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "len": len,
    "list": list,
    "map": map,
    "filter": filter,
    "iter": iter,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,  # 添加：字符转ASCII
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,  # 添加：反转迭代器
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,  # 添加：类型检查
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,  # 添加：索引错误
    "AttributeError": AttributeError,  # 添加：属性错误
    "AssertionError": AssertionError,
    "RuntimeError": RuntimeError,
    "OSError": OSError,
    "StopIteration": StopIteration,
    "SyntaxError": SyntaxError,
    "IndentationError": IndentationError,
    "TabError": TabError,
    "ImportError": ImportError,
    "ModuleNotFoundError": ModuleNotFoundError,
    "NameError": NameError,
    "TimeoutError": TimeoutError,
}


def _execute_tool_code(code: str, inputs: dict) -> Dict:
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS}
    exec(code, sandbox_globals)
    run_fn = sandbox_globals.get("run")
    if run_fn is None or not callable(run_fn):
        raise ValueError("Tool code must define callable run(inputs)")
    output = run_fn(inputs)
    if not isinstance(output, dict):
        raise TypeError("Tool run(inputs) must return dict")
    return output


def _sandbox_worker(
    code: str,
    inputs: dict,
    result_queue: mp.Queue,
    http_timeout_s: Optional[float],
) -> None:
    try:
        _patch_httpx_timeout(http_timeout_s)
        _patch_openai_timeout(http_timeout_s)
        output = _execute_tool_code(code, inputs)
        result_queue.put({"ok": True, "output": output, "error": None})
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "output": None,
                "error": {"code": type(exc).__name__, "message": str(exc)},
            }
        )


def _read_tool_http_timeout() -> Optional[float]:
    raw = str(os.getenv("TOOL_HTTP_TIMEOUT_S") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if value <= 0:
        return None
    return value


def _resolve_tool_http_timeout(execution_timeout_s: float) -> Optional[float]:
    env_timeout = _read_tool_http_timeout()
    if env_timeout is not None:
        return env_timeout
    try:
        value = float(execution_timeout_s)
    except Exception:
        return None
    if value <= 0:
        return None
    return value


def _patch_httpx_timeout(timeout_s: Optional[float]) -> None:
    if timeout_s is None:
        return
    try:
        import httpx  # type: ignore
    except Exception:
        return

    if getattr(httpx, "_TOOL_HTTP_TIMEOUT_PATCHED", False):
        return

    original_post = getattr(httpx, "post", None)
    if callable(original_post):
        def _patched_post(*args, **kwargs):
            kwargs["timeout"] = timeout_s
            return original_post(*args, **kwargs)

        httpx.post = _patched_post  # type: ignore[attr-defined]

    original_request = getattr(httpx, "request", None)
    if callable(original_request):
        def _patched_request(*args, **kwargs):
            kwargs["timeout"] = timeout_s
            return original_request(*args, **kwargs)

        httpx.request = _patched_request  # type: ignore[attr-defined]

    original_client = getattr(httpx, "Client", None)
    if isinstance(original_client, type):
        class _PatchedClient(original_client):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["timeout"] = timeout_s
                super().__init__(*args, **kwargs)

        _PatchedClient.__name__ = original_client.__name__
        httpx.Client = _PatchedClient  # type: ignore[attr-defined]

    original_async_client = getattr(httpx, "AsyncClient", None)
    if isinstance(original_async_client, type):
        class _PatchedAsyncClient(original_async_client):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["timeout"] = timeout_s
                super().__init__(*args, **kwargs)

        _PatchedAsyncClient.__name__ = original_async_client.__name__
        httpx.AsyncClient = _PatchedAsyncClient  # type: ignore[attr-defined]

    setattr(httpx, "_TOOL_HTTP_TIMEOUT_PATCHED", True)


def _patch_openai_timeout(timeout_s: Optional[float]) -> None:
    if timeout_s is None:
        return
    try:
        import openai  # type: ignore
        import httpx  # type: ignore
    except Exception:
        return

    if getattr(openai, "_TOOL_OPENAI_TIMEOUT_PATCHED", False):
        return

    trust_env = _env_bool("TOOL_TRUST_ENV", _env_bool("LLM_TRUST_ENV", True))

    original_openai = getattr(openai, "OpenAI", None)
    if isinstance(original_openai, type):
        class _PatchedOpenAI(original_openai):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["timeout"] = timeout_s
                kwargs.setdefault(
                    "http_client",
                    httpx.Client(timeout=timeout_s, http2=False, trust_env=trust_env),
                )
                super().__init__(*args, **kwargs)

        _PatchedOpenAI.__name__ = original_openai.__name__
        openai.OpenAI = _PatchedOpenAI  # type: ignore[attr-defined]

    original_async_openai = getattr(openai, "AsyncOpenAI", None)
    if isinstance(original_async_openai, type):
        class _PatchedAsyncOpenAI(original_async_openai):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["timeout"] = timeout_s
                kwargs.setdefault(
                    "http_client",
                    httpx.AsyncClient(timeout=timeout_s, http2=False, trust_env=trust_env),
                )
                super().__init__(*args, **kwargs)

        _PatchedAsyncOpenAI.__name__ = original_async_openai.__name__
        openai.AsyncOpenAI = _PatchedAsyncOpenAI  # type: ignore[attr-defined]

    setattr(openai, "_TOOL_OPENAI_TIMEOUT_PATCHED", True)


def execute_tool_code(code: str, inputs: dict, timeout_s: float = 1.0) -> Dict:
    result_queue: mp.Queue = mp.Queue()
    http_timeout_s = _resolve_tool_http_timeout(timeout_s)
    process = mp.Process(target=_sandbox_worker, args=(code, inputs, result_queue, http_timeout_s))
    process.start()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "ok": False,
            "output": None,
            "error": {"code": "timeout", "message": f"Execution exceeded {timeout_s}s"},
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "no_result", "message": "No result from sandbox"},
        }
