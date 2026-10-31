from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "nodes") and hasattr(obj, "edges"):
        try:
            return {"nodes": list(obj.nodes()), "edges": list(obj.edges())}
        except Exception:
            pass
    return str(obj)


def write_event(
    task_id: str,
    workflow_version: str,
    role: str,
    selected_main: Optional[str],
    selected_shadow: Optional[str],
    candidates_topk: Any,
    output: Any,
    validation: Any,
    executor_result: Any,
    failure_type: Optional[str],
    action: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
    runs_dir: str = "runs",
) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{task_id}.jsonl")
    record: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "workflow_version": workflow_version,
        "role": role,
        "selected_main": selected_main,
        "selected_shadow": selected_shadow,
        "candidates_topk": candidates_topk,
        "output": output,
        "validation": validation,
        "executor_result": executor_result,
        "failure_type": failure_type,
        "action": action,
    }
    if meta:
        record["meta"] = meta
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=_json_fallback) + "\n")
    return path
