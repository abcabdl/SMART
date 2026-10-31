from __future__ import annotations

from typing import Any, Dict, List, Optional

_ROLE_ALIASES = {
    "code-generation": "builder",
    "code-planner": "planner",
    "code-testing": "checker",
    "tester": "checker",
    "code-refactoring": "refactor",
    "refractor": "refactor",
    "researcher": "evidence_finder",
    "evidence-finder": "evidence_finder",
    "evidence-reader": "evidence_finder",
    "context-finder": "evidence_finder",
}


def _is_empty_list(value: Any) -> bool:
    return not value or (isinstance(value, list) and len(value) == 0)


def _normalize_role(role: str) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIASES.get(key, key)


def _missing_required_fields(role: str, output: Dict[str, Any]) -> bool:
    role_key = _normalize_role(role)
    if role_key == "planner":
        return _is_empty_list(output.get("steps")) or _is_empty_list(output.get("acceptance_criteria"))
    if role_key == "evidence_finder":
        return (
            _is_empty_list(output.get("evidence_targets") or output.get("search_queries"))
            or _is_empty_list(output.get("sources"))
            or _is_empty_list(output.get("evidence_points"))
        )
    if role_key in {"builder", "refactor"}:
        return (
            _is_empty_list(output.get("runnable_plan"))
            or not output.get("code_or_commands")
            or _is_empty_list(output.get("self_test"))
        )
    if role_key == "checker":
        return (
            _is_empty_list(output.get("test_cases"))
            or _is_empty_list(output.get("verdicts"))
            or not output.get("failure_localization")
        )
    return True


def route_failure(
    role: str,
    output: Optional[Dict[str, Any]],
    validation_errors: Optional[List[str]],
    executor_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    if validation_errors:
        return {"failure_type": "A_contract", "action": "rewrite_format"}

    if executor_result is not None and not executor_result.get("ok", True):
        return {"failure_type": "C_capability", "action": "swap_agent"}

    if output is None:
        return {"failure_type": "B_missing_info", "action": "request_more_info"}

    if _missing_required_fields(role, output):
        return {"failure_type": "B_missing_info", "action": "request_more_info"}

    return {"failure_type": "C_capability", "action": "workflow_update"}

