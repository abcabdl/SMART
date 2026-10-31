from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_REQUIRED_FIELDS = {
    "planner": ["steps", "acceptance_criteria"],
    "evidence_finder": ["evidence_targets", "sources", "evidence_points"],
    "builder": ["runnable_plan", "code_or_commands", "self_test"],
    "refactor": ["runnable_plan", "code_or_commands", "self_test"],
    "checker": ["test_cases", "verdicts", "failure_localization"],
}

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


def _normalize_role(role: str) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIASES.get(key, key)


class MockLLMClient:
    """Mock LLM client for probe outputs."""

    def generate_probe(
        self,
        role: str,
        task_text: str,
        constraints: Optional[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        role_key = _normalize_role(role)
        brief_tags = candidate.get("brief_tags", {})
        tag_snippet = ", ".join(sum([brief_tags.get("domain_tags", []), brief_tags.get("tool_tags", [])], []))
        if role_key == "planner":
            return {
                "steps": ["Plan scope", f"Consider {tag_snippet}"],
                "acceptance_criteria": ["Meets constraints", "Sequenced steps"],
            }
        if role_key == "evidence_finder":
            return {
                "evidence_targets": [f"find provided-context evidence about {tag_snippet}"],
                "sources": ["provided_context", "retrieved_passages"],
                "evidence_points": [f"Evidence mentions {tag_snippet}"],
            }
        if role_key == "builder":
            return {
                "runnable_plan": ["Implement core changes", f"Integrate {tag_snippet}"],
                "code_or_commands": "run build",
                "self_test": ["unit tests", "smoke tests"],
            }
        if role_key == "refactor":
            return {
                "runnable_plan": ["Apply minimal refactor", f"Preserve behavior with {tag_snippet}"],
                "code_or_commands": "run refactor",
                "self_test": ["regression tests", "smoke tests"],
            }
        if role_key == "checker":
            return {
                "test_cases": ["happy path", f"constraint {tag_snippet}"],
                "verdicts": ["pass", "review"],
                "failure_localization": f"check {tag_snippet}",
            }
        return {"note": f"probe for {task_text}"}


def _flatten_constraints(constraints: Optional[Dict[str, Any]]) -> List[str]:
    if not constraints:
        return []
    values: List[str] = []
    for val in constraints.values():
        if isinstance(val, list):
            values.extend([str(item) for item in val])
        elif val is not None:
            values.append(str(val))
    return values


def _field_score(role: str, output: Dict[str, Any]) -> int:
    required = _REQUIRED_FIELDS.get(_normalize_role(role), [])
    score = 0
    for field in required:
        value = output.get(field)
        if isinstance(value, list) and value:
            score += 1
        elif isinstance(value, str) and value.strip():
            score += 1
    return score


def _constraint_score(constraints: Optional[Dict[str, Any]], output: Dict[str, Any]) -> int:
    values = _flatten_constraints(constraints)
    if not values:
        return 0
    blob = json.dumps(output, ensure_ascii=True).lower()
    hits = 0
    for value in values:
        if value.lower() in blob:
            hits += 1
    return hits


def probe_commit(
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    top_probe: int = 5,
    max_shadows: int = 2,
) -> Dict[str, Any]:
    if not candidates:
        return {"selected_main": None, "selected_shadows": [], "probe_scores": []}

    llm = MockLLMClient()
    scored: List[Dict[str, Any]] = []

    for candidate in candidates[:top_probe]:
        output = llm.generate_probe(role, task_text, constraints, candidate)
        score = _field_score(role, output) + _constraint_score(constraints, output)
        scored.append(
            {
                "card_id": candidate.get("card_id"),
                "score": score,
                "output": output,
                "base_score": candidate.get("score", 0.0),
            }
        )

    scored.sort(key=lambda item: (item["score"], item["base_score"]), reverse=True)

    selected_main = scored[0]["card_id"] if scored else None
    shadows = [item["card_id"] for item in scored[1 : 1 + max_shadows]]

    return {
        "selected_main": selected_main,
        "selected_shadows": shadows,
        "probe_scores": scored,
    }
