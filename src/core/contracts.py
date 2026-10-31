from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError


class PlannerOutput(BaseModel):
    steps: List[str] = []
    acceptance_criteria: List[str] = []
    ready_to_handoff: Optional[bool] = None
    next_role: Optional[str] = None
    final_answer: Optional[str] = None


class EvidenceFinderOutput(BaseModel):
    evidence_targets: List[str] = []
    sources: List[str] = []
    evidence_points: List[str] = []


class ResearcherOutput(EvidenceFinderOutput):
    search_queries: List[str] = []


class BuilderOutput(BaseModel):
    runnable_plan: List[str] = []
    code_or_commands: str = ""
    self_test: List[str] = []
    final_answer: Optional[str] = None


class CheckerOutput(BaseModel):
    test_cases: List[str] = []
    verdicts: List[str] = []
    failure_localization: str = ""
    final_answer: Optional[str] = None


class RefactorOutput(BaseModel):
    runnable_plan: List[str] = []
    code_or_commands: str = ""
    self_test: List[str] = []
    final_answer: Optional[str] = None


class ManagerOutput(BaseModel):
    status: str
    next_role: Optional[str] = None
    instruction: Optional[str] = None
    summary: Optional[str] = None
    notes: List[str] = []


_ROLE_MODELS: Dict[str, Type[BaseModel]] = {
    "planner": PlannerOutput,
    "evidence_finder": EvidenceFinderOutput,
    "researcher": ResearcherOutput,
    "builder": BuilderOutput,
    "checker": CheckerOutput,
    "refactor": RefactorOutput,
    "manager": ManagerOutput,
}

_ROLE_ALIASES: Dict[str, str] = {
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
    role_key = str(role or "").strip().lower()
    if not role_key:
        return ""
    return _ROLE_ALIASES.get(role_key, role_key)


def validate_output(
    role: str,
    data: Any,
    allow_unknown: bool = False,
) -> Tuple[bool, List[str], Optional[BaseModel]]:
    role_key = _normalize_role(role)
    model_cls = _ROLE_MODELS.get(role_key)
    if model_cls is None:
        if allow_unknown:
            if isinstance(data, dict):
                return True, [], None
            return False, [f"Unknown role '{role}' expects a JSON object"], None
        return False, [f"Unknown role: {role}"], None

    if isinstance(data, model_cls):
        return True, [], data

    try:
        model = model_cls(**(data or {}))
    except ValidationError as exc:
        errors = [f"{err['loc']}: {err['msg']}" for err in exc.errors()]
        return False, errors, None

    return True, [], model
