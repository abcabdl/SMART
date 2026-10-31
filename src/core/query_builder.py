from __future__ import annotations

from typing import Any, Optional

from core.constraints import Constraints


def _constraints_text(constraints: Optional[Constraints]) -> str:
    if constraints is None:
        return ""
    pieces = []
    if constraints.domain_tags:
        pieces.append(f"Domain tags: {', '.join(constraints.domain_tags)}")
    if constraints.role_tags:
        pieces.append(f"Role tags: {', '.join(constraints.role_tags)}")
    if constraints.tool_tags:
        pieces.append(f"Tool tags: {', '.join(constraints.tool_tags)}")
    if constraints.modalities:
        pieces.append(f"Modalities: {', '.join(constraints.modalities)}")
    if constraints.output_formats:
        pieces.append(f"Output formats: {', '.join(constraints.output_formats)}")
    if constraints.permissions:
        pieces.append(f"Permissions: {', '.join(constraints.permissions)}")
    if constraints.cost_tier:
        pieces.append(f"Cost tier: {constraints.cost_tier}")
    if constraints.latency_tier:
        pieces.append(f"Latency tier: {constraints.latency_tier}")
    if constraints.reliability_prior is not None:
        pieces.append(f"Reliability prior >= {constraints.reliability_prior}")
    return " | ".join(pieces)


def _artifact_context_text(artifact_context: Optional[Any]) -> str:
    if artifact_context is None:
        return ""
    if isinstance(artifact_context, str):
        text = " ".join(artifact_context.split())
        return text[:400]
    if not isinstance(artifact_context, dict):
        text = " ".join(str(artifact_context).split())
        return text[:400]

    parts = []
    scalar_keys = [
        "entry_point",
        "candidate_answer",
        "final_answer",
        "formatted_answer",
        "normalized_code",
        "draft_code",
    ]
    for key in scalar_keys:
        value = artifact_context.get(key)
        if value:
            text = " ".join(str(value).split())
            parts.append(f"{key}: {text[:160]}")

    verification = artifact_context.get("verification")
    if verification:
        parts.append(f"verification: {' '.join(str(verification).split())[:160]}")

    diagnostics = artifact_context.get("diagnostics")
    if diagnostics:
        parts.append(f"diagnostics: {' '.join(str(diagnostics).split())[:160]}")

    evidence_points = artifact_context.get("evidence_points")
    if isinstance(evidence_points, list) and evidence_points:
        snippets = []
        for item in evidence_points[:3]:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                sent = " ".join(str(item.get("sentence") or "").split())
                bit = f"{title}: {sent[:80]}".strip(": ")
                if bit:
                    snippets.append(bit)
            else:
                snippets.append(" ".join(str(item).split())[:80])
        if snippets:
            parts.append("evidence: " + " | ".join(snippets))

    return " | ".join(parts)[:600]


def build_role_query(
    task_text: str,
    role: str,
    constraints: Optional[Constraints] = None,
    artifact_context: Optional[Any] = None,
) -> str:
    role_key = role.strip().lower()
    role_key = {
        "researcher": "evidence_finder",
        "evidence-finder": "evidence_finder",
        "evidence-reader": "evidence_finder",
        "context-finder": "evidence_finder",
    }.get(role_key, role_key)
    if constraints is not None and not isinstance(constraints, Constraints):
        constraints = Constraints(**constraints)
    constraints_text = _constraints_text(constraints)
    artifact_text = _artifact_context_text(artifact_context)
    task_text = task_text.strip()

    if role_key in {"planner", "code-planner"}:
        template = (
            "You are a code planner. Break down the task into milestones, dependencies, and checkpoints. "
            "Focus on sequencing and feasibility. Task: {task}. {constraints}"
        )
    elif role_key == "evidence_finder":
        template = (
            "You are an evidence finder. Find useful facts only in the provided context, "
            "extract supporting evidence, and note missing evidence. Do not browse the web. "
            "Task: {task}. {constraints}"
        )
    elif role_key in {"builder", "code-generation"}:
        template = (
            "You are a code generator. Generate the solution implementation in code. "
            "Focus on correct logic, clear structure, and runnable output. Task: {task}. {constraints}"
        )
    elif role_key in {"tester", "checker", "code-testing"}:
        template = (
            "You are a code tester. Validate outputs, test assumptions, and look for edge cases or failures. "
            "Focus on correctness, coverage, and risks. Task: {task}. {constraints}"
        )
    elif role_key in {"refractor", "refactor", "code-refactoring"}:
        template = (
            "You are a code refactoring expert. Improve and refactor the solution. "
            "Fix errors, polish style, and restructure code for clarity and maintainability. "
            "Focus on minimal, correct changes. Task: {task}. {constraints}"
        )
    elif role_key == "manager":
        template = (
            "You are a manager. Coordinate roles, delegate tasks, and synthesize progress. "
            "Focus on sequencing and decision-making. Task: {task}. {constraints}"
        )
    else:
        template = "Role: {role}. Task: {task}. {constraints}"

    rendered = template.format(task=task_text, role=role, constraints=constraints_text).strip()
    if artifact_text:
        rendered = f"{rendered} Recent artifact context: {artifact_text}"
    return rendered.strip()
