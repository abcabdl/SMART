from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from core.cards import AgentCard
from core.registry import SQLiteRegistry
from generation.tool_generator import register_tool_from_spec


_CODE_CANDIDATE_EXTRACTOR = r'''def run(inputs):
    output = inputs.get("output") or inputs.get("candidate") or inputs.get("artifact_bag") or {}
    if not isinstance(output, dict):
        output = {"text": str(output)}
    artifact = inputs.get("artifact_bag") if isinstance(inputs.get("artifact_bag"), dict) else {}
    code = (
        output.get("normalized_code")
        or output.get("draft_code")
        or output.get("code_or_commands")
        or artifact.get("normalized_code")
        or artifact.get("draft_code")
        or ""
    )
    answer = output.get("final_answer") or output.get("candidate_answer") or artifact.get("final_answer") or artifact.get("candidate_answer") or ""
    return {"normalized_code": str(code).strip(), "candidate_answer": str(answer).strip()}
'''


_ANSWER_NORMALIZER = r'''import re
from decimal import Decimal, InvalidOperation


def _normalize(value):
    text = str(value or "").strip()
    matches = re.findall(r"####\s*([^\n\r]+)", text)
    if matches:
        text = matches[-1].strip()
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    try:
        return str(Decimal(text.replace(",", "")).normalize())
    except (InvalidOperation, ValueError):
        return text.lower()


def run(inputs):
    value = inputs.get("answer") or inputs.get("candidate_answer") or inputs.get("final_answer") or inputs.get("text") or ""
    return {"normalized_answer": _normalize(value), "answer": str(value).strip()}
'''


_CONSENSUS_VOTER = r'''import re
from decimal import Decimal, InvalidOperation


def _normalize(value):
    text = str(value or "").strip()
    matches = re.findall(r"####\s*([^\n\r]+)", text)
    if matches:
        text = matches[-1].strip()
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    try:
        return str(Decimal(text.replace(",", "")).normalize())
    except (InvalidOperation, ValueError):
        return text.lower()


def run(inputs):
    candidates = inputs.get("candidates") or []
    groups = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        answer = item.get("answer") or item.get("final_answer") or item.get("candidate_answer") or ""
        key = _normalize(answer)
        if not key:
            continue
        groups.setdefault(key, {"answer": answer, "count": 0, "branches": []})
        groups[key]["count"] += 1
        if item.get("branch_id"):
            groups[key]["branches"].append(item.get("branch_id"))
    if not groups:
        return {"consensus": None, "votes": []}
    winner = sorted(groups.items(), key=lambda item: (-item[1]["count"], str(item[0])))[0]
    return {"consensus": winner[1]["answer"], "normalized": winner[0], "votes": list(groups.values())}
'''


_BRANCH_DIAGNOSTICS = r'''def run(inputs):
    branches = inputs.get("branches") or []
    diagnostics = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        error = branch.get("error") or {}
        output = branch.get("output") or {}
        diagnostics.append({
            "branch_id": branch.get("branch_id"),
            "role": branch.get("role"),
            "selected_main": branch.get("selected_main"),
            "error": error or output.get("error"),
            "test_passed": bool(output.get("test_passed") or output.get("status") == "success"),
        })
    return {"diagnostics": diagnostics}
'''


_RESULT_SELECTOR = r'''def run(inputs):
    branches = inputs.get("branches") or []
    for branch in branches:
        output = branch.get("output") if isinstance(branch, dict) else {}
        if isinstance(output, dict) and (output.get("test_passed") or output.get("status") == "success"):
            return {"winner_branch_id": branch.get("branch_id"), "reason": "test_passed"}
    for branch in branches:
        if isinstance(branch, dict) and not branch.get("error"):
            return {"winner_branch_id": branch.get("branch_id"), "reason": "first_non_error"}
    return {"winner_branch_id": branches[0].get("branch_id") if branches else None, "reason": "fallback"}
'''


def _tool_specs(dataset_family: str) -> List[tuple[dict, str, str]]:
    common = {
        "domain_tags": ["parallel", "benchmark", dataset_family],
        "role_tags": ["checker", "aggregator"],
        "modalities": ["text"],
        "output_formats": ["json"],
        "permissions": ["read"],
        "dataset_family": dataset_family,
        "deterministic": True,
        "cost_tier": "low",
        "latency_tier": "low",
        "reliability_prior": 0.9,
    }
    return [
        (
            {
                **common,
                "name": "ParallelCodeCandidateExtractor",
                "description": "Extract code and answer candidates from branch artifacts.",
                "tool_tags": ["parallel", "candidate-extractor"],
                "capability_kind": "candidate-extractor",
                "stage": "aggregate",
                "consumes": ["branch_outputs"],
                "produces": ["normalized_code", "candidate_answer"],
            },
            "parallel-code-candidate-extractor",
            _CODE_CANDIDATE_EXTRACTOR,
        ),
        (
            {
                **common,
                "name": "ParallelAnswerNormalizer",
                "description": "Normalize code or math answers for voting.",
                "tool_tags": ["parallel", "answer-normalizer"],
                "capability_kind": "answer-normalizer",
                "stage": "aggregate",
                "consumes": ["candidate_answer"],
                "produces": ["normalized_answer"],
            },
            "parallel-answer-normalizer",
            _ANSWER_NORMALIZER,
        ),
        (
            {
                **common,
                "name": "ParallelConsensusVoter",
                "description": "Group equivalent branch answers and select a consensus answer.",
                "tool_tags": ["parallel", "consensus-voter"],
                "capability_kind": "consensus-voter",
                "stage": "aggregate",
                "consumes": ["candidate_answers"],
                "produces": ["consensus"],
            },
            "parallel-consensus-voter",
            _CONSENSUS_VOTER,
        ),
        (
            {
                **common,
                "name": "ParallelBranchDiagnostics",
                "description": "Summarize branch failures for debugging and paper traces.",
                "tool_tags": ["parallel", "branch-diagnostics"],
                "capability_kind": "branch-diagnostics",
                "stage": "diagnose",
                "consumes": ["branch_outputs"],
                "produces": ["diagnostics"],
            },
            "parallel-branch-diagnostics",
            _BRANCH_DIAGNOSTICS,
        ),
        (
            {
                **common,
                "name": "ParallelResultSelector",
                "description": "Select a final branch result by deterministic policy.",
                "tool_tags": ["parallel", "result-selector"],
                "capability_kind": "result-selector",
                "stage": "aggregate",
                "consumes": ["branches"],
                "produces": ["winner_branch_id"],
            },
            "parallel-result-selector",
            _RESULT_SELECTOR,
        ),
    ]


def _agent(
    *,
    agent_id: str,
    name: str,
    role_tags: List[str],
    domain_tags: List[str],
    tool_ids: List[str],
    dataset_family: str,
    description: str,
) -> AgentCard:
    now = datetime.utcnow()
    return AgentCard(
        id=agent_id,
        name=name,
        kind="agent",
        version="1.0",
        updated_at=now,
        domain_tags=domain_tags + ["parallel", dataset_family],
        role_tags=role_tags,
        tool_tags=list(tool_ids),
        modalities=["text"],
        output_formats=["json"],
        permissions=["read"],
        cost_tier="low",
        latency_tier="low",
        reliability_prior=0.82,
        description=description,
        examples=["Run as one independent branch in a parallel benchmark workflow."],
        embedding_text=f"{name}. {description}. Roles: {', '.join(role_tags)}. Tools: {', '.join(tool_ids)}.",
        available_tool_ids=list(tool_ids),
        dataset_family=dataset_family,
    )


def register_parallel_catalog(
    registry: SQLiteRegistry,
    dataset_family: str = "benchmark",
    skip_existing: bool = True,
) -> dict:
    tool_ids: List[str] = []
    for spec, tool_id, code in _tool_specs(dataset_family):
        card = register_tool_from_spec(
            registry=registry,
            spec=spec,
            code=code,
            tool_id=tool_id,
            skip_existing=skip_existing,
        )
        tool_ids.append(card.id)

    agents = [
        _agent(
            agent_id="agent-parallel-code-precise-builder",
            name="ParallelCodePreciseBuilder",
            role_tags=["builder", "code-generation", "parallel-worker"],
            domain_tags=["code", "humaneval", "mbpp"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Precise code builder for minimal benchmark implementations.",
        ),
        _agent(
            agent_id="agent-parallel-code-creative-builder",
            name="ParallelCodeCreativeBuilder",
            role_tags=["builder", "code-generation", "parallel-worker"],
            domain_tags=["code", "humaneval", "mbpp"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Alternative code builder that explores a different implementation strategy.",
        ),
        _agent(
            agent_id="agent-parallel-code-repair-refactor",
            name="ParallelCodeRepairRefactor",
            role_tags=["refactor", "code-refactoring", "parallel-worker"],
            domain_tags=["code", "humaneval", "mbpp"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Repair and refactor branch for candidate code.",
        ),
        _agent(
            agent_id="agent-parallel-code-edge-checker",
            name="ParallelCodeEdgeChecker",
            role_tags=["checker", "code-testing", "parallel-verifier"],
            domain_tags=["code", "humaneval", "mbpp"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Edge-case checker for branch validation.",
        ),
        _agent(
            agent_id="agent-parallel-math-symbolic-solver",
            name="ParallelMathSymbolicSolver",
            role_tags=["builder", "math-solver", "parallel-worker"],
            domain_tags=["math", "gsm8k"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Math branch for direct symbolic or arithmetic solving.",
        ),
        _agent(
            agent_id="agent-parallel-math-verifier",
            name="ParallelMathVerifier",
            role_tags=["checker", "math-verifier", "parallel-verifier"],
            domain_tags=["math", "gsm8k"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Math branch for independent verification and answer normalization.",
        ),
        _agent(
            agent_id="agent-parallel-consensus-aggregator",
            name="ParallelConsensusAggregator",
            role_tags=["checker", "aggregator", "parallel-aggregator"],
            domain_tags=["parallel", "benchmark"],
            tool_ids=tool_ids,
            dataset_family=dataset_family,
            description="Deterministic aggregation agent for parallel branch outputs.",
        ),
    ]

    registered_agents: List[str] = []
    for agent in agents:
        if skip_existing and registry.get(agent.id) is not None:
            registered_agents.append(agent.id)
            continue
        registry.register(agent)
        registered_agents.append(agent.id)

    return {"agents": registered_agents, "tools": tool_ids}
