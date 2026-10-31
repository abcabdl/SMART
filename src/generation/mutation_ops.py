from __future__ import annotations

from datetime import datetime
from typing import List

from core.cards import AgentCard


def _compose_embedding_text(
    role: str,
    domain: str,
    strengths: str,
    output_style: str,
    failure_modes: str,
    preferred_tools: List[str],
) -> str:
    tools = ", ".join(preferred_tools) if preferred_tools else "none"
    return (
        f"Role preference: {role}. "
        f"Domain: {domain}. "
        f"Strengths: {strengths}. "
        f"Output style: {output_style}. "
        f"Failure modes: {failure_modes}. "
        f"Preferred tools: {tools}."
    )


def _ensure_list(values: List[str]) -> List[str]:
    return list(values) if values else []


def mutate_agent(agent: AgentCard, op: str) -> AgentCard:
    operation = op.strip().lower()
    if operation not in {"role", "style", "toolset", "domain"}:
        raise ValueError(f"Unknown mutation op: {op}")

    now = datetime.utcnow()
    role = agent.role_tags[0] if agent.role_tags else "generalist"
    domain = agent.domain_tags[0] if agent.domain_tags else "general"
    output_style = "concise" if "json" in agent.output_formats else "detailed"
    strengths = "adaptable reasoning and quality control"
    failure_modes = "may be overcautious"

    new_role_tags = _ensure_list(agent.role_tags)
    new_domain_tags = _ensure_list(agent.domain_tags)
    new_output_formats = _ensure_list(agent.output_formats)
    new_tool_tags = _ensure_list(agent.tool_tags)
    new_available_tools = _ensure_list(agent.available_tool_ids)

    if operation == "role":
        role = f"{role}-alt"
        new_role_tags = [role]
    elif operation == "style":
        if "json" in new_output_formats:
            new_output_formats = ["markdown"]
            output_style = "structured"
        else:
            new_output_formats = ["json"]
            output_style = "concise"
    elif operation == "toolset":
        if new_available_tools:
            new_available_tools = new_available_tools[:-1]
        else:
            new_available_tools = ["tool-placeholder"]
        new_tool_tags = list(new_available_tools)
    elif operation == "domain":
        domain = f"{domain}-adjacent"
        new_domain_tags = [domain]

    preferred_tools = new_available_tools or new_tool_tags
    embedding_text = _compose_embedding_text(
        role,
        domain,
        strengths,
        output_style,
        failure_modes,
        preferred_tools,
    )

    lineage_note = f"mutation_type={operation}; parent_id={agent.id}"
    description = agent.description
    description = f"{description} | {lineage_note}" if description else lineage_note

    new_id = f"{agent.id}-{operation}-{int(now.timestamp())}"
    return AgentCard(
        id=new_id,
        name=f"{agent.name} ({operation})",
        kind="agent",
        version=f"{agent.version}-m1",
        updated_at=now,
        domain_tags=new_domain_tags,
        role_tags=new_role_tags,
        tool_tags=new_tool_tags,
        modalities=_ensure_list(agent.modalities),
        output_formats=new_output_formats,
        permissions=_ensure_list(agent.permissions),
        cost_tier=agent.cost_tier,
        latency_tier=agent.latency_tier,
        reliability_prior=agent.reliability_prior,
        description=description,
        examples=_ensure_list(agent.examples),
        embedding_text=embedding_text,
        available_tool_ids=new_available_tools,
    )
