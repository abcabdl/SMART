from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from core.cards import AgentCard
from core.registry import SQLiteRegistry


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned or "domain"


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


def generate_agents(
    domains: list[str],
    n_per_domain: int,
    roles: list[str],
    registry: SQLiteRegistry,
    skip_existing: bool = False,
) -> List[AgentCard]:
    if not domains:
        raise ValueError("domains must be non-empty")
    if not roles:
        raise ValueError("roles must be non-empty")
    if n_per_domain <= 0:
        return []

    tool_ids = [card.id for card in registry.list({"kind": "tool"})]
    agents: List[AgentCard] = []
    role_count = len(roles)
    idx = 0

    for domain in domains:
        for i in range(n_per_domain):
            role = roles[idx % role_count]
            idx += 1
            output_style = "concise" if i % 2 == 0 else "detailed"
            strengths = "structured reasoning and data synthesis"
            failure_modes = "may overgeneralize or miss edge cases"
            preferred_tools = list(tool_ids)
            embedding_text = _compose_embedding_text(
                role,
                domain,
                strengths,
                output_style,
                failure_modes,
                preferred_tools,
            )
            agent_id = f"agent-{_slug(domain)}-{_slug(role)}-{i + 1}"
            name = f"{domain.title()} {role.title()} {i + 1}"
            description = (
                f"Role: {role}; Domain: {domain}; Output style: {output_style}. "
                f"Strengths: {strengths}. Failure modes: {failure_modes}. "
                f"Preferred tools: {', '.join(preferred_tools) if preferred_tools else 'none'}."
            )
            output_formats = ["json"] if output_style == "concise" else ["markdown"]
            agent = AgentCard(
                id=agent_id,
                name=name,
                kind="agent",
                version="1.0",
                updated_at=datetime.utcnow(),
                domain_tags=[domain],
                role_tags=[role],
                tool_tags=list(preferred_tools),
                modalities=["text"],
                output_formats=output_formats,
                permissions=["read"],
                cost_tier="low",
                latency_tier="medium",
                reliability_prior=0.75,
                description=description,
                examples=[f"Handle {domain} requests"],
                embedding_text=embedding_text,
                available_tool_ids=list(preferred_tools),
            )
            if skip_existing and registry.get(agent_id) is not None:
                continue
            registry.register(agent)
            agents.append(agent)
    return agents


def generate_agents_from_themes(
    themes: List[dict],
    n_per_theme: int,
    roles: List[str],
    registry: SQLiteRegistry,
    tools_by_theme: Optional[Dict[str, List[str]]] = None,
    skip_existing: bool = False,
) -> List[AgentCard]:
    if not themes:
        raise ValueError("themes must be non-empty")
    if not roles:
        raise ValueError("roles must be non-empty")
    if n_per_theme <= 0:
        return []

    all_tool_ids = [card.id for card in registry.list({"kind": "tool"})]
    tools_by_theme = tools_by_theme or {}
    agents: List[AgentCard] = []
    role_count = len(roles)
    idx = 0

    for theme in themes:
        theme_id = str(theme.get("id") or theme.get("name") or "theme").strip()
        theme_name = str(theme.get("name") or theme_id).strip()
        domain_tags = list(theme.get("domain_tags") or []) or [_slug(theme_name)]
        theme_tools = tools_by_theme.get(theme_id) or tools_by_theme.get(theme_name)
        if theme_tools is None:
            theme_tools = [] if tools_by_theme else all_tool_ids
        for i in range(n_per_theme):
            role = roles[idx % role_count]
            idx += 1
            output_style = "concise" if i % 2 == 0 else "detailed"
            strengths = str(theme.get("strengths") or "structured reasoning and data synthesis")
            failure_modes = str(theme.get("failure_modes") or "may overgeneralize or miss edge cases")
            preferred_tools = list(theme_tools)
            embedding_text = _compose_embedding_text(
                role,
                theme_name,
                strengths,
                output_style,
                failure_modes,
                preferred_tools,
            )
            agent_id = f"agent-{_slug(theme_name)}-{_slug(role)}-{i + 1}"
            name = f"{theme_name.title()} {role.title()} {i + 1}"
            description = (
                f"Role: {role}; Theme: {theme_name}; Domain: {', '.join(domain_tags)}; "
                f"Output style: {output_style}. Strengths: {strengths}. "
                f"Failure modes: {failure_modes}. Preferred tools: "
                f"{', '.join(preferred_tools) if preferred_tools else 'none'}."
            )
            output_formats = ["json"] if output_style == "concise" else ["markdown"]
            agent = AgentCard(
                id=agent_id,
                name=name,
                kind="agent",
                version="1.0",
                updated_at=datetime.utcnow(),
                domain_tags=list(domain_tags),
                role_tags=[role],
                tool_tags=list(preferred_tools),
                modalities=["text"],
                output_formats=output_formats,
                permissions=["read"],
                cost_tier="low",
                latency_tier="medium",
                reliability_prior=0.75,
                description=description,
                examples=[f"Handle {theme_name} requests"],
                embedding_text=embedding_text,
                available_tool_ids=list(preferred_tools),
            )
            if skip_existing and registry.get(agent_id) is not None:
                continue
            registry.register(agent)
            agents.append(agent)
    return agents


def generate_agents_from_profiles(
    theme: Dict[str, Any],
    profiles: List[Dict[str, Any]],
    roles: List[str],
    registry: SQLiteRegistry,
    skip_existing: bool = False,
    start_index: int = 0,
) -> List[AgentCard]:
    if not profiles:
        return []
    if not roles:
        raise ValueError("roles must be non-empty")

    theme_name = str(theme.get("name") or theme.get("id") or "theme").strip()
    domain_tags = list(theme.get("domain_tags") or []) or [_slug(theme_name)]
    agents: List[AgentCard] = []
    role_count = len(roles)
    idx = 0

    for i, profile in enumerate(profiles, start=start_index):
        role = str(profile.get("role") or "").strip()
        if not role:
            role = roles[idx % role_count]
        idx += 1
        output_style = str(profile.get("output_style") or "").strip() or ("concise" if i % 2 == 0 else "detailed")
        strengths = str(profile.get("strengths") or "structured reasoning and data synthesis")
        failure_modes = str(profile.get("failure_modes") or "may overgeneralize or miss edge cases")
        tool_ids = list(profile.get("tool_ids") or profile.get("tools") or [])
        focus_tags = list(profile.get("focus_tags") or [])
        name = str(profile.get("name") or f"{theme_name} {role} {i + 1}").strip()
        description = str(profile.get("description") or "")
        if not description:
            description = (
                f"Role: {role}; Theme: {theme_name}; Domain: {', '.join(domain_tags)}; "
                f"Focus: {', '.join(focus_tags) if focus_tags else 'general'}."
            )
        embedding_text = _compose_embedding_text(
            role,
            theme_name,
            strengths,
            output_style,
            failure_modes,
            tool_ids,
        )
        agent_id = f"agent-{_slug(theme_name)}-{_slug(role)}-{i + 1}"
        output_formats = ["json"] if output_style == "concise" else ["markdown"]
        agent = AgentCard(
            id=agent_id,
            name=name,
            kind="agent",
            version="1.0",
            updated_at=datetime.utcnow(),
            domain_tags=list(domain_tags),
            role_tags=[role],
            tool_tags=list(tool_ids),
            modalities=["text"],
            output_formats=output_formats,
            permissions=["read"],
            cost_tier="low",
            latency_tier="medium",
            reliability_prior=0.75,
            description=description,
            examples=[f"Handle {theme_name} requests"],
            embedding_text=embedding_text,
            available_tool_ids=list(tool_ids),
        )
        if skip_existing and registry.get(agent_id) is not None:
            continue
        registry.register(agent)
        agents.append(agent)
    return agents
