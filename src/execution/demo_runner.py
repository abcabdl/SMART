from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure `core`, `generation`, etc. are importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from execution.orchestrator import run_workflow
from generation.agent_generator import generate_agents, generate_agents_from_profiles, generate_agents_from_themes
from generation.functional_tool_catalog import code_from_record, select_functional_tools, spec_from_record
from generation.llm_client import LLMClient
from generation.parallel_agent_catalog import register_parallel_catalog
from generation.tool_generator import generate_tool_code, register_tool, register_tool_from_spec
from generation.toolbench_catalog import endpoint_to_code, endpoint_to_spec, select_toolbench_endpoints
from generation.tool_planner import (
    plan_agent_profiles_for_theme,
    plan_agent_themes,
    plan_tools_for_theme,
    themes_from_domains,
)
from retrieval.build_index import build_index
from retrieval.embedder import build_embedder

DEFAULT_DOMAINS = [
    "GSM8K Math Reasoning",
    "Competition Math",
    "HumanEval Code Generation",
    "MBPP Code Tasks",
]
# DEFAULT_DOMAINS = [
#     "Code Generation ",
#     "Code Refactoring",
#     "Code Testing",
#     "Mathematics & Scientific Calc",
#     "Math & Problem Solving",
#     "Commonsense & QA",
#     "Knowledge & Reasoning",
#     "Planning & Task Decomposition",
# ]

# Guidance injected into tool planning so generated tools call the correct API endpoint.
TOOL_PLANNING_GUIDANCE = (
    "Tool spec guidance: if a tool requires an external LLM API, set requires_api=true and "
    "set api_notes to instruct: use OpenAI-compatible Chat Completions at "
    "${LLM_API_BASE}/chat/completions with Authorization: Bearer ${LLM_API_KEY} "
    "and model gpt-4o. "
    "Do NOT invent custom endpoints like /generate_algorithm. Prefer returning generated code "
    "as a string in a code_or_commands field."
)

def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().strip().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned or "item"

def _normalize_tags(values: object) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return [str(values)]
    tags: list[str] = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            tags.append(text)
    return tags


def _append_planning_guidance(base: str, guidance: str) -> str:
    base = (base or "").strip()
    if base:
        return f"{base}\n\n{guidance}"
    return guidance


def _should_require_llm_api(spec: dict) -> bool:
    text_parts: list[str] = []
    for key in ("name", "description"):
        value = spec.get(key)
        if value:
            text_parts.append(str(value))
    for key in ("domain_tags", "tool_tags", "role_tags", "inputs", "outputs"):
        text_parts.extend(_normalize_tags(spec.get(key)))
    hay = " ".join(text_parts).lower()
    keywords = [
        "code",
        "program",
        "refactor",
        "rewrite",
        "generator",
        "generate",
        "math",
        "algebra",
        "equation",
        "reason",
        "logic",
        "qa",
        "question",
        "answer",
        "summarize",
        "explain",
    ]
    return any(keyword in hay for keyword in keywords)


def _enforce_llm_api_specs(specs: list[dict]) -> None:
    for spec in specs:
        needs_api = bool(spec.get("requires_api")) or _should_require_llm_api(spec)
        if not needs_api:
            continue
        spec["requires_api"] = True
        requirements = str(spec.get("requirements") or "").strip()
        if "httpx" not in requirements.lower():
            requirements = (requirements + ", httpx").strip(", ").strip()
        if "llm" not in requirements.lower():
            requirements = (requirements + ", LLM API").strip(", ").strip()
        spec["requirements"] = requirements
        api_notes = str(spec.get("api_notes") or "").strip()
        must_include = (
            "Use ${LLM_API_BASE}/chat/completions with Authorization: Bearer ${LLM_API_KEY} "
            "and model gpt-4o. Do not invent custom endpoints."
        )
        if not api_notes:
            spec["api_notes"] = must_include
        elif "chat/completions" not in api_notes.lower() or "gpt-4o" not in api_notes.lower():
            spec["api_notes"] = f"{api_notes} {must_include}".strip()


def _score_tool(profile: dict, tool: dict) -> float:
    focus = set(_normalize_tags(profile.get("focus_tags")))
    preferred = set(_normalize_tags(profile.get("preferred_tool_tags")))
    tool_tags = set(_normalize_tags(tool.get("tool_tags"))) | set(_normalize_tags(tool.get("domain_tags")))
    score = 0.0
    score += 2.0 * len(preferred & tool_tags)
    score += 1.0 * len(focus & tool_tags)
    name = str(tool.get("name") or "").lower()
    desc = str(tool.get("description") or "").lower()
    for tag in focus | preferred:
        tag_l = tag.lower()
        if tag_l and (tag_l in name or tag_l in desc):
            score += 0.5
    return score


def _select_tools_for_profiles(
    profiles: list[dict],
    tools: list[dict],
    tools_per_agent: int,
) -> list[dict]:
    if not profiles:
        return profiles
    if not tools:
        for profile in profiles:
            profile["tool_ids"] = []
        return profiles
    limit = max(1, tools_per_agent) if tools_per_agent else len(tools)
    usage: dict[str, int] = {tool["id"]: 0 for tool in tools if tool.get("id")}
    for profile in profiles:
        scored = []
        for tool in tools:
            tool_id = tool.get("id")
            if not tool_id:
                continue
            base = _score_tool(profile, tool)
            penalty = 0.1 * usage.get(tool_id, 0)
            scored.append((base - penalty, usage.get(tool_id, 0), tool.get("name") or "", tool_id))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: list[str] = []
        for _, _, _, tool_id in scored:
            if tool_id in selected:
                continue
            selected.append(tool_id)
            if len(selected) >= limit:
                break
        if not selected:
            selected = [tool["id"] for tool in tools[:limit] if tool.get("id")]
        for tool_id in selected:
            usage[tool_id] = usage.get(tool_id, 0) + 1
        profile["tool_ids"] = selected
    return profiles


def _plan_tools_batched(
    planning_text: str,
    theme: dict,
    total_tools: int,
    batch_size: int,
    llm: LLMClient,
    debug_llm: bool,
    debug_dir: str | None,
    verbose: bool,
) -> list[dict]:
    if total_tools <= 0:
        return []
    if batch_size <= 0:
        batch_size = total_tools
    collected: list[dict] = []
    seen: set[str] = set()
    remaining = total_tools
    batch_idx = 1
    while remaining > 0:
        current = min(remaining, batch_size)
        if verbose and total_tools > batch_size:
            print(
                f"[progress] tool batch {batch_idx} (target={current}, remaining={remaining})",
                flush=True,
            )
        specs = plan_tools_for_theme(
            planning_text,
            theme,
            current,
            llm,
            debug=debug_llm,
            debug_dir=debug_dir,
            avoid_names=sorted(seen),
            debug_label=f"plan_tools_{_slug(str(theme.get('name') or theme.get('id') or 'theme'))}_b{batch_idx}",
        )
        if not specs:
            break
        for spec in specs:
            name = str(spec.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            collected.append(spec)
        remaining = total_tools - len(collected)
        batch_idx += 1
        if len(specs) == 0:
            break
    return collected


def _chunk_list(items: list[dict], batch_size: int) -> list[list[dict]]:
    if not items:
        return []
    if not batch_size or batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _build_default_profiles(theme_id: str, roles: list[str], agents_per_theme: int, theme_slug: str) -> list[dict]:
    profiles: list[dict] = []
    if not roles:
        return profiles
    role_focus = {
        "planner": ["analysis", "decompose", "strategy"],
        "builder": ["implementation", "execution", "synthesis"],
        "checker": ["verification", "tests", "debug"],
        "refactor": ["cleanup", "repair", "normalization"],
    }
    for i in range(agents_per_theme):
        role = roles[i % len(roles)]
        focus_tags = [role, theme_slug]
        focus_tags.extend(role_focus.get(role, []))
        profiles.append(
            {
                "name": f"{theme_id} {role} {i + 1}",
                "description": f"Role: {role}; Theme: {theme_id}; Focus: {', '.join(focus_tags)}.",
                "focus_tags": focus_tags,
                "preferred_tool_tags": [theme_slug, role],
                "role": role,
            }
        )
    return profiles


def _plan_profile_batches(
    *,
    theme: dict,
    total: int,
    roles: list[str],
    llm: LLMClient,
    batch_size: int,
    debug_llm: bool,
    debug_dir: str | None,
) -> list[list[dict]]:
    if total <= 0:
        return []
    if not batch_size or batch_size <= 0:
        profiles = plan_agent_profiles_for_theme(
            theme,
            total,
            roles,
            llm,
            debug=debug_llm,
            debug_dir=debug_dir,
            batch_size=0,
        )
        return [profiles] if profiles else []
    batches: list[list[dict]] = []
    seen: set[str] = set()
    remaining = total
    stall = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        profiles = plan_agent_profiles_for_theme(
            theme,
            current,
            roles,
            llm,
            debug=debug_llm,
            debug_dir=debug_dir,
            batch_size=0,
        )
        if not profiles:
            break
        unique: list[dict] = []
        for profile in profiles:
            name = str(profile.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            unique.append(profile)
        if not unique:
            stall += 1
            if stall >= 2:
                break
            continue
        batches.append(unique)
        remaining -= len(unique)
        stall = 0
    return batches


def _toolbench_request_text(base_text: str, theme: dict, profile: dict | None = None) -> str:
    parts = [
        base_text,
        str(theme.get("name") or ""),
        str(theme.get("description") or ""),
        " ".join(_normalize_tags(theme.get("domain_tags"))),
        " ".join(_normalize_tags(theme.get("capability_hints"))),
    ]
    if profile:
        parts.extend(
            [
                str(profile.get("name") or ""),
                str(profile.get("description") or ""),
                " ".join(_normalize_tags(profile.get("focus_tags"))),
                " ".join(_normalize_tags(profile.get("preferred_tool_tags"))),
                str(profile.get("role") or ""),
            ]
        )
    return "\n".join(part for part in parts if part)


def _register_toolbench_cards(
    *,
    registry: SQLiteRegistry,
    selected: list[dict],
    tool_output_dir: str | None,
    skip_existing: bool,
) -> tuple[list[str], list[dict]]:
    tool_ids: list[str] = []
    tools_meta: list[dict] = []
    for endpoint in selected:
        spec = endpoint_to_spec(endpoint)
        code = endpoint_to_code(endpoint)
        card = register_tool_from_spec(
            registry,
            spec,
            code,
            tool_id=endpoint["tool_id"],
            save_dir=tool_output_dir,
            skip_existing=skip_existing,
        )
        tool_ids.append(card.id)
        tools_meta.append(
            {
                "id": card.id,
                "name": card.name,
                "tool_tags": spec.get("tool_tags"),
                "domain_tags": spec.get("domain_tags"),
                "description": spec.get("description"),
            }
        )
    return tool_ids, tools_meta


def _register_functional_cards(
    *,
    registry: SQLiteRegistry,
    selected: list[dict],
    tool_output_dir: str | None,
    skip_existing: bool,
) -> tuple[list[str], list[dict]]:
    tool_ids: list[str] = []
    tools_meta: list[dict] = []
    for record in selected:
        spec = spec_from_record(record)
        code = code_from_record(record)
        card = register_tool_from_spec(
            registry,
            spec,
            code,
            tool_id=record["tool_id"],
            save_dir=tool_output_dir,
            skip_existing=skip_existing,
        )
        tool_ids.append(card.id)
        tools_meta.append(
            {
                "id": card.id,
                "name": card.name,
                "tool_tags": spec.get("tool_tags"),
                "domain_tags": spec.get("domain_tags"),
                "description": spec.get("description"),
            }
        )
    return tool_ids, tools_meta


def _emit_specs_payload(
    payload: dict,
    *,
    print_specs: bool,
    spec_out_dir: str | None,
    filename: str,
    verbose: bool,
) -> None:
    if not print_specs:
        return
    if spec_out_dir:
        os.makedirs(spec_out_dir, exist_ok=True)
        out_path = os.path.join(spec_out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        if verbose:
            print(f"[progress] saved specs: {out_path}", flush=True)
        return
    print(json.dumps(payload, ensure_ascii=True, indent=2), flush=True)


def run_demo(
    task_text: str = "Summarize quarterly performance and risks",
    db_path: str = "demo_registry.sqlite",
    index_dir: str = "./index",
    tool_count: int = 20,
    domains: list[str] | None = None,
    n_per_domain: int = 40,
    roles: list[str] | None = None,
    dim: int = 64,
    seed: int = 7,
    embedder_kind: str = "sentence-transformer",
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedder_device: str | None = None,
    embedder_normalize: bool = False,
    workflow_version: str = "v1",
    real_tools: bool = False,
    theme_count: int = 5,
    tools_per_theme: int = 5,
    agents_per_theme: int = 10,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: float = 60.0,
    llm_retries: int = 2,
    tool_output_dir: str | None = None,
    plan_themes: bool = False,
    plan_agents: bool = True,
    domain_start: int = 0,
    domain_limit: int = 0,
    tools_per_agent: int = 5,
    tools_per_profile: int = 0,
    verbose: bool = False,
    debug_llm: bool = False,
    debug_llm_dir: str | None = None,
    tools_batch_size: int = 10,
    profiles_batch_size: int = 0,
    reset_db: bool = False,
    print_specs: bool = False,
    spec_out_dir: str | None = None,
    enforce_llm_api: bool = True,
    tool_source: str = "functional",
    toolbench_dir: str = "tools",
    paper_mode: bool = False,
    task_family: str | None = None,
    dataset: str | None = None,
    parallel_execution: bool = False,
    parallel_width: int = 4,
    parallel_branch_roles: list[str] | None = None,
    parallel_consensus: str = "test_first_then_majority",
    parallel_branch_timeout_s: float | None = None,
    parallel_fail_fast: bool = True,
    register_parallel_catalog_flag: bool = False,
) -> dict:
    domains = domains or list(DEFAULT_DOMAINS)
    if domain_start < 0:
        domain_start = 0
    if domain_limit and domain_limit < 0:
        domain_limit = 0
    if domain_start or domain_limit:
        end = None if not domain_limit else domain_start + domain_limit
        domains = domains[domain_start:end]
    roles = roles or ["planner", "builder", "checker", "refactor"]

    if reset_db and os.path.exists(db_path):
        os.remove(db_path)

    with SQLiteRegistry(db_path) as registry:
        if register_parallel_catalog_flag:
            register_parallel_catalog(
                registry,
                dataset_family=dataset or task_family or "benchmark",
                skip_existing=not reset_db,
            )
        if real_tools:
            llm = LLMClient(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
                timeout_s=llm_timeout,
                max_retries=llm_retries,
            )
            debug_dir = debug_llm_dir if debug_llm else None
            if plan_themes:
                if verbose:
                    print("[progress] planning themes...", flush=True)
                themes = plan_agent_themes(
                    task_text,
                    theme_count,
                    llm,
                    debug=debug_llm,
                    debug_dir=debug_dir,
                )
                if not themes:
                    raise ValueError("No themes generated. Check LLM config or prompt.")
            else:
                themes = themes_from_domains(domains)
            if verbose:
                print(f"[progress] themes: {len(themes)}", flush=True)
            used_toolbench_ids: set[str] = set()
            functional_usage_counts: dict[str, int] = {}
            for theme_idx, theme in enumerate(themes, start=1):
                theme_id = str(theme.get("id") or theme.get("name") or "").strip()
                theme_slug = _slug(theme_id) if theme_id else "theme"
                if verbose:
                    print(
                        f"[progress] theme {theme_idx}/{len(themes)}: {theme_id}",
                        flush=True,
                    )
                profile_batches: list[list[dict]] = []
                if plan_agents:
                    if verbose:
                        print(
                            f"[progress] planning agent profiles for {theme_id}...",
                            flush=True,
                        )
                    profile_batches = _plan_profile_batches(
                        theme=theme,
                        total=agents_per_theme,
                        roles=roles,
                        llm=llm,
                        batch_size=profiles_batch_size,
                        debug_llm=debug_llm,
                        debug_dir=debug_dir,
                    )
                if not profile_batches:
                    profiles = _build_default_profiles(theme_id, roles, agents_per_theme, theme_slug)
                    profile_batches = _chunk_list(profiles, profiles_batch_size)
                total_profiles = sum(len(batch) for batch in profile_batches)
                if verbose:
                    print(
                        f"[progress] agent profiles: {total_profiles} "
                        f"(batches={len(profile_batches)}, tools_per_agent={tools_per_agent or 'all'})",
                        flush=True,
                    )
                planning_text = task_text if plan_themes else ""
                if tool_source == "generated":
                    planning_text = _append_planning_guidance(planning_text, TOOL_PLANNING_GUIDANCE)
                tools_meta: list[dict] = []
                tool_ids: list[str] = []
                profile_offset = 0

                if not tools_per_profile:
                    if tool_source == "functional":
                        pool_target = max(tools_per_theme, tools_per_agent * 2, min(12, agents_per_theme + 2))
                        if verbose:
                            print(
                                f"[progress] selecting functional tools for {theme_id} (target={pool_target})...",
                                flush=True,
                            )
                        selected = select_functional_tools(
                            request_text=_toolbench_request_text(task_text, theme),
                            count=pool_target,
                            usage_counts=functional_usage_counts,
                        )
                        for record in selected:
                            functional_usage_counts[record["tool_id"]] = functional_usage_counts.get(record["tool_id"], 0) + 1
                        if verbose:
                            print(
                                f"[progress] tool specs: {len(selected)} for {theme_id}",
                                flush=True,
                            )
                        _emit_specs_payload(
                            {"theme": theme_id, "specs": [spec_from_record(record) for record in selected]},
                            print_specs=print_specs,
                            spec_out_dir=spec_out_dir,
                            filename=f"{theme_slug}_specs.json",
                            verbose=verbose,
                        )
                        registered_ids, registered_meta = _register_functional_cards(
                            registry=registry,
                            selected=selected,
                            tool_output_dir=tool_output_dir,
                            skip_existing=not reset_db,
                        )
                        tool_ids.extend(registered_ids)
                        tools_meta.extend(registered_meta)
                        if verbose:
                            for tool_id in registered_ids:
                                print(f"[progress] saved tool: {tool_id}", flush=True)
                    elif tool_source == "toolbench":
                        if verbose:
                            print(
                                f"[progress] selecting ToolBench tools for {theme_id} (target={tools_per_theme})...",
                                flush=True,
                            )
                        selected = select_toolbench_endpoints(
                            toolbench_dir,
                            request_text=_toolbench_request_text(task_text, theme),
                            count=tools_per_theme,
                            exclude_ids=sorted(used_toolbench_ids),
                        )
                        for endpoint in selected:
                            used_toolbench_ids.add(endpoint["tool_id"])
                        if verbose:
                            print(
                                f"[progress] tool specs: {len(selected)} for {theme_id}",
                                flush=True,
                            )
                        _emit_specs_payload(
                            {"theme": theme_id, "specs": [endpoint_to_spec(endpoint) for endpoint in selected]},
                            print_specs=print_specs,
                            spec_out_dir=spec_out_dir,
                            filename=f"{theme_slug}_specs.json",
                            verbose=verbose,
                        )
                        registered_ids, registered_meta = _register_toolbench_cards(
                            registry=registry,
                            selected=selected,
                            tool_output_dir=tool_output_dir,
                            skip_existing=not reset_db,
                        )
                        tool_ids.extend(registered_ids)
                        tools_meta.extend(registered_meta)
                        if verbose:
                            for tool_id in registered_ids:
                                print(f"[progress] saved tool: {tool_id}", flush=True)
                    else:
                        if verbose:
                            print(
                                f"[progress] planning tool specs for {theme_id} (target={tools_per_theme})...",
                                flush=True,
                            )
                        specs = _plan_tools_batched(
                            planning_text,
                            theme,
                            tools_per_theme,
                            tools_batch_size,
                            llm,
                            debug_llm,
                            debug_dir,
                            verbose,
                        )
                        if verbose:
                            print(
                                f"[progress] tool specs: {len(specs)} for {theme_id}",
                                flush=True,
                            )
                        if enforce_llm_api:
                            _enforce_llm_api_specs(specs)
                        _emit_specs_payload(
                            {"theme": theme_id, "specs": specs},
                            print_specs=print_specs,
                            spec_out_dir=spec_out_dir,
                            filename=f"{theme_slug}_specs.json",
                            verbose=verbose,
                        )
                        for spec_idx, spec in enumerate(specs, start=1):
                            base_name = str(spec.get("name") or "tool")
                            tool_id = f"{theme_slug}-{_slug(base_name)}"
                            if verbose:
                                print(
                                    f"[progress] generating tool {spec_idx}/{len(specs)}: {tool_id}",
                                    flush=True,
                                )
                            tool_tags = list(spec.get("tool_tags") or [])
                            if tool_id not in tool_tags:
                                tool_tags.append(tool_id)
                            if theme_slug not in tool_tags:
                                tool_tags.append(theme_slug)
                            spec["tool_tags"] = tool_tags
                            if not spec.get("domain_tags"):
                                spec["domain_tags"] = [theme_slug]
                            code = generate_tool_code(spec, llm)
                            card = register_tool_from_spec(
                                registry,
                                spec,
                                code,
                                tool_id=tool_id,
                                save_dir=tool_output_dir,
                                skip_existing=not reset_db,
                            )
                            tool_ids.append(card.id)
                            tools_meta.append(
                                {
                                    "id": card.id,
                                    "name": card.name,
                                    "tool_tags": spec.get("tool_tags"),
                                    "domain_tags": spec.get("domain_tags"),
                                    "description": spec.get("description"),
                                }
                            )
                            if verbose:
                                print(
                                    f"[progress] saved tool: {card.id}",
                                    flush=True,
                                )

                for batch_idx, profiles in enumerate(profile_batches, start=1):
                    if not profiles:
                        continue
                    if tools_per_profile:
                        if verbose:
                            print(
                                f"[progress] planning tool specs per profile for {theme_id} "
                                f"(batch {batch_idx}, target={tools_per_profile})...",
                                flush=True,
                            )
                        for profile_idx, profile in enumerate(profiles, start=1):
                            profile_name = str(profile.get("name") or f"profile-{profile_idx}")
                            profile_slug = _slug(profile_name)
                            profile_desc = str(profile.get("description") or "").strip()
                            focus_tags = list(profile.get("focus_tags") or [])
                            profile_theme = {
                                "id": f"{theme_id}-{profile_slug}",
                                "name": f"{theme_id} - {profile_name}",
                                "description": profile_desc
                                or f"Profile focus: {', '.join(focus_tags) if focus_tags else 'general'}",
                                "domain_tags": list(theme.get("domain_tags") or []) or [theme_slug],
                                "capability_hints": focus_tags,
                            }
                            profile_planning = _append_planning_guidance(
                                planning_text,
                                f"Profile focus: {', '.join(focus_tags) if focus_tags else 'general'}.",
                            )
                            profile_tool_ids: list[str] = []
                            if tool_source == "functional":
                                if verbose:
                                    print(
                                        f"[progress] selecting functional tools for profile {profile_name} "
                                        f"(target={tools_per_profile})...",
                                        flush=True,
                                    )
                                selected = select_functional_tools(
                                    request_text=_toolbench_request_text(task_text, theme, profile),
                                    count=tools_per_profile,
                                    usage_counts=functional_usage_counts,
                                )
                                for record in selected:
                                    functional_usage_counts[record["tool_id"]] = functional_usage_counts.get(record["tool_id"], 0) + 1
                                if verbose:
                                    print(
                                        f"[progress] tool specs: {len(selected)} for profile {profile_name}",
                                        flush=True,
                                    )
                                _emit_specs_payload(
                                    {
                                        "theme": theme_id,
                                        "profile": profile_name,
                                        "specs": [spec_from_record(record) for record in selected],
                                    },
                                    print_specs=print_specs,
                                    spec_out_dir=spec_out_dir,
                                    filename=f"{theme_slug}_{profile_slug}_specs.json",
                                    verbose=verbose,
                                )
                                registered_ids, registered_meta = _register_functional_cards(
                                    registry=registry,
                                    selected=selected,
                                    tool_output_dir=tool_output_dir,
                                    skip_existing=not reset_db,
                                )
                                profile_tool_ids.extend(registered_ids)
                                tool_ids.extend(registered_ids)
                                tools_meta.extend(registered_meta)
                                if verbose:
                                    for tool_id in registered_ids:
                                        print(f"[progress] saved tool: {tool_id}", flush=True)
                            elif tool_source == "toolbench":
                                if verbose:
                                    print(
                                        f"[progress] selecting ToolBench tools for profile {profile_name} "
                                        f"(target={tools_per_profile})...",
                                        flush=True,
                                    )
                                selected = select_toolbench_endpoints(
                                    toolbench_dir,
                                    request_text=_toolbench_request_text(task_text, theme, profile),
                                    count=tools_per_profile,
                                    exclude_ids=sorted(used_toolbench_ids),
                                )
                                for endpoint in selected:
                                    used_toolbench_ids.add(endpoint["tool_id"])
                                if verbose:
                                    print(
                                        f"[progress] tool specs: {len(selected)} for profile {profile_name}",
                                        flush=True,
                                    )
                                _emit_specs_payload(
                                    {
                                        "theme": theme_id,
                                        "profile": profile_name,
                                        "specs": [endpoint_to_spec(endpoint) for endpoint in selected],
                                    },
                                    print_specs=print_specs,
                                    spec_out_dir=spec_out_dir,
                                    filename=f"{theme_slug}_{profile_slug}_specs.json",
                                    verbose=verbose,
                                )
                                registered_ids, registered_meta = _register_toolbench_cards(
                                    registry=registry,
                                    selected=selected,
                                    tool_output_dir=tool_output_dir,
                                    skip_existing=not reset_db,
                                )
                                profile_tool_ids.extend(registered_ids)
                                tool_ids.extend(registered_ids)
                                tools_meta.extend(registered_meta)
                                if verbose:
                                    for tool_id in registered_ids:
                                        print(f"[progress] saved tool: {tool_id}", flush=True)
                            else:
                                specs = _plan_tools_batched(
                                    profile_planning,
                                    profile_theme,
                                    tools_per_profile,
                                    tools_batch_size,
                                    llm,
                                    debug_llm,
                                    debug_dir,
                                    verbose,
                                )
                                if verbose:
                                    print(
                                        f"[progress] tool specs: {len(specs)} for profile {profile_name}",
                                        flush=True,
                                    )
                                if enforce_llm_api:
                                    _enforce_llm_api_specs(specs)
                                _emit_specs_payload(
                                    {"theme": theme_id, "profile": profile_name, "specs": specs},
                                    print_specs=print_specs,
                                    spec_out_dir=spec_out_dir,
                                    filename=f"{theme_slug}_{profile_slug}_specs.json",
                                    verbose=verbose,
                                )
                                for spec_idx, spec in enumerate(specs, start=1):
                                    base_name = str(spec.get("name") or "tool")
                                    tool_id = f"{theme_slug}-{profile_slug}-{_slug(base_name)}"
                                    if verbose:
                                        print(
                                            f"[progress] generating tool {spec_idx}/{len(specs)}: {tool_id}",
                                            flush=True,
                                        )
                                    tool_tags = list(spec.get("tool_tags") or [])
                                    if tool_id not in tool_tags:
                                        tool_tags.append(tool_id)
                                    if theme_slug not in tool_tags:
                                        tool_tags.append(theme_slug)
                                    if profile_slug not in tool_tags:
                                        tool_tags.append(profile_slug)
                                    spec["tool_tags"] = tool_tags
                                    if not spec.get("domain_tags"):
                                        spec["domain_tags"] = list(theme.get("domain_tags") or []) or [theme_slug]
                                    code = generate_tool_code(spec, llm)
                                    card = register_tool_from_spec(
                                        registry,
                                        spec,
                                        code,
                                        tool_id=tool_id,
                                        save_dir=tool_output_dir,
                                        skip_existing=not reset_db,
                                    )
                                    profile_tool_ids.append(card.id)
                                    tool_ids.append(card.id)
                                    tools_meta.append(
                                        {
                                            "id": card.id,
                                            "name": card.name,
                                            "tool_tags": spec.get("tool_tags"),
                                            "domain_tags": spec.get("domain_tags"),
                                            "description": spec.get("description"),
                                        }
                                    )
                                    if verbose:
                                        print(
                                            f"[progress] saved tool: {card.id}",
                                            flush=True,
                                        )
                            preferred = set(profile.get("preferred_tool_tags") or [])
                            preferred.add(profile_slug)
                            profile["preferred_tool_tags"] = list(preferred)
                            profile["tool_ids"] = list(profile_tool_ids)
                    else:
                        _select_tools_for_profiles(profiles, tools_meta, tools_per_agent)

                    generate_agents_from_profiles(
                        theme=theme,
                        profiles=profiles,
                        roles=roles,
                        registry=registry,
                        skip_existing=not reset_db,
                        start_index=profile_offset,
                    )
                    profile_offset += len(profiles)

                if verbose:
                    print(
                        f"[progress] agents created: {profile_offset} for {theme_id}",
                        flush=True,
                    )
        else:
            for i in range(tool_count):
                tool_name = "text_cleaner" if i % 2 == 0 else "basic_stats"
                register_tool(registry, tool_name, tool_id=f"{tool_name}-{i}", skip_existing=not reset_db)
            generate_agents(
                domains=domains,
                n_per_domain=n_per_domain,
                roles=roles,
                registry=registry,
                skip_existing=not reset_db,
            )

    index = build_index(
        db_path=db_path,
        kind="agent",
        out_dir=index_dir,
        dim=dim,
        seed=seed,
        embedder_kind=embedder_kind,
        embedder_model=embedder_model,
        embedder_device=embedder_device,
        embedder_normalize=embedder_normalize,
    )
    embedder = build_embedder(
        kind=embedder_kind,
        dim=dim,
        seed=seed,
        model_name=embedder_model,
        device=embedder_device,
        normalize=embedder_normalize,
    )
    if embedder.dim != index.dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} != index dim {index.dim}. "
            "Rebuild index with matching embedder/model."
        )

    registry = SQLiteRegistry(db_path)
    try:
        result = run_workflow(
            task_text=task_text,
            roles=roles,
            constraints_per_role={},
            workflow_version=workflow_version,
            registry=registry,
            index=index,
            embedder=embedder,
            top_n=10,
            top_k=5,
            mmr_lambda=0.5,
            paper_mode=paper_mode,
            task_family=task_family,
            dataset=dataset,
            parallel_execution=parallel_execution,
            parallel_width=parallel_width,
            parallel_branch_roles=parallel_branch_roles,
            parallel_consensus=parallel_consensus,
            parallel_branch_timeout_s=parallel_branch_timeout_s,
            parallel_fail_fast=parallel_fail_fast,
        )
    finally:
        registry.close()

    return result


def _parse_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tools/agents + run demo workflow")
    parser.add_argument("--task_text", type=str, default="Summarize quarterly performance and risks")
    parser.add_argument("--db", type=str, default="demo_registry.sqlite")
    parser.add_argument("--index_dir", type=str, default="./index")
    parser.add_argument("--tool_count", type=int, default=20)
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--n_per_domain", type=int, default=40)
    parser.add_argument("--roles", type=str, default="planner,builder,checker,refactor")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", type=str, default="sentence-transformer", help="dummy|sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embedder_device", type=str, default=None)
    parser.add_argument("--embedder_normalize", action="store_true")
    parser.add_argument("--workflow_version", type=str, default="v1")
    parser.add_argument("--real_tools", action="store_true")
    parser.add_argument("--theme_count", type=int, default=5)
    parser.add_argument("--tools_per_theme", type=int, default=5)
    parser.add_argument("--agents_per_theme", type=int, default=10)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_timeout", type=float, default=60.0)
    parser.add_argument("--llm_retries", type=int, default=5)
    parser.add_argument("--tool_output_dir", type=str, default=None)
    parser.add_argument("--plan_themes", action="store_true")
    parser.add_argument("--no_plan_agents", action="store_true")
    parser.add_argument("--domain_start", type=int, default=0)
    parser.add_argument("--domain_limit", type=int, default=0)
    parser.add_argument("--tools_per_agent", type=int, default=5)
    parser.add_argument("--tools_per_profile", type=int, default=0, help="generate tools per profile if >0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug_llm", action="store_true")
    parser.add_argument("--debug_llm_dir", type=str, default="runs/llm_debug")
    parser.add_argument("--tools_batch_size", type=int, default=10)
    parser.add_argument("--profiles_batch_size", type=int, default=0, help="generate profiles in smaller batches")
    parser.add_argument("--reset_db", action="store_true", help="delete existing db before generating")
    parser.add_argument("--print_specs", action="store_true", help="print tool specs as JSON")
    parser.add_argument("--spec_out_dir", type=str, default="", help="optional dir to save tool specs")
    parser.add_argument("--no_enforce_llm_api", action="store_true", help="do not force LLM API for reasoning/code tools")
    parser.add_argument("--tool_source", choices=["functional", "toolbench", "generated"], default="functional")
    parser.add_argument("--toolbench_dir", type=str, default="tools")
    parser.add_argument("--paper_mode", action="store_true")
    parser.add_argument("--task_family", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--parallel_width", type=int, default=4)
    parser.add_argument("--parallel_roles", type=str, default="")
    parser.add_argument("--parallel_consensus", type=str, default="test_first_then_majority")
    parser.add_argument("--parallel_branch_timeout_s", type=float, default=0.0)
    parser.add_argument("--no_parallel_fail_fast", action="store_true")
    parser.add_argument("--register_parallel_catalog", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(
        task_text=args.task_text,
        db_path=args.db,
        index_dir=args.index_dir,
        tool_count=args.tool_count,
        domains=_parse_list(args.domains) or None,
        n_per_domain=args.n_per_domain,
        roles=_parse_list(args.roles) or None,
        dim=args.dim,
        seed=args.seed,
        embedder_kind=args.embedder,
        embedder_model=args.embedder_model,
        embedder_device=args.embedder_device,
        embedder_normalize=args.embedder_normalize,
        workflow_version=args.workflow_version,
        real_tools=args.real_tools,
        theme_count=args.theme_count,
        tools_per_theme=args.tools_per_theme,
        agents_per_theme=args.agents_per_theme,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout=args.llm_timeout,
        llm_retries=args.llm_retries,
        tool_output_dir=args.tool_output_dir,
        plan_themes=args.plan_themes,
        plan_agents=not args.no_plan_agents,
        domain_start=args.domain_start,
        domain_limit=args.domain_limit,
        tools_per_agent=args.tools_per_agent,
        tools_per_profile=args.tools_per_profile,
        verbose=args.verbose,
        debug_llm=args.debug_llm,
        debug_llm_dir=args.debug_llm_dir,
        tools_batch_size=args.tools_batch_size,
        profiles_batch_size=args.profiles_batch_size,
        reset_db=args.reset_db,
        print_specs=args.print_specs,
        spec_out_dir=args.spec_out_dir or None,
        enforce_llm_api=not args.no_enforce_llm_api,
        tool_source=args.tool_source,
        toolbench_dir=args.toolbench_dir,
        paper_mode=args.paper_mode,
        task_family=args.task_family or None,
        dataset=args.dataset or None,
        parallel_execution=args.parallel,
        parallel_width=args.parallel_width,
        parallel_branch_roles=_parse_list(args.parallel_roles) or None,
        parallel_consensus=args.parallel_consensus,
        parallel_branch_timeout_s=args.parallel_branch_timeout_s if args.parallel_branch_timeout_s > 0 else None,
        parallel_fail_fast=not args.no_parallel_fail_fast,
        register_parallel_catalog_flag=args.register_parallel_catalog,
    )
    print(result.get("answer"))
    print(f"log_path: {result.get('log_path')}")


if __name__ == "__main__":
    main()
