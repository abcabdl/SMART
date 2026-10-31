from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cards import AgentCard
from core.registry import SQLiteRegistry
from evaluation.smart_pdf_preset import load_smart_pdf_preset
from generation.real_dataset_bulk_catalog import (
    DATASET_CONFIG,
    _capability_metadata,
    _dataset_keywords,
    _focus_specific_domain_tags,
    _make_tool_code,
    _title_case,
)
from generation.tool_generator import register_tool_from_spec
from retrieval.build_index import build_index


PAPER_DATASETS = ("humaneval", "mbpp", "gsm8k", "math")
PAPER_ROLE_TAGS = {
    "humaneval": {
        "planner": ["planner", "code-planner"],
        "builder": ["builder", "code-generation"],
        "checker": ["checker", "code-testing"],
        "refactor": ["refactor", "code-refactoring"],
    },
    "mbpp": {
        "planner": ["planner", "code-planner"],
        "builder": ["builder", "code-generation"],
        "checker": ["checker", "code-testing"],
        "refactor": ["refactor", "code-refactoring"],
    },
    "gsm8k": {
        "planner": ["planner"],
        "builder": ["builder"],
        "checker": ["checker"],
        "refactor": ["refactor"],
    },
    "math": {
        "planner": ["planner"],
        "builder": ["builder"],
        "checker": ["checker"],
        "refactor": ["refactor"],
    },
}
PAPER_ROLE_ORDER = ("planner", "builder", "checker", "refactor")
PAPER_FULL_FOCUS_COUNT = 18
PAPER_PARTIAL_FOCUS_COUNT = 1


def _resolve_output_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "repo":  # legacy "" prefix
        raw = Path(*raw.parts[1:])
    return ROOT / raw


def _backup_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = f"{path}.bak.{stamp}"
    os.replace(path, backup)
    return backup


def _clear_generated_tool_dir(tools_dir: Path) -> int:
    removed = 0
    if not tools_dir.exists():
        return removed
    for path in tools_dir.glob("paper-*.py"):
        path.unlink()
        removed += 1
    return removed


def _paper_focuses(dataset: str) -> List[Dict[str, Any]]:
    focuses = list(DATASET_CONFIG[dataset]["focuses"])
    return focuses[: PAPER_FULL_FOCUS_COUNT + PAPER_PARTIAL_FOCUS_COUNT]


def _paper_roles_for_focus(index: int) -> List[str]:
    if index < PAPER_FULL_FOCUS_COUNT:
        return list(PAPER_ROLE_ORDER)
    return list(PAPER_ROLE_ORDER[:-1])


def _tool_record(
    *,
    tool_id: str,
    name: str,
    dataset: str,
    description: str,
    code: str,
    domain_tags: List[str],
    role_tags: List[str],
    tool_tags: List[str],
    dataset_family: str,
    focus_slug: str,
    primary_role: str,
    capability_kind: str,
    stage: str,
    consumes: List[str],
    produces: List[str],
) -> Dict[str, Any]:
    return {
        "id": tool_id,
        "dataset": dataset,
        "spec": {
            "name": name,
            "description": description,
            "domain_tags": list(domain_tags),
            "role_tags": list(role_tags),
            "tool_tags": list(tool_tags),
            "modalities": ["text"],
            "output_formats": ["json"],
            "permissions": ["read", "compute"],
            "cost_tier": "low",
            "latency_tier": "low",
            "reliability_prior": 0.88,
            "examples": [description],
            "embedding_text": f"{name}. {description}. Tags: {' '.join(tool_tags)}",
            "dataset_family": dataset_family,
            "focus_slug": focus_slug,
            "primary_role": primary_role,
            "capability_kind": capability_kind,
            "stage": stage,
            "deterministic": True,
            "consumes": list(consumes),
            "produces": list(produces),
        },
        "code": code,
    }


def build_paper_tool_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset in PAPER_DATASETS:
        config = DATASET_CONFIG[dataset]
        dataset_keywords = _dataset_keywords(dataset)
        role_tags_map = PAPER_ROLE_TAGS[dataset]
        for focus_idx, focus in enumerate(_paper_focuses(dataset)):
            focus_slug = str(focus["slug"])
            focus_title = str(focus["title"])
            keywords = list(dict.fromkeys(list(focus["keywords"]) + dataset_keywords))
            for primary_role in _paper_roles_for_focus(focus_idx):
                role_tags = list(dict.fromkeys(role_tags_map[primary_role] + [primary_role]))
                for kind_slug, kind_title in config["tool_kinds"]:
                    tool_id = f"paper-{dataset}-{focus_slug}-{primary_role}-{kind_slug}"
                    if tool_id in seen_ids:
                        raise ValueError(f"Duplicate paper tool id: {tool_id}")
                    seen_ids.add(tool_id)
                    tool_tags = list(
                        dict.fromkeys(
                            [
                                "paper-benchmark",
                                dataset,
                                focus_slug,
                                primary_role,
                                kind_slug,
                                "real-tool",
                            ]
                            + keywords[:4]
                        )
                    )
                    capability_meta = _capability_metadata(dataset, kind_slug)
                    records.append(
                        _tool_record(
                            tool_id=tool_id,
                            name=f"Paper{config['display']}{_title_case(focus_slug)}{primary_role.title()}{_title_case(kind_slug)}Tool",
                            dataset=dataset,
                            description=(
                                f"Paper benchmark deterministic {config['display']} tool for {focus_title.lower()} tasks. "
                                f"Primary role={primary_role}. Capability={kind_title.lower()}."
                            ),
                            code=_make_tool_code(dataset, focus_title, keywords, primary_role, kind_slug),
                            domain_tags=list(
                                dict.fromkeys(
                                    list(config["domain_tags"])
                                    + _focus_specific_domain_tags(dataset, focus_slug)
                                    + ["paper-benchmark"]
                                )
                            ),
                            role_tags=role_tags,
                            tool_tags=tool_tags,
                            dataset_family=dataset,
                            focus_slug=focus_slug,
                            primary_role=primary_role,
                            capability_kind=capability_meta["capability_kind"],
                            stage=capability_meta["stage"],
                            consumes=capability_meta["consumes"],
                            produces=capability_meta["produces"],
                        )
                    )
    return records


def build_paper_agent_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset in PAPER_DATASETS:
        config = DATASET_CONFIG[dataset]
        role_tags_map = PAPER_ROLE_TAGS[dataset]
        for focus_idx, focus in enumerate(_paper_focuses(dataset)):
            focus_slug = str(focus["slug"])
            focus_title = str(focus["title"])
            for primary_role in _paper_roles_for_focus(focus_idx):
                agent_id = f"paper-agent-{dataset}-{focus_slug}-{primary_role}"
                if agent_id in seen_ids:
                    raise ValueError(f"Duplicate paper agent id: {agent_id}")
                seen_ids.add(agent_id)
                tool_ids = [
                    f"paper-{dataset}-{focus_slug}-{primary_role}-{kind_slug}"
                    for kind_slug, _ in config["tool_kinds"]
                ]
                records.append(
                    {
                        "id": agent_id,
                        "name": f"Paper{config['display']}{_title_case(focus_slug)}{primary_role.title()}-RealTools",
                        "description": (
                            f"Paper benchmark specialized {primary_role} agent for {config['display']} "
                            f"focused on {focus_title.lower()}. Uses 5 deterministic local tools."
                        ),
                        "domain_tags": list(
                            dict.fromkeys(
                                list(config["domain_tags"])
                                + _focus_specific_domain_tags(dataset, focus_slug)
                                + ["paper-benchmark"]
                            )
                        ),
                        "role_tags": list(dict.fromkeys(role_tags_map[primary_role] + [primary_role])),
                        "tool_ids": tool_ids,
                        "embedding_text": (
                            f"Paper benchmark {config['display']} specialist for {focus_title.lower()}. "
                            f"Primary role {primary_role}. Tools: {' '.join(tool_ids)}. "
                            f"Keywords: {' '.join(focus['keywords'])}."
                        ),
                        "dataset_family": dataset,
                        "focus_slug": focus_slug,
                        "primary_role": primary_role,
                    }
                )
    return records


def build_paper_manifest(agent_records: List[Dict[str, Any]], tool_records: List[Dict[str, Any]], preset: Dict[str, Any]) -> Dict[str, Any]:
    dataset_distribution = dict(Counter(str(item.get("dataset_family") or "") for item in agent_records))
    role_distribution = dict(Counter(str(item.get("primary_role") or "") for item in agent_records))
    all_tools_unique = len({tool_id for agent in agent_records for tool_id in agent["tool_ids"]}) == sum(
        len(agent["tool_ids"]) for agent in agent_records
    )
    return {
        "source_of_truth": str(preset.get("source_of_truth") or ""),
        "agent_count": len(agent_records),
        "tool_count": len(tool_records),
        "tools_per_agent": int(preset.get("tools_per_agent") or 5),
        "dataset_distribution": dataset_distribution,
        "role_distribution": role_distribution,
        "domains_or_benchmarks": list(PAPER_DATASETS),
        "all_tools_unique_across_agents": all_tools_unique,
        "paths": dict(preset.get("paths") or {}),
    }


def generate_paper_assets(*, build_agent_index: bool = True) -> Dict[str, Any]:
    preset = load_smart_pdf_preset()
    runtime = dict(preset.get("runtime") or {})
    paths = dict(preset.get("paths") or {})
    db_path = _resolve_output_path(str(paths.get("benchmark_db") or "paper_registry.sqlite"))
    index_dir = _resolve_output_path(str(paths.get("benchmark_index_dir") or "index_paper"))
    manifest_path = _resolve_output_path(str(paths.get("paper_manifest") or "paper_agent_manifest.json"))
    tools_dir = ROOT / "generated_tools_paper"

    os.makedirs(tools_dir, exist_ok=True)
    backup_path = _backup_file(str(db_path))
    removed_tool_files = _clear_generated_tool_dir(tools_dir)

    tool_records = build_paper_tool_records()
    agent_records = build_paper_agent_records()
    tool_name_by_id: Dict[str, str] = {}

    with SQLiteRegistry(str(db_path)) as registry:
        for record in tool_records:
            card = register_tool_from_spec(
                registry,
                record["spec"],
                record["code"],
                tool_id=record["id"],
                save_dir=str(tools_dir),
                skip_existing=False,
            )
            tool_name_by_id[card.id] = card.name

        for record in agent_records:
            tool_ids = list(record["tool_ids"])
            card = AgentCard(
                id=record["id"],
                name=record["name"],
                kind="agent",
                version="1.0",
                updated_at=datetime.now(timezone.utc),
                domain_tags=list(record["domain_tags"]),
                role_tags=list(record["role_tags"]),
                tool_tags=list(tool_ids),
                modalities=["text"],
                output_formats=["json"],
                permissions=["read", "compute"],
                cost_tier="low",
                latency_tier="low",
                reliability_prior=0.9,
                description=record["description"] + " Tools: " + ", ".join(tool_name_by_id[tool_id] for tool_id in tool_ids),
                examples=[record["description"]],
                embedding_text=record["embedding_text"],
                available_tool_ids=tool_ids,
                dataset_family=str(record["dataset_family"]),
                focus_slug=str(record["focus_slug"]),
                primary_role=str(record["primary_role"]),
            )
            registry.register(card)

    if build_agent_index:
        build_index(
            db_path=str(db_path),
            kind="agent",
            out_dir=str(index_dir),
            dim=int(runtime.get("dim") or 64),
            seed=int(runtime.get("seed") or 7),
            embedder_kind=str(runtime.get("embedder") or "dummy"),
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            embedder_device=None,
            embedder_normalize=False,
        )

    manifest = build_paper_manifest(agent_records, tool_records, preset)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["db_path"] = str(db_path.resolve())
    manifest["index_dir"] = str(index_dir.resolve())
    manifest["tools_dir"] = str(tools_dir.resolve())
    manifest["backup_db_path"] = backup_path
    manifest["cleared_tool_file_count"] = removed_tool_files
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    return manifest


def main() -> None:
    manifest = generate_paper_assets(build_agent_index=True)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
