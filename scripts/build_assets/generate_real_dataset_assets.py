from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cards import AgentCard
from core.registry import SQLiteRegistry
from generation.real_dataset_bulk_catalog import (
    build_agent_design_manifest,
    build_manifest_payload,
    build_real_dataset_agent_records,
    build_real_dataset_tool_records,
)
from generation.tool_generator import register_tool_from_spec
from retrieval.build_index import build_index


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


def _count_matching_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _count_files_with_markers(root: Path, suffix: str, markers: List[str]) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob(suffix):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if any(marker in text for marker in markers):
            total += 1
    return total


def _clear_generated_tool_dir(tools_dir: Path) -> int:
    removed = 0
    if not tools_dir.exists():
        return removed
    for path in tools_dir.glob("*.py"):
        path.unlink()
        removed += 1
    return removed


def _build_audit_report(repo_root: Path, tool_records: List[Dict[str, Any]], tools_dir: Path) -> Dict[str, Any]:
    specialized_dir = repo_root / "generated_tools_specialized"
    legacy_generated_dir = repo_root / "generated_tools"
    external_tools_dir = repo_root / "tools"
    return {
        "generated_tools_specialized_status": {
            "classification": "template-local-executable",
            "note": "Executable local Python tools, but template-generated and not used as the new primary tool pool.",
            "python_file_count": _count_matching_files(specialized_dir, "spec-*.py"),
        },
        "legacy_llm_dependent_generated_tools": {
            "classification": "llm-dependent",
            "note": "Legacy generated tools that depend on remote LLM/API configuration and are kept as legacy assets.",
            "python_file_count": _count_matching_files(legacy_generated_dir, "*.py"),
            "llm_marker_file_count": _count_files_with_markers(
                legacy_generated_dir,
                "*.py",
                ["LLM_API_KEY", "LLM_API_BASE", "urlopen(", "openai", "httpx.post"],
            ),
        },
        "external_api_toolbench_assets": {
            "classification": "external-api",
            "note": "Imported HTTP/OpenAPI-style tools under tools/, not the primary path for benchmark solving.",
            "api_python_file_count": _count_matching_files(external_tools_dir, "api.py"),
            "json_spec_count": _count_matching_files(external_tools_dir, "*.json"),
        },
        "new_real_deterministic_tools": {
            "classification": "real-deterministic-local",
            "note": "New deterministic tool pool for Humaneval, MBPP, GSM8K, MATH, and HotpotQA. No remote API calls.",
            "tool_count": len(tool_records),
            "tool_ids": [record["id"] for record in tool_records],
            "generated_file_count": _count_matching_files(tools_dir, "*.py"),
        },
    }


def generate_real_dataset_assets(
    *,
    db_path: str,
    tools_dir: str,
    index_dir: str,
    build_agent_index: bool = False,
    index_embedder: str = "dummy",
    index_dim: int = 64,
) -> Dict[str, Any]:
    repo_root = ROOT
    db_path_resolved = _resolve_output_path(db_path)
    tools_dir_path = _resolve_output_path(tools_dir)
    index_dir_path = _resolve_output_path(index_dir)
    db_path = str(db_path_resolved)
    os.makedirs(tools_dir_path, exist_ok=True)
    backup_path = _backup_file(db_path)
    removed_tool_files = _clear_generated_tool_dir(tools_dir_path)

    tool_records = build_real_dataset_tool_records()
    agent_records = build_real_dataset_agent_records()
    tool_name_by_id: Dict[str, str] = {}

    with SQLiteRegistry(db_path) as registry:
        for record in tool_records:
            card = register_tool_from_spec(
                registry,
                record["spec"],
                record["code"],
                tool_id=record["id"],
                save_dir=str(tools_dir_path),
                skip_existing=False,
            )
            tool_name_by_id[card.id] = card.name

        for record in agent_records:
            tool_ids = list(record["tool_ids"])
            tool_names = [tool_name_by_id[tool_id] for tool_id in tool_ids]
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
                description=record["description"] + " Tools: " + ", ".join(tool_names),
                examples=[record["description"]],
                embedding_text=record["embedding_text"],
                available_tool_ids=list(tool_ids),
                dataset_family=str(record.get("dataset_family") or ""),
                focus_slug=str(record.get("focus_slug") or ""),
                primary_role=str(record.get("primary_role") or ""),
            )
            registry.register(card)

    if build_agent_index:
        build_index(
            db_path=db_path,
            kind="agent",
            out_dir=str(index_dir_path),
            dim=index_dim,
            embedder_kind=index_embedder,
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            embedder_device=None,
            embedder_normalize=False,
        )

    design_manifest_path = repo_root / "agent_tool_design_real_v1.json"
    design_manifest_path.write_text(
        json.dumps(build_agent_design_manifest(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    manifest_payload = build_manifest_payload()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(Path(db_path).resolve()),
        "backup_db_path": backup_path,
        "cleared_tool_file_count": removed_tool_files,
        "tools_dir": str(tools_dir_path.resolve()),
        "index_dir": str(index_dir_path.resolve()),
        "tool_count": len(tool_records),
        "agent_count": len(agent_records),
        "tools_per_agent": 5,
        "all_tools_unique_across_agents": len({tool_id for agent in agent_records for tool_id in agent["tool_ids"]})
        == sum(len(agent["tool_ids"]) for agent in agent_records),
        "agent_ids": [record["id"] for record in agent_records],
        "tool_ids": [record["id"] for record in tool_records],
        "design_path": str(design_manifest_path.resolve()),
        "build_index_requested": bool(build_agent_index),
        "index_embedder": index_embedder if build_agent_index else None,
        "index_dim": index_dim if build_agent_index else None,
    }
    manifest_payload["summary"] = summary
    manifest_payload["paths"] = {
        "db_path": summary["db_path"],
        "tools_dir": summary["tools_dir"],
        "index_dir": summary["index_dir"],
        "design_path": summary["design_path"],
    }

    manifest_path = repo_root / "real_dataset_agent_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    audit_report = _build_audit_report(repo_root, tool_records, tools_dir_path)
    audit_path = repo_root / "real_tool_audit_report.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=True, indent=2), encoding="utf-8")

    summary["manifest_path"] = str(manifest_path.resolve())
    summary["audit_path"] = str(audit_path.resolve())
    return summary


def _print_registry_summary(db_path: str) -> None:
    resolved = _resolve_output_path(db_path)
    conn = sqlite3.connect(str(resolved))
    cur = conn.cursor()
    cur.execute("SELECT kind, COUNT(*) FROM cards GROUP BY kind ORDER BY kind")
    counts = cur.fetchall()
    print(json.dumps({"db_path": str(resolved), "counts": counts}, ensure_ascii=True))
    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real deterministic dataset tools and agents")
    parser.add_argument("--db", default="real_dataset_registry.sqlite")
    parser.add_argument("--tools_dir", default="generated_tools_real")
    parser.add_argument("--index_dir", default="index_real_dataset")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--index_embedder", default="dummy", choices=["dummy", "sentence-transformer"])
    parser.add_argument("--index_dim", type=int, default=64)
    parser.add_argument("--print_registry_summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_real_dataset_assets(
        db_path=args.db,
        tools_dir=args.tools_dir,
        index_dir=args.index_dir,
        build_agent_index=args.build_index,
        index_embedder=args.index_embedder,
        index_dim=args.index_dim,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.print_registry_summary:
        _print_registry_summary(args.db)


if __name__ == "__main__":
    main()
