from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cards import AgentCard
from core.registry import SQLiteRegistry
from generation.specialized_functional_catalog import build_specialized_agents, load_specialized_tool_records
from generation.tool_generator import register_tool_from_spec
from retrieval.build_index import build_index


def _backup_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup = f"{path}.bak.{stamp}"
    os.replace(path, backup)
    return backup


def generate_specialized_assets(
    *,
    db_path: str,
    tools_dir: str,
    index_dir: str,
    build_agent_index: bool = False,
) -> dict:
    _backup_file(db_path)
    os.makedirs(tools_dir, exist_ok=True)

    tool_records = load_specialized_tool_records()
    agent_records = build_specialized_agents()
    tool_name_by_id: dict[str, str] = {}

    with SQLiteRegistry(db_path) as registry:
        for record in tool_records:
            card = register_tool_from_spec(
                registry,
                record["spec"],
                record["code"],
                tool_id=record["id"],
                save_dir=tools_dir,
                skip_existing=True,
            )
            tool_name_by_id[card.id] = card.name

        for record in agent_records:
            tool_ids = list(record["tool_ids"])
            tool_names = [tool_name_by_id.get(tool_id, tool_id) for tool_id in tool_ids]
            card = AgentCard(
                id=record["id"],
                name=record["name"],
                kind="agent",
                version="1.0",
                updated_at=datetime.utcnow(),
                domain_tags=list(record["domain_tags"]),
                role_tags=list(record["role_tags"]),
                tool_tags=list(tool_ids),
                modalities=["text"],
                output_formats=["json"],
                permissions=["read", "compute"],
                cost_tier="low",
                latency_tier="low",
                reliability_prior=0.82,
                description=record["description"] + " Tools: " + ", ".join(tool_names),
                examples=[record["description"]],
                embedding_text=record["embedding_text"],
                available_tool_ids=tool_ids,
            )
            existing = registry.get(card.id)
            if existing is None:
                registry.register(card)
            else:
                registry.update(card)

    if build_agent_index:
        build_index(
            db_path=db_path,
            kind="agent",
            out_dir=index_dir,
            dim=64,
            embedder_kind="sentence-transformer",
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            embedder_device=None,
            embedder_normalize=False,
        )

    summary = {
        "db_path": db_path,
        "tools_dir": tools_dir,
        "tool_count": len(tool_records),
        "agent_count": len(agent_records),
        "specialization_count": len(agent_records) // 3,
        "tools_per_agent": 5,
    }
    summary_path = os.path.join(os.path.dirname(db_path) or ".", "specialized_generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    summary["summary_path"] = summary_path
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 600 specialized real tools and 300 specialized agents")
    parser.add_argument("--db", default="specialized_registry.sqlite")
    parser.add_argument("--tools_dir", default="generated_tools_specialized")
    parser.add_argument("--index_dir", default="index_specialized")
    parser.add_argument("--build_index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_specialized_assets(
        db_path=args.db,
        tools_dir=args.tools_dir,
        index_dir=args.index_dir,
        build_agent_index=args.build_index,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
