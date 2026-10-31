from __future__ import annotations

import json
import os
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
from evaluation.smart_pdf_preset import load_smart_pdf_preset
from generation.real_dataset_bulk_catalog import build_real_dataset_agent_records, build_real_dataset_tool_records
from generation.tool_generator import register_tool_from_spec
from retrieval.build_index import build_index


def _resolve_output_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "repo":  # legacy "" prefix
        raw = Path(*raw.parts[1:])
    return ROOT / raw


def generate_hotpotqa_appendix_assets() -> Dict[str, Any]:
    preset = load_smart_pdf_preset()
    runtime = dict(preset.get("runtime") or {})
    paths = dict(preset.get("paths") or {})
    db_path = _resolve_output_path(str(paths.get("hotpotqa_appendix_db") or "assets/hotpotqa_registry.sqlite"))
    index_dir = _resolve_output_path(str(paths.get("hotpotqa_appendix_index_dir") or "assets/hotpotqa_index"))
    tools_dir = ROOT / "generated_tools_hotpotqa_appendix"
    os.makedirs(tools_dir, exist_ok=True)

    tool_records = [item for item in build_real_dataset_tool_records() if str(item.get("dataset") or "") == "hotpotqa"]
    agent_records = [item for item in build_real_dataset_agent_records() if str(item.get("dataset_family") or "") == "hotpotqa"]
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
                dataset_family="hotpotqa",
                focus_slug=str(record.get("focus_slug") or ""),
                primary_role=str(record.get("primary_role") or ""),
            )
            registry.register(card)

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

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hotpotqa",
        "agent_count": len(agent_records),
        "tool_count": len(tool_records),
        "db_path": str(db_path.resolve()),
        "index_dir": str(index_dir.resolve()),
        "tools_dir": str(tools_dir.resolve()),
        "note": "HotpotQA appendix diagnostic assets only; not part of the SMART paper main benchmark.",
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return summary


if __name__ == "__main__":
    generate_hotpotqa_appendix_assets()
