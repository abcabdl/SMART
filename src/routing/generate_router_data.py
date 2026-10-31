from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure `core`, `retrieval`, etc. are importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from generation.llm_client import LLMClient
from retrieval.embedder import build_embedder
from retrieval.faiss_index import HNSWIndex
from retrieval.search_service import get_candidates


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_id_map(index_dir: str) -> List[str]:
    path = os.path.join(index_dir, "id_map.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return []
    try:
        ordered_keys = sorted(raw.keys(), key=lambda k: int(k))
    except ValueError:
        ordered_keys = sorted(raw.keys())
    return [raw[key] for key in ordered_keys]


def _load_queries(path: Optional[str], inline: Optional[str]) -> List[str]:
    queries: List[str] = []
    if inline:
        queries.extend(_parse_list(inline))
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.endswith(".jsonl"):
                    record = json.loads(line)
                    if isinstance(record, dict):
                        text = record.get("query") or record.get("task") or record.get("text")
                        if text:
                            queries.append(str(text))
                            continue
                queries.append(line)
    return queries


def _normalize_constraints(raw: str) -> Dict[str, Dict[str, object]]:
    if not raw:
        return {}
    return json.loads(raw)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return text[start : end + 1]
    return None


def _safe_load_json(text: str, fallback: dict) -> dict:
    blob = _extract_json_blob(text) or text
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return fallback
    return data if isinstance(data, dict) else fallback


def _heuristic_topology(task_text: str) -> str:
    text = task_text.lower()
    if any(token in text for token in ["brainstorm", "multi-angle", "debate", "discussion", "对比", "多角度", "讨论"]):
        return "decentralized"
    if any(token in text for token in ["then", "first", "next", "after", "并且", "然后", "最后", "步骤", "流程"]):
        return "centralized"
    return "single"


def _normalize_meta_config(raw: dict, roles: List[str], max_steps: int) -> dict:
    topology = str(raw.get("topology") or "").strip().lower()
    if topology not in {"single", "centralized", "decentralized", "chain"}:
        topology = _heuristic_topology(raw.get("task_text", ""))
    selected_roles = raw.get("roles") or roles
    if not isinstance(selected_roles, list) or not selected_roles:
        selected_roles = roles
    manager_role = raw.get("manager_role")
    if topology == "centralized" and not manager_role:
        manager_role = "planner" if "planner" in selected_roles else selected_roles[0]
    entry_role = raw.get("entry_role") or (selected_roles[0] if selected_roles else None)
    return {
        "topology": topology,
        "roles": selected_roles,
        "manager_role": manager_role,
        "entry_role": entry_role,
        "max_steps": int(raw.get("max_steps") or max_steps),
        "flow_type": raw.get("flow_type"),
    }


def _format_candidates(candidates: List[dict]) -> str:
    parts: List[str] = []
    for cand in candidates:
        parts.append(
            "\n".join(
                [
                    f"- id: {cand.get('id')}",
                    f"  name: {cand.get('name')}",
                    f"  domain_tags: {cand.get('domain_tags')}",
                    f"  role_tags: {cand.get('role_tags')}",
                    f"  tool_tags: {cand.get('tool_tags')}",
                    f"  description: {cand.get('description')}",
                ]
            )
        )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate router SFT data using a labeler LLM")
    parser.add_argument("--queries_file", type=str, default=None)
    parser.add_argument("--queries", type=str, default="")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agent_router", "meta_router", "combined"],
        default="agent_router",
        help="generate agent-routing data, meta-router data, or both",
    )
    parser.add_argument("--roles", type=str, default="planner,builder,checker,refactor")
    parser.add_argument("--constraints", type=str, default="", help="JSON string per role")
    parser.add_argument("--db", type=str, default="demo_registry.sqlite")
    parser.add_argument("--index_dir", type=str, default="index")
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
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mmr_lambda", type=float, default=0.5)
    parser.add_argument("--llm_model", type=str, default="qwen3-8b")
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_timeout", type=float, default=60.0)
    parser.add_argument("--llm_retries", type=int, default=5)
    parser.add_argument("--out", type=str, default="data/router_sft.jsonl")
    parser.add_argument("--max_steps", type=int, default=6, help="max steps for meta-router outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = _load_queries(args.queries_file, args.queries)
    if not queries:
        raise ValueError("No queries provided. Use --queries or --queries_file.")
    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]

    index = HNSWIndex.load(os.path.join(args.index_dir, "faiss.index"))
    id_map = _load_id_map(args.index_dir)
    if id_map:
        index.ids = list(id_map)
    embedder = build_embedder(
        kind=args.embedder,
        dim=args.dim,
        seed=args.seed,
        model_name=args.embedder_model,
        device=args.embedder_device,
        normalize=args.embedder_normalize,
    )
    if embedder.dim != index.dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} != index dim {index.dim}. "
            "Rebuild index with matching embedder/model."
        )

    llm = LLMClient(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout_s=args.llm_timeout,
        max_retries=args.llm_retries,
    )

    roles = _parse_list(args.roles)
    constraints = _normalize_constraints(args.constraints)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with SQLiteRegistry(args.db) as registry, open(args.out, "w", encoding="utf-8") as out:
        for query in queries:
            if args.mode in {"meta_router", "combined"}:
                system_msg = (
                    "You are a meta-router. Decide the best agent topology for the task. "
                    "Return ONLY JSON with keys: topology, roles, manager_role, entry_role, max_steps, flow_type."
                )
                user_msg = (
                    f"Task: {query}\n"
                    f"Available roles: {json.dumps(roles, ensure_ascii=True)}\n"
                    "Topology options: single, centralized, decentralized, chain.\n"
                )
                response = llm.chat(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    temperature=0.0,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                data = _safe_load_json(str(response), {})
                data["task_text"] = query
                config = _normalize_meta_config(data, roles, args.max_steps)
                record = {
                    "sample_type": "meta_router",
                    "query": query,
                    "topology": config.get("topology"),
                    "roles": config.get("roles"),
                    "manager_role": config.get("manager_role"),
                    "entry_role": config.get("entry_role"),
                    "max_steps": config.get("max_steps"),
                    "flow_type": config.get("flow_type"),
                    "output": json.dumps(config, ensure_ascii=True),
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": json.dumps(config, ensure_ascii=True)},
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=True) + "\n")

            if args.mode in {"agent_router", "combined"}:
                for role in roles:
                    candidates = get_candidates(
                        task_text=query,
                        role=role,
                        constraints=constraints.get(role),
                        kind="agent",
                        top_n=args.top_n,
                        top_k=args.top_k,
                        mmr_lambda=args.mmr_lambda,
                        registry=registry,
                        index=index,
                        embedder=embedder,
                    )
                    candidate_cards: List[dict] = []
                    for cand in candidates:
                        card = registry.get(cand.get("card_id"))
                        if card is None:
                            continue
                        candidate_cards.append(
                            {
                                "id": card.id,
                                "name": card.name,
                                "description": card.description,
                                "domain_tags": card.domain_tags,
                                "role_tags": card.role_tags,
                                "tool_tags": card.tool_tags,
                            }
                        )
                    if not candidate_cards:
                        continue

                    candidates_text = _format_candidates(candidate_cards)
                    system_msg = (
                        "You are a routing model. Select the single best agent id from the candidate list. "
                        "Return ONLY JSON: {\"selected_id\": \"...\"}."
                    )
                    user_msg = (
                        f"Role: {role}\n"
                        f"Task: {query}\n"
                        f"Candidates:\n{candidates_text}\n"
                    )
                    response = llm.chat(
                        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                        temperature=0.0,
                        max_tokens=200,
                        response_format={"type": "json_object"},
                    )
                    data = _safe_load_json(str(response), {})
                    selected_id = data.get("selected_id")
                    valid_ids = {c["id"] for c in candidate_cards}
                    if selected_id not in valid_ids:
                        selected_id = candidate_cards[0]["id"]

                    record = {
                        "sample_type": "agent_router",
                        "role": role,
                        "query": query,
                        "candidates": candidate_cards,
                        "label": selected_id,
                        "input": f"{system_msg}\n\n{user_msg}",
                        "output": selected_id,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": json.dumps({"selected_id": selected_id})},
                        ],
                    }
                    out.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
