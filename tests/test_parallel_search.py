import importlib
import sys
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_parallel_search_batch_endpoint_matches_single_shape(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")
    service_mod = _import_from_src("retrieval.search_service")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder
    create_app = service_mod.create_app

    db_path = tmp_path / "search.sqlite"
    registry = SQLiteRegistry(str(db_path))
    for i in range(12):
        role = "builder" if i % 2 == 0 else "checker"
        domain = "code" if i < 8 else "math"
        registry.register(
            AgentCard(
                id=f"agent-{i}",
                name=f"Agent {i}",
                kind="agent",
                version="1.0",
                updated_at=datetime(2026, 1, 1),
                domain_tags=[domain],
                role_tags=[role],
                description=f"{role} {domain} agent",
                embedding_text=f"{role} {domain} candidate {i}",
            )
        )

    out_dir = tmp_path / "index"
    index = build_index(db_path=str(db_path), kind="agent", out_dir=str(out_dir), dim=24, seed=11)
    embedder = DummyEmbedder(dim=24, seed=11)
    client = TestClient(create_app(registry=registry, index=index, embedder=embedder))

    single = client.post(
        "/candidates",
        json={
            "task_text": "implement code",
            "role": "builder",
            "kind": "agent",
            "top_n": 8,
            "top_k": 3,
        },
    )
    batch = client.post(
        "/candidates/batch",
        json={
            "requests": [
                {
                    "branch_id": "builder-0",
                    "task_text": "implement code",
                    "role": "builder",
                    "kind": "agent",
                    "top_n": 8,
                    "top_k": 3,
                },
                {
                    "branch_id": "checker-1",
                    "task_text": "verify math",
                    "role": "checker",
                    "constraints": {"role_tags": ["checker"], "domain_tags": ["math"]},
                    "kind": "agent",
                    "top_n": 8,
                    "top_k": 3,
                },
            ]
        },
    )

    assert single.status_code == 200
    assert batch.status_code == 200
    payload = batch.json()
    assert set(payload) == {"builder-0", "checker-1"}
    assert payload["builder-0"]
    assert set(payload["builder-0"][0]) == {"card_id", "score", "brief_tags"}
    assert payload["builder-0"][0] == single.json()[0]
    assert payload["checker-1"]
    assert all("checker" in item["brief_tags"]["role_tags"] for item in payload["checker-1"])
    assert all("math" in item["brief_tags"]["domain_tags"] for item in payload["checker-1"])

    registry.close()
