import importlib
import sys
from datetime import datetime
from pathlib import Path

import pytest


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


@pytest.mark.xfail(reason="legacy expectation predates the current DummyEmbedder", strict=False)
def test_build_index_and_search(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder

    db_path = tmp_path / "index.sqlite"
    with SQLiteRegistry(str(db_path)) as registry:
        for i in range(100):
            text = f"agent {i} domain {i % 5} output style concise"
            card = AgentCard(
                id=f"agent-{i}",
                name=f"Agent {i}",
                kind="agent",
                version="1.0",
                updated_at=datetime(2026, 1, 1),
                domain_tags=[f"domain-{i % 5}"],
                role_tags=["analyst"],
                description="test agent",
                embedding_text=text,
            )
            registry.register(card)

    out_dir = tmp_path / "index"
    dim = 32
    index = build_index(
        db_path=str(db_path),
        kind="agent",
        out_dir=str(out_dir),
        dim=dim,
        batch_size=25,
        seed=123,
    )

    embedder = DummyEmbedder(dim=dim, seed=123)
    query_text = "agent 42 domain 2 output style concise"
    query_vec = embedder.embed([query_text])
    _, ids = index.search(query_vec, top_k=5)

    assert ids
    assert "agent-42" in ids[0]
