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


def test_search_service_candidates(tmp_path) -> None:
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
    for i in range(20):
        card = AgentCard(
            id=f"agent-{i}",
            name=f"Agent {i}",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["finance" if i % 2 == 0 else "health"],
            role_tags=["evidence_finder"],
            description="test agent",
            embedding_text=f"agent {i} domain finance" if i % 2 == 0 else f"agent {i} domain health",
        )
        registry.register(card)

    out_dir = tmp_path / "index"
    dim = 32
    index = build_index(
        db_path=str(db_path),
        kind="agent",
        out_dir=str(out_dir),
        dim=dim,
        batch_size=10,
        seed=9,
    )
    embedder = DummyEmbedder(dim=dim, seed=9)
    app = create_app(registry=registry, index=index, embedder=embedder)
    client = TestClient(app)

    response = client.post(
        "/candidates",
        json={
            "task_text": "analyze finance trends",
            "role": "evidence_finder",
            "constraints": {"domain_tags": ["finance"]},
            "kind": "agent",
            "top_n": 10,
            "top_k": 5,
            "mmr_lambda": 0.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) <= 5
    assert payload
    assert all("finance" in item["brief_tags"]["domain_tags"] for item in payload)

    registry.close()


def test_dummy_embedder_constrained_search_uses_lexical_priority(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")
    service_mod = _import_from_src("retrieval.search_service")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder
    get_candidates = service_mod.get_candidates

    db_path = tmp_path / "mbpp_search.sqlite"
    registry = SQLiteRegistry(str(db_path))
    cards = [
        AgentCard(
            id="paper-agent-mbpp-set-operations-builder",
            name="Paper MBPP Set Operations Builder",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["mbpp", "set-operations", "code"],
            role_tags=["builder", "code-generation"],
            description="Builds set union intersection dedupe solutions.",
            embedding_text="mbpp set operations union intersection dedupe builder",
        ),
        AgentCard(
            id="paper-agent-mbpp-tuple-number-builder",
            name="Paper MBPP Tuple Number Builder",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["mbpp", "tuple-number", "code"],
            role_tags=["builder", "code-generation"],
            description="Builds tuple integer digit number conversion solutions.",
            embedding_text="mbpp tuple number integer digits conversion builder",
        ),
    ]
    for card in cards:
        registry.register(card)

    dim = 32
    index = build_index(
        db_path=str(db_path),
        kind="agent",
        out_dir=str(tmp_path / "index"),
        dim=dim,
        batch_size=2,
        seed=7,
    )
    embedder = DummyEmbedder(dim=dim, seed=7)
    payload = get_candidates(
        task_text="Write a function tuple_to_int that converts a tuple of digits into an integer number.",
        role="builder",
        constraints={"domain_tags": ["mbpp"], "role_tags": ["builder"]},
        kind="agent",
        top_n=1,
        top_k=1,
        mmr_lambda=0.5,
        registry=registry,
        index=index,
        embedder=embedder,
        use_mmr=True,
    )

    assert payload[0]["card_id"] == "paper-agent-mbpp-tuple-number-builder"
    registry.close()


def test_dummy_embedder_lexical_rank_splits_identifiers_and_ignores_generic_code_words(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")
    service_mod = _import_from_src("retrieval.search_service")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder
    get_candidates = service_mod.get_candidates

    db_path = tmp_path / "identifier_search.sqlite"
    registry = SQLiteRegistry(str(db_path))
    for card in [
        AgentCard(
            id="paper-agent-mbpp-regex-matching-builder",
            name="Paper MBPP Regex Matching Builder",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["mbpp", "regex-matching", "code"],
            role_tags=["builder", "code-generation"],
            embedding_text="mbpp regex match pattern string python function code builder",
        ),
        AgentCard(
            id="paper-agent-mbpp-geometry-formula-builder",
            name="Paper MBPP Geometry Formula Builder",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["mbpp", "geometry-formula", "code"],
            role_tags=["builder", "code-generation"],
            embedding_text="mbpp geometry area perimeter volume builder",
        ),
    ]:
        registry.register(card)

    dim = 32
    index = build_index(
        db_path=str(db_path),
        kind="agent",
        out_dir=str(tmp_path / "index"),
        dim=dim,
        batch_size=2,
        seed=7,
    )
    embedder = DummyEmbedder(dim=dim, seed=7)
    payload = get_candidates(
        task_text="Write a Python function area_tetrahedron that returns the surface area.",
        role="builder",
        constraints={"domain_tags": ["mbpp"], "role_tags": ["builder"]},
        kind="agent",
        top_n=2,
        top_k=1,
        mmr_lambda=0.5,
        registry=registry,
        index=index,
        embedder=embedder,
        use_mmr=True,
    )

    assert payload[0]["card_id"] == "paper-agent-mbpp-geometry-formula-builder"
    registry.close()


def test_search_text_strips_dataset_names_from_retrieval_fields() -> None:
    search_text_mod = _import_from_src("retrieval.search_text")

    card = type(
        "Card",
        (),
        {
            "name": "GSM8KRatioChainBuilder-RealTools",
            "description": "Deterministic local GSM8K tool for ratio chain tasks.",
            "embedding_text": "GSM8K specialized agent. Tools: gsm8k-ratio-chain-builder-equation-engine",
            "domain_tags": ["gsm8k", "math", "gsm8k-ratio-chain"],
            "role_tags": ["builder"],
            "tool_tags": ["gsm8k-ratio-chain", "ratio"],
            "modalities": ["text"],
            "output_formats": ["json"],
        },
    )()
    text = search_text_mod.card_search_text(card).lower()

    for token in ("gsm8k", "mbpp", "humaneval", "hotpotqa", "math"):
        assert token not in text
    assert "ratio" in text
    assert "builder" in text
