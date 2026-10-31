import importlib
import sys
from datetime import datetime
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_registry_filter_and_query(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    constraints_mod = _import_from_src("core.constraints")

    ToolCard = cards_mod.ToolCard
    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    Constraints = constraints_mod.Constraints
    filter_candidates = constraints_mod.filter_candidates

    db_path = tmp_path / "registry.sqlite"

    card1 = ToolCard(
        id="tool-1",
        name="VectorSearch",
        kind="tool",
        version="1.0",
        updated_at=datetime(2026, 1, 1),
        domain_tags=["search", "nlp"],
        role_tags=["retrieval"],
        tool_tags=["vector"],
        modalities=["text"],
        output_formats=["json"],
        permissions=["read"],
        cost_tier="low",
        latency_tier="medium",
        reliability_prior=0.8,
        description="Vector search tool",
        examples=["find docs"],
        embedding_text="vector search",
    )
    card2 = AgentCard(
        id="agent-1",
        name="Planner",
        kind="agent",
        version="2.0",
        updated_at=datetime(2026, 1, 2),
        domain_tags=["planning"],
        role_tags=["planner"],
        tool_tags=["orchestrator"],
        modalities=["text"],
        output_formats=["json"],
        permissions=["write"],
        cost_tier="medium",
        latency_tier="low",
        reliability_prior=0.9,
        description="Planning agent",
        examples=["build plan"],
        embedding_text="planning",
    )
    card3 = AgentCard(
        id="agent-2",
        name="Analyst",
        kind="agent",
        version="1.1",
        updated_at=datetime(2026, 1, 3),
        domain_tags=["finance", "analysis"],
        role_tags=["analyst"],
        tool_tags=["calc"],
        modalities=["text"],
        output_formats=["markdown"],
        permissions=["read"],
        cost_tier="low",
        latency_tier="high",
        reliability_prior=0.7,
        description="Financial analyst",
        examples=["analyze report"],
        embedding_text="finance analyst",
    )

    with SQLiteRegistry(str(db_path)) as registry:
        registry.register(card1)
        registry.register(card2)
        registry.register(card3)

        tools = registry.list({"kind": "tool"})
        assert len(tools) == 1
        assert tools[0].id == "tool-1"

        low_cost = registry.list({"cost_tier": "low"})
        assert {card.id for card in low_cost} == {"tool-1", "agent-2"}

        analyst = registry.get("agent-2")
        assert analyst is not None
        assert analyst.name == "Analyst"

        updated = analyst.model_copy(update={"latency_tier": "medium"})
        registry.update(updated)
        fetched = registry.get("agent-2")
        assert fetched is not None
        assert fetched.latency_tier == "medium"

        registry.remove("agent-1")
        assert registry.get("agent-1") is None

        constraints = Constraints(domain_tags=["finance"], role_tags=["analyst"])
        filtered = filter_candidates(registry.list(), constraints)
        assert [card.id for card in filtered] == ["agent-2"]


def test_registry_persists_extended_tool_metadata(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")

    ToolCard = cards_mod.ToolCard
    SQLiteRegistry = registry_mod.SQLiteRegistry

    db_path = tmp_path / "registry_extended.sqlite"

    card = ToolCard(
        id="tool-bench-1",
        name="PublicTestRunner",
        kind="tool",
        version="1.0",
        updated_at=datetime(2026, 1, 4),
        domain_tags=["code"],
        role_tags=["checker"],
        tool_tags=["tests"],
        description="Runs benchmark public tests.",
        embedding_text="public tests benchmark tool",
        dataset_family="humaneval",
        focus_slug="public-test-runner",
        primary_role="checker",
        capability_kind="public-test-runner",
        stage="execute",
        deterministic=True,
        consumes=["normalized_code", "tests_public"],
        produces=["verification"],
    )

    with SQLiteRegistry(str(db_path)) as registry:
        registry.register(card)
        fetched = registry.get("tool-bench-1")

    assert fetched is not None
    assert fetched.dataset_family == "humaneval"
    assert fetched.focus_slug == "public-test-runner"
    assert fetched.primary_role == "checker"
    assert fetched.capability_kind == "public-test-runner"
    assert fetched.stage == "execute"
    assert fetched.deterministic is True
    assert fetched.consumes == ["normalized_code", "tests_public"]
    assert fetched.produces == ["verification"]
