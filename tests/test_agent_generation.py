import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_agent_generation_and_mutation(tmp_path) -> None:
    registry_mod = _import_from_src("core.registry")
    tool_gen_mod = _import_from_src("generation.tool_generator")
    agent_gen_mod = _import_from_src("generation.agent_generator")
    mutation_mod = _import_from_src("generation.mutation_ops")

    SQLiteRegistry = registry_mod.SQLiteRegistry
    register_tool = tool_gen_mod.register_tool
    generate_agents = agent_gen_mod.generate_agents
    mutate_agent = mutation_mod.mutate_agent

    db_path = tmp_path / "agents.sqlite"
    with SQLiteRegistry(str(db_path)) as registry:
        register_tool(registry, "text_cleaner")
        register_tool(registry, "basic_stats")

        agents = generate_agents(
            domains=["finance", "health"],
            n_per_domain=5,
            roles=["analyst", "planner"],
            registry=registry,
        )
        assert len(agents) == 10
        assert len(registry.list({"kind": "agent"})) == 10

        mutated = mutate_agent(agents[0], "style")
        registry.register(mutated)

        fetched = registry.get(mutated.id)
        assert fetched is not None
        assert "parent_id" in fetched.description

        finance_agents = registry.list({"domain_tags": "finance"})
        assert len(finance_agents) >= 5
        assert all(isinstance(card.available_tool_ids, list) for card in finance_agents)
