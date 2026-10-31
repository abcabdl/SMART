import importlib
import sys
import time
from datetime import datetime
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def _make_registry(tmp_path, roles=("builder", "builder", "checker", "refactor")):
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder

    db_path = tmp_path / "parallel.sqlite"
    registry = SQLiteRegistry(str(db_path))
    for idx, role in enumerate(roles):
        registry.register(
            AgentCard(
                id=f"agent-{role}-{idx}",
                name=f"{role.title()} Agent {idx}",
                kind="agent",
                version="1.0",
                updated_at=datetime(2026, 1, 1),
                domain_tags=["code", "parallel"],
                role_tags=[role],
                tool_tags=[],
                description=f"{role} benchmark parallel agent",
                embedding_text=f"{role} code benchmark parallel {idx}",
            )
        )
    index = build_index(db_path=str(db_path), kind="agent", out_dir=str(tmp_path / "index"), dim=16, seed=5)
    embedder = DummyEmbedder(dim=16, seed=5)
    return registry, index, embedder


def _make_hotpotqa_registry(tmp_path):
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder

    db_path = tmp_path / "hotpotqa_parallel.sqlite"
    registry = SQLiteRegistry(str(db_path))
    roles = ["planner", "builder", "builder", "checker", "builder"]
    for idx, role in enumerate(roles):
        registry.register(
            AgentCard(
                id=f"agent-hotpotqa-{role}-{idx}",
                name=f"HotpotQA {role.title()} Agent {idx}",
                kind="agent",
                version="1.0",
                updated_at=datetime(2026, 1, 1),
                domain_tags=["hotpotqa", "multi-hop-qa", "evidence-grounded-qa"],
                role_tags=[role],
                tool_tags=[],
                description=f"{role} HotpotQA benchmark parallel agent",
                embedding_text=f"{role} hotpotqa multi-hop evidence benchmark {idx}",
                dataset_family="hotpotqa",
            )
        )
    index = build_index(db_path=str(db_path), kind="agent", out_dir=str(tmp_path / "hotpotqa_index"), dim=16, seed=7)
    embedder = DummyEmbedder(dim=16, seed=7)
    return registry, index, embedder


def test_parallel_run_workflow_returns_parallel_payload_and_unique_agents(tmp_path) -> None:
    orch_mod = _import_from_src("execution.orchestrator")
    registry, index, embedder = _make_registry(tmp_path, roles=("builder", "builder", "builder", "checker"))
    try:
        result = orch_mod.run_workflow(
            task_text="implement a simple function",
            roles=["builder", "checker"],
            registry=registry,
            index=index,
            embedder=embedder,
            parallel_execution=True,
            parallel_width=3,
            parallel_branch_roles=["builder", "builder", "checker"],
            execute_tools=False,
            runs_dir=str(tmp_path / "runs"),
        )
    finally:
        registry.close()

    assert result["parallel"]["enabled"] is True
    assert len(result["parallel"]["branches"]) == 3
    branch_ids = [item.get("branch_id") for item in result["execution_trace"] if item.get("branch_id") != "aggregate"]
    assert len(set(branch_ids)) >= 2
    selected = [item["selected_main"] for item in result["selections"] if item.get("selected_main")]
    assert len(selected) == len(set(selected))


def test_parallel_run_workflow_keeps_success_when_another_branch_fails(tmp_path, monkeypatch) -> None:
    orch_mod = _import_from_src("execution.orchestrator")
    registry, index, embedder = _make_registry(tmp_path, roles=("builder", "checker"))

    def fake_execute(**kwargs):
        role = kwargs["role"]
        if role == "checker":
            return {
                "output": {"error": "simulated_failure"},
                "raw_output": {"error": "simulated_failure"},
                "executor_result": None,
                "artifact_bag": {},
                "log_path": None,
                "selected_main": kwargs["selection"].get("selected_main"),
                "selected_shadows": [],
            }
        return {
            "output": {"code_or_commands": "return 1", "test_passed": True},
            "raw_output": {"code_or_commands": "return 1", "test_passed": True},
            "executor_result": None,
            "artifact_bag": {"normalized_code": "return 1", "verification": {"test_passed": True}},
            "log_path": None,
            "selected_main": kwargs["selection"].get("selected_main"),
            "selected_shadows": [],
        }

    monkeypatch.setattr(orch_mod, "_execute_role_with_selection", fake_execute)
    try:
        result = orch_mod.run_workflow(
            task_text="implement a simple function",
            roles=["builder", "checker"],
            registry=registry,
            index=index,
            embedder=embedder,
            parallel_execution=True,
            parallel_width=2,
            parallel_branch_roles=["builder", "checker"],
            execute_tools=False,
            runs_dir=str(tmp_path / "runs"),
            parallel_fail_fast=False,
        )
    finally:
        registry.close()

    assert result["parallel"]["winner_branch_id"] == "builder-0"
    assert result["checker_ok"] is True


def test_parallel_run_workflow_records_branch_timeout(tmp_path, monkeypatch) -> None:
    orch_mod = _import_from_src("execution.orchestrator")
    registry, index, embedder = _make_registry(tmp_path, roles=("builder", "checker"))

    def slow_execute(**kwargs):
        if kwargs["role"] == "checker":
            time.sleep(0.05)
        return {
            "output": {"code_or_commands": "return 1"},
            "raw_output": {"code_or_commands": "return 1"},
            "executor_result": None,
            "artifact_bag": {"normalized_code": "return 1"},
            "log_path": None,
            "selected_main": kwargs["selection"].get("selected_main"),
            "selected_shadows": [],
        }

    monkeypatch.setattr(orch_mod, "_execute_role_with_selection", slow_execute)
    try:
        result = orch_mod.run_workflow(
            task_text="implement a simple function",
            roles=["builder", "checker"],
            registry=registry,
            index=index,
            embedder=embedder,
            parallel_execution=True,
            parallel_width=2,
            parallel_branch_roles=["builder", "checker"],
            execute_tools=False,
            runs_dir=str(tmp_path / "runs"),
            parallel_branch_timeout_s=0.001,
            parallel_fail_fast=False,
        )
    finally:
        registry.close()

    errors = [branch.get("error") for branch in result["parallel"]["branches"]]
    assert {"code": "branch_timeout", "message": "Branch exceeded 0.001s"} in errors


def test_hotpotqa_parallel_workflow_defaults_to_qa_branches_and_constraints(tmp_path, monkeypatch) -> None:
    orch_mod = _import_from_src("execution.orchestrator")
    registry, index, embedder = _make_hotpotqa_registry(tmp_path)

    def fake_execute(**kwargs):
        role = kwargs["role"]
        branch_id = kwargs["meta"]["branch_id"]
        return {
            "output": {"candidate_answer": "pan-Asian life insurance group", "final_answer": "pan-Asian life insurance group"},
            "raw_output": {"candidate_answer": "pan-Asian life insurance group"},
            "executor_result": None,
            "artifact_bag": {
                "task_family": "qa",
                "dataset": "hotpotqa",
                "candidate_answer": "pan-Asian life insurance group",
                "final_answer": "pan-Asian life insurance group",
                "expected_answer": "pan-Asian life insurance group",
                "verification": {"matches_expected": True, "supporting_fact_coverage": 1.0},
            },
            "log_path": None,
            "selected_main": kwargs["selection"].get("selected_main"),
            "selected_shadows": [],
        }

    monkeypatch.setattr(orch_mod, "_execute_role_with_selection", fake_execute)
    try:
        result = orch_mod.run_workflow(
            task_text="What kind of company is this group whose headquarters are in AIA Central?",
            roles=["planner", "builder", "checker"],
            constraints_per_role={
                "planner": {"domain_tags": ["hotpotqa"], "role_tags": ["planner"]},
                "builder": {"domain_tags": ["hotpotqa"], "role_tags": ["builder"]},
                "checker": {"domain_tags": ["hotpotqa"], "role_tags": ["checker"]},
            },
            registry=registry,
            index=index,
            embedder=embedder,
            parallel_execution=True,
            parallel_width=4,
            execute_tools=False,
            runs_dir=str(tmp_path / "runs"),
            dataset="hotpotqa",
            task_family="qa",
            parallel_fail_fast=False,
        )
    finally:
        registry.close()

    assert result["parallel"]["enabled"] is True
    assert [item["role"] for item in result["selections"]] == ["planner", "builder", "builder", "checker"]
    assert all(str(item["selected_main"]).startswith("agent-hotpotqa-") for item in result["selections"])
    assert result["answer"] == "pan-Asian life insurance group"
