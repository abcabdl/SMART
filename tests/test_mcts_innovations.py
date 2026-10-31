import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execution import orchestrator
from execution.orchestrator import (
    TopologyConfig,
    TopologyType,
    _compute_gap_state,
    _gap_reduction_score,
    _select_agent_with_mcts_dynamic,
)


class _Registry:
    def __init__(self, cards):
        self._cards = dict(cards)

    def get(self, card_id):
        return self._cards.get(card_id)


def _agent(card_id: str, *, tools, role_tags=None, cost=0.1, latency=0.1, quality=0.8):
    return SimpleNamespace(
        id=card_id,
        name=card_id,
        available_tool_ids=list(tools),
        role_tags=list(role_tags or []),
        cost=cost,
        latency=latency,
        quality=quality,
        input_schema={},
        output_schema={},
        domain_tags=[],
        tool_tags=[],
        description=card_id,
    )


def _tool(card_id: str, *, stage="", capability="", produces=None):
    return SimpleNamespace(
        id=card_id,
        name=card_id,
        stage=stage,
        capability_kind=capability,
        produces=list(produces or []),
    )


def test_compute_gap_state_for_code_and_math() -> None:
    code_gap = _compute_gap_state(
        {
            "task_family": "code",
            "dataset": "humaneval",
            "entry_point": "",
            "draft_code": "",
            "normalized_code": "",
            "verification": {},
            "diagnostics": {"error_summary": "Traceback: syntax error"},
        }
    )
    assert code_gap["gap_vector"]["missing_entry_point"] is True
    assert code_gap["gap_vector"]["missing_draft_code"] is True
    assert code_gap["failure_flags"]["has_failure_trace"] is True

    math_gap = _compute_gap_state(
        {
            "task_family": "math",
            "dataset": "math",
            "reasoning_trace": ["numbers extracted"],
            "candidate_answer": "42",
        }
    )
    assert math_gap["gap_vector"]["missing_structure_parse"] is False
    assert math_gap["gap_vector"]["missing_exact_relation"] is True
    assert math_gap["gap_vector"]["missing_verification"] is True


def test_gap_reduction_uses_equal_active_gap_weights() -> None:
    gap_state = _compute_gap_state(
        {
            "task_family": "gsm8k",
            "dataset": "gsm8k",
            "reasoning_trace": ["numbers extracted"],
            "candidate_answer": "",
        }
    )
    relation_agent = _agent("agent-rel", tools=[])
    formatter_agent = _agent("agent-format", tools=[])
    relation_score = _gap_reduction_score(
        gap_state,
        relation_agent,
        "checker",
        {
            "stage:solve_support": 1.0,
            "capability:relation-calculator": 1.0,
        },
    )
    formatter_score = _gap_reduction_score(
        gap_state,
        formatter_agent,
        "checker",
        {
            "stage:format": 1.0,
            "capability:answer-formatter": 1.0,
        },
    )
    assert abs(relation_score - formatter_score) < 1e-9


def test_gap_conditioned_mcts_emits_richer_trace(monkeypatch) -> None:
    registry = _Registry(
        {
            "checker-rel": _agent("checker-rel", tools=["rel-tool"], role_tags=["checker"], quality=0.9),
            "checker-format": _agent("checker-format", tools=["format-tool"], role_tags=["checker"], quality=0.9),
            "rel-tool": _tool(
                "rel-tool",
                stage="solve_support",
                capability="relation-calculator",
                produces=["reasoning_trace"],
            ),
            "format-tool": _tool(
                "format-tool",
                stage="format",
                capability="answer-formatter",
                produces=["formatted_answer"],
            ),
        }
    )

    def _fake_prepare_role_candidates(**kwargs):
        role = kwargs["role"]
        if role == "checker":
            return [], [
                {"card_id": "checker-format", "score": 0.90},
                {"card_id": "checker-rel", "score": 0.90},
            ]
        return [], []

    monkeypatch.setattr(orchestrator, "_prepare_role_candidates", _fake_prepare_role_candidates)

    selection = _select_agent_with_mcts_dynamic(
        task_text="Solve the word problem and verify the relation between quantities.",
        role="checker",
        step_idx=0,
        config=TopologyConfig(topology=TopologyType.SINGLE, roles=["checker"], max_steps=1).normalized(default_roles=["checker"]),
        constraints_per_role={"checker": {}},
        selected_agent_ids=[],
        selected_agent_roles=[],
        registry=registry,
        index=None,
        embedder=None,
        reranker=None,
        top_n=5,
        top_k=5,
        rerank_top_m=5,
        mmr_lambda=0.5,
        router_no_rerank=True,
        team_lambda_compat=1.0,
        team_mu_cost=0.05,
        team_nu_latency=0.05,
        mcts_iterations=24,
        mcts_rollout_depth=2,
        mcts_exploration=1.2,
        mcts_discount=0.95,
        mcts_max_candidates=5,
        pair_success_priors=None,
        role_success_priors={
            ("checker", "checker-rel"): 0.92,
            ("checker", "checker-format"): 0.45,
        },
        history=[],
        artifact_bag={
            "task_family": "gsm8k",
            "dataset": "gsm8k",
            "reasoning_trace": ["numbers extracted"],
            "candidate_answer": "",
        },
        task_family="gsm8k",
        dataset="gsm8k",
    )

    assert selection is not None
    assert selection["selected_main"] == "checker-rel"
    assert selection["mcts"]["current_gap_state"]["gap_vector"]["missing_relations"] is True
    rel_child = selection["mcts"]["children"]["checker-rel"]
    fmt_child = selection["mcts"]["children"]["checker-format"]
    assert "unary_score" in rel_child
    assert "transition_score" in rel_child
    assert "io_score" in rel_child
    assert "gap_state" in rel_child
    assert abs(rel_child["gap_reduction"] - fmt_child["gap_reduction"]) < 1e-9
