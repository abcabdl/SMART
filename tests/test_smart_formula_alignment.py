import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.mmr import mmr_rerank
from routing.smart_mcts import select_with_smart_mcts
from routing.smart_reward import build_smart_candidates, io_chain_score, score_smart_reward


def test_mmr_first_step_has_zero_diversity_and_lambda_edges() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    candidates = np.asarray([
        [0.9, 0.0],
        [0.6, 0.8],
        [0.1, 0.99],
    ], dtype=np.float32)

    relevant, _ = mmr_rerank(query, candidates, lambda_param=1.0, top_k=2)
    diverse, _ = mmr_rerank(query, candidates, lambda_param=0.0, top_k=2)

    assert relevant[0] == 0
    assert diverse[0] == 0
    assert diverse[1] == 2


def test_smart_unary_candidate_normalization_and_top_k() -> None:
    candidates = [
        {"card_id": "a", "score": 0.2, "brief_tags": {}},
        {"card_id": "b", "score": 0.5, "brief_tags": {}},
        {"card_id": "c", "score": 0.9, "brief_tags": {}},
    ]
    smart = build_smart_candidates(
        candidates,
        raw_unary_scores={"a": 0.1, "b": 0.4, "c": 0.9},
        final_k=2,
    )

    assert [item.card_id for item in smart] == ["c", "b"]
    assert smart[0].unary_score == 1.0
    assert 0.0 < smart[1].unary_score < 1.0


def test_reward_uses_neutral_transition_and_io_chain_components() -> None:
    prev = SimpleNamespace(output_formats=["code"], modalities=["text"], tool_tags=["parse", "test"])
    curr = SimpleNamespace(output_formats=["code"], modalities=["text"], tool_tags=["test", "repair"])

    io_score = io_chain_score(prev, curr)
    reward = score_smart_reward(
        unary_score=1.0,
        transition_score=0.5,
        io_score=io_score,
        gap_reduction=0.5,
        use_transition_term=True,
        use_iochain=True,
        use_gap_term=True,
    )

    assert 0.5 <= io_score <= 1.0
    assert reward.components == ("Unary", "Trans", "IO", "Gap")
    assert abs(reward.reward - ((1.0 + 0.5 + io_score + 0.5) / 4.0)) < 1e-9


def test_smart_mcts_selects_by_visit_count_without_rollout_side_effects() -> None:
    calls = {"score": 0}

    def action_ids(role, selected_ids, selected_roles, gap_state, depth):
        return ["strong", "weak"] if depth == 0 else []

    def score_action(prev_ids, prev_roles, role, card_id, gap_state, depth):
        calls["score"] += 1
        reward = 0.9 if card_id == "strong" else 0.1
        return {
            "reward": reward,
            "unary_score": reward,
            "transition_score": 0.5,
            "io_score": 0.5,
            "gap_reduction": reward,
            "gap_state": {"gap_count": 0, "gap_vector": {}},
        }

    result = select_with_smart_mcts(
        initial_role="builder",
        initial_selected_ids=(),
        initial_selected_roles=(),
        initial_gap_state={"gap_count": 1, "gap_vector": {"missing_draft_code": True}},
        iterations=8,
        rollout_depth=2,
        exploration=1.414,
        discount=0.95,
        max_candidates=2,
        action_ids_for_node=action_ids,
        score_action=score_action,
        predict_next_role=lambda role, depth, gap_state: None,
    )

    assert result is not None
    assert result["selected_main"] in {"strong", "weak"}
    assert "visits" in result["children"][result["selected_main"]]
    assert calls["score"] > 0
