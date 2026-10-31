import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execution.orchestrator import (
    _extract_candidate_final_answer_from_tool_history,
    _extract_final_answer_from_tool_history,
    _should_replace_candidate_final_answer,
)


def test_orchestrator_prefers_solve_candidate_over_later_verify_and_format() -> None:
    tool_history = [
        {
            "tool_id": "math-planner-solve-countingarrangement-4o-t025-r944-v001",
            "result": {
                "ok": True,
                "output": {
                    "output": {
                        "final_answer": "#### 8",
                    }
                },
            },
        },
        {
            "tool_id": "math-planner-verify-countingarrangement-4o-t010-r756-v001",
            "result": {
                "ok": True,
                "output": {
                    "output": {
                        "final_answer": "#### 18",
                        "payload": {"recomputed_answer": "#### 18"},
                    }
                },
            },
        },
        {
            "tool_id": "math-planner-format-countingarrangement-4omini-t003-r413-v001",
            "result": {
                "ok": True,
                "output": {
                    "output": {
                        "final_answer": "#### 9",
                    }
                },
            },
        },
    ]

    answer, source = _extract_candidate_final_answer_from_tool_history(tool_history) or ("", "")

    assert answer == "8"
    assert source == "math-planner-solve-countingarrangement-4o-t025-r944-v001"
    assert _extract_final_answer_from_tool_history(tool_history) == "8"


def test_orchestrator_prefers_metadata_solve_support_with_sanitized_tool_id() -> None:
    tool_history = [
        {
            "tool_id": "counting-arrangement-planner-relation-calculator",
            "stage": "solve_support",
            "result": {"ok": True, "output": {"output": {"final_answer": "#### 8"}}},
        },
        {
            "tool_id": "counting-arrangement-planner-answer-verifier",
            "stage": "verify",
            "result": {"ok": True, "output": {"output": {"final_answer": "#### 18"}}},
        },
    ]

    answer, source = _extract_candidate_final_answer_from_tool_history(tool_history) or ("", "")

    assert answer == "8"
    assert source == "counting-arrangement-planner-relation-calculator|stage=solve_support"
    assert _extract_final_answer_from_tool_history(tool_history) == "8"


def test_orchestrator_does_not_replace_existing_solve_with_conflicting_later_candidate() -> None:
    current_state = {
        "answer": "41",
        "source": "math-planner-solve-worktimemotion-4o-t024-r959-v009",
    }

    assert (
        _should_replace_candidate_final_answer(
            current_state,
            "74",
            "math-checker-solve-sequencestatistics-4o-t012-r501-v013",
        )
        is False
    )


def test_orchestrator_does_not_replace_existing_metadata_solve_support() -> None:
    current_state = {
        "answer": "41",
        "source": "linear-equation-planner-relation-calculator|stage=solve_support",
    }

    assert (
        _should_replace_candidate_final_answer(
            current_state,
            "74",
            "sequence-statistics-checker-relation-calculator|stage=solve_support",
        )
        is False
    )


def test_orchestrator_replaces_non_solve_candidate_with_solve_candidate() -> None:
    current_state = {
        "answer": "2/5",
        "source": "math-planner-verify-unitconversion-4omini-t007-r374-v006",
    }

    assert (
        _should_replace_candidate_final_answer(
            current_state,
            "2/5",
            "math-checker-solve-unitconversion-4o-t008-r406-v006",
        )
        is True
    )
