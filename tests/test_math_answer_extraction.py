import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.eval_math import (
    _ensure_math_default_constraints,
    _extract_prediction_from_workflow_result,
    _prediction_is_resolved,
    parse_args,
)


def test_math_default_constraints_use_sanitized_domain_tags() -> None:
    constraints = _ensure_math_default_constraints({}, ["planner", "builder", "checker", "refactor"])

    for role in ("planner", "builder", "checker", "refactor"):
        assert constraints[role]["domain_tags"] == ["exact-competition"]
        joined = " ".join(constraints[role]["domain_tags"])
        assert "math" not in joined.lower()


def test_extract_prediction_allows_results_free_text_answer() -> None:
    result = {"results": {"builder": "Final answer is 129."}}

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "129"
    assert source == "results.builder.free_text_answer"
    assert _prediction_is_resolved(pred, source) is True


def test_extract_prediction_filters_tool_artifacts_even_with_convergence_marker() -> None:
    result = {
        "results": {
            "builder": (
                "Final answer: is formatted appropriately using "
                "math-builder-format-primedivisibility-4omini-t000-r880-v007."
            )
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "UNRESOLVED_NO_FINAL_ANSWER"
    assert source == "unresolved_no_final_answer"


def test_extract_prediction_strips_trailing_punctuation_for_numeric_answer() -> None:
    result = {"results": {"checker": "Answer is 11/16."}}

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "11/16"
    assert source == "results.checker.free_text_answer"
    assert _prediction_is_resolved(pred, source) is True


def test_extract_prediction_prefers_checker_over_refactor_outputs() -> None:
    result = {
        "results": {
            "refactor": "#### -1198",
            "checker": "#### -449",
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "-449"
    assert source == "results.checker.hash_answer"


def test_extract_prediction_prefers_solve_stage_over_later_math_verify_and_format() -> None:
    result = {
        "tool_exec": {
            "planner": [
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
                                "payload": {"formatted": "9"},
                            }
                        },
                    },
                },
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "8"
    assert source == "tool_exec.planner.solve.final_answer"


def test_extract_prediction_prefers_metadata_solve_support_with_sanitized_tool_id() -> None:
    result = {
        "tool_exec": {
            "planner": [
                {
                    "tool_id": "counting-arrangement-planner-relation-calculator",
                    "stage": "solve_support",
                    "result": {
                        "ok": True,
                        "output": {"output": {"final_answer": "#### 8"}},
                    },
                },
                {
                    "tool_id": "counting-arrangement-planner-answer-verifier",
                    "stage": "verify",
                    "result": {
                        "ok": True,
                        "output": {"output": {"final_answer": "#### 18"}},
                    },
                },
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "8"
    assert source == "tool_exec.planner.solve_support.final_answer"


def test_extract_prediction_prefers_earliest_solve_across_roles_when_later_solves_conflict() -> None:
    result = {
        "tool_exec": {
            "planner": [
                {
                    "tool_id": "math-planner-solve-worktimemotion-4o-t024-r959-v009",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "final_answer": "#### 41",
                            }
                        },
                    },
                },
                {
                    "tool_id": "math-planner-verify-worktimemotion-4omini-t008-r526-v009",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "final_answer": "18",
                                "payload": {"recomputed_answer": 18},
                            }
                        },
                    },
                },
            ],
            "checker": [
                {
                    "tool_id": "math-checker-solve-sequencestatistics-4o-t012-r501-v013",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "final_answer": "#### 74",
                            }
                        },
                    },
                }
            ],
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "41"
    assert source == "tool_exec.planner.solve.final_answer"


def test_extract_prediction_uses_verify_when_no_solve_answer_exists() -> None:
    result = {
        "tool_exec": {
            "planner": [
                {
                    "tool_id": "math-planner-verify-unitconversion-4omini-t007-r374-v006",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "final_answer": "2/5",
                                "payload": {"recomputed_answer": "2/5"},
                            }
                        },
                    },
                }
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "2/5"
    assert source == "tool_exec.planner.verify.final_answer"


def test_extract_prediction_avoids_checks_rhs_when_math_tool_has_direct_answer() -> None:
    result = {
        "tool_exec": {
            "planner": [
                {
                    "tool_id": "math-planner-solve-geometryareaperimeter-4o-t025-r592-v004",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "final_answer": "135",
                            }
                        },
                    },
                },
                {
                    "tool_id": "math-planner-verify-geometryareaperimeter-4o-t010-r506-v004",
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "payload": {
                                    "recomputed_answer": "117",
                                    "checks": [
                                        {
                                            "step": "Equation setup",
                                            "details": "3mn = 12 * 3m",
                                        }
                                    ],
                                }
                            }
                        },
                    },
                },
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "135"
    assert source == "tool_exec.planner.solve.final_answer"


def test_parse_args_math_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_math.py", "--tasks", "dummy.jsonl"])

    args = parse_args()

    assert args.roles == "builder,checker"
    assert args.workflow_max_runs == 1
    assert args.strict_workflow_completion is False
