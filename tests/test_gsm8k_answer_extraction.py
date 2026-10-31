import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.eval_gsm8k import (
    _ensure_gsm_default_constraints,
    _extract_numeric_from_text,
    _extract_prediction_from_workflow_result,
)


def test_gsm_default_constraints_use_sanitized_domain_tags() -> None:
    constraints = _ensure_gsm_default_constraints({}, ["planner", "builder", "checker", "refactor"])

    for role in ("planner", "builder", "checker", "refactor"):
        assert constraints[role]["domain_tags"] == ["arithmetic-word-problem"]
        joined = " ".join(constraints[role]["domain_tags"])
        assert "gsm8k" not in joined.lower()


def test_extract_prediction_prefers_checker_final_answer_over_temperature() -> None:
    result = {
        "tool_trace": {
            "checker": [
                {
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "stage": "analyze",
                                "temperature": 0.02,
                                "payload": {"notes": "analysis"},
                            }
                        },
                    }
                },
                {
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "stage": "solve",
                                "temperature": 0.1,
                                "payload": {"numeric_answer": 25, "final_answer": 25},
                            }
                        },
                    }
                },
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "25"
    assert source == "tool_trace.checker.final_answer"


def test_extract_prediction_does_not_use_metadata_numeric_scalars() -> None:
    result = {
        "tool_trace": {
            "checker": [
                {
                    "result": {
                        "ok": True,
                        "output": {
                            "output": {
                                "tool_id": "gsm8k-verifier-analyze-x",
                                "family": "verifier",
                                "stage": "analyze",
                                "temperature": 0.02,
                                "confidence": 0.8,
                                "payload_source": "llm",
                            }
                        },
                    }
                }
            ]
        }
    }

    pred, source = _extract_prediction_from_workflow_result(result)

    assert pred == "UNRESOLVED_NO_FINAL_ANSWER"
    assert source == "unresolved_no_final_answer"


def test_extract_numeric_from_text_keeps_decimal_token_intact() -> None:
    value, token = _extract_numeric_from_text("The result is 7.5")
    assert token == "7.5"
    assert str(value) == "7.5"
