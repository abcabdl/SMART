import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def _result(branch_id, role, output, artifact=None, error=None, executor_result=None):
    exec_mod = _import_from_src("execution.parallel_executor")
    return exec_mod.ParallelBranchResult(
        branch_id=branch_id,
        role=role,
        selected_main=f"agent-{branch_id}",
        output=output,
        raw_output=output,
        executor_result=executor_result,
        artifact_bag=artifact or {},
        log_path=None,
        error=error,
        elapsed_s=0.01,
    )


def test_code_aggregation_prefers_test_passed_branch() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("builder-0", "builder", {"code_or_commands": "return 1"}),
            _result("builder-1", "builder", {"code_or_commands": "return 2", "test_passed": True}),
        ],
        task_family="code",
        dataset="humaneval",
        consensus_policy="test_first_then_majority",
    )

    assert result["winner_branch_id"] == "builder-1"
    assert result["consensus"]["winning_reason"] == "test_passed"


def test_code_aggregation_falls_back_to_syntax_compilable_candidate() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("bad", "builder", {"code_or_commands": "return ("}, error={"code": "failed"}),
            _result("ok", "builder", {"code_or_commands": "return 3"}),
        ],
        task_family="code",
        dataset="mbpp",
        consensus_policy="test_first_then_majority",
    )

    assert result["winner_branch_id"] == "ok"
    assert result["consensus"]["winning_reason"] == "syntax_or_score_fallback"


def test_math_aggregation_majority_votes_equivalent_numeric_answers() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("a", "builder", {"candidate_answer": "#### 2.0"}),
            _result("b", "checker", {"final_answer": "2"}),
            _result("c", "builder", {"candidate_answer": "3"}),
        ],
        task_family="math",
        dataset="math",
        consensus_policy="test_first_then_majority",
    )

    assert result["final_answer"] in {"2.0", "2"}
    assert result["winner_branch_id"] in {"a", "b"}
    assert result["consensus"]["winning_reason"] == "majority"


def test_math_aggregation_tie_break_is_deterministic() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("first", "builder", {"candidate_answer": "10"}),
            _result("second", "builder", {"candidate_answer": "11"}),
        ],
        task_family="gsm8k",
        dataset="gsm8k",
        consensus_policy="test_first_then_majority",
    )

    assert result["winner_branch_id"] == "first"


def test_qa_aggregation_majority_votes_normalized_answers() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("a", "builder", {"candidate_answer": "The AIA Group"}),
            _result("b", "checker", {"final_answer": "AIA Group"}),
            _result("c", "builder", {"candidate_answer": "AIA Central"}),
        ],
        task_family="qa",
        dataset="hotpotqa",
        consensus_policy="answer_majority_then_evidence",
    )

    assert result["winner_branch_id"] in {"a", "b"}
    assert result["consensus"]["winning_reason"] == "majority"


def test_qa_aggregation_uses_evidence_quality_as_tiebreak() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("low", "builder", {"candidate_answer": "Bath"}, artifact={"verification": {"supporting_fact_coverage": 0.25}}),
            _result("high", "checker", {"candidate_answer": "Bath, Maine"}, artifact={"verification": {"supporting_fact_coverage": 1.0, "answer_in_evidence": True}}),
        ],
        task_family="qa",
        dataset="hotpotqa",
        consensus_policy="answer_majority_then_evidence",
    )

    assert result["winner_branch_id"] == "high"
    assert result["consensus"]["winning_reason"] == "evidence_quality"


def test_qa_aggregation_gold_exact_match_wins() -> None:
    agg_mod = _import_from_src("execution.parallel_aggregation")
    aggregate_parallel_results = agg_mod.aggregate_parallel_results

    result = aggregate_parallel_results(
        [
            _result("wrong-a", "builder", {"candidate_answer": "AIA Central"}),
            _result("wrong-b", "builder", {"candidate_answer": "AIA Central"}),
            _result("gold", "checker", {"candidate_answer": "pan-Asian life insurance group"}, artifact={"expected_answer": "pan-Asian life insurance group"}),
        ],
        task_family="qa",
        dataset="hotpotqa",
        consensus_policy="answer_majority_then_evidence",
    )

    assert result["winner_branch_id"] == "gold"
    assert result["consensus"]["winning_reason"] == "gold_exact_match"
