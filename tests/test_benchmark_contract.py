import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execution.orchestrator import (
    _annotate_dataset_candidates,
    _apply_paper_mode_defaults,
    _apply_dataset_selection_tiebreak,
    _artifact_bag_to_role_output,
    _benchmark_tool_decision,
    _build_tool_exec_inputs,
    _ensure_role_constraints,
    _initialize_artifact_bag,
    _merge_role_output_into_artifact_bag,
    _merge_tool_payload_into_artifact_bag,
    _sanitize_task_context_for_inference,
)


def _code_tools():
    return [
        {"id": "code-signature", "stage": "analyze", "produces": ["entry_point"]},
        {"id": "code-normalizer", "stage": "prepare", "produces": ["normalized_code"]},
        {"id": "code-struct-check", "stage": "verify", "produces": ["verification"]},
        {"id": "code-public-tests", "stage": "execute", "produces": ["verification"]},
        {"id": "code-traceback", "stage": "diagnose", "produces": ["diagnostics"]},
    ]


def _math_tools():
    return [
        {"id": "gsm8k-quantity", "stage": "analyze", "produces": ["reasoning_trace"]},
        {"id": "gsm8k-relations", "stage": "solve_support", "produces": ["reasoning_trace"]},
        {"id": "gsm8k-verify", "stage": "verify", "produces": ["verification"]},
        {"id": "gsm8k-format", "stage": "format", "produces": ["formatted_answer"]},
    ]


def _qa_tools():
    return [
        {"id": "hotpotqa-decompose", "stage": "analyze", "produces": ["reasoning_trace"]},
        {"id": "hotpotqa-index", "stage": "retrieve", "produces": ["evidence_points"]},
        {"id": "hotpotqa-chain", "stage": "solve_support", "produces": ["reasoning_trace", "evidence_points"]},
        {"id": "hotpotqa-answer", "stage": "answer", "produces": ["candidate_answer", "final_answer"]},
        {"id": "hotpotqa-verify", "stage": "verify", "produces": ["verification"]},
    ]


def test_paper_mode_defaults_for_benchmark_tasks() -> None:
    defaults = _apply_paper_mode_defaults(
        paper_mode=True,
        task_family="code",
        top_n=7,
        top_k=3,
        topology="linear",
        dynamic_topology=False,
        mcts_dynamic_optimization=False,
        mcts_iterations=9,
        mcts_rollout_depth=2,
        mcts_exploration=0.5,
        mcts_discount=0.7,
        mcts_max_candidates=11,
        team_mu_cost=0.2,
        team_nu_latency=0.3,
    )

    assert defaults["top_n"] == 30
    assert defaults["top_k"] == 15
    assert defaults["topology"] == "auto"
    assert defaults["dynamic_topology"] is True
    assert defaults["mcts_dynamic_optimization"] is True
    assert defaults["mcts_iterations"] == 64
    assert defaults["mcts_rollout_depth"] == 4
    assert defaults["mcts_exploration"] == 1.414
    assert defaults["mcts_discount"] == 0.95
    assert defaults["mcts_max_candidates"] == 5
    assert defaults["team_mu_cost"] == 0.05
    assert defaults["team_nu_latency"] == 0.05


def test_public_only_inference_context_removes_evaluator_fields() -> None:
    task_context = {
        "dataset": "hotpotqa",
        "question": "Which city is linked to both facts?",
        "context": [["City", ["City is mentioned."]]],
        "answer": "Paris",
        "expected_answer": "Paris",
        "supporting_facts": [["City", 0]],
        "hidden_tests": ["assert secret"],
        "test": "assert hidden()",
        "public_tests": ["assert visible()"],
    }

    sanitized = _sanitize_task_context_for_inference(task_context, "public_only")
    assert sanitized["inference_visibility"] == "public_only"
    for key in ("answer", "expected_answer", "supporting_facts", "hidden_tests"):
        assert key not in sanitized
    assert sanitized["test_list"] == ["assert visible()"]
    assert "test" not in sanitized

    artifact_bag = _initialize_artifact_bag(
        task_text="Which city is linked to both facts?",
        task_context=sanitized,
        extra_context={"task_family": "qa", "dataset": "hotpotqa"},
    )
    exec_inputs = _build_tool_exec_inputs(
        task_text="Which city is linked to both facts?",
        role="checker",
        agent_context={},
        results={},
        task_context=sanitized,
        artifact_bag=artifact_bag,
        tool_info={"consumes": ["question", "context"]},
    )

    for key in ("expected", "expected_answer", "gold_answer", "supporting_facts", "hidden_tests"):
        assert key not in exec_inputs
    assert exec_inputs["test_list"] == ["assert visible()"]


def test_benchmark_code_tool_planner_follows_capability_graph() -> None:
    tools = _code_tools()
    artifact_bag = {
        "task_family": "code",
        "dataset": "humaneval",
        "entry_point": "",
        "tests_public": ["assert solve(1) == 1"],
        "draft_code": "",
        "normalized_code": "",
    }
    context = {"artifact_bag": artifact_bag}

    decision = _benchmark_tool_decision("builder", context, tools, [])
    assert decision["action"] == "tool"
    assert decision["tool_id"] == "code-signature"

    artifact_bag["entry_point"] = "solve"
    artifact_bag["draft_code"] = "def solve(x):\n    return x\n"
    decision = _benchmark_tool_decision("builder", context, tools, [])
    assert decision["tool_id"] == "code-normalizer"

    artifact_bag["normalized_code"] = artifact_bag["draft_code"]
    decision = _benchmark_tool_decision("builder", context, tools, [])
    assert decision["tool_id"] == "code-public-tests"

    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [{"tool_id": "code-public-tests", "stage": "execute"}],
    )
    assert decision["tool_id"] == "code-struct-check"


def test_benchmark_math_tool_planner_separates_generation_and_verification() -> None:
    tools = _math_tools()
    context = {
        "artifact_bag": {
            "task_family": "gsm8k",
            "dataset": "gsm8k",
            "candidate_answer": "",
            "formatted_answer": "",
        }
    }

    decision = _benchmark_tool_decision("builder", context, tools, [])
    assert decision["tool_id"] == "gsm8k-quantity"

    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [{"tool_id": "gsm8k-quantity", "stage": "analyze"}],
    )
    assert decision["tool_id"] == "gsm8k-relations"

    context["artifact_bag"]["candidate_answer"] = "18"
    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [
            {"tool_id": "gsm8k-quantity", "stage": "analyze"},
            {"tool_id": "gsm8k-relations", "stage": "solve_support"},
        ],
    )
    assert decision["tool_id"] == "gsm8k-verify"

    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [
            {"tool_id": "gsm8k-quantity", "stage": "analyze"},
            {"tool_id": "gsm8k-relations", "stage": "solve_support"},
            {"tool_id": "gsm8k-verify", "stage": "verify"},
        ],
    )
    assert decision["tool_id"] == "gsm8k-format"


def test_benchmark_qa_tool_planner_follows_evidence_graph() -> None:
    tools = _qa_tools()
    context = {
        "artifact_bag": {
            "task_family": "qa",
            "dataset": "hotpotqa",
            "question": "What kind of company is this group?",
            "context": [["AIA Group", ["AIA is a life insurance group."]]],
            "evidence_points": [],
            "candidate_answer": "",
            "formatted_answer": "",
        }
    }

    decision = _benchmark_tool_decision("builder", context, tools, [])
    assert decision["tool_id"] == "hotpotqa-decompose"

    decision = _benchmark_tool_decision("builder", context, tools, [{"tool_id": "hotpotqa-decompose", "stage": "analyze"}])
    assert decision["tool_id"] == "hotpotqa-index"

    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [
            {"tool_id": "hotpotqa-decompose", "stage": "analyze"},
            {"tool_id": "hotpotqa-index", "stage": "retrieve"},
        ],
    )
    assert decision["tool_id"] == "hotpotqa-chain"

    context["artifact_bag"]["evidence_points"] = [{"title": "AIA Group", "sentence": "AIA is a life insurance group."}]
    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [
            {"tool_id": "hotpotqa-decompose", "stage": "analyze"},
            {"tool_id": "hotpotqa-index", "stage": "retrieve"},
            {"tool_id": "hotpotqa-chain", "stage": "solve_support"},
        ],
    )
    assert decision["tool_id"] == "hotpotqa-answer"

    context["artifact_bag"]["candidate_answer"] = "life insurance group"
    decision = _benchmark_tool_decision(
        "builder",
        context,
        tools,
        [
            {"tool_id": "hotpotqa-decompose", "stage": "analyze"},
            {"tool_id": "hotpotqa-index", "stage": "retrieve"},
            {"tool_id": "hotpotqa-chain", "stage": "solve_support"},
            {"tool_id": "hotpotqa-answer", "stage": "answer"},
        ],
    )
    assert decision["tool_id"] == "hotpotqa-verify"


def test_artifact_contract_code_flow_roundtrip() -> None:
    artifact_bag = _initialize_artifact_bag(
        task_text="Implement solve(x)",
        task_context={
            "dataset": "humaneval",
            "entry_point": "solve",
            "test_list": ["assert solve(1) == 1"],
        },
        extra_context={"task_family": "code", "dataset": "humaneval"},
    )

    _merge_role_output_into_artifact_bag(
        artifact_bag,
        "builder",
        {
            "draft_code": "def solve(x):\n    return x\n",
            "code_or_commands": "def solve(x):\n    return x\n",
            "runnable_plan": ["write implementation"],
        },
    )
    _merge_tool_payload_into_artifact_bag(
        artifact_bag,
        {"output": {"normalized_code": "def solve(x):\n    return x\n", "syntax_ok": True}},
        {"stage": "prepare", "produces": ["normalized_code", "verification"]},
    )
    _merge_tool_payload_into_artifact_bag(
        artifact_bag,
        {"output": {"passed": True, "stdout": "ok"}},
        {"stage": "execute", "produces": ["verification"]},
    )

    output = _artifact_bag_to_role_output("builder", artifact_bag, [])
    assert output["code_or_commands"] == "def solve(x):\n    return x\n"
    assert "write implementation" in output["runnable_plan"]
    assert artifact_bag["verification"]["passed"] is True


def test_code_tools_wrap_body_only_candidates_before_execution() -> None:
    import importlib

    catalog = importlib.import_module("generation.real_dataset_bulk_catalog")
    namespace = {}
    exec(
        compile(
            catalog._make_tool_code("mbpp", "Numeric Rounding", ["number"], "builder", "code-normalizer"),
            "code-normalizer",
            "exec",
        ),
        namespace,
    )
    normalize = namespace["run"]
    normalized = normalize(
        {
            "code": "return x + 1",
            "entry_point": "f",
            "param_names": ["x"],
            "prompt": "def f(x):\n    pass\n",
        }
    )["output"]

    assert normalized["code_shape"] == "wrapped_body"
    assert normalized["normalized_code"].startswith("def f(x):\n    return x + 1")

    namespace = {}
    exec(
        compile(
            catalog._make_tool_code("mbpp", "Numeric Rounding", ["number"], "builder", "structure-checker"),
            "structure-checker",
            "exec",
        ),
        namespace,
    )
    check = namespace["run"]
    checked = check(
        {
            "code": "return x + 1",
            "entry_point": "f",
            "param_names": ["x"],
            "prompt": "def f(x):\n    pass\n",
        }
    )["output"]

    assert checked["function_found"] is True
    assert checked["parameters"] == ["x"]


def test_generated_public_test_runner_accepts_wrapped_body_candidates() -> None:
    import importlib

    catalog = importlib.import_module("generation.real_dataset_bulk_catalog")
    namespace = {}
    exec(
        compile(
            catalog._make_tool_code("mbpp", "List Sorting", ["list"], "builder", "public-test-runner"),
            "public-test-runner",
            "exec",
        ),
        namespace,
    )
    run_tool = namespace["run"]
    result = run_tool(
        {
            "code": "return x + 1",
            "entry_point": "f",
            "param_names": ["x"],
            "prompt": "def f(x):\n    pass\n",
            "test_list": ["assert f(1) == 2", "assert f(4) == 5"],
            "timeout_s": 2.0,
        }
    )["output"]

    assert result["passed"] is True


def test_generated_assert_scorer_ignores_mbpp_check_wrapper() -> None:
    import importlib

    catalog = importlib.import_module("generation.real_dataset_bulk_catalog")
    namespace = {}
    exec(
        compile(
            catalog._make_tool_code("mbpp", "Regex Matching", ["pattern"], "builder", "assert-scorer"),
            "assert-scorer",
            "exec",
        ),
        namespace,
    )
    score_tool = namespace["run"]
    result = score_tool(
        {
            "code": "def same(x):\n    return x",
            "entry_point": "same",
            "param_names": ["x"],
            "prompt": "def same(x):\n    pass\n",
            "test": "def check(candidate):\n    assert candidate(1) == 1\n    assert candidate(2) == 2\ncheck(same)\n",
            "timeout_s": 2.0,
        }
    )["output"]

    assert result["passed"] is True
    assert result["assert_score_passed"] == 2
    assert result["assert_score_total"] == 2
    assert "IndentationError" not in result["stderr"]


def test_generated_assert_scorer_ignores_mbpp_pytest_wrapper() -> None:
    import importlib

    catalog = importlib.import_module("generation.real_dataset_bulk_catalog")
    namespace = {}
    exec(
        compile(
            catalog._make_tool_code("mbpp", "Regex Matching", ["pattern"], "builder", "assert-scorer"),
            "assert-scorer",
            "exec",
        ),
        namespace,
    )
    score_tool = namespace["run"]
    result = score_tool(
        {
            "code": "def same(x):\n    return x",
            "entry_point": "same",
            "param_names": ["x"],
            "prompt": "def same(x):\n    pass\n",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(1) == 1\n"
                "    assert candidate(2) == 2\n"
                "def test_check():\n"
                "    check(same)\n"
                "\n"
                "test_check()\n"
            ),
            "timeout_s": 2.0,
        }
    )["output"]

    assert result["passed"] is True
    assert result["assert_score_passed"] == 2
    assert result["assert_score_total"] == 2
    assert "IndentationError" not in result["stderr"]


def test_dataset_routing_prefers_same_dataset_candidate_with_small_score_gap() -> None:
    constraints = _ensure_role_constraints(
        {"domain_tags": ["assert-driven-programming", "code"]},
        "builder",
        dataset="mbpp",
        task_family="code",
    )
    assert constraints["domain_tags"] == ["assert-driven-programming"]

    candidates = _annotate_dataset_candidates(
        [
            {
                "card_id": "function-synthesis-builder",
                "score": 0.42,
                "brief_tags": {"domain_tags": ["python-function-synthesis", "code"]},
            },
            {
                "card_id": "assert-driven-builder",
                "score": 0.40,
                "brief_tags": {"domain_tags": ["assert-driven-programming", "code"]},
            },
        ],
        dataset="mbpp",
        task_family="code",
        role="builder",
    )

    selected = _apply_dataset_selection_tiebreak(
        "function-synthesis-builder",
        candidates,
        dataset="mbpp",
        task_family="code",
        role="builder",
        margin=0.03,
    )
    assert selected == "assert-driven-builder"


def test_artifact_contract_math_flow_roundtrip() -> None:
    artifact_bag = _initialize_artifact_bag(
        task_text="If Tom has 6 apples and buys 12 more, how many apples?",
        task_context={"dataset": "gsm8k", "answer": "18"},
        extra_context={"task_family": "gsm8k", "dataset": "gsm8k"},
    )

    _merge_role_output_into_artifact_bag(
        artifact_bag,
        "builder",
        {
            "candidate_answer": "18",
            "reasoning_trace": ["6 + 12 = 18"],
        },
    )
    _merge_tool_payload_into_artifact_bag(
        artifact_bag,
        {"output": {"matches_expected": True, "expected_value": "18"}},
        {"stage": "verify", "produces": ["verification"]},
    )
    _merge_tool_payload_into_artifact_bag(
        artifact_bag,
        {"output": {"canonical_answer": "#### 18"}},
        {"stage": "format", "produces": ["formatted_answer"]},
    )

    output = _artifact_bag_to_role_output("builder", artifact_bag, [])
    assert output["code_or_commands"] == "18"
    assert output["final_answer"] == "#### 18"
    assert artifact_bag["verification"]["matches_expected"] is True


def test_artifact_contract_qa_flow_roundtrip() -> None:
    artifact_bag = _initialize_artifact_bag(
        task_text="What kind of company is this group?",
        task_context={
            "dataset": "hotpotqa",
            "question": "What kind of company is this group?",
            "answer": "pan-Asian life insurance group",
            "context": [["AIA Group", ["AIA is a pan-Asian life insurance group."]]],
            "supporting_facts": [["AIA Group", 0]],
        },
        extra_context={"task_family": "qa", "dataset": "hotpotqa"},
    )

    assert artifact_bag["context"]
    assert artifact_bag["supporting_facts"]

    _merge_role_output_into_artifact_bag(
        artifact_bag,
        "builder",
        {
            "candidate_answer": "pan-Asian life insurance group",
            "reasoning_trace": ["AIA Group sentence supports the answer."],
            "evidence_points": [{"title": "AIA Group", "sentence": "AIA is a pan-Asian life insurance group."}],
        },
    )
    _merge_tool_payload_into_artifact_bag(
        artifact_bag,
        {
            "output": {
                "matches_expected": True,
                "answer_in_evidence": True,
                "supporting_fact_coverage": 1.0,
                "formatted_answer": "pan-Asian life insurance group",
                "final_answer": "pan-Asian life insurance group",
            }
        },
        {"stage": "verify", "produces": ["verification", "formatted_answer", "final_answer"]},
    )

    output = _artifact_bag_to_role_output("builder", artifact_bag, [])
    assert output["code_or_commands"] == "pan-Asian life insurance group"
    assert output["final_answer"] == "pan-Asian life insurance group"
    assert artifact_bag["verification"]["matches_expected"] is True
