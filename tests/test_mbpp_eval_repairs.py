import ast
import sys
from pathlib import Path

import pytest


def _add_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_humaneval_program_builder_imports_test_helper_modules() -> None:
    _add_src()
    from evaluation import humaneval_postprocess as hp

    program = hp._build_program(
        "def area(r):\n",
        "return math.pi * r * r",
        "def check(candidate):\n    assert math.isclose(candidate(2), 12.566370614359172)\ncheck(area)\n",
        "area",
    )

    ok, message = hp._run_python(program, timeout_s=2.0)
    assert ok, message


def test_fix_missing_indents_recovers_nested_mbpp_body() -> None:
    _add_src()
    from evaluation import humaneval_postprocess as hp

    broken = "\n".join(
        [
            "sum_of_powers = 0",
            "power = 1",
            "while sum_of_powers <= n:",
            "if sum_of_powers == n:",
            "    return True",
            "sum_of_powers += power",
            "power *= 2",
            "return False",
        ]
    )
    fixed = hp._fix_missing_indents(broken, "is_Sum_Of_Powers_Of_Two")
    wrapped = (
        "def is_Sum_Of_Powers_Of_Two(n):\n"
        f"{fixed}\n"
    )

    namespace = {}
    exec(wrapped, namespace)
    fn = namespace["is_Sum_Of_Powers_Of_Two"]
    assert fn(7) is True
    assert fn(10) is False


def test_fix_missing_indents_recovers_if_else_clause_alignment() -> None:
    _add_src()
    from evaluation import humaneval_postprocess as hp

    broken = "\n".join(
        [
            "if n % 2 == 0:",
            "return n",
            "else:",
            "return -1",
        ]
    )
    fixed = hp._fix_missing_indents(broken, "score_nums")
    wrapped = (
        "def score_nums(n):\n"
        f"{fixed}\n"
    )

    ast.parse(wrapped)
    namespace = {}
    exec(wrapped, namespace)
    fn = namespace["score_nums"]
    assert fn(8) == 8
    assert fn(3) == -1


def test_fix_missing_indents_recovers_try_except_clause_alignment() -> None:
    _add_src()
    from evaluation import humaneval_postprocess as hp

    broken = "\n".join(
        [
            "try:",
            "return nums[idx]",
            "except IndexError:",
            "return -1",
        ]
    )
    fixed = hp._fix_missing_indents(broken, "safe_get")
    wrapped = (
        "def safe_get(nums, idx):\n"
        f"{fixed}\n"
    )

    ast.parse(wrapped)
    namespace = {}
    exec(wrapped, namespace)
    fn = namespace["safe_get"]
    assert fn([10, 20], 1) == 20
    assert fn([10, 20], 4) == -1


def test_repair_hard_gate_accepts_body_only_completion() -> None:
    _add_src()
    from evaluation import eval_mbpp

    prompt = (
        "def replace_spaces(text):\n"
        "    \"\"\"\n"
        "    Replace spaces with underscores when needed.\n"
        "    \"\"\"\n"
    )
    completion = "\n".join(
        [
            "    if ' ' in text:",
            "        return text.replace(' ', '_')",
            "    return text.replace('_', ' ')",
        ]
    )

    error = eval_mbpp._repair_hard_gate_error(prompt, completion, "replace_spaces")
    assert error is None
    assert eval_mbpp._mbpp_hard_gate_mode(completion, "replace_spaces") == "wrapped_body"


def test_repair_hard_gate_rejects_full_def_signature_mismatch() -> None:
    _add_src()
    from evaluation import eval_mbpp

    prompt = (
        "def add_pair(a, b):\n"
        "    \"\"\"\n"
        "    Add two values.\n"
        "    \"\"\"\n"
    )
    completion = "\n".join(
        [
            "def add_pair(a):",
            "    return a",
        ]
    )

    error = eval_mbpp._repair_hard_gate_error(prompt, completion, "add_pair")
    assert error is not None
    assert "entry_point_signature_mismatch" in error


def test_mbpp_completion_diagnostics_detect_flattened_control_body() -> None:
    _add_src()
    from evaluation import eval_mbpp

    completion = "\n".join(
        [
            "    total = 0",
            "    for n in nums:",
            "    total += n",
            "    return total",
        ]
    )

    assert eval_mbpp._mbpp_completion_shape(completion, "sum_nums") == "body_with_control"
    assert eval_mbpp._mbpp_has_flattened_control_flow(completion, "sum_nums") is True


def test_mbpp_return_risk_detects_loop_internal_return() -> None:
    _add_src()
    from evaluation import eval_mbpp

    completion = "\n".join(
        [
            "    total = 0",
            "    for n in nums:",
            "        total += n",
            "        return total",
        ]
    )

    risk = eval_mbpp._mbpp_return_risk(completion, "sum_nums")
    assert risk["return_in_loop"] is True
    assert risk["return_lines"]


@pytest.mark.xfail(reason="legacy expectation predates the current MBPP repair prompt", strict=False)
def test_mbpp_failure_context_includes_return_risk_and_rewrite_mode() -> None:
    _add_src()
    from evaluation import eval_mbpp

    completion = "\n".join(
        [
            "    result = []",
            "    for item in items:",
            "        result.append(item)",
            "        return result",
        ]
    )
    test_code = (
        "def check(candidate):\n"
        "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
        "def test_check():\n"
        "    check(keep_all)\n"
        "test_check()\n"
    )

    context = eval_mbpp._failure_context_message(
        "AssertionError",
        test_code,
        current_completion=completion,
        entry_point="keep_all",
        rewrite_mode=True,
    )

    assert "return statement appears inside a loop" in context
    assert "Rewrite mode" in context
    assert "assert candidate([1, 2, 3]) == [1, 2, 3]" in context


def test_apply_default_constraints_prefers_mbpp_agents_for_mbpp_tasks() -> None:
    _add_src()
    from evaluation import eval_humaneval

    constraints = eval_humaneval._apply_humaneval_default_constraints(
        {},
        tasks=[{"source": "mbpp", "name": "MBPP_1"}],
    )

    assert constraints["builder"]["domain_tags"] == ["assert-driven-programming"]
    assert constraints["planner"]["domain_tags"] == ["assert-driven-programming"]
    assert constraints["checker"]["domain_tags"] == ["assert-driven-programming"]
