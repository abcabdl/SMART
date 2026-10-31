import ast
import importlib.util
import sys
from pathlib import Path


def _load_tool_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_select_tools_disables_param_and_scope_on_assertion() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from evaluation import humaneval_postprocess as hp

    tools = hp._select_postprocess_tools(
        "Traceback ... AssertionError",
        ["sentence"],
    )
    assert "postprocess-param-consistency" not in tools
    assert "postprocess-name-scope-repair" not in tools


def test_name_scope_only_enabled_for_name_resolution_errors() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from evaluation import humaneval_postprocess as hp

    tools = hp._select_postprocess_tools("Traceback ... TypeError: bad operand type", ["n"])
    assert "postprocess-name-scope-repair" not in tools


def test_param_consistency_gate_only_for_related_nameerror() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from evaluation import humaneval_postprocess as hp

    unrelated = (
        "Traceback (most recent call last):\n"
        "NameError: name 'n' is not defined"
    )
    related = (
        "Traceback (most recent call last):\n"
        "NameError: name 'num' is not defined"
    )
    assert hp._should_use_param_consistency(unrelated, ["sentence"]) is False
    assert hp._should_use_param_consistency(related, ["n"]) is True


def test_error_regression_gate_rejects_worse_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from evaluation import humaneval_postprocess as hp

    before = "Traceback ... NameError: name 'n' is not defined"
    worse = "Traceback ... TypeError: '<=' not supported between instances of 'str' and 'int'"
    better = "Traceback ... AssertionError"
    assert hp._is_error_class_regressed(before, worse) is True
    assert hp._is_error_class_regressed(before, better) is False


def test_param_consistency_uses_ast_local_replace() -> None:
    root = Path(__file__).resolve().parents[1]
    tool = _load_tool_module(root / "generated_tools" / "postprocess-param-consistency.py")

    completion = "\n".join(
        [
            "value = num + 1",
            "msg = 'num should stay'",
            "return value",
        ]
    )
    out = tool.run(
        {
            "completion": completion,
            "entry_point": "prime_fib",
            "param_names": ["n"],
            "error_message": "NameError: name 'num' is not defined",
        }
    )
    assert out["changed"] is True
    assert "value = n + 1" in out["completion"]
    assert "'num should stay'" in out["completion"]


def test_name_scope_repair_helper_keeps_relative_indent() -> None:
    root = Path(__file__).resolve().parents[1]
    tool = _load_tool_module(root / "generated_tools" / "postprocess-name-scope-repair.py")

    completion = "\n".join(
        [
            "    if is_prime(intersection_length):",
            "        return 'YES'",
            "    return 'NO'",
        ]
    )
    out = tool.run(
        {
            "completion": completion,
            "entry_point": "intersection",
            "error_message": "NameError: name 'is_prime' is not defined",
        }
    )
    assert out["changed"] is True
    lines = out["completion"].splitlines()
    helper_idx = lines.index("    def is_prime(n):")
    assert lines[helper_idx + 1].startswith("        if n <= 1:")
    wrapped = "def intersection(interval1, interval2):\n" + out["completion"] + "\n"
    ast.parse(wrapped)


def test_param_consistency_prefers_python_hint_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    tool = _load_tool_module(root / "generated_tools" / "postprocess-param-consistency.py")

    completion = "\n".join(
        [
            "value = lst + 1",
            "return value",
        ]
    )
    out = tool.run(
        {
            "completion": completion,
            "entry_point": "total_match",
            "param_names": ["lst1"],
            "error_message": "NameError: name 'lst' is not defined. Did you mean: 'lst1'?",
        }
    )
    assert out["changed"] is True
    assert "value = lst1 + 1" in out["completion"]


def test_name_scope_repair_can_fix_nameerror_with_hint() -> None:
    root = Path(__file__).resolve().parents[1]
    tool = _load_tool_module(root / "generated_tools" / "postprocess-name-scope-repair.py")

    completion = "\n".join(
        [
            "if num <= 1:",
            "    return False",
            "return True",
        ]
    )
    out = tool.run(
        {
            "completion": completion,
            "entry_point": "is_prime",
            "param_names": ["n"],
            "error_message": "NameError: name 'num' is not defined. Did you mean: 'n'?",
        }
    )
    assert out["changed"] is True
    assert "if n <= 1:" in out["completion"]
