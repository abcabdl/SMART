import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_failure_router_contract() -> None:
    router = _import_from_src("core.failure_router")
    result = router.route_failure(
        role="planner",
        output=None,
        validation_errors=["missing steps"],
        executor_result=None,
    )
    assert result["failure_type"] == "A_contract"
    assert result["action"] == "rewrite_format"


def test_failure_router_missing_info() -> None:
    router = _import_from_src("core.failure_router")
    result = router.route_failure(
        role="evidence_finder",
        output={"evidence_targets": [], "sources": [], "evidence_points": []},
        validation_errors=None,
        executor_result=None,
    )
    assert result["failure_type"] == "B_missing_info"
    assert result["action"] == "request_more_info"


def test_failure_router_capability() -> None:
    router = _import_from_src("core.failure_router")
    result = router.route_failure(
        role="builder",
        output={"runnable_plan": ["do"], "code_or_commands": "run", "self_test": ["ok"]},
        validation_errors=None,
        executor_result={"ok": False, "error": {"code": "timeout", "message": "slow"}},
    )
    assert result["failure_type"] == "C_capability"
    assert result["action"] == "swap_agent"
