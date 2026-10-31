import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_demo_runner_end2end(tmp_path, monkeypatch) -> None:
    demo_mod = _import_from_src("execution.demo_runner")
    monkeypatch.chdir(tmp_path)

    result = demo_mod.run_demo(
        task_text="quick test",
        db_path=str(tmp_path / "demo.sqlite"),
        index_dir=str(tmp_path / "index"),
        tool_count=4,
        domains=["finance", "health"],
        n_per_domain=2,
        roles=["planner", "evidence_finder"],
        dim=16,
        seed=3,
        embedder_kind="dummy",
        workflow_version="test-v1",
    )

    log_path = result.get("log_path")
    assert log_path is not None
    assert Path(log_path).exists()


def test_demo_runner_parallel_end2end(tmp_path, monkeypatch) -> None:
    demo_mod = _import_from_src("execution.demo_runner")
    monkeypatch.chdir(tmp_path)

    result = demo_mod.run_demo(
        task_text="quick test",
        db_path=str(tmp_path / "parallel_demo.sqlite"),
        index_dir=str(tmp_path / "parallel_index"),
        tool_count=4,
        domains=["finance", "health"],
        n_per_domain=2,
        roles=["planner", "builder", "checker", "refactor"],
        dim=16,
        seed=3,
        embedder_kind="dummy",
        workflow_version="test-parallel-v1",
        parallel_execution=True,
        parallel_width=3,
    )

    assert result.get("parallel", {}).get("enabled") is True
    assert result.get("log_path") is not None
    assert Path(result["log_path"]).exists()
