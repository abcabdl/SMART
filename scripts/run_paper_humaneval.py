from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import eval_humaneval
from evaluation.smart_pdf_preset import load_smart_pdf_preset, resolve_project_path


def main() -> None:
    preset = load_smart_pdf_preset()
    runtime = dict(preset.get("runtime") or {})
    paths = dict(preset.get("paths") or {})
    roles = ",".join(str(item) for item in (preset.get("roles") or []))
    argv = [
        "run_paper_humaneval",
        "--use_orchestrator",
        "--db",
        resolve_project_path(str(paths.get("benchmark_db"))),
        "--index_dir",
        resolve_project_path(str(paths.get("benchmark_index_dir"))),
        "--roles",
        roles,
        "--dynamic_topology",
        "--topology",
        "auto",
        "--mcts_dynamic_optimization",
        "--embedder",
        str(runtime.get("embedder") or "dummy"),
        "--dim",
        str(runtime.get("dim") or 64),
        "--seed",
        str(runtime.get("seed") or 7),
        "--disable_orchestrator_postprocess_repair",
    ]
    argv.extend(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = argv
        eval_humaneval.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
