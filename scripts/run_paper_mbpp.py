from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import eval_mbpp
from evaluation.smart_pdf_preset import load_smart_pdf_preset, resolve_project_path


def _semantic_index_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def _index_dim(index_dir: str) -> int:
    try:
        from retrieval.faiss_index import HNSWIndex
    except Exception:
        return 0
    index_path = Path(resolve_project_path(index_dir)) / "faiss.index"
    if not index_path.exists():
        return 0
    try:
        return int(HNSWIndex.load(str(index_path)).dim)
    except Exception:
        return 0


def main() -> None:
    preset = load_smart_pdf_preset()
    runtime = dict(preset.get("runtime") or {})
    paths = dict(preset.get("paths") or {})
    # MBPP benefits from direct implementation + checking; refactor is reserved for
    # post-failure repair rather than an unconditional initial role.
    roles = ",".join(str(item) for item in (preset.get("roles") or []) if str(item) != "refactor")
    embedder = str(runtime.get("embedder") or "dummy")
    embedder_model = str(runtime.get("embedder_model") or "sentence-transformers/all-MiniLM-L6-v2")
    dim = int(runtime.get("dim") or 64)
    index_dir = str(paths.get("benchmark_index_dir"))
    wants_semantic = embedder.lower() in {"sentence-transformer", "sentence_transformer", "sbert", "st"}
    semantic_ready = _semantic_index_available() and _index_dim(index_dir) == dim
    if wants_semantic and not semantic_ready:
        embedder = "dummy"
        dim = 64
        index_dir = str(paths.get("benchmark_index_dir_dummy") or "assets/agent_index_dummy")
    argv = [
        "run_paper_mbpp",
        "--db",
        resolve_project_path(str(paths.get("benchmark_db"))),
        "--index_dir",
        resolve_project_path(index_dir),
        "--roles",
        roles,
        "--dynamic_topology",
        "--topology",
        "auto",
        "--mcts_dynamic_optimization",
        "--embedder",
        embedder,
        "--embedder_model",
        embedder_model,
        "--dim",
        str(dim),
        "--seed",
        str(runtime.get("seed") or 7),
        "--use_orchestrator",
    ]
    argv.extend(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = argv
        eval_mbpp.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
