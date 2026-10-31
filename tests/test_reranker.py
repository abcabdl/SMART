import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_reranker_prefers_relevant_candidate() -> None:
    reranker_mod = _import_from_src("routing.reranker_model")
    TrainingExample = reranker_mod.TrainingExample
    TfidfLinearReranker = reranker_mod.TfidfLinearReranker

    examples = [
        TrainingExample(query="finance report", candidate="finance analysis", label=1),
        TrainingExample(query="finance report", candidate="sports news", label=0),
    ]
    model = TfidfLinearReranker()
    model.fit(examples, epochs=10, lr=0.5)

    indices, scores = model.rank("finance report", ["finance analysis", "sports news"], top_m=2)
    assert indices[0] == 0
    assert scores[0] >= scores[1]
