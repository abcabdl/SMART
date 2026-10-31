import importlib
import sys
from pathlib import Path
import random


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_bandit_store_update_and_select(tmp_path) -> None:
    bandit_mod = _import_from_src("routing.bandit_store")
    BanditStore = bandit_mod.BanditStore

    db_path = tmp_path / "bandit.sqlite"
    with BanditStore(str(db_path)) as store:
        store.update("v1", "planner", "agent-a", reward=1.0, confidence=1.0)
        store.update("v1", "planner", "agent-b", reward=0.0, confidence=1.0)

        alpha_a, beta_a = store.get("v1", "planner", "agent-a")
        alpha_b, beta_b = store.get("v1", "planner", "agent-b")
        assert alpha_a > alpha_b
        assert beta_b > beta_a

        rng = random.Random(42)
        scores = store.select_thompson("v1", "planner", ["agent-a", "agent-b"], rng=rng)
        assert set(scores.keys()) == {"agent-a", "agent-b"}
