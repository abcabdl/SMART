import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_probe_commit_selects_best_candidate() -> None:
    probe_mod = _import_from_src("execution.probe_commit")
    probe_commit = probe_mod.probe_commit

    candidates = [
        {
            "card_id": "agent-finance",
            "score": 0.4,
            "brief_tags": {"domain_tags": ["finance"], "tool_tags": ["text_cleaner"]},
        },
        {
            "card_id": "agent-health",
            "score": 0.9,
            "brief_tags": {"domain_tags": ["health"], "tool_tags": ["basic_stats"]},
        },
    ]

    result = probe_commit(
        task_text="analyze finance report",
        role="evidence_finder",
        constraints={"domain_tags": ["finance"]},
        candidates=candidates,
        top_probe=5,
        max_shadows=2,
    )

    assert result["selected_main"] == "agent-finance"
    assert result["selected_shadows"][0] == "agent-health"
