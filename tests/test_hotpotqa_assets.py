import importlib
import json
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def _load_first_hotpotqa_sample():
    path = Path(__file__).resolve().parents[1] / "data" / "datas" / "hotpotqa_test.jsonl"
    with path.open("r", encoding="utf-8-sig") as f:
        return json.loads(f.readline())


def _run_generated_tool(record, payload):
    namespace = {}
    exec(compile(record["code"], record["id"], "exec"), namespace)
    return namespace["run"](payload)["output"]


def test_hotpotqa_catalog_shape_and_unique_five_tool_agents() -> None:
    catalog = _import_from_src("generation.real_dataset_bulk_catalog")
    tools = [record for record in catalog.build_real_dataset_tool_records() if record["dataset"] == "hotpotqa"]
    agents = [record for record in catalog.build_real_dataset_agent_records() if record["id"].startswith("agent-real-hotpotqa-")]

    assert len(tools) == 375
    assert len(agents) == 75

    tool_ids = {record["id"] for record in tools}
    used_ids = []
    for agent in agents:
        assert agent["dataset_family"] == "hotpotqa"
        assert "hotpotqa" in agent["domain_tags"]
        assert len(agent["tool_ids"]) == 5
        assert set(agent["tool_ids"]).issubset(tool_ids)
        used_ids.extend(agent["tool_ids"])

    assert len(set(used_ids)) == 375
    assert len(used_ids) == 375


def test_hotpotqa_generated_tools_execute_on_sample() -> None:
    catalog = _import_from_src("generation.real_dataset_bulk_catalog")
    records = {
        record["id"]: record
        for record in catalog.build_real_dataset_tool_records()
        if record["dataset"] == "hotpotqa"
    }
    sample = _load_first_hotpotqa_sample()
    prefix = "hotpotqa-bridge-entity-linking-planner"

    decomposed = _run_generated_tool(records[f"{prefix}-question-decomposer"], sample)
    assert decomposed["question_type"] in {"bridge", "comparison"}

    indexed = _run_generated_tool(records[f"{prefix}-context-indexer"], sample)
    assert indexed["evidence_points"]

    chained = _run_generated_tool(
        records[f"{prefix}-evidence-chain-builder"],
        {**sample, "evidence_points": indexed["evidence_points"]},
    )
    assert chained["reasoning_trace"]

    extracted = _run_generated_tool(
        records[f"{prefix}-answer-extractor"],
        {**sample, "evidence_points": chained["evidence_points"]},
    )
    assert extracted["candidate_answer"]

    verified = _run_generated_tool(
        records[f"{prefix}-support-verifier"],
        {
            **sample,
            "candidate_answer": sample["answer"],
            "evidence_points": chained["evidence_points"],
        },
    )
    assert "matches_expected" not in verified
    assert "expected_value" not in verified.get("verification", {})
    assert verified["answer_in_evidence"] in {True, False}
    assert 0.0 <= float(verified["supporting_fact_coverage"]) <= 1.0
