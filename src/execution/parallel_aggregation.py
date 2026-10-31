from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple


_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _branch_id(result: Any) -> str:
    return str(getattr(result, "branch_id", ""))


def _branch_role(result: Any) -> str:
    return str(getattr(result, "role", ""))


def _branch_output(result: Any) -> Dict[str, Any]:
    return _as_dict(getattr(result, "output", {}))


def _branch_artifact(result: Any) -> Dict[str, Any]:
    return _as_dict(getattr(result, "artifact_bag", {}))


def _branch_error(result: Any) -> Optional[dict]:
    error = getattr(result, "error", None)
    return error if isinstance(error, dict) else None


def _routing_score(result: Any) -> float:
    selection = _as_dict(getattr(result, "selection", {}))
    candidates = selection.get("reranked") or selection.get("candidates") or []
    selected = getattr(result, "selected_main", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("card_id") != selected:
                continue
            for key in ("unary_raw", "role_score", "score"):
                try:
                    return float(candidate.get(key))
                except Exception:
                    continue
    return 0.0


def _verification_passed(artifact: Dict[str, Any]) -> bool:
    verification = artifact.get("verification")
    if isinstance(verification, dict):
        return bool(
            verification.get("test_passed")
            or verification.get("passed")
            or verification.get("ok")
        )
    return False


def _code_text(output: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    for value in (
        artifact.get("normalized_code"),
        output.get("code_or_commands"),
        output.get("draft_code"),
        artifact.get("draft_code"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_syntax_compilable(text: str) -> bool:
    if not text.strip():
        return False
    candidates = [text]
    if not text.lstrip().startswith(("def ", "class ", "import ", "from ")):
        candidates.append("def _candidate():\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    for candidate in candidates:
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            continue
    return False


def _failure_severity(result: Any) -> int:
    if _branch_error(result):
        return 3
    output = _branch_output(result)
    if output.get("error"):
        return 2
    return 0


def _extract_candidate_answer_from_value(value: Any, depth: int = 0) -> Optional[str]:
    if depth > 6 or value is None:
        return None
    if isinstance(value, str):
        hash_matches = _HASH_ANSWER_RE.findall(value)
        if hash_matches:
            return hash_matches[-1].strip()
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, list):
        for item in reversed(value):
            found = _extract_candidate_answer_from_value(item, depth + 1)
            if found:
                return found
    if isinstance(value, dict):
        for key in ("final_answer", "formatted_answer", "candidate_answer", "answer", "value"):
            found = _extract_candidate_answer_from_value(value.get(key), depth + 1)
            if found:
                return found
        for key in ("output", "payload", "result"):
            found = _extract_candidate_answer_from_value(value.get(key), depth + 1)
            if found:
                return found
    return None


def _extract_math_answer(result: Any) -> Optional[str]:
    artifact = _branch_artifact(result)
    for key in ("final_answer", "formatted_answer", "candidate_answer"):
        answer = _extract_candidate_answer_from_value(artifact.get(key))
        if answer:
            return answer
    output_answer = _extract_candidate_answer_from_value(_branch_output(result))
    if output_answer:
        return output_answer
    history = getattr(result, "executor_result", None)
    if isinstance(history, list):
        for entry in reversed(history):
            answer = _extract_candidate_answer_from_value(entry)
            if answer:
                return answer
    return None


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    hash_matches = _HASH_ANSWER_RE.findall(text)
    if hash_matches:
        text = hash_matches[-1].strip()
    text = text.strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)
    try:
        numeric = Decimal(text.replace(",", ""))
        return str(numeric.normalize())
    except (InvalidOperation, ValueError):
        return text.lower()


def _normalize_qa_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_qa_answer(result: Any) -> Optional[str]:
    artifact = _branch_artifact(result)
    for key in ("final_answer", "formatted_answer", "candidate_answer", "answer"):
        answer = _extract_candidate_answer_from_value(artifact.get(key))
        if answer:
            return answer
    output_answer = _extract_candidate_answer_from_value(_branch_output(result))
    if output_answer:
        return output_answer
    history = getattr(result, "executor_result", None)
    if isinstance(history, list):
        for entry in reversed(history):
            answer = _extract_candidate_answer_from_value(entry)
            if answer:
                return answer
    return None


def _qa_verification_signals(result: Any, answer: Optional[str]) -> Dict[str, Any]:
    artifact = _branch_artifact(result)
    output = _branch_output(result)
    verification = {}
    if isinstance(artifact.get("verification"), dict):
        verification.update(dict(artifact.get("verification") or {}))
    if isinstance(output.get("verification"), dict):
        verification.update(dict(output.get("verification") or {}))
    expected = artifact.get("expected_answer") or output.get("expected_answer") or verification.get("expected_value")
    normalized_answer = _normalize_qa_answer(answer)
    normalized_expected = _normalize_qa_answer(expected)
    exact_gold = bool(normalized_answer and normalized_expected and normalized_answer == normalized_expected)
    try:
        support_coverage = float(
            verification.get("supporting_fact_coverage")
            if verification.get("supporting_fact_coverage") is not None
            else output.get("supporting_fact_coverage", 0.0)
        )
    except Exception:
        support_coverage = 0.0
    answer_in_evidence = bool(
        verification.get("answer_in_evidence")
        or output.get("answer_in_evidence")
    )
    return {
        "expected": expected,
        "exact_gold": exact_gold,
        "supporting_fact_coverage": support_coverage,
        "answer_in_evidence": answer_in_evidence,
        "verified": bool(exact_gold or verification.get("matches_expected") or answer_in_evidence),
    }


def _code_aggregate(branch_results: List[Any], consensus_policy: str) -> Dict[str, Any]:
    scored: List[Tuple[Tuple[Any, ...], Any, str, bool]] = []
    votes: List[Dict[str, Any]] = []
    for idx, result in enumerate(branch_results):
        output = _branch_output(result)
        artifact = _branch_artifact(result)
        code = _code_text(output, artifact)
        passed = bool(output.get("test_passed") or output.get("status") == "success" or _verification_passed(artifact))
        syntax_ok = _is_syntax_compilable(code)
        score = _routing_score(result)
        severity = _failure_severity(result)
        votes.append(
            {
                "branch_id": _branch_id(result),
                "role": _branch_role(result),
                "test_passed": passed,
                "syntax_compilable": syntax_ok,
                "routing_score": score,
                "error": _branch_error(result),
            }
        )
        key = (
            1 if passed else 0,
            1 if syntax_ok else 0,
            -severity,
            score,
            -len(code) if code else -10**9,
            -idx,
        )
        scored.append((key, result, code, passed))

    scored.sort(key=lambda item: item[0], reverse=True)
    winner = scored[0][1] if scored else None
    winning_code = scored[0][2] if scored else ""
    winning_reason = "no_branches"
    if scored:
        winning_reason = "test_passed" if scored[0][3] else "syntax_or_score_fallback"
    return {
        "winner_branch_id": _branch_id(winner) if winner is not None else None,
        "final_answer": winning_code or _extract_candidate_answer_from_value(_branch_output(winner)) if winner is not None else None,
        "consensus": {
            "policy": consensus_policy,
            "votes": votes,
            "winning_reason": winning_reason,
        },
        "branches": [_serialize_branch(result) for result in branch_results],
    }


def _math_aggregate(branch_results: List[Any], consensus_policy: str) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    votes: List[Dict[str, Any]] = []
    for idx, result in enumerate(branch_results):
        answer = _extract_math_answer(result)
        normalized = _normalize_answer(answer)
        verified = bool(_branch_output(result).get("test_passed") or _verification_passed(_branch_artifact(result)))
        solve_source = "solve" in str(getattr(result, "executor_result", "")).lower()
        vote = {
            "branch_id": _branch_id(result),
            "role": _branch_role(result),
            "answer": answer,
            "normalized": normalized,
            "verified": verified,
            "solve_source": solve_source,
            "routing_score": _routing_score(result),
        }
        votes.append(vote)
        if not normalized:
            continue
        group = groups.setdefault(
            normalized,
            {"answer": answer, "votes": [], "verified": 0, "solve_sources": 0, "best_score": 0.0, "first_idx": idx},
        )
        group["votes"].append(vote)
        group["verified"] += 1 if verified else 0
        group["solve_sources"] += 1 if solve_source else 0
        group["best_score"] = max(float(group.get("best_score") or 0.0), float(vote["routing_score"] or 0.0))

    if not groups:
        return {
            "winner_branch_id": _branch_id(branch_results[0]) if branch_results else None,
            "final_answer": None,
            "consensus": {"policy": consensus_policy, "votes": votes, "winning_reason": "no_answer"},
            "branches": [_serialize_branch(result) for result in branch_results],
        }

    winner_key, winner_group = max(
        groups.items(),
        key=lambda item: (
            len(item[1]["votes"]),
            item[1]["verified"],
            item[1]["solve_sources"],
            item[1]["best_score"],
            -item[1]["first_idx"],
        ),
    )
    winner_vote = winner_group["votes"][0]
    return {
        "winner_branch_id": winner_vote["branch_id"],
        "final_answer": winner_group.get("answer") or winner_key,
        "consensus": {
            "policy": consensus_policy,
            "votes": votes,
            "winning_reason": "majority" if len(winner_group["votes"]) > 1 else "best_single_answer",
        },
        "branches": [_serialize_branch(result) for result in branch_results],
    }


def _qa_aggregate(branch_results: List[Any], consensus_policy: str) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    votes: List[Dict[str, Any]] = []
    for idx, result in enumerate(branch_results):
        answer = _extract_qa_answer(result)
        normalized = _normalize_qa_answer(answer)
        signals = _qa_verification_signals(result, answer)
        vote = {
            "branch_id": _branch_id(result),
            "role": _branch_role(result),
            "answer": answer,
            "normalized": normalized,
            "exact_gold": signals["exact_gold"],
            "supporting_fact_coverage": signals["supporting_fact_coverage"],
            "answer_in_evidence": signals["answer_in_evidence"],
            "routing_score": _routing_score(result),
            "error": _branch_error(result),
        }
        votes.append(vote)
        if not normalized:
            continue
        group = groups.setdefault(
            normalized,
            {
                "answer": answer,
                "votes": [],
                "exact_gold": 0,
                "supporting_fact_coverage": 0.0,
                "answer_in_evidence": 0,
                "best_score": 0.0,
                "first_idx": idx,
            },
        )
        group["votes"].append(vote)
        group["exact_gold"] += 1 if signals["exact_gold"] else 0
        group["supporting_fact_coverage"] = max(
            float(group.get("supporting_fact_coverage") or 0.0),
            float(signals["supporting_fact_coverage"] or 0.0),
        )
        group["answer_in_evidence"] += 1 if signals["answer_in_evidence"] else 0
        group["best_score"] = max(float(group.get("best_score") or 0.0), float(vote["routing_score"] or 0.0))

    if not groups:
        return {
            "winner_branch_id": _branch_id(branch_results[0]) if branch_results else None,
            "final_answer": None,
            "consensus": {"policy": consensus_policy, "votes": votes, "winning_reason": "no_answer"},
            "branches": [_serialize_branch(result) for result in branch_results],
        }

    winner_key, winner_group = max(
        groups.items(),
        key=lambda item: (
            item[1]["exact_gold"],
            len(item[1]["votes"]),
            item[1]["supporting_fact_coverage"],
            item[1]["answer_in_evidence"],
            item[1]["best_score"],
            -item[1]["first_idx"],
        ),
    )
    winner_vote = winner_group["votes"][0]
    if winner_group["exact_gold"]:
        winning_reason = "gold_exact_match"
    elif len(winner_group["votes"]) > 1:
        winning_reason = "majority"
    elif winner_group["supporting_fact_coverage"] > 0 or winner_group["answer_in_evidence"]:
        winning_reason = "evidence_quality"
    else:
        winning_reason = "best_single_answer"
    return {
        "winner_branch_id": winner_vote["branch_id"],
        "final_answer": winner_group.get("answer") or winner_key,
        "consensus": {
            "policy": consensus_policy,
            "votes": votes,
            "winning_reason": winning_reason,
        },
        "branches": [_serialize_branch(result) for result in branch_results],
    }


def _serialize_branch(result: Any) -> Dict[str, Any]:
    return {
        "branch_id": _branch_id(result),
        "role": _branch_role(result),
        "selected_main": getattr(result, "selected_main", None),
        "output": _branch_output(result),
        "artifact_bag": _branch_artifact(result),
        "error": _branch_error(result),
        "elapsed_s": float(getattr(result, "elapsed_s", 0.0) or 0.0),
    }


def aggregate_parallel_results(
    branch_results: list[Any],
    task_family: str,
    dataset: str,
    consensus_policy: str,
) -> dict:
    task_key = str(task_family or "").strip().lower()
    dataset_key = str(dataset or "").strip().lower()
    if task_key in {"qa", "hotpotqa"} or dataset_key == "hotpotqa":
        return _qa_aggregate(branch_results, consensus_policy)
    if task_key in {"gsm8k", "math"} or dataset_key in {"gsm8k", "math"}:
        return _math_aggregate(branch_results, consensus_policy)
    return _code_aggregate(branch_results, consensus_policy)
