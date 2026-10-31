from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ParallelBranchSpec:
    branch_id: str
    role: str
    task_text: str
    constraints: dict
    selection: dict
    reason: str
    diversity_hint: str


@dataclass
class ParallelBranchResult:
    branch_id: str
    role: str
    selected_main: Optional[str]
    output: dict
    raw_output: Any
    executor_result: Any
    artifact_bag: dict
    log_path: Optional[str]
    error: Optional[dict]
    elapsed_s: float
    selection: Optional[dict] = None


BranchRunner = Callable[[ParallelBranchSpec], ParallelBranchResult]


def _branch_succeeded(result: ParallelBranchResult) -> bool:
    if result.error:
        return False
    output = result.output if isinstance(result.output, dict) else {}
    if output.get("test_passed") or output.get("status") == "success":
        return True
    verification = result.artifact_bag.get("verification") if isinstance(result.artifact_bag, dict) else {}
    return bool(isinstance(verification, dict) and verification.get("test_passed"))


def _timeout_result(spec: ParallelBranchSpec, timeout_s: float) -> ParallelBranchResult:
    return ParallelBranchResult(
        branch_id=spec.branch_id,
        role=spec.role,
        selected_main=(spec.selection or {}).get("selected_main"),
        output={"error": "branch_timeout"},
        raw_output=None,
        executor_result=None,
        artifact_bag={},
        log_path=None,
        error={"code": "branch_timeout", "message": f"Branch exceeded {timeout_s}s"},
        elapsed_s=float(timeout_s),
        selection=spec.selection,
    )


def run_parallel_branches(
    specs: List[ParallelBranchSpec],
    *,
    branch_runner: BranchRunner,
    parallel_width: int,
    branch_timeout_s: Optional[float] = None,
    fail_fast_on_test_pass: bool = True,
) -> List[ParallelBranchResult]:
    if not specs:
        return []

    max_workers = max(1, min(int(parallel_width or 1), len(specs)))
    timeout_s = None if branch_timeout_s is None or branch_timeout_s <= 0 else float(branch_timeout_s)
    started_at = time.monotonic()
    results: List[ParallelBranchResult] = []
    result_by_branch: Dict[str, ParallelBranchResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {executor.submit(branch_runner, spec): spec for spec in specs}
        pending = set(future_to_spec.keys())

        while pending:
            remaining_timeout = None
            if timeout_s is not None:
                elapsed = time.monotonic() - started_at
                remaining_timeout = max(0.0, timeout_s - elapsed)
                if remaining_timeout == 0.0:
                    break

            done, pending = wait(
                pending,
                timeout=remaining_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break

            for future in done:
                spec = future_to_spec[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ParallelBranchResult(
                        branch_id=spec.branch_id,
                        role=spec.role,
                        selected_main=(spec.selection or {}).get("selected_main"),
                        output={"error": "branch_exception"},
                        raw_output=None,
                        executor_result=None,
                        artifact_bag={},
                        log_path=None,
                        error={"code": type(exc).__name__, "message": str(exc)},
                        elapsed_s=max(0.0, time.monotonic() - started_at),
                        selection=spec.selection,
                    )
                result_by_branch[result.branch_id] = result
                results.append(result)
                if fail_fast_on_test_pass and _branch_succeeded(result):
                    pending = set()
                    break

        for future in pending:
            spec = future_to_spec[future]
            future.cancel()
            timeout_value = timeout_s if timeout_s is not None else max(0.0, time.monotonic() - started_at)
            result = _timeout_result(spec, timeout_value)
            result_by_branch[result.branch_id] = result
            results.append(result)

    ordered: List[ParallelBranchResult] = []
    for spec in specs:
        item = result_by_branch.get(spec.branch_id)
        if item is not None:
            ordered.append(item)
    return ordered
