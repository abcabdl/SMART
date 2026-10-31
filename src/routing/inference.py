from __future__ import annotations

from typing import List, Tuple

from routing.reranker_model import TfidfLinearReranker


def rerank_candidates(
    model_path: str,
    query: str,
    candidates: List[str],
    top_m: int,
) -> Tuple[List[int], List[float]]:
    model = TfidfLinearReranker.load(model_path)
    return model.rank(query, candidates, top_m)
