from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def mmr_rerank(
    query: np.ndarray,
    candidates: np.ndarray,
    lambda_param: float = 0.5,
    top_k: int = 5,
) -> Tuple[List[int], List[float]]:
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query.reshape(-1)

    query = _l2_normalize(query.astype(np.float32))
    candidates = _l2_normalize(candidates.astype(np.float32))

    n = candidates.shape[0]
    if n == 0:
        return [], []

    top_k = min(top_k, n)
    lambda_param = float(lambda_param)
    lambda_param = max(0.0, min(1.0, lambda_param))

    query_scores = candidates @ query
    selected: List[int] = []
    selected_scores: List[float] = []
    remaining = set(range(n))

    for _ in range(top_k):
        if not selected:
            best = int(np.argmax(query_scores))
            selected.append(best)
            selected_scores.append(float(query_scores[best]))
            remaining.discard(best)
            continue

        best_idx = None
        best_score = None
        for idx in list(remaining):
            candidate_vec = candidates[idx]
            if selected:
                selected_vecs = candidates[selected]
                diversity = float(np.max(selected_vecs @ candidate_vec))
            else:
                diversity = 0.0
            score = lambda_param * float(query_scores[idx]) - (1.0 - lambda_param) * diversity
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_scores.append(float(query_scores[best_idx]))
        remaining.discard(best_idx)

    return selected, selected_scores
