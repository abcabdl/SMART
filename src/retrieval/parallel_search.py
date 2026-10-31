from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from core.constraints import Constraints, filter_candidates
from core.query_builder import build_role_query
from core.registry import SQLiteRegistry
from retrieval.embedder import BaseEmbedder
from retrieval.faiss_index import HNSWIndex
from retrieval.mmr import mmr_rerank
from retrieval.search_text import card_search_text, sanitize_search_text


@dataclass(frozen=True)
class ParallelCandidateRequest:
    branch_id: str
    task_text: str
    role: str
    constraints: Optional[dict] = None
    artifact_context: Optional[dict] = None
    kind: str = "agent"
    top_n: int = 20
    top_k: int = 5
    mmr_lambda: float = 0.5
    use_mmr: bool = True


def _brief_tags(card: Any) -> Dict[str, Any]:
    return {
        "domain_tags": card.domain_tags,
        "role_tags": card.role_tags,
        "tool_tags": card.tool_tags,
        "modalities": card.modalities,
        "output_formats": card.output_formats,
    }


def _card_text(card: Any) -> str:
    return card_search_text(card)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "candidate",
    "code",
    "correct",
    "def",
    "for",
    "from",
    "function",
    "generate",
    "implementation",
    "in",
    "is",
    "agent",
    "assert",
    "benchmark",
    "builder",
    "checker",
    "clear",
    "dataset",
    "deterministic",
    "domain",
    "driven",
    "focus",
    "focused",
    "generation",
    "generator",
    "local",
    "logic",
    "mbpp",
    "num",
    "number",
    "numbers",
    "nums",
    "of",
    "on",
    "output",
    "paper",
    "primary",
    "python",
    "real",
    "refactor",
    "role",
    "runnable",
    "return",
    "solution",
    "specialist",
    "specialized",
    "structure",
    "tag",
    "tags",
    "task",
    "that",
    "the",
    "to",
    "tool",
    "tools",
    "uses",
    "with",
    "write",
    "you",
}


def _tokenize_for_lexical_rank(text: str) -> set[str]:
    normalized = _CAMEL_BOUNDARY_RE.sub(" ", str(text or "").replace("_", " ").replace("-", " "))
    tokens = {
        token.lower()
        for token in _TOKEN_RE.findall(normalized)
        if len(token) > 1 and token.lower() not in _LEXICAL_STOPWORDS
    }
    expanded = set(tokens)
    for token in list(tokens):
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            expanded.add(token[:-1])
    return expanded


def _lexical_scores(query_text: str, candidate_texts: List[str]) -> np.ndarray:
    query_tokens = _tokenize_for_lexical_rank(query_text)
    if not candidate_texts:
        return np.zeros((0,), dtype=np.float32)
    if not query_tokens:
        return np.zeros((len(candidate_texts),), dtype=np.float32)
    candidate_token_sets = [_tokenize_for_lexical_rank(text) for text in candidate_texts]
    df: Dict[str, int] = {}
    for tokens in candidate_token_sets:
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    n_candidates = float(max(1, len(candidate_token_sets)))

    def weight(token: str) -> float:
        return 1.0 + float(np.log((1.0 + n_candidates) / (1.0 + float(df.get(token, 0)))))

    query_weight = sum(weight(token) for token in query_tokens)
    scores: List[float] = []
    for candidate_tokens in candidate_token_sets:
        if not candidate_tokens:
            scores.append(0.0)
            continue
        overlap_tokens = query_tokens.intersection(candidate_tokens)
        overlap = sum(weight(token) for token in overlap_tokens)
        candidate_weight = sum(weight(token) for token in candidate_tokens)
        coverage = overlap / float(max(1e-6, query_weight))
        precision = overlap / float(max(1e-6, candidate_weight))
        scores.append((0.75 * coverage) + (0.25 * precision))
    return np.asarray(scores, dtype=np.float32)


def _uses_dummy_embedder(embedder: BaseEmbedder) -> bool:
    return embedder.__class__.__name__.lower().startswith("dummy")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _single_filtered_candidates(
    request: ParallelCandidateRequest,
    *,
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
) -> List[Dict[str, Any]]:
    constraints_obj = Constraints(**(request.constraints or {}))
    query_text = sanitize_search_text(
        build_role_query(
            request.task_text,
            request.role,
            constraints_obj,
            artifact_context=request.artifact_context,
        )
    )
    query_vec = embedder.embed([query_text]).astype(np.float32)

    search_k = min(
        len(index.ids),
        max(
            max(1, int(request.top_n or 1)) * 8,
            max(1, int(request.top_k or 1)) * 8,
            128,
        ),
    )
    with SQLiteRegistry(registry.db_path) as local_registry:
        scores, id_rows = index.search(query_vec, top_k=search_k)
        filtered_pairs: List[tuple[Any, float]] = []
        for idx, card_id in enumerate(id_rows[0]):
            if card_id is None:
                continue
            card = local_registry.get(card_id)
            if card is None or card.kind != request.kind:
                continue
            if filter_candidates([card], constraints_obj):
                filtered_pairs.append((card, float(scores[0][idx])))
        if not filtered_pairs and search_k < len(index.ids):
            scores, id_rows = index.search(query_vec, top_k=len(index.ids))
            for idx, card_id in enumerate(id_rows[0]):
                if card_id is None:
                    continue
                card = local_registry.get(card_id)
                if card is None or card.kind != request.kind:
                    continue
                if filter_candidates([card], constraints_obj):
                    filtered_pairs.append((card, float(scores[0][idx])))

    filtered_cards = [card for card, _ in filtered_pairs]
    if not filtered_cards:
        return []

    candidate_texts = [_card_text(card) for card in filtered_cards]
    candidate_vecs = embedder.embed(candidate_texts).astype(np.float32)
    query_norm = _l2_normalize(query_vec)[0]
    candidate_norms = _l2_normalize(candidate_vecs)
    dense_scores = candidate_norms @ query_norm
    lexical_scores = _lexical_scores(query_text, candidate_texts)
    lexical_max = float(np.max(lexical_scores)) if lexical_scores.size else 0.0
    lexical_priority = _uses_dummy_embedder(embedder) and lexical_max > 0.0
    if lexical_priority:
        query_scores = lexical_scores
    else:
        query_scores = (0.85 * dense_scores) + (0.15 * lexical_scores)

    top_n = max(1, int(request.top_n or 1))
    if len(filtered_cards) > top_n:
        top_idx = np.argsort(-query_scores)[:top_n]
        filtered_cards = [filtered_cards[i] for i in top_idx]
        candidate_vecs = candidate_vecs[top_idx]
        query_scores = query_scores[top_idx]

    if request.use_mmr and not lexical_priority:
        selected_indices, selected_scores = mmr_rerank(
            query_vec[0],
            candidate_vecs,
            lambda_param=float(request.mmr_lambda),
            top_k=max(1, int(request.top_k or 1)),
        )
    else:
        selected_indices = list(np.argsort(-query_scores)[: max(1, int(request.top_k or 1))])
        selected_scores = [float(query_scores[idx]) for idx in selected_indices]
    responses: List[Dict[str, Any]] = []
    for rank_idx, candidate_idx in enumerate(selected_indices):
        card = filtered_cards[candidate_idx]
        score = (
            float(query_scores[candidate_idx])
            if candidate_idx < len(query_scores)
            else float(selected_scores[rank_idx])
        )
        responses.append({"card_id": card.id, "score": score, "brief_tags": _brief_tags(card)})
    return responses


def _mmr_unconstrained_response(
    *,
    request: ParallelCandidateRequest,
    query_vec: np.ndarray,
    candidate_pairs: List[tuple[Any, float]],
    embedder: BaseEmbedder,
) -> List[Dict[str, Any]]:
    if not candidate_pairs:
        return []
    filtered_cards = [card for card, _ in candidate_pairs]
    score_map = {card.id: score for card, score in candidate_pairs}
    candidate_vecs = embedder.embed([_card_text(card) for card in filtered_cards])
    if request.use_mmr:
        selected_indices, selected_scores = mmr_rerank(
            query_vec,
            candidate_vecs,
            lambda_param=float(request.mmr_lambda),
            top_k=max(1, int(request.top_k or 1)),
        )
    else:
        ordered = sorted(
            range(len(candidate_pairs)),
            key=lambda idx: float(candidate_pairs[idx][1]),
            reverse=True,
        )
        selected_indices = ordered[: max(1, int(request.top_k or 1))]
        selected_scores = [float(candidate_pairs[idx][1]) for idx in selected_indices]
    responses: List[Dict[str, Any]] = []
    for rank_idx, candidate_idx in enumerate(selected_indices):
        card = filtered_cards[candidate_idx]
        score = score_map.get(card.id, float(selected_scores[rank_idx]))
        responses.append({"card_id": card.id, "score": score, "brief_tags": _brief_tags(card)})
    return responses


def get_candidates_batch(
    requests: List[ParallelCandidateRequest],
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    if not requests:
        return results
    if not getattr(index, "ids", None):
        return {request.branch_id: [] for request in requests}

    constrained: List[ParallelCandidateRequest] = []
    unconstrained: List[ParallelCandidateRequest] = []
    for request in requests:
        if request.constraints:
            constrained.append(request)
        else:
            unconstrained.append(request)

    for request in constrained:
        results[request.branch_id] = _single_filtered_candidates(
            request,
            registry=registry,
            index=index,
            embedder=embedder,
        )

    if unconstrained:
        query_texts = [
            sanitize_search_text(
                build_role_query(
                    request.task_text,
                    request.role,
                    None,
                    artifact_context=request.artifact_context,
                )
            )
            for request in unconstrained
        ]
        query_vecs = embedder.embed(query_texts).astype(np.float32)
        max_top_n = max(1, max(int(request.top_n or 1) for request in unconstrained))
        scores, id_rows = index.search(query_vecs, top_k=max_top_n)

        with SQLiteRegistry(registry.db_path) as local_registry:
            for row_idx, request in enumerate(unconstrained):
                candidate_pairs: List[tuple[Any, float]] = []
                for col_idx, card_id in enumerate(id_rows[row_idx][: max(1, int(request.top_n or 1))]):
                    if card_id is None:
                        continue
                    card = local_registry.get(card_id)
                    if card is None or card.kind != request.kind:
                        continue
                    candidate_pairs.append((card, float(scores[row_idx][col_idx])))
                results[request.branch_id] = _mmr_unconstrained_response(
                    request=request,
                    query_vec=query_vecs[row_idx],
                    candidate_pairs=candidate_pairs,
                    embedder=embedder,
                )

    for request in requests:
        results.setdefault(request.branch_id, [])
    return results
