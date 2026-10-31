from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.constraints import Constraints, filter_candidates
from core.query_builder import build_role_query
from core.registry import SQLiteRegistry
from retrieval.embedder import BaseEmbedder, build_embedder
from retrieval.faiss_index import HNSWIndex
from retrieval.mmr import mmr_rerank
from retrieval.build_index import build_index
from retrieval.search_text import card_search_text, sanitize_search_text
from retrieval.parallel_search import ParallelCandidateRequest, get_candidates_batch


class CandidateRequest(BaseModel):
    branch_id: Optional[str] = None
    task_text: str
    role: str
    constraints: Optional[Dict[str, Any]] = None
    artifact_context: Optional[Dict[str, Any]] = None
    kind: str = Field(default="agent")
    top_n: int = Field(default=50, ge=1)
    top_k: int = Field(default=10, ge=1)
    mmr_lambda: float = Field(default=0.5)
    use_mmr: bool = Field(default=True)


class CandidateResponse(BaseModel):
    card_id: str
    score: float
    brief_tags: Dict[str, Any]


class BatchCandidateRequest(BaseModel):
    requests: List[CandidateRequest]


class BatchCandidateResponse(BaseModel):
    root: Dict[str, List[CandidateResponse]]


def _brief_tags(card) -> Dict[str, Any]:
    return {
        "domain_tags": card.domain_tags,
        "role_tags": card.role_tags,
        "tool_tags": card.tool_tags,
        "modalities": card.modalities,
        "output_formats": card.output_formats,
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _card_text(card) -> str:
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


def get_candidates(
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    kind: str,
    top_n: int,
    top_k: int,
    mmr_lambda: float,
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
    use_mmr: bool = True,
    artifact_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    constraints_obj = Constraints(**constraints) if constraints else None
    query_text = sanitize_search_text(
        build_role_query(task_text, role, constraints_obj, artifact_context=artifact_context)
    )
    query_vec = embedder.embed([query_text])
    if not getattr(index, "ids", None):
        return []

    if constraints_obj is not None:
        search_k = min(
            len(index.ids),
            max(
                max(1, int(top_n or 1)) * 8,
                max(1, int(top_k or 1)) * 8,
                128,
            ),
        )
        scores, id_rows = index.search(query_vec, top_k=search_k)
        retrieved: List[tuple[Any, float]] = []
        with SQLiteRegistry(registry.db_path) as local_registry:
            for idx, card_id in enumerate(id_rows[0]):
                if card_id is None:
                    continue
                card = local_registry.get(card_id)
                if card is None or card.kind != kind:
                    continue
                retrieved.append((card, float(scores[0][idx])))

            filtered_pairs = [
                (card, score)
                for card, score in retrieved
                if filter_candidates([card], constraints_obj)
            ]
            if not filtered_pairs and search_k < len(index.ids):
                scores, id_rows = index.search(query_vec, top_k=len(index.ids))
                for idx, card_id in enumerate(id_rows[0]):
                    if card_id is None:
                        continue
                    card = local_registry.get(card_id)
                    if card is None or card.kind != kind:
                        continue
                    if filter_candidates([card], constraints_obj):
                        filtered_pairs.append((card, float(scores[0][idx])))

        filtered_cards = [card for card, _ in filtered_pairs]
        if not filtered_cards:
            return []

        candidate_texts = [_card_text(card) for card in filtered_cards]
        candidate_vecs = embedder.embed(candidate_texts).astype(np.float32)
        query_vec_f = query_vec.astype(np.float32)
        query_norm = _l2_normalize(query_vec_f)[0]
        candidate_norms = _l2_normalize(candidate_vecs)
        dense_scores = candidate_norms @ query_norm
        lexical_scores = _lexical_scores(query_text, candidate_texts)
        lexical_max = float(np.max(lexical_scores)) if lexical_scores.size else 0.0
        lexical_priority = _uses_dummy_embedder(embedder) and lexical_max > 0.0
        if lexical_priority:
            query_scores = lexical_scores
        else:
            query_scores = (0.85 * dense_scores) + (0.15 * lexical_scores)

        if top_n and len(filtered_cards) > top_n:
            top_idx = np.argsort(-query_scores)[:top_n]
            filtered_cards = [filtered_cards[i] for i in top_idx]
            candidate_vecs = candidate_vecs[top_idx]
            query_scores = query_scores[top_idx]

        if use_mmr and not lexical_priority:
            selected_indices, selected_scores = mmr_rerank(
                query_vec_f[0],
                candidate_vecs,
                lambda_param=mmr_lambda,
                top_k=top_k,
            )
        else:
            selected_indices = list(np.argsort(-query_scores)[: max(1, int(top_k or 1))])
            selected_scores = [float(query_scores[idx]) for idx in selected_indices]

        responses: List[Dict[str, Any]] = []
        for rank_idx, candidate_idx in enumerate(selected_indices):
            card = filtered_cards[candidate_idx]
            score = float(query_scores[candidate_idx]) if candidate_idx < len(query_scores) else float(
                selected_scores[rank_idx]
            )
            responses.append(
                {
                    "card_id": card.id,
                    "score": score,
                    "brief_tags": _brief_tags(card),
                }
            )
        return responses

    scores, id_rows = index.search(query_vec, top_k=top_n)
    ids = id_rows[0]
    score_row = scores[0]

    candidates_list = []
    with SQLiteRegistry(registry.db_path) as local_registry:
        for idx, card_id in enumerate(ids):
            if card_id is None:
                continue
            card = local_registry.get(card_id)
            if card is None:
                continue
            if card.kind != kind:
                continue
            candidates_list.append((card, float(score_row[idx])))

    filtered_cards = [card for card, _ in candidates_list]
    if constraints_obj is not None:
        filtered_cards = filter_candidates(filtered_cards, constraints_obj)

    if not filtered_cards:
        return []

    score_map = {card.id: score for card, score in candidates_list}
    candidate_texts = [_card_text(card) for card in filtered_cards]
    candidate_vecs = embedder.embed(candidate_texts)

    if use_mmr:
        selected_indices, selected_scores = mmr_rerank(
            query_vec[0],
            candidate_vecs,
            lambda_param=mmr_lambda,
            top_k=top_k,
        )
    else:
        selected_indices = list(range(min(max(1, int(top_k or 1)), len(filtered_cards))))
        selected_scores = [score_map.get(filtered_cards[idx].id, 0.0) for idx in selected_indices]

    responses: List[Dict[str, Any]] = []
    for rank_idx, candidate_idx in enumerate(selected_indices):
        card = filtered_cards[candidate_idx]
        score = score_map.get(card.id, float(selected_scores[rank_idx]))
        responses.append(
            {
                "card_id": card.id,
                "score": score,
                "brief_tags": _brief_tags(card),
            }
        )
    return responses


def create_app(
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
) -> FastAPI:
    app = FastAPI()

    @app.post("/candidates", response_model=List[CandidateResponse])
    def candidates(request: CandidateRequest) -> List[CandidateResponse]:
        payload = get_candidates(
            task_text=request.task_text,
            role=request.role,
            constraints=request.constraints,
            artifact_context=request.artifact_context,
            kind=request.kind,
            top_n=request.top_n,
            top_k=request.top_k,
            mmr_lambda=request.mmr_lambda,
            registry=registry,
            index=index,
            embedder=embedder,
            use_mmr=request.use_mmr,
        )
        return [CandidateResponse(**item) for item in payload]

    @app.post("/candidates/batch", response_model=Dict[str, List[CandidateResponse]])
    def candidates_batch(request: BatchCandidateRequest) -> Dict[str, List[CandidateResponse]]:
        batch_requests = []
        for idx, item in enumerate(request.requests):
            branch_id = str(getattr(item, "branch_id", "") or f"branch-{idx}")
            batch_requests.append(
                ParallelCandidateRequest(
                    branch_id=branch_id,
                    task_text=item.task_text,
                    role=item.role,
                    constraints=item.constraints,
                    artifact_context=item.artifact_context,
                    kind=item.kind,
                    top_n=item.top_n,
                    top_k=item.top_k,
                    mmr_lambda=item.mmr_lambda,
                    use_mmr=item.use_mmr,
                )
            )
        payload = get_candidates_batch(
            batch_requests,
            registry=registry,
            index=index,
            embedder=embedder,
        )
        return {
            branch_id: [CandidateResponse(**candidate) for candidate in candidates]
            for branch_id, candidates in payload.items()
        }

    return app


def _build_default_app() -> FastAPI:
    try:
        db_path = os.getenv("SEARCH_DB", "registry.sqlite")
        index_dir = os.getenv("SEARCH_INDEX_DIR", "./index")
        dim = int(os.getenv("SEARCH_DIM", "256"))
        seed = int(os.getenv("SEARCH_SEED", "7"))

        index_path = os.path.join(index_dir, "faiss.index")

        registry = SQLiteRegistry(db_path)
        if os.path.exists(index_path):
            index = HNSWIndex.load(index_path)
        else:
            index = build_index(db_path=db_path, kind="agent", out_dir=index_dir, dim=dim, seed=seed)
        embedder_kind = os.getenv("SEARCH_EMBEDDER", "sentence-transformer")
        embedder_model = os.getenv("SEARCH_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedder_device = os.getenv("SEARCH_EMBEDDER_DEVICE", None)
        embedder_normalize = os.getenv("SEARCH_EMBEDDER_NORMALIZE", "false").lower() in {"1", "true", "yes"}
        embedder = build_embedder(
            kind=embedder_kind,
            dim=dim,
            seed=seed,
            model_name=embedder_model,
            device=embedder_device,
            normalize=embedder_normalize,
        )
        if embedder.dim != index.dim:
            raise ValueError(
                f"Embedder dim {embedder.dim} != index dim {index.dim}. "
                "Rebuild index with matching embedder/model."
            )
        return create_app(registry=registry, index=index, embedder=embedder)
    except Exception:
        app = FastAPI()

        @app.get("/health")
        def health() -> Dict[str, Any]:
            return {"ok": False}

        return app


app = _build_default_app()
