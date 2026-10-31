from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np

try:
    from core.registry import SQLiteRegistry
    from retrieval.embedder import build_embedder
    from retrieval.faiss_index import HNSWIndex
    from retrieval.search_text import card_search_text
except ImportError:  # pragma: no cover
    from src.core.registry import SQLiteRegistry
    from src.retrieval.embedder import build_embedder
    from src.retrieval.faiss_index import HNSWIndex
    from src.retrieval.search_text import card_search_text


def _chunked(values: List[str], size: int) -> List[List[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def build_index(
    db_path: str,
    kind: str,
    out_dir: str,
    dim: int,
    batch_size: int = 64,
    seed: int = 7,
    embedder_kind: str = "dummy",
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedder_device: Optional[str] = None,
    embedder_normalize: bool = False,
) -> HNSWIndex:
    with SQLiteRegistry(db_path) as registry:
        cards = registry.list({"kind": kind})

    if not cards:
        raise ValueError("No cards found for index build")

    ids = [card.id for card in cards]
    texts = [card_search_text(card) for card in cards]

    embedder = build_embedder(
        kind=embedder_kind,
        dim=dim,
        seed=seed,
        model_name=embedder_model,
        device=embedder_device,
        normalize=embedder_normalize,
    )
    dim = embedder.dim
    batches = _chunked(texts, batch_size)
    vectors: List[np.ndarray] = []
    for batch in batches:
        vectors.append(embedder.embed(batch))
    matrix = np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)

    index = HNSWIndex(dim=dim)
    index.build(matrix, ids)

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "faiss.index")
    id_map_path = os.path.join(out_dir, "id_map.json")
    index.save(index_path)

    id_map = {str(i): card_id for i, card_id in enumerate(ids)}
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=True, indent=2)

    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--db", required=True, type=str, help="Path to sqlite registry")
    parser.add_argument("--kind", required=True, type=str, help="Card kind to index")
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", type=str, default="sentence-transformer", help="dummy|sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embedder_device", type=str, default=None)
    parser.add_argument("--embedder_normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        db_path=args.db,
        kind=args.kind,
        out_dir=args.out,
        dim=args.dim,
        batch_size=args.batch,
        seed=args.seed,
        embedder_kind=args.embedder,
        embedder_model=args.embedder_model,
        embedder_device=args.embedder_device,
        embedder_normalize=args.embedder_normalize,
    )


if __name__ == "__main__":
    main()
