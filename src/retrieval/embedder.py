from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np


class BaseEmbedder:
    """Base embedder interface."""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class DummyEmbedder(BaseEmbedder):
    """Deterministic embedder using a fixed seed."""

    def __init__(self, dim: int, seed: int = 7):
        super().__init__(dim)
        self.seed = int(seed)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            vectors[idx] = self._embed_one(text or "")
        return vectors

    def _embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        mix = int.from_bytes(digest[:8], "little", signed=False)
        seed = (self.seed + mix) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformers embedder for semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.device = device
        self.normalize = bool(normalize)
        self._model = SentenceTransformer(model_name, device=device)
        dim = int(self._model.get_sentence_embedding_dimension())
        super().__init__(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embeddings, dtype=np.float32)


def build_embedder(
    kind: str,
    dim: int,
    seed: int = 7,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    normalize: bool = False,
) -> BaseEmbedder:
    kind = (kind or "sentence-transformer").strip().lower()
    if kind in {"sentence-transformer", "sentence_transformer", "sbert", "st"}:
        return SentenceTransformerEmbedder(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            device=device,
            normalize=normalize,
        )
    return DummyEmbedder(dim=dim, seed=seed)
