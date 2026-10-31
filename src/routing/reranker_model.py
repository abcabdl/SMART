from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrainingExample:
    query: str
    candidate: str
    label: int


class TfidfVectorizer:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: List[str]) -> None:
        tokens_list = [self._tokenize(text) for text in texts]
        vocab = {}
        for tokens in tokens_list:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        doc_count = len(tokens_list)
        df = np.zeros(len(vocab), dtype=np.float32)
        for tokens in tokens_list:
            seen = set(tokens)
            for token in seen:
                df[self.vocab[token]] += 1.0
        self.idf = np.log((1.0 + doc_count) / (1.0 + df)) + 1.0

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.idf is None:
            raise ValueError("Vectorizer not fitted")
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                idx = self.vocab.get(token)
                if idx is None:
                    continue
                matrix[i, idx] += 1.0
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in text.lower().split() if token]


class TfidfLinearReranker:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.weights: np.ndarray | None = None
        self._fitted = False

    def fit(self, examples: List[TrainingExample], epochs: int = 5, lr: float = 0.1) -> None:
        texts = []
        for ex in examples:
            texts.append(ex.query)
            texts.append(ex.candidate)
        self.vectorizer.fit(texts)

        weights = np.zeros(len(self.vectorizer.vocab), dtype=np.float32)
        for _ in range(epochs):
            for ex in examples:
                q_vec = self.vectorizer.transform([ex.query])[0]
                c_vec = self.vectorizer.transform([ex.candidate])[0]
                features = q_vec * c_vec
                score = float(np.dot(weights, features))
                prob = 1.0 / (1.0 + math.exp(-score))
                grad = (ex.label - prob) * features
                weights += lr * grad
        self.weights = weights
        self._fitted = True

    def ensure_fitted(self, texts: List[str]) -> None:
        if self._fitted:
            return
        self.vectorizer.fit(texts)
        self.weights = np.ones(len(self.vectorizer.vocab), dtype=np.float32)
        self._fitted = True

    def score(self, query: str, candidates: List[str]) -> List[float]:
        texts = [query] + candidates
        self.ensure_fitted(texts)
        q_vec = self.vectorizer.transform([query])[0]
        c_vecs = self.vectorizer.transform(candidates)
        features = c_vecs * q_vec
        weights = self.weights if self.weights is not None else np.ones(c_vecs.shape[1], dtype=np.float32)
        return (features @ weights).tolist()

    def rank(self, query: str, candidates: List[str], top_m: int) -> Tuple[List[int], List[float]]:
        scores = self.score(query, candidates)
        indices = list(range(len(scores)))
        indices.sort(key=lambda i: scores[i], reverse=True)
        top_m = min(top_m, len(indices))
        return indices[:top_m], [scores[i] for i in indices[:top_m]]

    def save(self, path: str) -> None:
        payload = {
            "vocab": self.vectorizer.vocab,
            "idf": self.vectorizer.idf.tolist() if self.vectorizer.idf is not None else [],
            "weights": self.weights.tolist() if self.weights is not None else [],
            "fitted": self._fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "TfidfLinearReranker":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls()
        obj.vectorizer.vocab = {k: int(v) for k, v in payload.get("vocab", {}).items()}
        idf_list = payload.get("idf", [])
        obj.vectorizer.idf = np.array(idf_list, dtype=np.float32) if idf_list else None
        weights_list = payload.get("weights", [])
        obj.weights = np.array(weights_list, dtype=np.float32) if weights_list else None
        obj._fitted = bool(payload.get("fitted", False))
        return obj
