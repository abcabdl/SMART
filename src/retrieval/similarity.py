import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity via L2 normalize + inner product."""

    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))
