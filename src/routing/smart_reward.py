from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SmartCandidate:
    card_id: str
    score: float
    role_score: float
    unary_raw: float
    unary_score: float
    brief_tags: Dict[str, Any]


@dataclass(frozen=True)
class SmartRewardBreakdown:
    reward: float
    unary_score: float
    transition_score: float
    io_score: float
    gap_reduction: float
    gap_state: Dict[str, Any]
    components: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reward": float(self.reward),
            "unary_score": float(self.unary_score),
            "transition_score": float(self.transition_score),
            "io_score": float(self.io_score),
            "posterior_score": float(self.transition_score),
            "uncertainty_score": 0.0,
            "gap_reduction": float(self.gap_reduction),
            "gap_state": dict(self.gap_state or {}),
            "components": list(self.components),
        }


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def tier_penalty(value: Optional[str]) -> float:
    if not value:
        return 0.25
    normalized = str(value).strip().lower()
    if normalized in {"low", "cheap", "economy", "fast"}:
        return 0.0
    if normalized in {"medium", "mid", "normal", "balanced"}:
        return 0.5
    if normalized in {"high", "expensive", "slow", "premium"}:
        return 1.0
    return 0.25


def jaccard_overlap(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = {str(item).strip().lower() for item in (left or []) if str(item).strip()}
    right_set = {str(item).strip().lower() for item in (right or []) if str(item).strip()}
    if not left_set and not right_set:
        return 0.0
    union = left_set.union(right_set)
    if not union:
        return 0.0
    return float(len(left_set.intersection(right_set)) / len(union))


def io_chain_score(prev_card: Any, curr_card: Any) -> float:
    if prev_card is None or curr_card is None:
        return 0.5
    format_overlap = jaccard_overlap(
        list(getattr(prev_card, "output_formats", []) or []),
        list(getattr(curr_card, "output_formats", []) or []),
    )
    modality_overlap = jaccard_overlap(
        list(getattr(prev_card, "modalities", []) or []),
        list(getattr(curr_card, "modalities", []) or []),
    )
    tool_overlap = jaccard_overlap(
        list(getattr(prev_card, "tool_tags", []) or []),
        list(getattr(curr_card, "tool_tags", []) or []),
    )
    return clamp01((format_overlap + modality_overlap + tool_overlap) / 3.0)


def normalize_scores(raw_scores: Dict[str, float], default: float = 0.5) -> Dict[str, float]:
    if not raw_scores:
        return {}
    values = [float(value) for value in raw_scores.values()]
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {key: float(default) for key in raw_scores}
    return {key: clamp01((float(value) - lo) / (hi - lo)) for key, value in raw_scores.items()}


def build_smart_candidates(
    candidates: List[Dict[str, Any]],
    *,
    raw_unary_scores: Dict[str, float],
    final_k: int,
) -> List[SmartCandidate]:
    normalized = normalize_scores(raw_unary_scores)
    smart_candidates: List[SmartCandidate] = []
    for candidate in candidates:
        card_id = str(candidate.get("card_id") or "").strip()
        if not card_id:
            continue
        role_score = float(candidate.get("role_score", candidate.get("score", 0.0)) or 0.0)
        unary_raw = float(raw_unary_scores.get(card_id, candidate.get("unary_raw", role_score)) or 0.0)
        smart_candidates.append(
            SmartCandidate(
                card_id=card_id,
                score=float(candidate.get("score", role_score) or 0.0),
                role_score=role_score,
                unary_raw=unary_raw,
                unary_score=float(normalized.get(card_id, 0.5)),
                brief_tags=dict(candidate.get("brief_tags") or {}),
            )
        )
    smart_candidates.sort(key=lambda item: item.unary_raw, reverse=True)
    return smart_candidates[: max(1, int(final_k or 1))]


def score_smart_reward(
    *,
    unary_score: float,
    transition_score: float = 0.5,
    io_score: float = 0.5,
    gap_reduction: float = 0.0,
    gap_state: Optional[Dict[str, Any]] = None,
    use_transition_term: bool = True,
    use_iochain: bool = True,
    use_gap_term: bool = True,
    reward_variant: str = "mean_active",
) -> SmartRewardBreakdown:
    unary = clamp01(unary_score)
    transition = clamp01(transition_score)
    io_component = clamp01(io_score)
    gap = clamp01(gap_reduction)

    components: List[Tuple[str, float]] = [("Unary", unary)]
    if use_transition_term:
        components.append(("Trans", transition))
    if use_iochain:
        components.append(("IO", io_component))
    if use_gap_term:
        components.append(("Gap", gap))

    variant = str(reward_variant or "mean_active").strip().lower()
    if variant in {"legacy_four_term", "four_term"}:
        reward = (unary + transition + io_component + gap) / 4.0
        names = ("Unary", "Trans", "IO", "Gap")
    else:
        reward = sum(value for _, value in components) / float(max(1, len(components)))
        names = tuple(name for name, _ in components)

    return SmartRewardBreakdown(
        reward=float(reward),
        unary_score=unary,
        transition_score=transition,
        io_score=io_component,
        gap_reduction=gap,
        gap_state=dict(gap_state or {}),
        components=names,
    )
