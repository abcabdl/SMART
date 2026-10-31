from __future__ import annotations

from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .cards import BaseCard


_ROLE_TAG_ALIASES = {
    "researcher": "evidence_finder",
    "evidence-finder": "evidence_finder",
    "evidence-reader": "evidence_finder",
    "context-finder": "evidence_finder",
}


def _normalize_role_tag(value: str) -> str:
    key = str(value or "").strip().lower()
    return _ROLE_TAG_ALIASES.get(key, key)


class Constraints(BaseModel):
    """Filter constraints for candidate cards."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Optional[str] = None
    domain_tags: List[str] = Field(default_factory=list)
    role_tags: List[str] = Field(default_factory=list)
    tool_tags: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    output_formats: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    cost_tier: Optional[str] = None
    latency_tier: Optional[str] = None
    reliability_prior: Optional[float] = None


def filter_candidates(cards: Iterable[BaseCard], constraints: Optional[Constraints]) -> List[BaseCard]:
    if constraints is None:
        return list(cards)

    def matches_list(card_values: List[str], required: List[str], *, role_tags: bool = False) -> bool:
        if not required:
            return True
        if not card_values:
            return False
        if role_tags:
            card_set = {_normalize_role_tag(value) for value in card_values}
            required_set = {_normalize_role_tag(value) for value in required}
            return bool(required_set.intersection(card_set))
        return bool(set(required).intersection(card_values))

    filtered: List[BaseCard] = []
    for card in cards:
        if constraints.kind and card.kind != constraints.kind:
            continue
        if constraints.cost_tier and card.cost_tier != constraints.cost_tier:
            continue
        if constraints.latency_tier and card.latency_tier != constraints.latency_tier:
            continue
        if constraints.reliability_prior is not None:
            if card.reliability_prior is None:
                continue
            if card.reliability_prior < constraints.reliability_prior:
                continue
        if not matches_list(card.domain_tags, constraints.domain_tags):
            continue
        if not matches_list(card.role_tags, constraints.role_tags, role_tags=True):
            continue
        if not matches_list(card.tool_tags, constraints.tool_tags):
            continue
        if not matches_list(card.modalities, constraints.modalities):
            continue
        if not matches_list(card.output_formats, constraints.output_formats):
            continue
        if not matches_list(card.permissions, constraints.permissions):
            continue
        filtered.append(card)
    return filtered
