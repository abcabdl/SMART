from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseCard(BaseModel):
    """Base card type for all structured artifacts."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    kind: str
    version: str
    updated_at: datetime
    domain_tags: List[str] = Field(default_factory=list)
    role_tags: List[str] = Field(default_factory=list)
    tool_tags: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    output_formats: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    cost_tier: Optional[str] = None
    latency_tier: Optional[str] = None
    reliability_prior: Optional[float] = None
    description: str = ""
    examples: List[str] = Field(default_factory=list)
    embedding_text: str = ""
    available_tool_ids: List[str] = Field(default_factory=list)
    dataset_family: str = ""
    focus_slug: str = ""
    primary_role: str = ""
    capability_kind: str = ""
    stage: str = ""
    deterministic: bool = True
    consumes: List[str] = Field(default_factory=list)
    produces: List[str] = Field(default_factory=list)
    embedding_vector: Optional[List[float]] = Field(default=None, exclude=True, repr=False)

    def to_json(self) -> str:
        return self.model_dump_json()


class ToolCard(BaseCard):
    """Tool card model."""


class AgentCard(BaseCard):
    """Agent card model."""
