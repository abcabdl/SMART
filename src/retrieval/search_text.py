from __future__ import annotations

import re
from typing import Any, List


_TASK_NAME_PATTERNS = (
    re.compile(r"humaneval", re.IGNORECASE),
    re.compile(r"hotpotqa", re.IGNORECASE),
    re.compile(r"gsm8k", re.IGNORECASE),
    re.compile(r"mbpp", re.IGNORECASE),
    re.compile(r"MATH(?=[A-Z_-])"),
    re.compile(r"\bmath\b", re.IGNORECASE),
)


def sanitize_search_text(text: Any) -> str:
    value = str(text or "")
    for pattern in _TASK_NAME_PATTERNS:
        value = pattern.sub(" ", value)
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def card_search_text(card: Any) -> str:
    tag_parts: List[str] = []
    for attr in ("domain_tags", "role_tags", "tool_tags", "modalities", "output_formats"):
        values = getattr(card, attr, []) or []
        tag_parts.extend(str(item) for item in values if str(item).strip())
    tags = " ".join(tag_parts)
    embedding_text = str(getattr(card, "embedding_text", "") or "")
    if embedding_text:
        return sanitize_search_text(f"{embedding_text} {tags}")
    name = str(getattr(card, "name", "") or "")
    desc = str(getattr(card, "description", "") or "")
    return sanitize_search_text(f"{name} {desc} {tags}") or sanitize_search_text(name or desc)
