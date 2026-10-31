from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .cards import AgentCard, BaseCard, ToolCard


class SQLiteRegistry:
    """SQLite-backed registry for cards."""

    TABLE_NAME = "cards"
    TOOL_CODE_TABLE = "tool_code"

    SCALAR_FIELDS = {
        "id",
        "name",
        "kind",
        "version",
        "updated_at",
        "cost_tier",
        "latency_tier",
        "reliability_prior",
        "description",
        "embedding_text",
        "dataset_family",
        "focus_slug",
        "primary_role",
        "capability_kind",
        "stage",
        "deterministic",
    }

    LIST_FIELDS = {
        "domain_tags",
        "role_tags",
        "tool_tags",
        "modalities",
        "output_formats",
        "permissions",
        "examples",
        "available_tool_ids",
        "consumes",
        "produces",
    }

    ALL_FIELDS = [
        "id",
        "name",
        "kind",
        "version",
        "updated_at",
        "domain_tags",
        "role_tags",
        "tool_tags",
        "modalities",
        "output_formats",
        "permissions",
        "cost_tier",
        "latency_tier",
        "reliability_prior",
        "description",
        "examples",
        "embedding_text",
        "available_tool_ids",
        "dataset_family",
        "focus_slug",
        "primary_role",
        "capability_kind",
        "stage",
        "deterministic",
        "consumes",
        "produces",
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                version TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                domain_tags TEXT,
                role_tags TEXT,
                tool_tags TEXT,
                modalities TEXT,
                output_formats TEXT,
                permissions TEXT,
                cost_tier TEXT,
                latency_tier TEXT,
                reliability_prior REAL,
                description TEXT,
                examples TEXT,
                embedding_text TEXT,
                available_tool_ids TEXT,
                dataset_family TEXT,
                focus_slug TEXT,
                primary_role TEXT,
                capability_kind TEXT,
                stage TEXT,
                deterministic INTEGER,
                consumes TEXT,
                produces TEXT
            )
            """
        )
        self._ensure_optional_columns()
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TOOL_CODE_TABLE} (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _ensure_optional_columns(self) -> None:
        cursor = self.conn.execute(f"PRAGMA table_info({self.TABLE_NAME})")
        existing = {str(row["name"]) for row in cursor.fetchall()}
        required_defs = {
            "dataset_family": "TEXT",
            "focus_slug": "TEXT",
            "primary_role": "TEXT",
            "capability_kind": "TEXT",
            "stage": "TEXT",
            "deterministic": "INTEGER",
            "consumes": "TEXT",
            "produces": "TEXT",
        }
        for name, column_type in required_defs.items():
            if name in existing:
                continue
            self.conn.execute(
                f"ALTER TABLE {self.TABLE_NAME} ADD COLUMN {name} {column_type}"
            )

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "SQLiteRegistry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register(self, card: BaseCard) -> None:
        if self.get(card.id) is not None:
            raise ValueError(f"Card already exists: {card.id}")
        self._upsert(card)

    def update(self, card: BaseCard) -> None:
        self._upsert(card)

    def remove(self, card_id: str) -> None:
        self.conn.execute(
            f"DELETE FROM {self.TABLE_NAME} WHERE id = ?", (card_id,)
        )
        self.conn.execute(
            f"DELETE FROM {self.TOOL_CODE_TABLE} WHERE id = ?", (card_id,)
        )
        self.conn.commit()

    def get(self, card_id: str) -> Optional[BaseCard]:
        cursor = self.conn.execute(
            f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?", (card_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_card(row)

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseCard]:
        filters = filters or {}
        where_clauses = []
        values: List[Any] = []
        for key, value in filters.items():
            if key in self.SCALAR_FIELDS and value is not None:
                where_clauses.append(f"{key} = ?")
                values.append(value)
        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        cursor = self.conn.execute(
            f"SELECT * FROM {self.TABLE_NAME}{where_sql}", values
        )
        rows = cursor.fetchall()
        cards = [self._row_to_card(row) for row in rows]
        return self._filter_cards(cards, filters)

    def register_tool_code(self, tool_id: str, code: str, updated_at: Optional[datetime] = None) -> None:
        stamp = updated_at or datetime.utcnow()
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.TOOL_CODE_TABLE} (id, code, updated_at) VALUES (?, ?, ?)",
            (tool_id, code, stamp.isoformat()),
        )
        self.conn.commit()

    def get_tool_code(self, tool_id: str) -> Optional[str]:
        cursor = self.conn.execute(
            f"SELECT code FROM {self.TOOL_CODE_TABLE} WHERE id = ?", (tool_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return row["code"]

    def _filter_cards(self, cards: Iterable[BaseCard], filters: Dict[str, Any]) -> List[BaseCard]:
        if not filters:
            return list(cards)
        filtered: List[BaseCard] = []
        for card in cards:
            if self._matches_filters(card, filters):
                filtered.append(card)
        return filtered

    def _matches_filters(self, card: BaseCard, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            if key in self.SCALAR_FIELDS:
                if getattr(card, key) != value:
                    return False
            elif key in self.LIST_FIELDS:
                card_values = getattr(card, key)
                if isinstance(value, list):
                    if not set(value).intersection(card_values):
                        return False
                else:
                    if value not in card_values:
                        return False
            else:
                if getattr(card, key, None) != value:
                    return False
        return True

    def _upsert(self, card: BaseCard) -> None:
        payload = self._serialize_card(card)
        columns = ", ".join(self.ALL_FIELDS)
        placeholders = ", ".join(["?"] * len(self.ALL_FIELDS))
        values = [payload[field] for field in self.ALL_FIELDS]
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

    def _serialize_card(self, card: BaseCard) -> Dict[str, Any]:
        data = card.model_dump(exclude={"embedding_vector"})
        serialized: Dict[str, Any] = {}
        for field in self.ALL_FIELDS:
            value = data.get(field)
            if field in self.LIST_FIELDS:
                serialized[field] = json.dumps(value or [], ensure_ascii=True)
            elif field == "updated_at":
                if isinstance(value, datetime):
                    serialized[field] = value.isoformat()
                else:
                    serialized[field] = value
            elif field == "deterministic":
                serialized[field] = 1 if bool(value) else 0
            else:
                serialized[field] = value
        return serialized

    def _row_to_card(self, row: sqlite3.Row) -> BaseCard:
        payload: Dict[str, Any] = {}
        for field in self.ALL_FIELDS:
            value = row[field]
            if field in self.LIST_FIELDS:
                payload[field] = json.loads(value) if value else []
            elif field == "updated_at":
                payload[field] = datetime.fromisoformat(value)
            elif field == "deterministic":
                payload[field] = bool(value) if value is not None else True
            else:
                payload[field] = value
        kind = payload.get("kind")
        if kind == "tool":
            return ToolCard(**payload)
        if kind == "agent":
            return AgentCard(**payload)
        return BaseCard(**payload)
