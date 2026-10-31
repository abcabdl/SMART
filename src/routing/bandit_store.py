from __future__ import annotations

import random
import sqlite3
from typing import Dict, Iterable, Tuple


class BanditStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bandits (
                workflow_version TEXT NOT NULL,
                role TEXT NOT NULL,
                card_id TEXT NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                PRIMARY KEY (workflow_version, role, card_id)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "BanditStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get(self, workflow_version: str, role: str, card_id: str) -> Tuple[float, float]:
        cursor = self.conn.execute(
            "SELECT alpha, beta FROM bandits WHERE workflow_version = ? AND role = ? AND card_id = ?",
            (workflow_version, role, card_id),
        )
        row = cursor.fetchone()
        if row is None:
            return 1.0, 1.0
        return float(row["alpha"]), float(row["beta"])

    def update(
        self,
        workflow_version: str,
        role: str,
        card_id: str,
        reward: float,
        confidence: float = 1.0,
    ) -> None:
        reward = max(0.0, min(1.0, float(reward)))
        confidence = max(0.0, min(1.0, float(confidence)))
        alpha, beta = self.get(workflow_version, role, card_id)
        alpha += reward * confidence
        beta += (1.0 - reward) * confidence
        self.conn.execute(
            "INSERT OR REPLACE INTO bandits (workflow_version, role, card_id, alpha, beta) VALUES (?, ?, ?, ?, ?)",
            (workflow_version, role, card_id, alpha, beta),
        )
        self.conn.commit()

    def select_thompson(
        self,
        workflow_version: str,
        role: str,
        candidates: Iterable[str],
        rng: random.Random | None = None,
    ) -> Dict[str, float]:
        rng = rng or random.Random()
        scores: Dict[str, float] = {}
        for card_id in candidates:
            alpha, beta = self.get(workflow_version, role, card_id)
            sample = rng.gammavariate(alpha, 1.0) / (
                rng.gammavariate(alpha, 1.0) + rng.gammavariate(beta, 1.0)
            )
            scores[card_id] = sample
        return scores
