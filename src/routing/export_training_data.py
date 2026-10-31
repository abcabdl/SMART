from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

from core.registry import SQLiteRegistry


def _load_runs(runs_dir: str) -> List[Dict]:
    records: List[Dict] = []
    for path in glob.glob(os.path.join(runs_dir, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def export_training_data(runs_dir: str, db_path: str, out_path: str) -> int:
    records = _load_runs(runs_dir)
    if not records:
        return 0

    with SQLiteRegistry(db_path) as registry:
        with open(out_path, "w", encoding="utf-8") as out:
            count = 0
            for record in records:
                role = record.get("role", "")
                output = record.get("output", {})
                query_text = f"{role} {json.dumps(output, ensure_ascii=True)}"
                selected_main = record.get("selected_main")
                candidates = record.get("candidates_topk", [])
                for cand in candidates:
                    card_id = cand.get("card_id")
                    if not card_id:
                        continue
                    card = registry.get(card_id)
                    if card is None:
                        continue
                    candidate_text = card.embedding_text or card.description or card.name
                    label = 1 if card_id == selected_main else 0
                    out.write(
                        json.dumps(
                            {
                                "query": query_text,
                                "candidate": candidate_text,
                                "label": label,
                                "role": role,
                                "card_id": card_id,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training data from runs")
    parser.add_argument("--runs", type=str, default="runs", help="runs directory")
    parser.add_argument("--db", type=str, required=True, help="sqlite registry")
    parser.add_argument("--out", type=str, required=True, help="output jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_training_data(args.runs, args.db, args.out)


if __name__ == "__main__":
    main()
