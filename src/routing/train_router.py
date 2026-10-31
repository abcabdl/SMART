from __future__ import annotations

import argparse
import json

from routing.reranker_model import TrainingExample, TfidfLinearReranker


def load_examples(path: str) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                TrainingExample(
                    query=record.get("query", ""),
                    candidate=record.get("candidate", ""),
                    label=int(record.get("label", 0)),
                )
            )
    return examples


def train_reranker(data_path: str, model_path: str, epochs: int = 5, lr: float = 0.1) -> None:
    examples = load_examples(data_path)
    if not examples:
        raise ValueError("No training data found")
    model = TfidfLinearReranker()
    model.fit(examples, epochs=epochs, lr=lr)
    model.save(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reranker")
    parser.add_argument("--data", required=True, type=str, help="training data jsonl")
    parser.add_argument("--out", required=True, type=str, help="output model path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_reranker(args.data, args.out, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
