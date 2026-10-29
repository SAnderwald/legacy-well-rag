"""Retrieval evaluation: hit@1, hit@k and MRR per configuration against labeled questions."""
from __future__ import annotations

import json
from pathlib import Path

from .index import Index

# (label, search mode, well-name routing). Routing is off here so each mode is scored on its own.
CONFIGS = [("bm25", "bm25", False), ("dense", "dense", False), ("hybrid", "hybrid", False)]


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(index: Index, questions: list[dict], k: int = 5, configs=CONFIGS) -> dict:
    results = {}
    for name, mode, routing in configs:
        hit1 = hitk = rr = 0.0
        for q in questions:
            hits = index.search(q["question"], k=k, mode=mode, auto_well=routing)
            kinds = set(q.get("kinds", []))
            rank = next(
                (i for i, h in enumerate(hits, 1) if h.source == q["source"] and (not kinds or h.kind in kinds)),
                None,
            )
            if rank:
                hitk += 1
                hit1 += rank == 1
                rr += 1 / rank
        n = len(questions)
        results[name] = {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n}
    return results


def to_markdown(results: dict) -> str:
    cols = list(next(iter(results.values())).keys())
    lines = ["| retrieval | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    return "\n".join(lines)
