"""Grounded answer generation with citations. Backends: Claude API, local Ollama, or retrieval-only."""
from __future__ import annotations

import os

from .ingest import Chunk

SYSTEM = (
    "You answer questions about oil and gas wells using only the numbered excerpts from LAS file headers "
    "and well reports provided. Cite every fact with its excerpt number, like [2]. Quote depths, rates and "
    "units exactly as written. If the excerpts do not contain the answer, say so plainly instead of guessing."
)


def build_prompt(question: str, hits: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i}] ({h.source}, {h.kind})\n{h.text}" for i, h in enumerate(hits, 1))
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}"


def answer(question: str, hits: list[Chunk], backend: str = "anthropic", model: str | None = None) -> str:
    if not hits:
        return "No matching excerpts were found in the index."
    if backend == "none":
        return "Retrieval only (no LLM). Top excerpts:\n\n" + "\n\n".join(
            f"[{i}] {h.source} ({h.kind})\n{h.text}" for i, h in enumerate(hits, 1)
        )

    prompt = build_prompt(question, hits)

    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or os.getenv("WELLRAG_MODEL", "claude-haiku-5-5"),
            max_tokens=800,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    if backend == "ollama":
        import requests

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.post(
            f"{host}/api/chat",
            json={
                "model": model or os.getenv("WELLRAG_OLLAMA_MODEL", "llama3.1:8b"),
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    raise ValueError(f"unknown backend {backend!r}")
