"""Hybrid index: BM25 keyword search + dense embeddings, fused with reciprocal rank fusion."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Embedder:
    """MiniLM sentence embeddings when available, TF-IDF + SVD fallback otherwise."""

    def __init__(self, kind: str = "auto", model_name: str = DEFAULT_MODEL):
        self.kind, self.model_name, self._model = kind, model_name, None
        if kind in {"auto", "minilm"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.kind = "minilm"
            except Exception:
                if kind == "minilm":
                    raise
                self.kind = "tfidf"

    def fit(self, texts: list[str]) -> None:
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "minilm":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        vecs = self._model.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, data_dir: str | None = None):
        self.chunks, self.vectors, self.embedder, self.data_dir = chunks, vectors, embedder, data_dir
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: str = "auto") -> "Index":
        emb = Embedder(embedder)
        texts = [c.text for c in chunks]
        emb.fit(texts)
        return cls(chunks, emb.encode(texts), emb)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        np.save(out / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder.kind, "model_name": self.embedder.model_name,
                "data_dir": str(Path(self.data_dir).resolve()) if self.data_dir else None}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.embedder.kind == "tfidf":
            with open(out / "tfidf.pkl", "wb") as f:
                pickle.dump(self.embedder._model, f)

    @classmethod
    def load(cls, index_dir: str | Path) -> "Index":
        d = Path(index_dir)
        meta = json.loads((d / "meta.json").read_text())
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        if meta["embedder"] == "tfidf":
            emb = Embedder.__new__(Embedder)
            emb.kind, emb.model_name = "tfidf", meta["model_name"]
            with open(d / "tfidf.pkl", "rb") as f:
                emb._model = pickle.load(f)
        else:
            emb = Embedder("minilm", meta["model_name"])
        return cls(chunks, np.load(d / "vectors.npy"), emb, meta.get("data_dir"))

    def _rank(self, query: str, mode: str) -> np.ndarray:
        if mode == "bm25":
            return np.argsort(-self.bm25.get_scores(tokenize(query)))
        if mode == "dense":
            return np.argsort(-(self.vectors @ self.embedder.encode([query])[0]))
        if mode == "hybrid":
            rrf = np.zeros(len(self.chunks))
            for m in ("bm25", "dense"):
                rrf[self._rank(query, m)] += 1.0 / (60 + np.arange(1, len(self.chunks) + 1))
            return np.argsort(-rrf)
        raise ValueError(f"unknown mode {mode!r}")

    def detect_well(self, query: str) -> str | None:
        """Entity routing: return the longest indexed well name that appears in the query."""
        q = " ".join(tokenize(query))
        names = {c.well for c in self.chunks}
        found = [n for n in names if f" {' '.join(tokenize(n))} " in f" {q} "]
        return max(found, key=len) if found else None

    def search(self, query: str, k: int = 5, mode: str = "hybrid", well: str | None = None,
               auto_well: bool = True) -> list[Chunk]:
        if well is None and auto_well:
            well = self.detect_well(query)
        hits = []
        for i in self._rank(query, mode):
            c = self.chunks[i]
            if well and well.lower() != c.well.lower() and well.lower() not in c.well.lower():
                continue
            hits.append(c)
            if len(hits) == k:
                break
        return hits
