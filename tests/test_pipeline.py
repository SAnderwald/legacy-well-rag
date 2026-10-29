import subprocess
import sys
from pathlib import Path

import pytest

from wellrag.evaluate import evaluate, load_questions
from wellrag.index import Index
from wellrag.ingest import load_corpus
from wellrag.llm import answer, build_prompt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(out), "--n", "6"], check=True)
    return out


@pytest.fixture(scope="module")
def index(corpus):
    return Index.build(load_corpus(corpus), embedder="tfidf")


def test_chunks_cover_las_and_reports(index):
    kinds = {c.kind for c in index.chunks}
    assert {"las_well", "las_curves", "las_params", "report"} <= kinds


def test_search_returns_named_well(index):
    well = index.chunks[0].well
    hits = index.search(f"What curves were logged on the {well.title()}?", k=3)
    assert hits and all(h.well == well for h in hits)


def test_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    again = Index.load(tmp_path)
    assert len(again.chunks) == len(index.chunks)
    q = f"initial potential {index.chunks[0].well}"
    assert [c.id for c in again.search(q, k=3)] == [c.id for c in index.search(q, k=3)]


def test_retrieval_quality(index, corpus):
    res = evaluate(index, load_questions(corpus / "questions.jsonl"), k=5)
    assert res["bm25"]["hit@5"] >= 0.9
    assert res["hybrid"]["hit@5"] >= 0.9


def test_prompt_numbers_excerpts_and_retrieval_only_backend(index):
    hits = index.search("perforated interval", k=2)
    prompt = build_prompt("Which interval was perforated?", hits)
    assert "[1]" in prompt and "[2]" in prompt
    assert answer("q", hits, backend="none").startswith("Retrieval only")
    assert "No matching excerpts" in answer("q", [], backend="none")
