import subprocess
import sys
from pathlib import Path

import pytest

from wellrag.agent import run_agent, text_tool_calls
from wellrag.index import Index
from wellrag.ingest import load_corpus
from wellrag.logs import LogStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(out), "--n", "4"], check=True)
    return Index.build(load_corpus(out), embedder="tfidf"), LogStore.load(out)


def test_interval_accepts_reversed_and_string_depths(setup):
    _, store = setup
    w = next(iter(store.wells.values()))
    a = store.curve_stats(w.name, next(iter(w.curves)), "5,500", 5000)
    assert (a["top"], a["base"]) == (5000.0, 5500.0)


def test_text_tool_call_is_recovered():
    reply = 'Let me try again.\n{"name": "plot_log", "parameters": {"base":"8500","top":"8000","well":"X"}}'
    calls = text_tool_calls(reply)
    assert calls == [{"function": {"name": "plot_log", "arguments": {"base": "8500", "top": "8000", "well": "X"}}}]
    assert text_tool_calls("No tools needed here {not json}") == []


class FakeResp:
    def __init__(self, msg):
        self.msg = msg

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": self.msg}


def test_ollama_loop_runs_text_tool_call(setup):
    index, store = setup
    w = next(iter(store.wells.values()))
    replies = iter([
        {"role": "assistant", "content": f'{{"name": "plot_log", "parameters": {{"well": "{w.name}", "top": "6000", "base": "5000"}}}}'},
        {"role": "assistant", "content": "Here is the plot."},
    ])
    res = run_agent("plot it", index, store, backend="ollama", client=lambda *a, **k: FakeResp(next(replies)))
    assert res.text == "Here is the plot."
    assert len(res.figures) == 1
    assert res.trace[0]["input"] == {"well": w.name, "top": 6000.0, "base": 5000.0}
    assert "error" not in res.trace[0]["output"]


def test_parse_plot_request(setup):
    _, store = setup
    w = next(w for w in store.wells.values() if w.perfs and w.tops)
    p_top, p_base, _ = w.perfs[0]
    name = w.name.title()
    assert store.parse_plot_request(f"Show me the {name} between 8,500-8000 ft with perfs")[1:3] == (8000.0, 8500.0)
    assert store.parse_plot_request(f"plot {name} around 6000")[1:3] == (5700.0, 6300.0)
    _, top, base, note = store.parse_plot_request(f"show {name} perfs")
    assert top == p_top - 300 and base == p_base + 300 and "perforations" in note
    f = next(iter(w.tops))
    assert store.parse_plot_request(f"{name} {f}")[1:3] == w.formation_interval(f)
    assert store.parse_plot_request("plot 7000 to 7400", default_well=w.name)[:3] == (w.name, 7000.0, 7400.0)
    with pytest.raises(KeyError):
        store.parse_plot_request("plot something")


def test_inventory_and_bad_files(tmp_path, setup):
    _, store = setup
    rows = store.inventory()
    assert len(rows) == len(store.wells)
    assert {"well", "file", "curves", "depth", "tops", "perfs", "report"} <= set(rows[0])
    assert all(r["report"] != "none" for r in rows)

    import shutil
    w = next(iter(store.wells.values()))
    shutil.copy(w.las_path, tmp_path / "a.las")
    shutil.copy(w.las_path, tmp_path / "b_duplicate.las")
    (tmp_path / "broken.las").write_text("not a las file at all")
    s = LogStore.load(tmp_path)
    assert len(s.wells) == 1
    assert any("duplicate" in msg for _, msg in s.errors)


def test_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        LogStore.load("/no/such/folder")
