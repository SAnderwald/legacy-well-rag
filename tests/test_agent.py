import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wellrag.agent import Toolbox, run_agent
from wellrag.index import Index
from wellrag.ingest import load_corpus
from wellrag.logs import LogStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(out), "--n", "6"], check=True)
    return out


@pytest.fixture(scope="module")
def index(data):
    return Index.build(load_corpus(data), embedder="tfidf")


@pytest.fixture(scope="module")
def store(data):
    return LogStore.load(data)


def a_well(store):
    return next(w for w in store.wells.values() if w.perfs)


def test_tops_and_perfs_parsed(store):
    w = a_well(store)
    assert len(w.tops) >= 3 and w.perfs
    top, base = w.formation_interval(next(iter(w.tops)))
    assert base > top


def test_curve_stats_match_numpy(store):
    w = a_well(store)
    mn = next(iter(w.curves))
    s = store.curve_stats(w.name, mn, 4000, 5000)
    v = w.curves[mn]["values"]
    mask = (w.depth >= 4000) & (w.depth <= 5000) & np.isfinite(v)
    assert s["mean"] == pytest.approx(v[mask].mean(), abs=1e-3)
    assert s["samples"] == mask.sum()


def test_pay_sand_is_resistive(store):
    w = next(w for w in store.wells.values() if w.perfs and {"ILD", "LN"} & set(w.curves))
    mn = "ILD" if "ILD" in w.curves else "LN"
    p_top, p_base, _ = w.perfs[0]
    assert store.curve_stats(w.name, mn, p_top, p_base)["mean"] > 3 * store.curve_stats(w.name, mn)["p50"]


def test_errors_are_returned_not_raised(index, store):
    tb = Toolbox(index, store)
    assert "Available wells" in tb.execute("curve_stats", {"well": "NO SUCH WELL", "curve": "GR"})["error"]
    assert "Available curves" in tb.execute("curve_stats", {"well": a_well(store).name, "curve": "XYZ"})["error"]


def test_plot_has_one_axis_per_track(store):
    w = a_well(store)
    fig = store.plot_log(w.name, w.perfs[0][0] - 200, w.perfs[0][1] + 200)
    assert len(fig.axes) >= 2


class FakeClaude:
    """Scripted stand-in for anthropic.Anthropic: asks for curve_stats and plot_log, then answers."""

    def __init__(self, well):
        self.well, self.calls = well, []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return SimpleNamespace(stop_reason="tool_use", content=[
                SimpleNamespace(type="tool_use", id="t1", name="curve_stats", input={"well": self.well, "curve": "GR"}),
                SimpleNamespace(type="tool_use", id="t2", name="plot_log", input={"well": self.well})])
        results = json.loads(kw["messages"][-1]["content"][0]["content"])
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=f"Mean GR is {results['mean']} API.")])


def test_agent_loop_runs_tools_and_answers(index, store):
    well = next(w.name for w in store.wells.values() if "GR" in w.curves)
    fake = FakeClaude(well)
    res = run_agent("Plot it and give the mean GR", index, store, client=fake)
    assert [t["tool"] for t in res.trace] == ["curve_stats", "plot_log"]
    assert len(res.figures) == 1
    assert str(store.curve_stats(well, "GR")["mean"]) in res.text
    assert fake.calls[0]["tools"] and fake.calls[1]["messages"][-1]["content"][1]["tool_use_id"] == "t2"
