"""Shiny for Python front end. Run: shiny run app.py

Data tab: choose a project folder of LAS files (and optional reports) and see what was loaded.
Assistant tab: ask questions, plot logs, and inspect the agent's tool calls.
"""
import json
import os
from pathlib import Path

import pandas as pd
from shiny import App, reactive, render, ui

from wellrag.agent import run_agent
from wellrag.index import Index
from wellrag.ingest import load_corpus
from wellrag.logs import LogStore

RECENT_FILE = Path.home() / ".wellrag_recent.json"


def read_recent() -> list[str]:
    try:
        return [p for p in json.loads(RECENT_FILE.read_text()) if Path(p).is_dir()][:10]
    except Exception:
        return []


def remember(folder: str) -> list[str]:
    recent = [folder] + [p for p in read_recent() if p != folder]
    try:
        RECENT_FILE.write_text(json.dumps(recent[:10], indent=2))
    except OSError:
        pass
    return recent[:10]


def load_project(folder: str, embedder: str = "auto"):
    """Read every LAS file and report under folder and build an in-memory search index."""
    store = LogStore.load(folder)
    if not store.wells:
        raise ValueError(f"No readable .las files found under {store.data_dir}")
    index = Index.build(load_corpus(store.data_dir), embedder)
    index.data_dir = str(store.data_dir)
    return store, index


def startup_folder() -> str:
    if os.getenv("WELLRAG_DATA"):
        return os.getenv("WELLRAG_DATA")
    try:
        meta = json.loads((Path(os.getenv("WELLRAG_INDEX", "index")) / "meta.json").read_text())
        if meta.get("data_dir"):
            return meta["data_dir"]
    except (OSError, ValueError):
        pass
    recent = read_recent()
    return recent[0] if recent else ""


START = startup_folder()

data_tab = ui.nav_panel(
    "Data",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Project folder"),
            ui.input_text("folder", "Folder with LAS files", value=START, placeholder="~/Projects/my_field/las",
                          width="100%"),
            ui.input_select("recent", "Recent folders", [""] + read_recent(), width="100%"),
            ui.input_select("embedder", "Search embeddings",
                            {"tfidf": "TF-IDF (fast, offline)", "minilm": "MiniLM (better, needs PyTorch)",
                             "auto": "Auto (MiniLM if available)"}),
            ui.input_action_button("load", "Load folder", class_="btn-primary w-100"),
            ui.help_text("Subfolders are included. Optional .txt/.md reports with a 'WELL:' line are linked to "
                         "their well and supply formation tops and perforations."),
            width=360,
        ),
        ui.output_ui("data_status"),
        ui.card(ui.card_header("LAS files"), ui.output_data_frame("files")),
        ui.output_ui("problems"),
    ),
)

assistant_tab = ui.nav_panel(
    "Assistant",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Ask the assistant"),
            ui.input_text_area("q", None, rows=4, width="100%",
                               placeholder="Show me the Kubena 2 log around the perforations and give the "
                                           "average GR in the perforated zone."),
            ui.input_select("backend", "LLM", {"ollama": "Ollama (local)", "anthropic": "Claude (API)"}),
            ui.layout_columns(
                ui.input_action_button("go", "Ask", class_="btn-primary w-100"),
                ui.input_action_button("qplot", "Plot", class_="btn-success w-100"),
                col_widths=(6, 6),
            ),
            ui.help_text("Plot reads the well and depths from your question and draws the log instantly, "
                         "no LLM needed. With no depths it centers on the perforations or a named formation."),
            ui.hr(),
            ui.h5("Or plot directly"),
            ui.input_select("well", "Well", []),
            ui.input_numeric("top", "Top (ft)", None),
            ui.input_numeric("base", "Base (ft)", None),
            ui.input_action_button("plot", "Plot log", class_="w-100"),
            width=340,
        ),
        ui.card(ui.card_header("Answer"), ui.output_ui("ans")),
        ui.card(ui.card_header("Log display"), ui.output_plot("log", height="820px")),
        ui.card(ui.card_header("Tool calls"), ui.output_data_frame("trace")),
    ),
)

app_ui = ui.page_navbar(data_tab, assistant_tab, title="Legacy Well Log Assistant", id="tabs")


def server(input, output, session):
    project = reactive.value(None)  # (store, index)
    status = reactive.value("Choose a folder of LAS files and click **Load folder**.")
    answer_text = reactive.value("Ask a question or plot a well.")
    figure, calls = reactive.value(None), reactive.value([])

    def do_load(folder: str, embedder: str):
        folder = str(Path(folder.strip()).expanduser()) if folder.strip() else ""
        if not folder:
            status.set("Enter a folder path first.")
            return
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Reading LAS files and building the search index...")
            try:
                store, index = load_project(folder, embedder)
            except (FileNotFoundError, ValueError, OSError) as e:
                status.set(f"**Could not load:** {e}")
                return
        project.set((store, index))
        names = sorted(store.names())
        ui.update_select("well", choices=names, selected=names[0])
        ui.update_select("recent", choices=[""] + remember(folder), selected="")
        figure.set(None)
        calls.set([])
        answer_text.set(f"Loaded {len(names)} wells. Ask a question or plot a well.")
        n_reports = sum(1 for w in store.wells.values() if w.reports)
        status.set(f"**{folder}**: {len(names)} wells loaded, {n_reports} with reports, "
                   f"{len(index.chunks)} searchable chunks ({index.embedder.kind} embeddings).")

    @reactive.effect
    def _initial_load():
        with reactive.isolate():
            if START and project() is None:
                do_load(START, "tfidf")

    @reactive.effect
    @reactive.event(input.recent)
    def _pick_recent():
        if input.recent():
            ui.update_text("folder", value=input.recent())

    @reactive.effect
    @reactive.event(input.load)
    def _load():
        do_load(input.folder(), input.embedder())

    @render.ui
    def data_status():
        return ui.markdown(status())

    @render.data_frame
    def files():
        if project() is None:
            return pd.DataFrame()
        return render.DataGrid(pd.DataFrame(project()[0].inventory()), filters=True, height="560px")

    @render.ui
    def problems():
        if project() is None or not project()[0].errors:
            return None
        rows = "".join(f"<li><code>{f}</code>: {msg}</li>" for f, msg in project()[0].errors)
        return ui.card(ui.card_header(f"Skipped files ({len(project()[0].errors)})"), ui.HTML(f"<ul>{rows}</ul>"))

    def need_project():
        if project() is None:
            answer_text.set("No data loaded. Open the **Data** tab and load a folder of LAS files.")
            return None
        return project()

    @reactive.effect
    @reactive.event(input.go)
    def _ask():
        p = need_project()
        if not p:
            return
        store, index = p
        try:
            res = run_agent(input.q(), index, store, backend=input.backend())
            answer_text.set(res.text)
            calls.set(res.trace)
            if res.figures:
                figure.set(res.figures[-1])
        except Exception as e:
            answer_text.set(f"LLM call failed ({e}). Check your API key or that Ollama is running.")

    @reactive.effect
    @reactive.event(input.qplot)
    def _quick_plot():
        p = need_project()
        if not p:
            return
        store = p[0]
        try:
            well, top, base, note = store.parse_plot_request(input.q() or "", default_well=input.well())
            figure.set(store.plot_log(well, top, base))
            answer_text.set(f"Plotted **{well}**, {note}.")
            ui.update_select("well", selected=well)
            ui.update_numeric("top", value=top)
            ui.update_numeric("base", value=base)
        except (KeyError, ValueError) as e:
            answer_text.set(str(e).strip("'\""))

    @reactive.effect
    @reactive.event(input.plot)
    def _plot():
        p = need_project()
        if not p:
            return
        try:
            figure.set(p[0].plot_log(input.well(), input.top(), input.base()))
        except (KeyError, ValueError) as e:
            answer_text.set(str(e).strip("'\""))

    @render.ui
    def ans():
        return ui.markdown(answer_text())

    @render.plot
    def log():
        return figure()

    @render.data_frame
    def trace():
        rows = [{"tool": t["tool"], "input": json.dumps(t["input"]), "output": json.dumps(t["output"])[:300]}
                for t in calls()]
        return pd.DataFrame(rows, columns=["tool", "input", "output"])


app = App(app_ui, server)
