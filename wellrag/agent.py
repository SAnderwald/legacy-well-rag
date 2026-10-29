"""Tool-calling agent: the LLM decides whether to search documents, compute curve statistics, or plot a log."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .index import Index
from .logs import LogStore

AGENT_SYSTEM = (
    "You are a well-log interpretation assistant for legacy oil and gas wells. You have tools:\n"
    "- search_documents: facts from LAS headers and completion reports (operator, dates, KB, mud weight, "
    "perforations, IP tests, drilling problems).\n"
    "- curve_stats and compare_wells: any number derived from curve data. Never estimate curve values yourself; "
    "always compute them with a tool.\n"
    "- plot_log: when the user asks to see, show, display or plot a log. The plot is shown to the user automatically. "
    "Perforations and formation tops are always drawn on the plot.\n"
    "- list_wells: well names, available curves, formation tops and perforations.\n"
    "Depths increase downhole: top is the shallower number, base the deeper one. Pass depths as plain numbers. "
    "Call tools through the tool-calling interface; never write a tool call as text in your answer. "
    "Use list_wells first if you are unsure of a well name, curve or formation. "
    "Cite document facts as [doc N] using the numbers search_documents returns, and say when a number was computed "
    "from LAS curve data. Report depths in feet and keep units. If the tools cannot answer, say so."
)

TOOL_SPECS = [
    {"name": "search_documents",
     "description": "Search indexed LAS headers and well reports. Returns numbered excerpts to cite.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look for"},
         "well": {"type": "string", "description": "Optional well name to restrict the search"}},
         "required": ["query"]}},
    {"name": "curve_stats",
     "description": "Exact statistics (mean, min, max, P10/P50/P90, depth of max) for one curve over a depth interval "
                    "or a named formation. Omit top/base and formation for the whole log.",
     "input_schema": {"type": "object", "properties": {
         "well": {"type": "string"}, "curve": {"type": "string", "description": "Mnemonic, e.g. GR, ILD, NPHI"},
         "top": {"type": "number", "description": "Top depth, ft"}, "base": {"type": "number", "description": "Base depth, ft"},
         "formation": {"type": "string", "description": "Formation name from the well's tops, e.g. Frio"}},
         "required": ["well", "curve"]}},
    {"name": "compare_wells",
     "description": "Compare one curve across wells over a formation or depth interval. Omit wells to use all wells.",
     "input_schema": {"type": "object", "properties": {
         "curve": {"type": "string"}, "formation": {"type": "string"},
         "wells": {"type": "array", "items": {"type": "string"}},
         "top": {"type": "number"}, "base": {"type": "number"}},
         "required": ["curve"]}},
    {"name": "plot_log",
     "description": "Draw a standard log display (GR/SP, resistivity, porosity tracks with tops and perforations) "
                    "for a well and optional depth interval.",
     "input_schema": {"type": "object", "properties": {
         "well": {"type": "string"}, "top": {"type": "number"}, "base": {"type": "number"}},
         "required": ["well"]}},
    {"name": "list_wells",
     "description": "List wells with their curves, depth ranges, formation tops and perforations.",
     "input_schema": {"type": "object", "properties": {}}},
]


@dataclass
class AgentResult:
    text: str
    figures: list = field(default_factory=list)
    trace: list = field(default_factory=list)  # [{"tool": name, "input": args, "output": result}]


class Toolbox:
    def __init__(self, index: Index, store: LogStore):
        self.index, self.store = index, store
        self.figures, self.trace, self._doc_n = [], [], 0

    def execute(self, name: str, args: dict) -> dict:
        args = _coerce(name, args)
        try:
            out = getattr(self, f"_{name}")(**(args or {}))
        except (KeyError, ValueError, TypeError) as e:
            out = {"error": str(e).strip("'\"")}
        except AttributeError:
            out = {"error": f"Unknown tool {name}"}
        self.trace.append({"tool": name, "input": args, "output": out})
        return out

    def _search_documents(self, query, well=None):
        hits = self.index.search(query, k=5, well=well)
        excerpts = []
        for h in hits:
            self._doc_n += 1
            excerpts.append({"doc": self._doc_n, "source": h.source, "kind": h.kind, "text": h.text})
        return {"excerpts": excerpts}

    def _curve_stats(self, well, curve, top=None, base=None, formation=None):
        return self.store.curve_stats(well, curve, top, base, formation)

    def _compare_wells(self, curve, formation=None, wells=None, top=None, base=None):
        return {"rows": self.store.compare_wells(curve, formation, wells, top, base)}

    def _plot_log(self, well, top=None, base=None):
        fig = self.store.plot_log(well, top, base)
        self.figures.append(fig)
        w = self.store.get(well)
        return {"plotted": w.name, "top": top, "base": base, "curves": list(w.curves),
                "note": "The plot is displayed to the user."}

    def _list_wells(self):
        return {"wells": self.store.summary()}


def _anthropic_loop(question, tb, model, client, max_steps):
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = client.messages.create(model=model or os.getenv("WELLRAG_MODEL", "claude-sonnet-5-5"), max_tokens=1500,
                                      system=AGENT_SYSTEM, tools=TOOL_SPECS, messages=msgs)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        msgs.append({"role": "assistant", "content": resp.content})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(tb.execute(b.name, b.input))}
            for b in resp.content if b.type == "tool_use"]})
    return "Stopped after the maximum number of tool steps."


TOOL_NAMES = {t["name"] for t in TOOL_SPECS}


def text_tool_calls(content: str) -> list[dict]:
    """Small local models sometimes write a tool call as JSON text instead of using the tool-calling API.
    Recover calls like {"name": "plot_log", "parameters": {...}} found anywhere in the reply."""
    if not content or "{" not in content:
        return []
    calls, dec, i = [], json.JSONDecoder(), 0
    while (i := content.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(content, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("parameters", obj.get("arguments", {})) or {}
            calls.append({"function": {"name": obj["name"], "arguments": args}})
        i = end
    return calls


def _coerce(name: str, args: dict) -> dict:
    """Turn numeric strings ("8,500") into numbers for fields the schema declares as numbers."""
    spec = next((t["input_schema"]["properties"] for t in TOOL_SPECS if t["name"] == name), {})
    out = {}
    for k, v in (args or {}).items():
        if spec.get(k, {}).get("type") == "number" and isinstance(v, str):
            try:
                v = float(v.replace(",", "").replace("ft", "").strip())
            except ValueError:
                pass
        out[k] = v
    return out


def _ollama_loop(question, tb, model, post, max_steps):
    if post is None:
        import requests

        post = requests.post
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                "parameters": t["input_schema"]}} for t in TOOL_SPECS]
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
    num_ctx = int(os.getenv("WELLRAG_NUM_CTX", "8192"))
    for _ in range(max_steps):
        r = post(f"{host}/api/chat", json={"model": model or os.getenv("WELLRAG_OLLAMA_MODEL", "llama3.1:8b"),
                                           "messages": msgs, "tools": tools, "stream": False,
                                           "options": {"temperature": 0, "num_ctx": num_ctx}}, timeout=300)
        r.raise_for_status()
        m = r.json()["message"]
        calls = m.get("tool_calls") or text_tool_calls(m.get("content", ""))
        if not calls:
            return m.get("content", "")
        m = {**m, "tool_calls": calls, "content": "" if not m.get("tool_calls") else m.get("content", "")}
        msgs.append(m)
        for c in calls:
            f = c["function"]
            args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
            msgs.append({"role": "tool", "content": json.dumps(tb.execute(f["name"], args)), "tool_name": f["name"]})
    return "Stopped after the maximum number of tool steps."


def run_agent(question: str, index: Index, store: LogStore, backend: str = "anthropic", model: str | None = None,
              client=None, max_steps: int = 8) -> AgentResult:
    tb = Toolbox(index, store)
    if backend == "anthropic":
        text = _anthropic_loop(question, tb, model, client, max_steps)
    elif backend == "ollama":
        text = _ollama_loop(question, tb, model, client, max_steps)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return AgentResult(text, tb.figures, tb.trace)
