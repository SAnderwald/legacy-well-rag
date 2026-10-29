"""Command line: wellrag ingest | ask | chat | plot | eval"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluate import evaluate, load_questions, to_markdown
from .index import Index
from .ingest import load_corpus
from .llm import answer


def _store(index: Index, data_dir: str | None):
    from .logs import LogStore

    path = data_dir or os.getenv("WELLRAG_DATA") or index.data_dir
    if not path:
        raise SystemExit("No data folder known. Re-run ingest or pass --data-dir.")
    return LogStore.load(path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="wellrag", description="AI assistant for legacy LAS logs and well reports")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="chunk a data folder and build the index")
    pi.add_argument("data_dir")
    pi.add_argument("--index-dir", default="index")
    pi.add_argument("--embedder", choices=["auto", "minilm", "tfidf"], default="auto")

    pa = sub.add_parser("ask", help="document Q&A (plain RAG, no tools)")
    pa.add_argument("question")
    pa.add_argument("--index-dir", default="index")
    pa.add_argument("--k", type=int, default=5)
    pa.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    pa.add_argument("--well", help="restrict to wells whose name contains this text")
    pa.add_argument("--backend", choices=["anthropic", "ollama", "none"], default="anthropic")
    pa.add_argument("--model")

    pc = sub.add_parser("chat", help="agent: searches documents, computes curve stats, plots logs")
    pc.add_argument("question")
    pc.add_argument("--index-dir", default="index")
    pc.add_argument("--data-dir")
    pc.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    pc.add_argument("--model")
    pc.add_argument("--plot-out", default="log_plot.png", help="where to save a plot if the agent draws one")
    pc.add_argument("--show-tools", action="store_true", help="print each tool call and result")

    pp = sub.add_parser("plot", help="draw a log display without an LLM")
    pp.add_argument("well")
    pp.add_argument("--top", type=float)
    pp.add_argument("--base", type=float)
    pp.add_argument("--index-dir", default="index")
    pp.add_argument("--data-dir")
    pp.add_argument("--out", default="log_plot.png")

    pe = sub.add_parser("eval", help="score retrieval against labeled questions")
    pe.add_argument("questions")
    pe.add_argument("--index-dir", default="index")
    pe.add_argument("--k", type=int, default=5)
    pe.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        chunks = load_corpus(args.data_dir)
        idx = Index.build(chunks, args.embedder)
        idx.data_dir = args.data_dir
        idx.save(args.index_dir)
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} files "
              f"({idx.embedder.kind} embeddings) -> {args.index_dir}/")

    elif args.cmd == "ask":
        idx = Index.load(args.index_dir)
        hits = idx.search(args.question, k=args.k, mode=args.mode, well=args.well)
        print(answer(args.question, hits, backend=args.backend, model=args.model))
        print("\nSources: " + ", ".join(sorted({h.source for h in hits})))

    elif args.cmd == "chat":
        from .agent import run_agent

        idx = Index.load(args.index_dir)
        res = run_agent(args.question, idx, _store(idx, args.data_dir), backend=args.backend, model=args.model)
        print(res.text)
        if args.show_tools:
            for t in res.trace:
                print(f"\n> {t['tool']}({json.dumps(t['input'])})\n{json.dumps(t['output'])[:600]}")
        print("\nTools used: " + (", ".join(t["tool"] for t in res.trace) or "none"))
        for i, fig in enumerate(res.figures):
            out = Path(args.plot_out)
            path = out if i == 0 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {path}")

    elif args.cmd == "plot":
        idx = Index.load(args.index_dir)
        _store(idx, args.data_dir).plot_log(args.well, args.top, args.base).savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {args.out}")

    elif args.cmd == "eval":
        res = evaluate(Index.load(args.index_dir), load_questions(args.questions), k=args.k)
        print(json.dumps(res, indent=2) if args.json else to_markdown(res))


if __name__ == "__main__":
    main()
