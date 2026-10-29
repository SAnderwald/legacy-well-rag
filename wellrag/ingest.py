"""Turn LAS files and text well reports into retrievable chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import lasio
import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    well: str
    kind: str  # las_well | las_curves | las_params | las_stats | report

    def to_dict(self) -> dict:
        return asdict(self)


def _val(item) -> str:
    return str(item.value).strip()


def las_chunks(path: Path) -> list[Chunk]:
    """One chunk per LAS header section, plus a curve-statistics chunk."""
    las = lasio.read(str(path), ignore_header_errors=True)
    well_items = {i.mnemonic.upper(): i for i in las.well}
    well = _val(well_items["WELL"]) if "WELL" in well_items and _val(well_items["WELL"]) else path.stem
    src = path.name
    head = f"Well {well} (LAS file {src})."

    def describe(item) -> str:
        label = item.descr.strip() or item.mnemonic
        return f"{label}: {_val(item)} {item.unit}".strip()

    chunks = []
    header = [describe(i) for i in las.well if _val(i)]
    remarks = (las.other or "").strip()
    header_text = f"{head} Well header. " + "; ".join(header) + (f". Remarks: {remarks}" if remarks else "")
    chunks.append(Chunk(f"{src}#well", header_text, src, well, "las_well"))

    curves = [f"{c.mnemonic} ({c.unit or 'no unit'}) - {c.descr.strip()}" for c in las.curves]
    chunks.append(Chunk(f"{src}#curves", f"{head} Logged curves: " + "; ".join(curves), src, well, "las_curves"))

    params = [describe(i) for i in las.params if _val(i)]
    if params:
        chunks.append(Chunk(f"{src}#params", f"{head} Parameters: " + "; ".join(params), src, well, "las_params"))

    depth = np.asarray(las.index, dtype=float)
    stats = []
    for c in las.curves[1:]:
        data = np.asarray(c.data, dtype=float)
        ok = np.isfinite(data)
        if not ok.any():
            continue
        d = depth[ok]
        stats.append(
            f"{c.mnemonic}: {d.min():.0f}-{d.max():.0f} {las.curves[0].unit or 'ft'}, "
            f"min {data[ok].min():.2f}, max {data[ok].max():.2f}, mean {data[ok].mean():.2f} {c.unit}".strip()
        )
    if stats:
        chunks.append(Chunk(f"{src}#stats", f"{head} Curve coverage and statistics: " + "; ".join(stats), src, well, "las_stats"))
    return chunks


WELL_LINE = re.compile(r"^\s*WELL(?: NAME)?\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def report_chunks(path: Path, max_chars: int = 900) -> list[Chunk]:
    """Paragraph-aware chunking of a plain-text well report."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = WELL_LINE.search(text)
    well = m.group(1) if m else path.stem
    src = path.name
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    groups, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > max_chars:
            groups.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        groups.append(buf)

    return [
        Chunk(f"{src}#{i}", f"Well {well} report {src}.\n{g}", src, well, "report")
        for i, g in enumerate(groups)
    ]


def load_corpus(data_dir: str | Path) -> list[Chunk]:
    data_dir = Path(data_dir)
    chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".las":
            chunks.extend(las_chunks(path))
        elif suffix in {".txt", ".md"}:
            chunks.extend(report_chunks(path))
    return chunks
