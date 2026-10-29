"""Numeric well-log access: curve arrays, formation tops, perforations, interval statistics and log plots."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import lasio
import numpy as np

TOPS_BLOCK = re.compile(r"FORMATION TOPS[^\n]*\n(.*?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
TOP_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z .'/-]*?)\s*:\s*([\d,]+(?:\.\d+)?)\s*ft\b", re.MULTILINE)
PERF = re.compile(r"Perforated (?:the )?(.+?) from ([\d,]+(?:\.\d+)?) to ([\d,]+(?:\.\d+)?) ft", re.IGNORECASE)
WELL_LINE = re.compile(r"^\s*WELL(?: NAME)?\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

GR_FAMILY = ["GR", "SGR", "CGR"]
SP_FAMILY = ["SP"]
RES_FAMILY = ["ILD", "LLD", "RT", "ILM", "LLS", "LN", "SN", "MSFL"]
POR_FAMILY = ["NPHI", "DPHI", "RHOB", "PEF", "DT"]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class Well:
    name: str
    las_path: Path
    depth: np.ndarray
    curves: dict  # mnemonic -> {"values": ndarray, "unit": str, "descr": str}
    depth_unit: str = "ft"
    tops: dict = field(default_factory=dict)  # formation -> top depth
    perfs: list = field(default_factory=list)  # (top, base, formation)
    header: dict = field(default_factory=dict)  # ~Well items, e.g. COMP, FLD, CNTY, DATE, UWI
    reports: list = field(default_factory=list)  # report files linked to this well

    def formation_interval(self, formation: str) -> tuple[float, float]:
        names = {k.lower(): k for k in self.tops}
        if formation.lower() not in names:
            raise KeyError(f"No top for '{formation}' in {self.name}. Known tops: {', '.join(self.tops) or 'none'}")
        f = names[formation.lower()]
        top = self.tops[f]
        deeper = sorted(d for d in self.tops.values() if d > top)
        return top, (deeper[0] if deeper else float(self.depth.max()))


class LogStore:
    def __init__(self, wells: dict[str, Well], data_dir: Path | None = None, errors: list | None = None):
        self.wells, self.data_dir, self.errors = wells, data_dir, errors or []

    @classmethod
    def load(cls, data_dir: str | Path) -> "LogStore":
        data_dir = Path(data_dir).expanduser()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {data_dir}")
        wells: dict[str, Well] = {}
        errors = []  # (file, problem) for files that could not be read or were skipped
        for path in sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".las"):
            try:
                las = lasio.read(str(path), ignore_header_errors=True)
            except Exception as e:  # malformed real-world files should not stop the rest loading
                errors.append((str(path.relative_to(data_dir)), f"could not read: {e}"))
                continue
            items = {i.mnemonic.upper(): str(i.value).strip() for i in las.well}
            name = items.get("WELL") or path.stem
            if name.upper() in wells:
                errors.append((str(path.relative_to(data_dir)),
                               f"duplicate well name {name}; already loaded from {wells[name.upper()].las_path.name}"))
                continue
            curves = {
                c.mnemonic.upper(): {"values": np.asarray(c.data, dtype=float), "unit": c.unit, "descr": c.descr.strip()}
                for c in las.curves[1:]
            }
            wells[name.upper()] = Well(name, path, np.asarray(las.index, dtype=float), curves,
                                       las.curves[0].unit or "ft", header=items)
        for path in sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}):
            text = path.read_text(encoding="utf-8", errors="replace")
            m = WELL_LINE.search(text)
            if not m or m.group(1).upper() not in wells:
                continue
            w = wells[m.group(1).upper()]
            w.reports.append(path)
            block = TOPS_BLOCK.search(text)
            if block:
                w.tops.update({f.strip(): _num(d) for f, d in TOP_LINE.findall(block.group(1))})
            w.perfs += [(_num(a), _num(b), f.strip()) for f, a, b in PERF.findall(text)]
        return cls(wells, data_dir, errors)

    def inventory(self) -> list[dict]:
        """One row per LAS file, for the Data tab."""
        rows = []
        for w in sorted(self.wells.values(), key=lambda x: x.name):
            h = w.header
            rows.append({
                "well": w.name,
                "file": str(w.las_path.relative_to(self.data_dir)) if self.data_dir else w.las_path.name,
                "operator": h.get("COMP", ""), "field": h.get("FLD", ""), "county": h.get("CNTY", ""),
                "date": h.get("DATE", ""), "uwi": h.get("UWI", "") or h.get("API", ""),
                "curves": ", ".join(w.curves),
                "depth": f"{w.depth.min():,.0f}-{w.depth.max():,.0f} {w.depth_unit}",
                "tops": len(w.tops), "perfs": len(w.perfs),
                "report": ", ".join(p.name for p in w.reports) or "none",
            })
        return rows

    def names(self) -> list[str]:
        return [w.name for w in self.wells.values()]

    def get(self, well: str) -> Well:
        key = " ".join(well.upper().split())
        if key in self.wells:
            return self.wells[key]
        matches = [w for k, w in self.wells.items() if key in k or k in key]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Unknown well '{well}'. Available wells: {', '.join(self.names())}")

    # ---- analysis -------------------------------------------------------
    def _interval(self, w: Well, top, base, formation):
        if formation:
            top, base = w.formation_interval(formation)
        top = float(w.depth.min()) if top in (None, "") else _num(str(top))
        base = float(w.depth.max()) if base in (None, "") else _num(str(base))
        if base < top:  # accept the interval in either order
            top, base = base, top
        if base == top:
            raise ValueError(f"Top and base are both {top}; give an interval.")
        return top, base

    def curve_stats(self, well: str, curve: str, top=None, base=None, formation: str | None = None) -> dict:
        w = self.get(well)
        mn = curve.upper()
        if mn not in w.curves:
            raise KeyError(f"{w.name} has no {curve} curve. Available curves: {', '.join(w.curves)}")
        top, base = self._interval(w, top, base, formation)
        vals = w.curves[mn]["values"]
        mask = (w.depth >= top) & (w.depth <= base) & np.isfinite(vals)
        if not mask.any():
            raise ValueError(f"No valid {mn} samples between {top} and {base} {w.depth_unit} in {w.name}.")
        v = vals[mask]
        r = lambda x: round(float(x), 3)
        return {
            "well": w.name, "curve": mn, "unit": w.curves[mn]["unit"], "top": top, "base": base,
            "formation": formation, "samples": int(mask.sum()), "mean": r(v.mean()), "min": r(v.min()),
            "max": r(v.max()), "p10": r(np.percentile(v, 10)), "p50": r(np.percentile(v, 50)),
            "p90": r(np.percentile(v, 90)), "depth_of_max": r(w.depth[mask][v.argmax()]),
        }

    def compare_wells(self, curve: str, formation: str | None = None, wells: list[str] | None = None,
                      top=None, base=None) -> list[dict]:
        rows = []
        for name in wells or self.names():
            try:
                s = self.curve_stats(name, curve, top, base, formation)
                rows.append({k: s[k] for k in ("well", "top", "base", "mean", "p10", "p90", "max", "unit")})
            except (KeyError, ValueError) as e:
                rows.append({"well": name, "skipped": str(e)})
        return rows

    def parse_plot_request(self, text: str, default_well: str | None = None, pad: float = 300.0):
        """Read a well name and depth interval from free text, without an LLM.

        Priority: explicit depths ("8,000-8,500 ft") > a named formation > the perforations > the whole log.
        Returns (well_name, top, base, note)."""
        norm = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
        found = [w for w in self.wells.values() if " " + re.sub(r"[^a-z0-9]+", " ", w.name.lower()).strip() + " " in norm]
        if found:
            w = max(found, key=lambda x: len(x.name))
        elif default_well:
            w = self.get(default_well)
        else:
            raise KeyError("No well name found in the question. Pick a well in the sidebar or name one.")

        rest = re.sub(re.escape(w.name), " ", text, flags=re.IGNORECASE)
        depths = [_num(d) for d in re.findall(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d{3,6})(?:\.\d+)?(?![\d.])", rest)]
        depths = [d for d in depths if w.depth.min() - pad <= d <= w.depth.max() + pad]
        if len(depths) >= 2:
            top, base = sorted(depths[:2])
            return w.name, top, base, f"{top:,.0f}-{base:,.0f} ft"
        if len(depths) == 1:
            return w.name, depths[0] - pad, depths[0] + pad, f"{pad:.0f} ft either side of {depths[0]:,.0f} ft"

        for f in sorted(w.tops, key=len, reverse=True):
            if re.search(rf"\b{re.escape(f.lower())}\b", text.lower()):
                top, base = w.formation_interval(f)
                return w.name, top, base, f"the {f} ({top:,.0f}-{base:,.0f} ft)"
        if w.perfs:
            p_top, p_base = min(p[0] for p in w.perfs), max(p[1] for p in w.perfs)
            return w.name, p_top - pad, p_base + pad, f"around the perforations ({p_top:,.0f}-{p_base:,.0f} ft)"
        return w.name, None, None, "the whole log"

    def summary(self) -> list[dict]:
        return [{"well": w.name, "curves": list(w.curves), "depth_range": [float(w.depth.min()), float(w.depth.max())],
                 "formation_tops": w.tops, "perforations": [list(p) for p in w.perfs]} for w in self.wells.values()]

    # ---- plotting -------------------------------------------------------
    def plot_log(self, well: str, top=None, base=None, gr_cutoff: float = 60.0):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.transforms import blended_transform_factory

        w = self.get(well)
        top, base = self._interval(w, top, base, None)
        m = (w.depth >= top) & (w.depth <= base)
        d = w.depth[m]
        pick = lambda fam: [c for c in fam if c in w.curves]
        tracks = [t for t in (("Gamma ray / SP", pick(GR_FAMILY) + pick(SP_FAMILY) + pick(["CALI"])),
                              ("Resistivity", pick(RES_FAMILY)),
                              ("Porosity / density", pick(POR_FAMILY))) if t[1]]

        fig, axes = plt.subplots(1, len(tracks), figsize=(3.0 * len(tracks), 9), sharey=True, squeeze=False)
        axes = axes[0]
        colors = ["#2E7D32", "#1565C0", "#C62828", "#6A1B9A", "#EF6C00"]
        for ax, (title, mns) in zip(axes, tracks):
            ax.set_title(title, fontsize=10, pad=34 if len(mns) > 1 and title != "Resistivity" else 8)
            ax.grid(True, which="both", alpha=0.25)
            if title == "Resistivity":
                ax.set_xscale("log")
                ax.set_xlim(0.2, 200)
                for i, mn in enumerate(mns):
                    ax.plot(w.curves[mn]["values"][m], d, color=colors[i % 5], lw=0.8, label=mn)
                ax.legend(loc="upper right", fontsize=7)
                ax.set_xlabel("ohm-m", fontsize=8)
                continue
            for i, mn in enumerate(mns[:3]):
                a = ax if i == 0 else ax.twiny()
                if i == 2:
                    a.spines["top"].set_position(("outward", 28))
                v = w.curves[mn]["values"][m]
                a.plot(v, d, color=colors[i], lw=0.8)
                a.set_xlabel(f"{mn} ({w.curves[mn]['unit']})", color=colors[i], fontsize=8)
                a.tick_params(axis="x", colors=colors[i], labelsize=7)
                if i:
                    a.xaxis.set_label_position("top")
                if mn in GR_FAMILY:
                    a.set_xlim(0, 150)
                    a.fill_betweenx(d, v, gr_cutoff, where=v < gr_cutoff, color="#F9A825", alpha=0.45, lw=0)
                    a.axvline(gr_cutoff, color="#F9A825", lw=0.6, ls="--")
                elif mn == "CALI":
                    a.set_xlim(6, 16)
                elif mn == "NPHI":
                    a.set_xlim(0.45, -0.15)
                elif mn == "RHOB":
                    a.set_xlim(1.95, 2.95)

        for name, depth in w.tops.items():
            if top <= depth <= base:
                for ax in axes:
                    ax.axhline(depth, color="#37474F", lw=0.9)
                axes[0].text(0.02, depth, f" {name}", transform=blended_transform_factory(axes[0].transAxes, axes[0].transData),
                             va="bottom", fontsize=8, color="#37474F", fontweight="bold")
        for p_top, p_base, _ in w.perfs:
            if p_base >= top and p_top <= base:
                for ax in axes:
                    ax.axhspan(p_top, p_base, color="#D32F2F", alpha=0.25, lw=0)
                axes[-1].text(0.98, p_top, "perfs", transform=blended_transform_factory(axes[-1].transAxes, axes[-1].transData),
                              ha="right", va="bottom", fontsize=7, color="#D32F2F")

        axes[0].set_ylim(base, top)
        axes[0].set_ylabel(f"Depth ({w.depth_unit})")
        fig.suptitle(f"{w.name}   {top:,.0f}-{base:,.0f} {w.depth_unit}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig
