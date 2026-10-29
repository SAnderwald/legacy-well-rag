"""Generate a synthetic legacy-well corpus: LAS files, completion reports and labeled eval questions.

All wells, operators and fields are fictional. Curve suites follow the logging era of each well
(electrical logs before 1960, induction + GR through the 1970s, porosity logs after).
Usage: python scripts/make_synthetic_data.py --out data/synthetic --n 15 --seed 7
"""
import argparse
import json
import random
from pathlib import Path

import lasio
import numpy as np

FIELDS = [("Cedar Bayou South", "Chambers"), ("Pecan Gap", "Wharton"), ("Willow Slough", "Brazoria"),
          ("Lone Oak North", "Matagorda"), ("Tres Palacios", "Jackson"), ("Sandy Point", "Fort Bend"),
          ("Big Caney", "Colorado"), ("Mustang Draw", "Lavaca")]
OPERATORS = ["Gulf Bend Oil Co.", "Coastal Prairie Petroleum", "Brazos Delta Exploration",
             "Tidewater Royalty Corp.", "Five Oaks Operating"]
LEASES = ["Prairie Rose Unit", "Harlan Estate", "McCrory", "Blessing Gas Unit", "Ruth Ann Fee",
          "Cotton Gin", "Halsey State", "Morrow Ranch", "Lake Austwell", "Kubena", "Old Mill",
          "Duke Heirs", "Peach Creek", "Sugar Valley", "Indianola"]
FORMATIONS = ["Beaumont", "Lissie", "Miocene", "Anahuac", "Frio", "Vicksburg", "Yegua", "Wilcox"]
PROBLEMS = [
    "Lost circulation at {d:,} ft; regained returns after pumping LCM pill.",
    "Stuck pipe at {d:,} ft while tripping; worked free after 6 hours with jars.",
    "Gas kick at {d:,} ft; raised mud weight 0.4 ppg and circulated out.",
    "Washout of hole below {d:,} ft noted on caliper; sidewall cores not attempted.",
]

ERAS = {
    "electrical": [("SP", "MV", "Spontaneous potential"), ("SN", "OHMM", "Short normal resistivity 16 in"),
                   ("LN", "OHMM", "Long normal resistivity 64 in")],
    "induction": [("SP", "MV", "Spontaneous potential"), ("GR", "GAPI", "Gamma ray"),
                  ("ILD", "OHMM", "Deep induction resistivity")],
    "porosity": [("GR", "GAPI", "Gamma ray"), ("CALI", "IN", "Caliper"), ("ILD", "OHMM", "Deep induction resistivity"),
                 ("NPHI", "V/V", "Neutron porosity"), ("RHOB", "G/C3", "Bulk density")],
}


def era_for(year: int) -> str:
    return "electrical" if year < 1960 else "induction" if year < 1976 else "porosity"


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(np.pad(x, n, mode="edge"), np.ones(n) / n, mode="same")[n:-n]


def lithology(depth: np.ndarray, w: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Shale volume (0 clean sand, 1 shale) built from random beds, plus a hydrocarbon flag at the perforated sand."""
    vsh, pay = np.ones_like(depth), np.zeros_like(depth, dtype=bool)
    d = depth[0]
    while d < depth[-1]:
        thick = rng.uniform(6, 60)
        sand = rng.random() < 0.4
        sel = (depth >= d) & (depth < d + thick)
        vsh[sel] = rng.uniform(0.05, 0.3) if sand else rng.uniform(0.65, 1.0)
        d += thick
    p_top, p_base = w["perf"]
    sel = (depth >= p_top - 12) & (depth <= p_base + 12)
    vsh[sel], pay[sel] = rng.uniform(0.05, 0.15), True
    return np.clip(smooth(vsh, 7) + rng.normal(0, 0.03, len(depth)), 0, 1), smooth(pay.astype(float), 5) > 0.5


def curves_for(mns: list[str], depth: np.ndarray, w: dict, rng: np.random.Generator) -> dict:
    vsh, pay = lithology(depth, w, rng)
    n, sand = len(depth), 1 - vsh
    rsand = np.where(pay, 22.0, 0.7)
    rt = np.exp(vsh * np.log(1.6) + sand * np.log(rsand)) * np.exp(rng.normal(0, 0.06, n))
    out = {
        "GR": 18 + 112 * vsh + rng.normal(0, 3, n),
        "SP": -65 * sand + 6 * np.sin(depth / 900) + rng.normal(0, 1.5, n),
        "ILD": rt,
        "LN": rt * 0.9,
        "SN": np.exp(vsh * np.log(1.5) + sand * np.log(np.where(pay, 6.0, 0.9))) * np.exp(rng.normal(0, 0.05, n)),
        "NPHI": 0.36 * vsh + 0.23 * sand - 0.09 * pay * sand + rng.normal(0, 0.008, n),
        "RHOB": 2.50 * vsh + 2.21 * sand - 0.06 * pay * sand + rng.normal(0, 0.012, n),
        "CALI": 8.75 + 0.9 * np.clip(vsh - 0.6, 0, None) * rng.uniform(0.5, 1.5, n),
    }
    return {mn: np.round(out[mn], 3) for mn in mns}


def make_well(i: int, rng: np.random.Generator, pyr: random.Random) -> dict:
    field, county = pyr.choice(FIELDS)
    year = pyr.randint(1947, 1988)
    td = pyr.randrange(6200, 12400, 10)
    top = pyr.randrange(1500, 3500, 50)
    tops, d = {}, top
    for f in FORMATIONS[: pyr.randint(5, 8)]:
        if d >= td - 300:
            break
        tops[f] = d
        d += pyr.randrange(500, 1600, 10)
    pay_form = pyr.choice(list(tops)[3:]) if len(tops) > 3 else list(tops)[-1]
    pay_top = min(tops[pay_form] + pyr.randrange(40, 300, 2), td - 200)
    status = pyr.choice(["Producing", "Plugged and abandoned", "Shut in"])
    return {
        "well": f"{LEASES[i % len(LEASES)].upper()} {pyr.randint(1, 9)}",
        "uwi": f"42-{pyr.randint(100, 499):03d}-{pyr.randint(10000, 39999):05d}",
        "operator": pyr.choice(OPERATORS), "field": field, "county": county, "year": year,
        "spud": f"{pyr.randint(1, 12):02d}/{pyr.randint(1, 28):02d}/{year}",
        "td": td, "kb": pyr.randint(18, 160), "log_top": top, "era": era_for(year),
        "tops": tops, "pay_form": pay_form, "perf": (pay_top, pay_top + pyr.randrange(8, 40, 2)),
        "ip_bopd": pyr.randint(0, 420), "ip_mcfd": pyr.randint(50, 6200), "choke": pyr.choice([8, 10, 12, 16, 20]),
        "mud_wt": round(pyr.uniform(9.2, 15.8), 1), "bht": pyr.randint(150, 290),
        "problem": pyr.choice(PROBLEMS).format(d=pyr.randrange(top + 1000, td - 300, 10)), "status": status,
    }


def write_las(w: dict, path: Path, rng: np.random.Generator) -> None:
    las = lasio.LASFile()
    step = 0.5
    depth = np.arange(w["log_top"], w["td"] + step, step)
    las.well["STRT"].value, las.well["STOP"].value, las.well["STEP"].value = depth[0], depth[-1], step
    las.well["NULL"].value = -999.25
    for mn, val, descr in [("COMP", w["operator"], "Company"), ("WELL", w["well"], "Well"),
                           ("FLD", w["field"], "Field"), ("CNTY", w["county"], "County"),
                           ("STAT", "Texas", "State"), ("UWI", w["uwi"], "Unique well identifier"),
                           ("DATE", w["spud"], "Spud date"), ("SRVC", "Legacy digitized log", "Service company")]:
        las.well[mn] = lasio.HeaderItem(mn, value=val, descr=descr)
    las.params["EKB"] = lasio.HeaderItem("EKB", unit="FT", value=w["kb"], descr="Kelly bushing elevation")
    las.params["MDWT"] = lasio.HeaderItem("MDWT", unit="PPG", value=w["mud_wt"], descr="Mud weight")
    las.params["BHT"] = lasio.HeaderItem("BHT", unit="DEGF", value=w["bht"], descr="Bottom hole temperature")
    las.params["TD"] = lasio.HeaderItem("TD", unit="FT", value=w["td"], descr="Total depth driller")
    las.append_curve("DEPT", depth, unit="FT", descr="Measured depth")
    suite = ERAS[w["era"]]
    data = curves_for([mn for mn, _, _ in suite], depth, w, rng)
    for mn, unit, descr in suite:
        las.append_curve(mn, data[mn], unit=unit, descr=descr)
    las.other = f"Digitized from paper log dated {w['year']}. Logging suite: {w['era']}."
    with open(path, "w") as f:
        las.write(f, version=2.0)


def write_report(w: dict, path: Path) -> None:
    tops = "\n".join(f"  {f}: {d:,} ft" for f, d in w["tops"].items())
    text = f"""WELL COMPLETION REPORT
WELL: {w['well']}
OPERATOR: {w['operator']}
FIELD: {w['field']}, {w['county']} County, Texas
API/UWI: {w['uwi']}

DRILLING SUMMARY
Spudded {w['spud']}. Drilled to a total depth of {w['td']:,} ft. Maximum mud weight {w['mud_wt']} ppg.
Bottom hole temperature {w['bht']} deg F. {w['problem']}

FORMATION TOPS (log picks)
{tops}

COMPLETION
Perforated the {w['pay_form']} from {w['perf'][0]:,} to {w['perf'][1]:,} ft.
Initial potential: {w['ip_bopd']} BOPD and {w['ip_mcfd']:,} MCFD on a {w['choke']}/64 in choke.

STATUS
Current status: {w['status']}.
"""
    path.write_text(text)


def questions(w: dict, las_name: str, rpt_name: str) -> list[dict]:
    n = w["well"].title()
    return [
        {"question": f"What logging curves were run on the {n}?", "source": las_name, "kinds": ["las_curves"], "answer": ERAS[w["era"]][0][0]},
        {"question": f"What was the initial potential of the {n}?", "source": rpt_name, "kinds": ["report"], "answer": f"{w['ip_mcfd']:,} MCFD"},
        {"question": f"What drilling problems were reported on the {n}?", "source": rpt_name, "kinds": ["report"], "answer": w["problem"][:20]},
        {"question": f"What is the kelly bushing elevation of the {n}?", "source": las_name, "kinds": ["las_params"], "answer": str(w["kb"])},
        {"question": f"Which interval was perforated in the {n}?", "source": rpt_name, "kinds": ["report"], "answer": f"{w['perf'][0]:,}"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "las").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    rng, pyr = np.random.default_rng(args.seed), random.Random(args.seed)
    qs = []
    for i in range(args.n):
        w = make_well(i, rng, pyr)
        stem = w["well"].lower().replace(" ", "_")
        write_las(w, out / "las" / f"{stem}.las", rng)
        write_report(w, out / "reports" / f"{stem}_completion.txt")
        qs += questions(w, f"{stem}.las", f"{stem}_completion.txt")
    with open(out / "questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")


if __name__ == "__main__":
    main()
