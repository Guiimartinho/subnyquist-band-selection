# ruff: noqa
# Written by the adversarial review (reviewer-2) of 2026-11-13; see the E26 README.
# Usage: uv run --no-project --quiet --with numpy python <script> <this directory>
"""Loads metrics.csv/calib.csv of E26 and reproduces the rules of analyze_paderborn.py."""

import csv, itertools, math
from collections import defaultdict
import numpy as np

from pathlib import Path

RES = str(Path(__file__).resolve().parents[2] / "results-paderborn")
TOL = 0.025
BPFO, BPFI = 3.0543, 4.9457  # recomputed below from the geometry
GR = 6.75 / 28.55
BPFO = 4 * (1 - GR)
BPFI = 4 * (1 + GR)
ORD = {"BPFO": BPFO, "BPFI": BPFI}
ARMS = {
    "kurtograma": ("bruto", "cego"),
    "mediana_kurtograma": ("mediana3", "cego"),
    "guiado": ("bruto", "guiado"),
    "mediana_guiado": ("mediana3", "guiado"),
}
ART = {
    "KA01",
    "KA03",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KI01",
    "KI03",
    "KI05",
    "KI07",
    "KI08",
}

with open(f"{RES}/metrics.csv", encoding="utf-8") as f:
    ROWS = [r for r in csv.DictReader(f) if not r["skipped"]]
with open(f"{RES}/calib.csv", encoding="utf-8") as f:
    CAL = list(csv.DictReader(f))
VALID = [c for c in CAL if c["ok"] == "1"]
ALLMED = {
    r: float(np.median([float(c["center"]) for c in VALID if c["race"] == r])) for r in ("OR", "IR")
}


def truth(b, rb, rc):
    if rc == rb:
        return float(
            np.median([float(c["center"]) for c in VALID if c["race"] == rc and c["bearing"] != b])
        )
    return ALLMED[rc]


def sc(r, c):
    return float(r[f"score_{c}"]) if r[f"score_{c}"] != "" else -np.inf


FACTORS = sorted({int(r["factor"]) for r in ROWS})
HEALTHY = sorted({r["bearing"] for r in ROWS if r["race"] == "H"})


def thresholds(q=95.0):
    tau = {}
    for f in FACTORS:
        for clean in ("bruto", "mediana3"):
            hs = defaultdict(list)
            for r in ROWS:
                if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == clean:
                    hs[r["bearing"]].append(max(sc(r, "BPFO"), sc(r, "BPFI")))
            tau[(f, clean, None)] = float(np.percentile([s for v in hs.values() for s in v], q))
            for h in HEALTHY:
                rest = [s for b, v in hs.items() if b != h for s in v]
                tau[(f, clean, h)] = float(np.percentile(rest, q))
    return tau


def pred_guided(r, tau):
    f, clean, race, b = int(r["factor"]), r["clean"], r["race"], r["bearing"]
    th = tau[(f, clean, b if race == "H" else None)]
    top = max(("BPFO", "BPFI"), key=lambda c: sc(r, c))
    return ("OR" if top == "BPFO" else "IR") if sc(r, top) >= th else "none"


def pred_blind(r, tol=TOL, rule="nested"):
    b, race = r["bearing"], r["race"]
    est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
    pred = "none"
    for rc in ("OR", "IR"):
        if rule == "nested":
            t = truth(b, race, rc)
            if np.isfinite(est) and abs(est - t) <= tol:
                pred = rc
        elif rule == "nominal":
            t = BPFO if rc == "OR" else BPFI
            if np.isfinite(est) and abs(est - t) <= tol:
                pred = rc
    return pred


def acc_table(predfun):
    """(arm,f) -> {(race,b): mean ok}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in ROWS:
        f, clean, race, b = int(r["factor"]), r["clean"], r["race"], r["bearing"]
        for arm, (cl, sel) in ARMS.items():
            if cl != clean:
                continue
            p = predfun(r, sel)
            tgt = "none" if race == "H" else race
            acc[(arm, f)][(race, b)].append(int(p == tgt))
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in acc.items()}


def balanced(pb_, races=("OR", "IR")):
    by = defaultdict(list)
    for (race, _b), v in pb_.items():
        if race in races:
            by[race].append(v)
    return 100 * float(np.mean([np.mean(v) for v in by.values()])) if by else math.nan


def signflip(a, b, two_sided=False, mc=200000, seed=1, exact_max=18):
    keys = sorted(set(a) & set(b))
    races = sorted({k[0] for k in keys})
    d = {r: np.array([a[k] - b[k] for k in keys if k[0] == r]) for r in races}
    obs = float(np.mean([v.mean() for v in d.values()]))
    nz = [(r, i) for r in races for i in np.flatnonzero(d[r] != 0)]
    if not nz:
        return obs, 1.0, 0, 0
    w = np.array([abs(d[r][i]) / (d[r].size * len(races)) for r, i in nz])
    up = sum(1 for r, i in nz if d[r][i] > 0)
    if len(nz) <= exact_max:
        s = np.array(list(itertools.product((1.0, -1.0), repeat=len(nz))))
        t = s @ w
        p = (
            float(np.mean(np.abs(t) >= abs(obs) - 1e-12))
            if two_sided
            else float(np.mean(t >= obs - 1e-12))
        )
    else:
        s = np.random.default_rng(seed).choice((1.0, -1.0), size=(mc, len(nz)))
        t = s @ w
        cnt = (
            np.count_nonzero(np.abs(t) >= abs(obs) - 1e-12)
            if two_sided
            else np.count_nonzero(t >= obs - 1e-12)
        )
        p = float((1 + cnt) / (1 + mc))
    return 100 * obs, p, len(nz), up


def ci_signflip(dd, mc=20000, seed=2, alpha=0.05):
    """CI by inversion of the stratified sign-flip test (dd: {(race,b): d})."""
    races = sorted({k[0] for k in dd})
    rng = np.random.default_rng(seed)
    keys = sorted(dd)

    def p_at(delta):
        d = {r: np.array([dd[k] - delta for k in keys if k[0] == r]) for r in races}
        obs = float(np.mean([v.mean() for v in d.values()]))
        vals, wts = [], []
        for r in races:
            for x in d[r]:
                vals.append(abs(x))
                wts.append(1.0 / (d[r].size * len(races)))
        w = np.array(vals) * np.array(wts)
        s = rng.choice((1.0, -1.0), size=(mc, w.size))
        t = s @ w
        return float((1 + np.count_nonzero(np.abs(t) >= abs(obs) - 1e-12)) / (1 + mc))

    grid = np.linspace(-0.6, 0.8, 281)
    ok = [g for g in grid if p_at(g) > alpha]
    return (100 * min(ok), 100 * max(ok)) if ok else (math.nan, math.nan)
