"""Pre-registered analysis of E26 on Paderborn (README, "Confirmation on Paderborn").

- **Arms** (cleaning × selector):
  - ``kurtogram`` (raw, blind) and ``median_kurtogram`` (median, blind);
  - ``guided`` (raw, guided) and ``median_guided`` (median, guided).
- **Identification.**
  - **Blind:** it is outer race if the estimated order lies ≤ ``tolerance_order`` from the true
    BPFO, inner race if it lies that close to the BPFI, and "none" otherwise.
    - The true order of the race itself is the nested ruler of E24: the median of the valid
      calibrations of the **other** bearings of the same race.
    - For the other race and for the healthy bearings, it is the median of all the valid
      calibrations of that race.
  - **Guided:** it is the fault with the highest score, if that score exceeds the threshold;
    otherwise, "none".
    - The threshold is the 95th percentile of the healthy scores, per factor and cleaning.
    - To evaluate a healthy bearing, the threshold comes from the other 5, leaving one healthy
      bearing out.
- **Per bearing.**
  - On the damaged bearings, it is the fraction of segments with the right fault. The balanced
    accuracy is the mean, over the races, of the mean over the bearings.
  - On the healthy bearings, it is the fraction with "none" (specificity).
- **Tests.**
  - H1: ``median_guided`` against ``kurtogram``. H2: ``median_guided`` against
    ``median_kurtogram``.
  - They are run at the decisive factors, by a sign-flip test stratified by race on the
    per-bearing differences: exact up to 2^``signflip_exact_max``, otherwise Monte Carlo.
  - Holm over the factors.
  - Passes if the Holm p is below α, the difference is positive and the specificity of
    ``median_guided`` at that factor is at least ``min_specificity``.

    uv run python experiments/E26-fault-order-guided-selector/analyze_paderborn.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parent
ARMS = {
    "kurtogram": ("raw", "blind"),
    "median_kurtogram": ("median3", "blind"),
    "guided": ("raw", "guided"),
    "median_guided": ("median3", "guided"),
}
RACE_FAULT = {"OR": "BPFO", "IR": "BPFI"}
# The labels as the runs wrote them, before they were renamed to English (tools/relabel_map.py).
# The analyses sort their output by these, so that rerunning them rewrites the versioned
# analysis.json files byte for byte.
LEGACY_LABELS = {
    "raw": "bruto",
    "median3": "mediana3",
    "blind": "cego",
    "kurtogram": "kurtograma",
    "median_kurtogram": "mediana_kurtograma",
    "guided": "guiado",
    "median_guided": "mediana_guiado",
    "kurtogram_rule": "kurtograma_regra",
    "ses_infogram_rule": "infograma_ses_regra",
    "wideband_rule": "banda_larga_regra",
    "random_band_rule": "sorteada_regra",
    "pure": "pura",
    "fleet_zone": "zona_frota",
    "kurtogram_zone": "zona_kurtograma",
    **{f"zone{k}": f"zona{k}" for k in range(1, 8)},
}


def legacy_label(label: str) -> str:
    """A label as the runs wrote it, part by part for a composite ``a|b``."""
    return "|".join(LEGACY_LABELS.get(p, p) for p in label.split("|"))


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def signflip(a: dict, b: dict, exact_max: int, mc: int, seed: int) -> dict:
    keys = sorted(set(a) & set(b))
    races = sorted({k[0] for k in keys})
    d = {r: np.array([a[k] - b[k] for k in keys if k[0] == r]) for r in races}
    obs = float(np.mean([v.mean() for v in d.values()]))
    nz = [(r, i) for r in races for i in np.flatnonzero(d[r] != 0)]
    up = sum(1 for r, i in nz if d[r][i] > 0)
    if not nz:
        return {"diff_points": 0.0, "bearings_differing": 0, "bearings_up": 0, "p_one_sided": 1.0}
    w = np.array([abs(d[r][i]) / (d[r].size * len(races)) for r, i in nz])
    if len(nz) <= exact_max:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(nz))))
        p = float(np.mean(signs @ w >= obs - 1e-12))
    else:
        signs = np.random.default_rng(seed).choice((1.0, -1.0), size=(mc, len(nz)))
        p = float((1 + np.count_nonzero(signs @ w >= obs - 1e-12)) / (1 + mc))
    return {
        "diff_points": round(100 * obs, 2),
        "bearings_differing": len(nz),
        "bearings_up": up,
        "p_one_sided": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    an, tol = cfg["analysis"], float(cfg["tolerance_order"])
    res = EXP / (args.results or cfg["output"]["results_dir"])
    with (res / "metrics.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["skipped"]]
    with (res / "calib.csv").open(encoding="utf-8") as f:
        calib = list(csv.DictReader(f))
    valid = [c for c in calib if c["ok"] == "1"]
    all_med = {
        race: float(np.median([float(c["center"]) for c in valid if c["race"] == race]))
        for race in ("OR", "IR")
    }

    def truth(bearing: str, race_of_bearing: str, race: str) -> float:
        if race == race_of_bearing:  # nested ruler: without the bearing itself
            peers = [
                float(c["center"]) for c in valid if c["race"] == race and c["bearing"] != bearing
            ]
            return float(np.median(peers))
        return all_med[race]

    def score(r, c):
        return float(r[f"score_{c}"]) if r[f"score_{c}"] != "" else -np.inf

    factors = sorted({int(r["factor"]) for r in rows})
    healthy = sorted({r["bearing"] for r in rows if r["race"] == "H"})
    tau: dict = {}
    for f in factors:
        for clean in ("raw", "median3"):
            hs = defaultdict(list)
            for r in rows:
                if r["race"] == "H" and int(r["factor"]) == f and r["clean"] == clean:
                    hs[r["bearing"]].append(max(score(r, "BPFO"), score(r, "BPFI")))
            q = 100 * (1 - float(cfg["guided"]["false_alarm"]))
            tau[(f, clean, None)] = float(np.percentile([s for v in hs.values() for s in v], q))
            for h in healthy:
                rest = [s for b, v in hs.items() if b != h for s in v]
                tau[(f, clean, h)] = float(np.percentile(rest, q))

    acc: dict = defaultdict(lambda: defaultdict(list))  # (arm, f) -> (race, bearing) -> [ok]
    for r in rows:
        f, clean, race, b = int(r["factor"]), r["clean"], r["race"], r["bearing"]
        for arm, (cl, sel) in ARMS.items():
            if cl != clean:
                continue
            if sel == "blind":
                est = float(r["blind_order"]) if r["blind_order"] != "" else np.nan
                pred = "none"
                for rc in ("OR", "IR"):
                    if np.isfinite(est) and abs(est - truth(b, race, rc)) <= tol:
                        pred = rc
            else:
                th = tau[(f, clean, b if race == "H" else None)]
                top = max(("BPFO", "BPFI"), key=lambda c: score(r, c))
                pred = ("OR" if top == "BPFO" else "IR") if score(r, top) >= th else "none"
            target = "none" if race == "H" else race
            acc[(arm, f)][(race, b)].append(int(pred == target))

    def per_bearing(arm, f, races):
        return {k: float(np.mean(v)) for k, v in acc[(arm, f)].items() if k[0] in races}

    def balanced(pb_):
        by = defaultdict(list)
        for (race, _b), v in pb_.items():
            by[race].append(v)
        return {race: round(100 * float(np.mean(v)), 1) for race, v in sorted(by.items())}

    summary = {}
    for arm in ARMS:
        for f in factors:
            fault = balanced(per_bearing(arm, f, ("OR", "IR")))
            spec = balanced(per_bearing(arm, f, ("H",))).get("H")
            summary[f"{arm}|{f}x"] = {
                **fault,
                "balanced": round(float(np.mean(list(fault.values()))), 1),
                "specificity": spec,
            }
    tests = []
    for name, (arm, base) in {
        "H1": ("median_guided", "kurtogram"),
        "H2": ("median_guided", "median_kurtogram"),
    }.items():
        cell = []
        for f in (int(x) for x in an["decisive_factors"] if int(x) in factors):  # only those run
            t = signflip(
                per_bearing(arm, f, ("OR", "IR")),
                per_bearing(base, f, ("OR", "IR")),
                int(an["signflip_exact_max"]),
                int(an["signflip_mc"]),
                int(an["seed"]) + f,
            )
            cell.append(
                {
                    "hypothesis": name,
                    "arm": arm,
                    "baseline": base,
                    "factor": f,
                    **t,
                    "specificity_arm": summary[f"{arm}|{f}x"]["specificity"],
                }
            )
        for t, pa in zip(cell, holm([t["p_one_sided"] for t in cell]), strict=True):
            t["p_holm"] = round(pa, 4)
            t["passes"] = int(
                pa < float(an["alpha"])
                and t["diff_points"] > 0
                and (t["specificity_arm"] or 0) >= 100 * float(an["min_specificity"])
            )
        tests += cell
    verdict = {
        h: [t["factor"] for t in tests if t["hypothesis"] == h and t["passes"]]
        for h in ("H1", "H2")
    }
    rep = {
        "thresholds": {
            f"{f}x|{c}": round(tau[(f, c, None)], 3) for f in factors for c in ("raw", "median3")
        },
        "nested_all_median": all_med,
        "summary": summary,
        "tests": tests,
        "verdict_factors_passed": verdict,
    }
    (res / "analysis.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), "utf-8")
    for k, v in summary.items():
        print(k, v)
    for t in tests:
        print(t)
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
