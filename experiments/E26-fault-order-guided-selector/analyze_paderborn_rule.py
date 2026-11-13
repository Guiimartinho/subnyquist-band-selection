"""Pre-registered analysis of E26b (README, "E26b: the rule stays fixed, the band changes").

- **Arms with the guided rule fixed.** The band changes between the arms:
  - ``guiado``: the maximum of the score over the 23 bands, read from
    ``results-paderborn/metrics.csv``;
  - ``kurtograma_regra``, ``infograma_ses_regra``, ``banda_larga_regra`` and ``sorteada_regra``:
    the four strategies of ``results-paderborn-regra/metrics.csv``.
  - Each arm exists for both cleanings, ``bruto`` (raw) and ``mediana3`` (median of 3).
- **Identification rule,** the same in all arms: the fault with the highest score, if that score
  exceeds the threshold; otherwise, "none".
  - The threshold is the 95th percentile of the maximum healthy scores, per arm, factor and
    cleaning.
  - To evaluate a healthy bearing, the threshold comes from the other 5.
- **Tests** (per-bearing sign-flip test, stratified by race, as in E26):
  - **H3:** the guided arm against ``kurtograma_regra``, on both cleanings and at the 4 decisive
    factors, with Holm over the 8.
  - **H4:** the guided arm against ``banda_larga_regra``, in the same way.
  - A test passes if the Holm p is below α and the difference is positive.
- **Integrity check.** Where the random band coincides with the band that the guided arm chose for
  a fault, the score of that fault has to be the same in the two files.

    uv run python experiments/E26-fault-order-guided-selector/analyze_paderborn_rule.py \
        --config experiments/E26-fault-order-guided-selector/config/paderborn_rule.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e26an", EXP / "analyze_paderborn.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)
CLEANS = ("bruto", "mediana3")
FAULTS = ("BPFO", "BPFI")


def score(r: dict, c: str) -> float:
    return float(r[f"score_{c}"]) if r[f"score_{c}"] != "" else -np.inf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    a = cfg["analysis_rule"]
    res = EXP / (args.results or cfg["output"]["results_dir"])
    with (EXP / a["guided_results"] / "metrics.csv").open(encoding="utf-8") as f:
        g_rows = [r for r in csv.DictReader(f) if not r["skipped"]]
    with (res / "metrics.csv").open(encoding="utf-8") as f:
        s_rows = list(csv.DictReader(f))

    # one record per (arm, cleaning): (bearing, race, setting, measurement, factor, scores)
    recs: dict = defaultdict(list)
    for r in g_rows:
        recs[("guiado", r["clean"])].append(r)
    for r in s_rows:
        recs[(f"{r['strategy']}_regra", r["clean"])].append(r)
    keyed = {
        arm: {(r["bearing"], r["setting"], r["measurement"], r["factor"]): r for r in rows}
        for arm, rows in recs.items()
    }

    # integrity: random band equal to the guided band of the fault -> same score
    diffs = []
    for clean in CLEANS:
        g = keyed[("guiado", clean)]
        for k, r in keyed[("sorteada_regra", clean)].items():
            if k in g:
                for c in FAULTS:
                    if r["band"] == g[k][f"band_{c}"] and r[f"score_{c}"] and g[k][f"score_{c}"]:
                        diffs.append(abs(float(r[f"score_{c}"]) - float(g[k][f"score_{c}"])))
    integrity = {"matches": len(diffs), "max_abs_diff": max(diffs) if diffs else None}

    factors = sorted({int(r["factor"]) for r in s_rows})
    q = 100 * (1 - float(cfg["guided"]["false_alarm"]))
    acc: dict = defaultdict(lambda: defaultdict(list))
    thresholds = {}
    for (arm, clean), rows in recs.items():
        for f in factors:
            sub = [r for r in rows if int(r["factor"]) == f]
            hs = defaultdict(list)
            for r in sub:
                if r["race"] == "H":
                    hs[r["bearing"]].append(max(score(r, c) for c in FAULTS))
            tau_all = float(np.percentile([s for v in hs.values() for s in v], q))
            thresholds[f"{arm}|{clean}|{f}x"] = round(tau_all, 3)
            tau_lobo = {
                h: float(np.percentile([s for b, v in hs.items() if b != h for s in v], q))
                for h in hs
            }
            for r in sub:
                race, b = r["race"], r["bearing"]
                th = tau_lobo[b] if race == "H" else tau_all
                top = max(FAULTS, key=lambda c: score(r, c))
                pred = ("OR" if top == "BPFO" else "IR") if score(r, top) >= th else "none"
                target = "none" if race == "H" else race
                acc[(arm, clean, f)][(race, b)].append(int(pred == target))

    def per_bearing(arm, clean, f, races):
        return {k: float(np.mean(v)) for k, v in acc[(arm, clean, f)].items() if k[0] in races}

    summary = {}
    for arm, clean in sorted(recs):
        for f in factors:
            fault = an_balanced(per_bearing(arm, clean, f, ("OR", "IR")))
            spec = an_balanced(per_bearing(arm, clean, f, ("H",))).get("H")
            summary[f"{arm}|{clean}|{f}x"] = {
                **fault,
                "balanced": round(float(np.mean(list(fault.values()))), 1),
                "specificity": spec,
            }

    tests = []
    for name, base in (("H3", "kurtograma_regra"), ("H4", "banda_larga_regra")):
        cell = []
        for clean in CLEANS:
            for f in (int(x) for x in a["decisive_factors"] if int(x) in factors):
                t = an.signflip(
                    per_bearing("guiado", clean, f, ("OR", "IR")),
                    per_bearing(base, clean, f, ("OR", "IR")),
                    int(a["signflip_exact_max"]),
                    int(a["signflip_mc"]),
                    int(a["seed"]) + f + (1000 if clean == "mediana3" else 0),
                )
                cell.append(
                    {"hypothesis": name, "baseline": base, "clean": clean, "factor": f, **t}
                )
        for t, pa in zip(cell, an.holm([t["p_one_sided"] for t in cell]), strict=True):
            t["p_holm"] = round(pa, 4)
            t["passes"] = int(pa < float(a["alpha"]) and t["diff_points"] > 0)
        tests += cell
    verdict = {
        h: [f"{t['clean']}|{t['factor']}x" for t in tests if t["hypothesis"] == h and t["passes"]]
        for h in ("H3", "H4")
    }
    rep = {
        "integrity": integrity,
        "thresholds": thresholds,
        "summary": summary,
        "tests": tests,
        "verdict_passed": verdict,
    }
    (res / "analysis.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), "utf-8")
    print("integrity:", integrity)
    for k, v in summary.items():
        print(k, v)
    for t in tests:
        print(t)
    print("verdict:", verdict)
    return 0


def an_balanced(pb_: dict) -> dict:
    by = defaultdict(list)
    for (race, _b), v in pb_.items():
        by[race].append(v)
    return {race: round(100 * float(np.mean(v)), 1) for race, v in sorted(by.items())}


if __name__ == "__main__":
    raise SystemExit(main())
