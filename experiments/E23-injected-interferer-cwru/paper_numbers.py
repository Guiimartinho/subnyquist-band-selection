"""E23 numbers that the paper cites, recomputed from metrics.csv.

Three numbers in the list "What the paper can claim" (README) came from the recomputation by
`reviewer-2` and had no script: the 97 % of losses on the burst rate, the 8.5 points of the
anti-alias and the per-cluster CI of the surrogate ratio. This script recomputes them with the
definitions below, written before running it. If a value differs from the README, the one from
here holds, and the README is corrected.

1. **Losses on the interferer line.** It applies to the uniform arm, at the reference ceiling,
   at factors ≥ 2 and with the impulsive at the highest dose. Among the cases (record, carrier,
   factor) in which the arm hits without injection and misses with it, it is the fraction whose
   estimate falls on the burst rate or on a harmonic up to the 3rd (`interferer_line`).
2. **Cost of the ideal anti-alias without injection.** It is the hit rate of `uniform` at the
   reference ceiling minus that of `aaf_ideal`, both without injection, in points: at 10× and 12×
   (what the README attributes to the reviewer) and at all factors ≥ 2 (what the base run reports).
3. **Surrogate/impulsive ratio at P\*.** It is the pre-registered statistic of H1', from
   `analyze.py`. The CI comes from the bootstrap of the 4 independent fault clusters (`CLUSTERS`),
   instead of the bootstrap by record.
4. **Folded share against the size of the drop.** It is the Spearman correlation, among the
   evaluable predicted-capture cells, at the highest dose, between the mean folded share of the
   cell and the drop of the impulsive (`Data.drop`).
5. **Dose-response of the spikes.** It is the maximum drop of the uniform arm at factors ≥ 2, per
   rate and amplitude, from `results-spikes/metrics.csv`.

    uv run python experiments/E23-injected-interferer-cwru/paper_numbers.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

E = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e23_analyze", E / "analyze.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)

# The 12 BPFO records are one 0.007" fault in 3 positions x 4 loads; 213 is the BPFI (README,
# adversarial review, "Independent units").
CLUSTERS = {
    "centred": range(135, 139),
    "orthogonal": range(148, 152),
    "opposite": range(161, 165),
    "bpfi_213": range(213, 214),
}


def cluster_of(record: str) -> str:
    n = int(record[:3])
    for name, span in CLUSTERS.items():
        if n in span:
            return name
    raise KeyError(record)


def losses_on_line(data, cfg: dict) -> dict:
    db = max(data.powers)
    n_loss = n_line = 0
    for (kind, _fc, d, rec, f, c), r in data.inj.items():
        if kind != "impulsive" or d != db or f < 2 or c != an.ceiling0(cfg, f):
            continue
        if data.base.get((rec, f, c)) == 1 and r["hit"] == 0:
            n_loss += 1
            n_line += int(r["line"] == 1)
    return {
        "dose_db_label": db,
        "losses": n_loss,
        "on_line": n_line,
        "share": round(n_line / n_loss, 4) if n_loss else None,
    }


def aaf_cost(rows: list[dict], cfg: dict, factors=(10, 12)) -> dict:
    uni = [
        r["hit"]
        for r in rows
        if r["arm"] == "uniform"
        and r["kind"] == "none"
        and r["factor"] in factors
        and r["search_ceiling_hz"] == an.ceiling0(cfg, r["factor"])
        and r["hit"] is not None
    ]
    aaf = [
        r["hit"]
        for r in rows
        if r["arm"] == "aaf_ideal"
        and r["kind"] == "none"
        and r["factor"] in factors
        and r["hit"] is not None
    ]
    return {
        "factors": list(factors),
        "n_uniform": len(uni),
        "n_aaf": len(aaf),
        "acc_uniform": round(100 * float(np.mean(uni)), 2),
        "acc_aaf_ideal": round(100 * float(np.mean(aaf)), 2),
        "cost_points": round(100 * (float(np.mean(uni)) - float(np.mean(aaf))), 2),
    }


def surrogate_ratio_cluster_ci(data, cfg: dict) -> dict:
    by: dict[str, list[str]] = defaultdict(list)
    for rec in data.records:
        by[cluster_of(rec)].append(rec)
    names = sorted(by)
    point = data.h1(data.records)["surrogate_ratio_at_p_star"]
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    boots = []
    for _ in range(int(cfg["analysis"]["bootstrap"])):
        sample = [rec for k in rng.choice(names, size=len(names)) for rec in by[k]]
        v = data.h1(sample)["surrogate_ratio_at_p_star"]
        if v is not None and np.isfinite(v):
            boots.append(v)
    return {
        "point": round(point, 4),
        "n_boot_valid": len(boots),
        "ci95_cluster": [round(float(np.percentile(boots, q)), 3) for q in (2.5, 97.5)],
        "clusters": {k: by[k] for k in names},
    }


def folded_share_vs_drop(data) -> dict:
    from scipy.stats import spearmanr

    db = max(data.powers)
    share: dict = defaultdict(list)
    for (kind, fc, d, _rec, f, c), r in data.inj.items():
        if kind == "impulsive" and d == db and r["folded_share"] not in ("", None):
            share[(fc, f, c)].append(float(r["folded_share"]))
    xs, ys = [], []
    for cell in data.cells:
        if data.pred.get(cell) != 1 or not data.evaluable(cell, data.records) or not share[cell]:
            continue
        dr = data.drop("impulsive", cell[0], db, cell[1], cell[2], data.records)
        if dr is not None:
            xs.append(float(np.mean(share[cell])))
            ys.append(dr)
    rho, p = spearmanr(xs, ys)
    return {"n_cells": len(xs), "spearman_rho": round(float(rho), 3), "p": round(float(p), 3)}


def spike_dose_response(path: Path, cfg: dict) -> dict:
    hit: dict = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["hit"] == "" or r["arm"] != "uniform":
            continue
        f = int(r["factor"])
        if float(r["search_ceiling_hz"]) != an.ceiling0(cfg, f):
            continue
        hit[(r["kind"], r["spike_rate_hz"], r["spike_amp_mad"], f)][r["record"]] = int(r["hit"])
    out = {}
    for (kind, rate, amp, f), v in hit.items():
        if kind == "none" or f < 2:
            continue
        base = hit.get(("none", "", "", f), {})
        common = sorted(set(v) & set(base))
        if not common:
            continue
        drop = 100.0 * (np.mean([base[k] for k in common]) - np.mean([v[k] for k in common]))
        key = f"rate={rate}|amp={amp}"
        out[key] = max(out.get(key, -1e9), round(float(drop), 1))
    return dict(sorted(out.items()))


def main() -> int:
    cfg = yaml.safe_load((E / "config" / "base.yaml").read_text(encoding="utf-8"))
    rows = an.load(E / "results" / "metrics.csv")
    data = an.Data(rows, cfg)
    unmapped = [r for r in data.records if r[:3].isdigit() is False]
    assert not unmapped, unmapped
    spikes_cfg = yaml.safe_load((E / "config" / "spikes.yaml").read_text(encoding="utf-8"))
    rep = {
        "losses_on_interferer_line": losses_on_line(data, cfg),
        "aaf_ideal_cost_no_injection": aaf_cost(rows, cfg),
        "aaf_ideal_cost_no_injection_factors_ge2": aaf_cost(
            rows, cfg, tuple(sorted({r["factor"] for r in rows if r["factor"] >= 2}))
        ),
        "surrogate_ratio_at_p_star": surrogate_ratio_cluster_ci(data, cfg),
        "folded_share_vs_drop": folded_share_vs_drop(data),
        "spike_max_drop_points_factor_ge2": spike_dose_response(
            E / "results-spikes" / "metrics.csv", spikes_cfg
        ),
    }
    out = E / "results" / "paper_numbers.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
