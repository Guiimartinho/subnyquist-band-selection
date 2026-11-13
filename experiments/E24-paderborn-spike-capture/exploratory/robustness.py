"""Exploratory (adversarial review of 2026-11-12, outside the pre-registration): do the tests hold?

The pre-registration uses a bootstrap by bearing with B = 4000. With 11 to 12 clusters per race,
that bootstrap can be anticonservative (Cameron, Gelbach and Miller, 2008). And H1 passed by a
single resample. This script redoes H1, H2 and H3, under the nested ruler, with two tests:

- **Exact sign-flip test, stratified by race.** Under H0, the per-bearing difference is symmetric
  around zero. The statistic is that of the analysis (mean, over the races, of the mean per-bearing
  difference), and the one-sided p is the fraction of the 2^n sign flips with statistic ≥ the
  observed one. Bearings with zero difference do not change the statistic.
- **The pre-registered bootstrap with a large B** (``cluster_test`` in ``run.py``), to remove
  the Monte Carlo error of B = 4000.

Holm over the 4 budgets, as in the pre-registration.

    uv run python experiments/E24-paderborn-spike-capture/exploratory/robustness.py [--boot N]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e24run", EXP / "run.py")
e24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e24)

TESTS = {
    "H1": ("results", "kurtograma_sem_picos", "kurtograma"),
    "H2": ("results", "infograma_ses", "kurtograma"),
    "H3": ("results-post-despike", "kurtograma_sem_picos_decimado", "kurtograma"),
}
FACTORS = (4, 8, 16, 32)


def load(name: str) -> list[dict]:
    with (EXP / name / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["factor"] = int(r["factor"])
    return rows


def sign_flip_p(a: dict, b: dict) -> tuple[float, int, int]:
    """Exact one-sided p; also returns how many bearings differ and how many are in favour."""
    keys = sorted(set(a) & set(b))
    races = sorted({k[0] for k in keys})
    d = {r: np.array([a[k] - b[k] for k in keys if k[0] == r]) for r in races}
    obs = float(np.mean([v.mean() for v in d.values()]))
    nz = [(r, i) for r in races for i in np.flatnonzero(d[r] != 0)]
    if not nz:
        return 1.0, 0, 0
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(nz)):
        dd = {r: v.copy() for r, v in d.items()}
        for s, (r, i) in zip(signs, nz, strict=True):
            dd[r][i] = s * abs(dd[r][i])
        count += float(np.mean([v.mean() for v in dd.values()])) >= obs - 1e-12
    up = sum(1 for r, i in nz if d[r][i] > 0)
    return count / 2 ** len(nz), len(nz), up


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200_000)
    args = ap.parse_args()
    cache = {name: load(name) for name in {v[0] for v in TESTS.values()}}
    out = []
    for hyp, (name, arm, base) in TESTS.items():
        rows = cache[name]
        res = []
        for f in FACTORS:
            a = e24.per_bearing(rows, arm, f, "nested")
            b = e24.per_bearing(rows, base, f, "nested")
            p_sf, n_nz, n_up = sign_flip_p(a, b)
            bt = e24.cluster_test(a, b, args.boot, 20261112 + 1000 * f)
            res.append(
                {
                    "hypothesis": hyp,
                    "factor": f,
                    "diff_points": bt["diff_points"],
                    "bearings_differing": n_nz,
                    "bearings_up": n_up,
                    "p_signflip": p_sf,
                    "p_boot": bt["p_one_sided"],
                }
            )
        for key in ("p_signflip", "p_boot"):
            adj = e24.holm([r[key] for r in res])
            for r, pa in zip(res, adj, strict=True):
                r[f"{key}_holm"] = round(pa, 4)
        out += res
    for r in out:
        print(
            f"{r['hypothesis']} {r['factor']:>2}x: {r['diff_points']:+6.2f} pts | "
            f"bearings differing {r['bearings_differing']} ({r['bearings_up']} in favour) | "
            f"sign flip p={r['p_signflip']:.4f} Holm {r['p_signflip_holm']:.4f} | "
            f"bootstrap B={args.boot} p={r['p_boot']:.5f} Holm {r['p_boot_holm']:.4f}"
        )
    dst = EXP / "results" / "exploratory" / "robustness.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"boot": args.boot, "tests": out}, indent=2), encoding="utf-8")
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
