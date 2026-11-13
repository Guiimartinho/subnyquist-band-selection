"""E22 - test of hypotheses H1 and H2 from results/metrics.csv.

    uv run python experiments/E22-equal-rate-baselines/analyze.py

H1: random_cs does not beat uniform_ceiling at any budget (refuted if the difference has a
per-record bootstrap CI above zero). H2: tat_rate stays within at most 5 points of the uniform
oracle at every budget. Both are pre-registered in the README.

The oracle chooses, among the grid factors >= budget, the one with the highest accuracy ON THE
TEST ITSELF, per group (inner / outer race). It is an optimistic upper bound; in the bootstrap it
is re-chosen in each resample.

Added after seeing that the outer-race oracle lands on an isolated peak (factor 46 among
45 noisy factors): ``uniform_sweep_cv``, the best uniform rate chosen OUTSIDE the test
(it chooses on segment 1 and evaluates on 2, and vice versa; a tie goes to the higher rate). It is
post hoc and is labelled as such in the README.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parent
N_BOOT = 4000
SEED = 20261111
GAP_POINTS = 5.0
ARMS = ("uniform", "despike", "uniform_ceiling", "tat_rate", "random_cs", "bandpass_lu")
TOL_ORDER = 0.06  # hit tolerance of E22 (config: tolerance_order)
PAIRS = (
    ("tat_rate", "uniform"),
    ("uniform_ceiling", "uniform"),
    ("bandpass_lu", "uniform"),
    ("random_cs", "uniform"),
)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided p for the discordant pairs (b: only the 1st hits; c: only the 2nd)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def ci95(d: list[float]) -> list[float]:
    return [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]


@dataclass
class Cells:
    """Hits of one dataset, by (record, segment, arm, budget) and by factor."""

    hit: dict
    grid: dict
    group_of: dict
    segs: list
    groups: list
    factors: list
    est: dict  # (record, segment, factor) -> order estimate (uniform_grid arm)
    calib: dict  # record -> calibrated order on segment 0 (manifest)
    local: bool = False  # local ruler: the grid hits are already free of leakage

    def nested_truth(self, rec: str, pool: list[str]) -> float:
        """Median of the calibrations of the OTHER records of the group, restricted to ``pool``."""
        peers = [self.calib[q] for q in pool if q != rec and self.group_of[q] == self.group_of[rec]]
        return float(np.median(peers)) if peers else float("nan")

    def grid_acc_nested(self, f: int, recs: list[str], pool: list[str]) -> tuple[int, int]:
        h = n = 0
        for rec in recs:
            t = self.nested_truth(rec, pool)
            for s in self.segs:
                e = self.est.get((rec, s, f))
                if e is None or not np.isfinite(t):
                    continue
                h += int(abs(e - t) < TOL_ORDER)
                n += 1
        return h, n

    def cv_by_record_nested(self, b: int, sample: list[str]) -> tuple[float, int, dict]:
        """Best uniform rate chosen on other records, with NESTED truth.

        3 folds by trial. The truth of the training records uses only training records, and so
        does the truth of the test records: no test label enters the choice. Fixes the leakage of
        the version with global truth (review of 2026-11-11). A tie goes to the higher rate.
        """
        h_tot = n_tot = 0
        chosen: dict = {}
        for g in self.groups:
            sub = [rec for rec in sample if self.group_of[rec] == g]
            for t in sorted({rec.rsplit("-", 1)[-1] for rec in sub}):
                train = [rec for rec in sub if rec.rsplit("-", 1)[-1] != t]
                test = [rec for rec in sub if rec.rsplit("-", 1)[-1] == t]
                best = (-1.0, None)
                for f in self.factors:
                    if f < b:
                        continue
                    h, n = self.grid_acc_nested(f, train, train)
                    if n and h / n > best[0]:
                        best = (h / n, f)
                if best[1] is None:
                    continue
                h, n = self.grid_acc_nested(best[1], test, train)
                chosen.setdefault(g, []).append(best[1])
                h_tot, n_tot = h_tot + h, n_tot + n
        return 100.0 * h_tot / max(n_tot, 1), n_tot, chosen

    def acc_arm(self, arm: str, b: int, sample: list[str]) -> tuple[int, int]:
        vals = [self.hit.get((rec, s, arm, b)) for rec in sample for s in self.segs]
        vals = [v for v in vals if v is not None]
        return sum(vals), len(vals)

    def grid_acc(self, f: int, recs: list[str], segs: list) -> tuple[int, int]:
        vals = [self.grid.get((rec, s, f)) for rec in recs for s in segs]
        vals = [v for v in vals if v is not None]
        return sum(vals), len(vals)

    def oracle(self, b: int, sample: list[str]) -> tuple[float, int, dict]:
        """Accuracy (points) over the groups, each with its highest-accuracy factor on the test."""
        h_tot = n_tot = 0
        chosen = {}
        for g in self.groups:
            sub = [rec for rec in sample if self.group_of[rec] == g]
            best = (-1.0, None, 0, 0)
            for f in self.factors:
                if f < b:
                    continue
                h, n = self.grid_acc(f, sub, self.segs)
                if n and h / n > best[0]:
                    best = (h / n, f, h, n)
            chosen[g] = best[1]
            h_tot, n_tot = h_tot + best[2], n_tot + best[3]
        return 100.0 * h_tot / max(n_tot, 1), n_tot, chosen

    def cv_uniform(self, b: int, sample: list[str]) -> tuple[float, int, dict]:
        """Best uniform rate chosen on one test segment and evaluated on the other (2 folds)."""
        h_tot = n_tot = 0
        chosen: dict = {}
        for g in self.groups:
            sub = [rec for rec in sample if self.group_of[rec] == g]
            for pick, ev in ((self.segs[0], self.segs[1]), (self.segs[1], self.segs[0])):
                best = (-1.0, None)
                for f in self.factors:  # ascending: a tie stays with the higher rate
                    if f < b:
                        continue
                    h, n = self.grid_acc(f, sub, [pick])
                    if n and h / n > best[0]:
                        best = (h / n, f)
                h, n = self.grid_acc(best[1], sub, [ev])
                chosen.setdefault(g, []).append(best[1])
                h_tot, n_tot = h_tot + h, n_tot + n
        return 100.0 * h_tot / max(n_tot, 1), n_tot, chosen

    def cv_by_record(self, b: int, sample: list[str]) -> tuple[float, int, dict]:
        """Best uniform rate chosen on other records: 3 folds by trial (suffix -1/-2/-3).

        This is the kind of choice that E15 will have to face, because E15 splits by record.
        A tie goes to the higher rate. Records repeated by the bootstrap count with their weight.
        """
        h_tot = n_tot = 0
        chosen: dict = {}
        for g in self.groups:
            sub = [rec for rec in sample if self.group_of[rec] == g]
            trials = sorted({rec.rsplit("-", 1)[-1] for rec in sub})
            for t in trials:
                train = [rec for rec in sub if rec.rsplit("-", 1)[-1] != t]
                test = [rec for rec in sub if rec.rsplit("-", 1)[-1] == t]
                best = (-1.0, None)
                for f in self.factors:
                    if f < b:
                        continue
                    h, n = self.grid_acc(f, train, self.segs)
                    if n and h / n > best[0]:
                        best = (h / n, f)
                if best[1] is None:
                    continue
                h, n = self.grid_acc(best[1], test, self.segs)
                chosen.setdefault(g, []).append(best[1])
                h_tot, n_tot = h_tot + h, n_tot + n
        return 100.0 * h_tot / max(n_tot, 1), n_tot, chosen


def load(path: Path, hit_col: str = "hit") -> dict[str, Cells]:
    """``hit_col``: ``hit`` (leave-one-out truth) or ``hit_local`` (the full-rate line of the
    segment itself; review of 2026-11-11). With the local ruler, the choice outside the test has no
    leakage by construction, because the label of a segment depends only on the segment itself."""
    raw: dict[str, dict] = {}
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    calib_all = manifest.get("calibrated_order", {})
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r.get(hit_col, "") == "":
            continue
        d = raw.setdefault(r["dataset"], {"hit": {}, "grid": {}, "group_of": {}, "est": {}})
        d["group_of"][r["record"]] = r["group"]
        b, h = int(r["budget"]), int(r[hit_col])
        if r["arm"] == "uniform_grid":
            d["grid"][(r["record"], r["segment"], int(r["chosen_factor"]))] = h
            if r["estimate"] != "":
                d["est"][(r["record"], r["segment"], int(r["chosen_factor"]))] = float(
                    r["estimate"]
                )
        else:
            d["hit"][(r["record"], r["segment"], r["arm"], b)] = h
    out = {}
    for ds, d in raw.items():
        out[ds] = Cells(
            local=hit_col == "hit_local",
            hit=d["hit"],
            grid=d["grid"],
            group_of=d["group_of"],
            segs=sorted({k[1] for k in d["hit"]}),
            groups=sorted(set(d["group_of"].values())),
            factors=sorted({k[2] for k in d["grid"]}),
            est=d["est"],
            calib={
                k.split(":", 1)[1]: float(v) for k, v in calib_all.items() if k.startswith(ds + ":")
            },
        )
    return out


def analyze(c: Cells, rng: np.random.Generator) -> dict:
    recs = sorted(c.group_of)
    by_g = {g: [rec for rec in recs if c.group_of[rec] == g] for g in c.groups}
    budgets = sorted({k[3] for k in c.hit})
    out: dict = {"records": len(recs), "groups": c.groups, "budgets": {}}
    for b in budgets:
        cell: dict = {}
        for arm in ARMS:
            h, n = c.acc_arm(arm, b, recs)
            cell[arm] = {"hits": h, "n": n, "acc": round(100.0 * h / max(n, 1), 1)}
        o_acc, o_n, o_f = c.oracle(b, recs)
        cell["uniform_oracle"] = {"acc": round(o_acc, 1), "n": o_n, "factor": o_f}
        v_acc, v_n, v_f = c.cv_uniform(b, recs)
        cell["uniform_sweep_cv"] = {"acc": round(v_acc, 1), "n": v_n, "factor": v_f}
        if c.local:
            # local ruler: the label of a segment depends only on itself, so choosing on the grid
            # hits of the training records leaks nothing from the test
            r_acc, r_n, r_f = c.cv_by_record(b, recs)
        else:
            r_acc, r_n, r_f = c.cv_by_record(b, recs)
            # global truth: it leaks; kept only as a reference (review of 2026-11-11)
            cell["uniform_sweep_record_leak"] = {
                "acc": round(r_acc, 1),
                "n": r_n,
                "factor": r_f,
            }
            r_acc, r_n, r_f = c.cv_by_record_nested(b, recs)
        cell["uniform_sweep_record"] = {"acc": round(r_acc, 1), "n": r_n, "factor": r_f}

        d_h1, d_h2, d_cv, d_rec = [], [], [], []
        for _ in range(N_BOOT):
            sample = [rec for g in c.groups for rec in rng.choice(by_g[g], size=len(by_g[g]))]
            hc, nc = c.acc_arm("random_cs", b, sample)
            ht, nt = c.acc_arm("uniform_ceiling", b, sample)
            hq, nq = c.acc_arm("tat_rate", b, sample)
            tat = 100.0 * hq / max(nq, 1)
            d_h1.append(100.0 * (hc / max(nc, 1) - ht / max(nt, 1)))
            d_h2.append(tat - c.oracle(b, sample)[0])
            d_cv.append(tat - c.cv_uniform(b, sample)[0])
            cv = c.cv_by_record if c.local else c.cv_by_record_nested
            d_rec.append(tat - cv(b, sample)[0])
        cell["H1_cs_minus_ceiling"] = {
            "points": round(cell["random_cs"]["acc"] - cell["uniform_ceiling"]["acc"], 1),
            "ci95": ci95(d_h1),
        }
        cell["H2_tat_minus_oracle"] = {
            "points": round(cell["tat_rate"]["acc"] - cell["uniform_oracle"]["acc"], 1),
            "ci95": ci95(d_h2),
        }
        cell["posthoc_tat_minus_record"] = {
            "points": round(cell["tat_rate"]["acc"] - cell["uniform_sweep_record"]["acc"], 1),
            "ci95": ci95(d_rec),
        }
        cell["posthoc_tat_minus_cv"] = {
            "points": round(cell["tat_rate"]["acc"] - cell["uniform_sweep_cv"]["acc"], 1),
            "ci95": ci95(d_cv),
        }
        pairs = {}
        for a, z in PAIRS:
            only_a = only_z = 0
            for rec in recs:
                for s in c.segs:
                    x, y = c.hit.get((rec, s, a, b)), c.hit.get((rec, s, z, b))
                    if x is None or y is None:
                        continue
                    only_a += int(x == 1 and y == 0)
                    only_z += int(x == 0 and y == 1)
            pairs[f"{a}_vs_{z}"] = {
                "only_first": only_a,
                "only_second": only_z,
                "p": mcnemar_exact(only_a, only_z),
            }
        cell["mcnemar"] = pairs
        out["budgets"][str(b)] = cell
    cells = out["budgets"].values()
    out["verdict"] = {
        "H1": "refuted"
        if any(x["H1_cs_minus_ceiling"]["ci95"][0] > 0 for x in cells)
        else "not refuted",
        "H2": "refuted"
        if any(x["H2_tat_minus_oracle"]["points"] < -GAP_POINTS for x in cells)
        else "not refuted",
    }
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", choices=("global", "local"), default="global")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data = load(E / "results" / "metrics.csv", "hit_local" if args.ruler == "local" else "hit")
    report = {
        "n_boot": N_BOOT,
        "seed": SEED,
        # provenance of the analysis (review of 2026-11-11): what it read and with which code
        "analyze_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256((E / "results" / "metrics.csv").read_bytes()).hexdigest(),
        "ruler": args.ruler,
        "datasets": {},
    }
    for ds, c in sorted(data.items()):
        report["datasets"][ds] = analyze(c, rng)
    dst = E / "results" / ("analysis.json" if args.ruler == "global" else "analysis-local.json")
    dst.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for ds, out in report["datasets"].items():
        v = out["verdict"]
        print(f"== {ds} ({out['records']} records) -> H1 {v['H1']}, H2 {v['H2']}")
        for b, x in out["budgets"].items():
            arms = "  ".join(
                f"{a} {x[a]['acc'] if x[a]['n'] else '—'}" for a in ARMS
            )  # '—': arm absent from this run
            print(f"  1/{b}: {arms}")
            print(
                f"        oracle {x['uniform_oracle']['acc']} {x['uniform_oracle']['factor']}"
                f" | sweep_cv {x['uniform_sweep_cv']['acc']}"
                f" | sweep_record {x['uniform_sweep_record']['acc']}"
                f" {x['uniform_sweep_record']['factor']}"
                f" {x['uniform_sweep_cv']['factor']}"
            )
            for k in (
                "H1_cs_minus_ceiling",
                "H2_tat_minus_oracle",
                "posthoc_tat_minus_cv",
                "posthoc_tat_minus_record",
            ):
                print(f"        {k}: {x[k]['points']} {x[k]['ci95']}")
            for k, m in x["mcnemar"].items():
                print(f"        {k}: {m['only_first']} x {m['only_second']}, p = {m['p']:.2g}")
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
