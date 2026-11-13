"""E23 - pre-registered criteria (config/base.yaml, section ``analysis``) on results/metrics.csv.

    uv run python experiments/E23-injected-interferer-cwru/analyze.py \
        --config experiments/E23-injected-interferer-cwru/config/base.yaml

Written BEFORE the 1st full run. Definitions:

- cell: (carrier, factor, effective ceiling) of the ``uniform`` arm; evaluable if, without
  injection, the hit rate over the records is >= ``evaluable_baseline``;
- predicted class of the cell: the mode, across records, of the class computed from the injected
  realization (1 capture, 0 no capture, -1 borderline); mixed classes become borderline;
- drop: hit rate without injection minus hit rate with injection, in points, paired by record;
- H1(a): at max power, the fraction of evaluable predicted (1) cells with drop >= drop_points;
- H1(b): at each power, fraction of the folded cells (factor >= 2) with no predicted capture (0)
  that are evaluable and have drop >= drop_points;
- H1': P* = lowest power with a mean impulsive drop >= drop_points in the predicted cells; at P*,
  ratio between the mean drop of the surrogate and that of the impulsive in the same cells;
- H2: balanced accuracy of the TAT (admissible -> predicts a hit) over all rows of the uniform
  arm at the reference ceiling; and the mean difference between ``tat_rate`` and the uniform
  oracle (best factor >= budget in the test, per condition), over conditions x budgets.

Intervals: bootstrap by record (``analysis.bootstrap`` resamples).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

E = Path(__file__).resolve().parent


def ceiling0(cfg: dict, factor: int) -> float:
    return round(min(float(cfg["ceiling0_hz"]), float(cfg["fs_hz"]) / factor / 2.0), 1)


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        r["factor"] = int(r["factor"])
        r["search_ceiling_hz"] = float(r["search_ceiling_hz"])
        r["hit"] = int(r["hit"]) if r["hit"] != "" else None
        r["fc"] = float(r["fc_hz"]) if r["fc_hz"] != "" else None
        r["db"] = float(r["rel_db"]) if r["rel_db"] != "" else None
        r["pred"] = int(r["predicted"]) if r["predicted"] != "" else None
        r["line"] = int(r["interferer_line"]) if r["interferer_line"] != "" else None
        r["tat"] = int(r["tat_admissible_here"])
    return rows


class Data:
    def __init__(self, rows: list[dict], cfg: dict) -> None:
        self.cfg = cfg
        self.records = sorted({r["record"] for r in rows})
        uni = [r for r in rows if r["arm"] == "uniform"]
        # base[(rec, factor, ceil)] ; inj[(kind, fc, db, rec, factor, ceil)]
        self.base = {
            (r["record"], r["factor"], r["search_ceiling_hz"]): r["hit"]
            for r in uni
            if r["kind"] == "none"
        }
        self.inj = {
            (r["kind"], r["fc"], r["db"], r["record"], r["factor"], r["search_ceiling_hz"]): r
            for r in uni
            if r["kind"] != "none"
        }
        self.cells = sorted(
            {(r["fc"], r["factor"], r["search_ceiling_hz"]) for r in uni if r["fc"]}
        )
        self.powers = sorted({r["db"] for r in uni if r["db"] is not None})
        self.kinds = sorted({r["kind"] for r in uni if r["kind"] != "none"})
        self.rows = rows
        pred_votes: dict = defaultdict(Counter)
        for (kind, fc, _db, _rec, f, c), r in self.inj.items():
            if kind == "impulsive" and r["pred"] is not None:
                pred_votes[(fc, f, c)][r["pred"]] += 1
        self.pred = {}
        for cell, votes in pred_votes.items():
            cls, n = votes.most_common(1)[0]
            self.pred[cell] = cls if n == sum(votes.values()) else -1

    def base_acc(self, f: int, c: float, sample: list[str]) -> float | None:
        v = [self.base.get((rec, f, c)) for rec in sample]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    def drop(self, kind: str, fc: float, db: float, f: int, c: float, sample: list[str]):
        """Drop (points) paired by record; None if there is no pair."""
        pairs = []
        for rec in sample:
            b = self.base.get((rec, f, c))
            r = self.inj.get((kind, fc, db, rec, f, c))
            if b is None or r is None or r["hit"] is None:
                continue
            pairs.append((b, r["hit"]))
        if not pairs:
            return None
        return 100.0 * (np.mean([p[0] for p in pairs]) - np.mean([p[1] for p in pairs]))

    def evaluable(self, cell, sample: list[str]) -> bool:
        acc = self.base_acc(cell[1], cell[2], sample)
        return acc is not None and acc >= float(self.cfg["analysis"]["evaluable_baseline"])

    def h1(self, sample: list[str]) -> dict:
        a = self.cfg["analysis"]
        dmin = float(a["drop_points"])
        p_max = max(self.powers)
        out: dict = {"per_power": {}}
        for db in self.powers:
            pos = [c for c in self.cells if self.pred.get(c) == 1 and self.evaluable(c, sample)]
            neg = [
                c
                for c in self.cells
                if self.pred.get(c) == 0 and c[1] >= 2 and self.evaluable(c, sample)
            ]
            d_pos = [self.drop("impulsive", *c[:1], db, *c[1:], sample) for c in pos]
            d_neg = [self.drop("impulsive", *c[:1], db, *c[1:], sample) for c in neg]
            d_sur = [self.drop("surrogate", *c[:1], db, *c[1:], sample) for c in pos]
            d_pos = [d for d in d_pos if d is not None]
            d_neg = [d for d in d_neg if d is not None]
            d_sur = [d for d in d_sur if d is not None]
            out["per_power"][db] = {
                "n_pred_cells": len(d_pos),
                "n_clear_folded_cells": len(d_neg),
                "frac_pred_dropping": float(np.mean([d >= dmin for d in d_pos])) if d_pos else None,
                "frac_clear_dropping": float(np.mean([d >= dmin for d in d_neg]))
                if d_neg
                else None,
                "mean_drop_impulsive_pred": float(np.mean(d_pos)) if d_pos else None,
                "mean_drop_surrogate_pred": float(np.mean(d_sur)) if d_sur else None,
                "mean_drop_impulsive_clear": float(np.mean(d_neg)) if d_neg else None,
            }
        pp = out["per_power"]
        out["sensitivity_at_pmax"] = pp[p_max]["frac_pred_dropping"]
        out["max_false_alarm"] = max(
            (v["frac_clear_dropping"] for v in pp.values() if v["frac_clear_dropping"] is not None),
            default=None,
        )
        p_star = next(
            (
                db
                for db in self.powers
                if pp[db]["mean_drop_impulsive_pred"] is not None
                and pp[db]["mean_drop_impulsive_pred"] >= dmin
            ),
            None,
        )
        out["p_star_db"] = p_star
        if p_star is not None and pp[p_star]["mean_drop_surrogate_pred"] is not None:
            out["surrogate_ratio_at_p_star"] = pp[p_star]["mean_drop_surrogate_pred"] / max(
                pp[p_star]["mean_drop_impulsive_pred"], 1e-9
            )
        else:
            out["surrogate_ratio_at_p_star"] = None
        return out

    def h2(self, sample: list[str]) -> dict:
        cnt = Counter(sample)
        uni0 = [
            r
            for r in self.rows
            if r["arm"] == "uniform"
            and r["hit"] is not None
            and r["search_ceiling_hz"] == ceiling0(self.cfg, r["factor"])
            and r["record"] in cnt
        ]
        tp = fn = tn = fp = 0
        for r in uni0:
            w = cnt[r["record"]]
            if r["hit"] == 1:
                tp, fn = tp + w * r["tat"], fn + w * (1 - r["tat"])
            else:
                tn, fp = tn + w * (1 - r["tat"]), fp + w * r["tat"]
        tpr = tp / (tp + fn) if tp + fn else float("nan")  # undefined if there is no hit
        tnr = tn / (tn + fp) if tn + fp else float("nan")  # undefined if there is no miss
        # oracle per condition
        acc_uni: dict = defaultdict(list)
        for r in uni0:
            acc_uni[(r["kind"], r["fc_hz"], r["rel_db"], r["factor"])] += [r["hit"]] * cnt[
                r["record"]
            ]
        acc_tat: dict = defaultdict(list)
        for r in self.rows:
            if r["arm"] == "tat_rate" and r["hit"] is not None and r["record"] in cnt:
                acc_tat[(r["kind"], r["fc_hz"], r["rel_db"], r["factor"])] += [r["hit"]] * cnt[
                    r["record"]
                ]
        gaps = []
        factors = sorted({k[3] for k in acc_uni})
        for (kind, fc, db, m), v in acc_tat.items():
            best = max(
                (
                    np.mean(acc_uni[(kind, fc, db, q)])
                    for q in factors
                    if q >= m and acc_uni[(kind, fc, db, q)]
                ),
                default=None,
            )
            if best is not None:
                gaps.append(100.0 * (np.mean(v) - best))
        return {
            "tat_balanced_accuracy": 0.5 * (tpr + tnr),
            "tat_tpr": tpr,
            "tat_tnr": tnr,
            "mean_gap_tat_minus_oracle_points": float(np.mean(gaps)) if gaps else None,
            "worst_gap_points": float(np.min(gaps)) if gaps else None,
        }

    def extras(self) -> dict:
        """Direct capture indicator and recovery by the ideal anti-alias (descriptive)."""
        line: dict = defaultdict(list)
        for (kind, fc, db, _rec, f, c), r in self.inj.items():
            if r["line"] is not None and c == ceiling0(self.cfg, f):
                line[(kind, db, self.pred.get((fc, f, c)))].append(r["line"])
        aaf: dict = defaultdict(list)
        for r in self.rows:
            if r["arm"] == "aaf_ideal" and r["hit"] is not None:
                aaf[(r["kind"], r["rel_db"])].append(r["hit"])
        return {
            "interferer_line_rate": {
                f"{k[0]}|{k[1]}|pred={k[2]}": round(float(np.mean(v)), 3)
                for k, v in sorted(line.items(), key=str)
            },
            "aaf_ideal_acc": {
                f"{k[0]}|{k[1]}": round(float(np.mean(v)), 3)
                for k, v in sorted(aaf.items(), key=str)
            },
        }


def verdict(h1: dict, h2: dict, a: dict) -> dict:
    sens, fa, ratio = (
        h1["sensitivity_at_pmax"],
        h1["max_false_alarm"],
        h1["surrogate_ratio_at_p_star"],
    )
    ba, gap = h2["tat_balanced_accuracy"], h2["mean_gap_tat_minus_oracle_points"]

    def three(ok: bool, bad: bool) -> str:
        return "supported" if ok else ("refuted" if bad else "inconclusive")

    h1a = three(
        sens is not None and sens >= a["predicted_fraction"],
        sens is None or sens < a["refute_fraction"],
    )
    h1b = three(
        fa is not None and fa <= a["clear_false_alarm"],
        fa is not None and fa > a["refute_fraction"],
    )
    h1p = (
        "not evaluable (no P*)"
        if ratio is None
        else three(ratio < a["surrogate_ratio"], ratio >= a["surrogate_refute_ratio"])
    )
    if not np.isfinite(ba):
        h2a = "not evaluable (no hit or no miss)"
    else:
        h2a = three(ba >= a["tat_min_balanced_accuracy"], ba < a["tat_refute_balanced_accuracy"])
    h2b = three(
        gap is not None and gap >= -a["oracle_gap_points"],
        gap is not None and gap < -a["oracle_gap_points"],
    )
    return {
        "H1a_sensitivity": h1a,
        "H1b_specificity": h1b,
        "H1_line_impulsiveness": h1p,
        "H2_tat_ba": h2a,
        "H2_tat_vs_oracle": h2b,
    }


def analyze_spikes(path: Path, cfg: dict) -> dict:
    """H-spikes (pre-registered in the README): drop per arm, factor, rate and amplitude.

    Drop = hit rate without injection minus hit rate with spikes, in points, paired by record, at
    the reference ceiling for the uniform arm. Verdict, at the Ottawa condition (rate and amplitude
    in ``analysis.spikes.ottawa_like``):
    - ``supported`` if the uniform arm drops >= ``drop_points`` at some factor >= 2 AND
      ``aaf_ideal`` and ``despike`` drop <= ``mitigation_max_drop_points`` at all factors;
    - ``refuted`` if no uniform factor >= 2 drops >= ``drop_points`` at any amplitude;
    - ``inconclusive`` in between.
    """
    sa = cfg["analysis"]["spikes"]
    hit: dict = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["hit"] == "":
            continue
        f = int(r["factor"])
        if r["arm"] == "uniform" and float(r["search_ceiling_hz"]) != ceiling0(cfg, f):
            continue
        key = (r["arm"], r["kind"], r["spike_rate_hz"], r["spike_amp_mad"], f)
        hit[key][r["record"]] = int(r["hit"])
    factors = sorted({k[4] for k in hit})
    arms = sorted({k[0] for k in hit})

    def drop(arm: str, rate: str, amp: str, f: int) -> float | None:
        b = hit.get((arm, "none", "", "", f), {})
        x = hit.get((arm, "spikes", rate, amp, f), {})
        common = [r for r in b if r in x]
        if not common:
            return None
        return 100.0 * (sum(b[r] for r in common) - sum(x[r] for r in common)) / len(common)

    rates = sorted({k[2] for k in hit if k[1] == "spikes"}, key=float)
    amps = sorted({k[3] for k in hit if k[1] == "spikes"}, key=float)
    table = {
        arm: {
            f"{rate}|{amp}": {str(f): drop(arm, rate, amp, f) for f in factors}
            for rate in rates
            for amp in amps
        }
        for arm in arms
    }
    o_rate = str(float(sa["ottawa_like"]["rate_hz"]))
    o_amp = str(float(sa["ottawa_like"]["amplitude_mad"]))
    dmin, mmax = float(sa["drop_points"]), float(sa["mitigation_max_drop_points"])
    uni = [drop("uniform", o_rate, o_amp, f) for f in factors if f >= 2]
    falls = any(d is not None and d >= dmin for d in uni)
    mitig = all(
        d is None or d <= mmax
        for arm in ("aaf_ideal", "despike")
        for d in (drop(arm, o_rate, o_amp, f) for f in factors)
    )
    any_fall = any(
        d is not None and d >= dmin
        for amp in amps
        for d in (drop("uniform", o_rate, amp, f) for f in factors if f >= 2)
    )
    verdict = "supported" if (falls and mitig) else ("refuted" if not any_fall else "inconclusive")
    return {
        "ottawa_like": {"rate": o_rate, "amp": o_amp},
        "uniform_falls_somewhere": falls,
        "mitigations_within_threshold": mitig,
        "verdict_H_spikes": verdict,
        "drops_points": table,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=E / "config" / "base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    a = cfg["analysis"]
    results = args.config.parent.parent / cfg["output"]["results_dir"]
    if cfg["injection"]["kinds"] == ["spikes"]:
        rep = {
            "analyze_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metrics_sha256": hashlib.sha256((results / "metrics.csv").read_bytes()).hexdigest(),
            **analyze_spikes(results / "metrics.csv", cfg),
        }
        (results / "analysis.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            {
                k: rep[k]
                for k in (
                    "ottawa_like",
                    "uniform_falls_somewhere",
                    "mitigations_within_threshold",
                    "verdict_H_spikes",
                )
            }
        )
        return 0
    data = Data(load(results / "metrics.csv"), cfg)
    recs = data.records
    h1, h2 = data.h1(recs), data.h2(recs)
    rng = np.random.default_rng(int(cfg["seed"]))
    boot = defaultdict(list)
    for _ in range(int(a["bootstrap"])):
        s = list(rng.choice(recs, size=len(recs)))
        b1, b2 = data.h1(s), data.h2(s)
        for k, v in (
            ("sensitivity_at_pmax", b1["sensitivity_at_pmax"]),
            ("max_false_alarm", b1["max_false_alarm"]),
            ("surrogate_ratio_at_p_star", b1["surrogate_ratio_at_p_star"]),
            ("tat_balanced_accuracy", b2["tat_balanced_accuracy"]),
            ("mean_gap_tat_minus_oracle_points", b2["mean_gap_tat_minus_oracle_points"]),
        ):
            if v is not None and np.isfinite(v):
                boot[k].append(v)
    ci = {
        k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
        for k, v in boot.items()
    }
    report = {
        # provenance of the analysis (review of 2026-11-11): what it read and with which code
        "analyze_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256((results / "metrics.csv").read_bytes()).hexdigest(),
        "n_records": len(recs),
        "cells_by_predicted_class": dict(Counter(data.pred.values())),
        "H1": h1,
        "H2": h2,
        "ci95_bootstrap_by_record": ci,
        "verdict": verdict(h1, h2, a),
        "descriptive": data.extras(),
    }
    dst = results / "analysis.json"
    dst.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("n_records", "cells_by_predicted_class", "verdict")}, indent=1
        )
    )
    print("H1 per power:")
    for db, v in h1["per_power"].items():
        print(f"  {db:+5.0f} dB: {v}")
    print(
        f"P* = {h1['p_star_db']}; surrogate/impulsive ratio at P* = "
        f"{h1['surrogate_ratio_at_p_star']}"
    )
    print(f"H2: {h2}")
    print(f"95 % CI: {ci}")
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
