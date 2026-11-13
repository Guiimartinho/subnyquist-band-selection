"""Exploratory pilot of E25 (2026-11-12, NOT pre-registered): co-prime co-array against
uniform decimation, on CWRU 48k with the impulsive interferer of E23. See the README.

    uv run python experiments/E25-coprime-coarray/pilot.py [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from snq_recon.coarray import band_ses, cyclic_autocorrelation, sample_mask

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
E23 = ROOT / "experiments/E23-injected-interferer-cwru"
_spec = importlib.util.spec_from_file_location("e23run", E23 / "run.py")
e23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e23)

PATTERNS = {  # name -> (pattern, step, second step)
    "full_rate": ("full", 1, None),
    "uniform_4": ("uniform", 4, None),
    "coprime_8_9": ("coprime", 8, 9),
}
CARRIERS_HZ = [13000.0, 16000.0, 19000.0, 22000.0]
REL_DB = 20.0  # E23 label; the real dose is 3 dB lower (E23 README)
MAX_LAG = 192
F_MIN_HZ, F_MAX_HZ = 20.0, 330.0  # co-prime (8, 9): each lag's product comes at fs/72 -> 333 Hz
TOL_HZ = 2.0


def estimate(x: np.ndarray, fs: float, mask: np.ndarray, band: tuple[float, float]):
    cc = cyclic_autocorrelation(x, mask, MAX_LAG, fs, F_MAX_HZ)
    ses = band_ses(cc, *band)
    sel = (cc.alphas_hz >= F_MIN_HZ) & (cc.alphas_hz <= F_MAX_HZ)
    a, s = cc.alphas_hz[sel], ses[sel]
    return float(a[int(np.argmax(s))]), a, s


def prominence(a: np.ndarray, s: np.ndarray, truth: float) -> float:
    near = np.abs(a - truth) <= TOL_HZ
    win = (a >= 0.8 * truth) & (a <= 1.2 * truth)
    if not near.any() or not win.any():
        return math.nan
    return float(s[near].max() / np.median(s[win]))


def run_record(task) -> list[dict]:
    cfg, number, point, fault = task
    catalog = e23.CwruCatalog.load(e23.at_repo(cfg["catalog"]))
    rec = e23.load_record(
        number, point, e23.at_repo(cfg["data_dir"]), catalog, fs_hz=float(cfg["fs_hz"])
    )
    x, fs = rec.signal.astype(float), float(cfg["fs_hz"])
    h = x.size // 2
    xc, xt = x[:h], x[h:]
    truth = float(e23.fault_frequencies_hz(e23.GEOM, rec.shaft_hz)[fault])
    k_cal = e23.kurtogram(xc, fs, e23.ceiling_at(fs, cfg), cfg)  # commissioning, no injection
    band = (float(k_cal.f_lo_hz), float(k_cal.f_hi_hz))
    p_ref = e23.band_power(xc, fs, *band)
    seed_rec = int(hashlib.sha256(f"{number}{point}".encode()).hexdigest()[:8], 16)
    burst = float(cfg["injection"]["burst_rate_hz"])
    rows = []
    for fc in [None, *CARRIERS_HZ]:
        if fc is None:
            xtest = xt
        else:  # the same test signal as E23 (seed and dose)
            amp = math.sqrt(p_ref * 10.0 ** (REL_DB / 10.0))
            parts = [int(cfg["seed"]), seed_rec, int(round(fc))]
            xtest = xt + amp * e23.interferer(xt.size, fs, "impulsive", fc, cfg, [*parts, 1])
        for name, (kind, m, n2) in PATTERNS.items():
            mask = sample_mask(xtest.size, kind, m, n2)
            est, a, s = estimate(xtest, fs, mask, band)
            on_burst = any(abs(est - k * burst) <= TOL_HZ for k in (1, 2, 3))
            rows.append(
                {
                    "record": f"{number}{point}",
                    "fault": fault,
                    "truth_hz": round(truth, 3),
                    "band_lo_hz": round(band[0], 1),
                    "band_hi_hz": round(band[1], 1),
                    "fc_hz": "" if fc is None else fc,
                    "pattern": name,
                    "rate": round(float(mask.mean()), 4),
                    "estimate_hz": round(est, 3),
                    "hit": int(abs(est - truth) <= TOL_HZ),
                    "on_burst_line": int(on_burst),
                    "prominence": round(prominence(a, s, truth), 3),
                }
            )
    return rows


def e23_reference(records: set[str]) -> dict[str, float]:
    """Accuracy of the E23 blind kurtogram at 4× (uniform arm, reference ceiling)."""
    hits: dict[str, list[int]] = defaultdict(list)
    with (E23 / "results/metrics.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] != "uniform" or int(r["factor"]) != 4 or r["hit"] == "":
                continue
            if float(r["search_ceiling_hz"]) != 6000.0 or r["record"] not in records:
                continue
            if r["kind"] == "none":
                hits["no injection"].append(int(r["hit"]))
            elif r["kind"] == "impulsive" and float(r["rel_db"]) == REL_DB:
                hits[f"{float(r['fc_hz']):.0f}"].append(int(r["hit"]))
    return {k: round(100.0 * float(np.mean(v)), 1) for k, v in sorted(hits.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load((E23 / "config/base.yaml").read_text(encoding="utf-8"))
    catalog = e23.CwruCatalog.load(e23.at_repo(cfg["catalog"]))
    sel = catalog.selection("classical", cfg["subset"], cfg["point"])
    tasks = [(cfg, n, p, f) for n, p, f in sel]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_record, tasks):
            rows += r
    out = EXP / "results-pilot"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary: dict = {}
    for fc in ["", *CARRIERS_HZ]:
        label = "no injection" if fc == "" else f"{fc:.0f}"
        for name in PATTERNS:
            cell = [r for r in rows if r["fc_hz"] == fc and r["pattern"] == name]
            summary[f"{label}|{name}"] = {
                "n": len(cell),
                "hits_pct": round(100.0 * np.mean([r["hit"] for r in cell]), 1),
                "on_burst_pct": round(100.0 * np.mean([r["on_burst_line"] for r in cell]), 1),
                "prominence_median": round(float(np.nanmedian([r["prominence"] for r in cell])), 2),
            }
    summary["e23_blind_kurtogram_4x_hits_pct"] = e23_reference({r["record"] for r in rows})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    for k, v in summary.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
