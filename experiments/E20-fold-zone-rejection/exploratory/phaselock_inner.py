"""Exploratory (adversarial review of 2026-11-12, outside the pre-registration): what the
cleaning replaces.

Question: are the samples that the ``despike`` criterion replaces in the **already-decimated**
stream, and that are not full-rate spikes, impacts of the fault?

- **Detection:** the same criterion as the E20 ``despike`` (10 MAD of the second difference,
  neighbours in the opposite direction and smaller than 0.8 times).
- **Phase locking:** R = |mean of exp(2πi · order · revolutions)| over the replaced samples, at
  the segment's true order. It is the local ruler when there is one; otherwise, the nested one.
- **Null:** the 95th percentile of R over 400 orders drawn in [2, 15].
- **Control:** the full-rate spikes should not lock to the fault order.

Written by ``reviewer-2`` and brought into the repository so that the quoted number regenerates.

    uv run python experiments/E20-fold-zone-rejection/exploratory/phaselock_inner.py inner_race
    (or outer_race)
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from snq_data.ottawa import FS_HZ, load
from snq_recon.highres import phase_revolutions

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RAW = ROOT / "datasets/raw/ottawa"
METRICS = EXP / "results-post-despike/metrics.csv"
OUT = EXP / "results-post-despike/exploratory"
SEG = 500_000
CODE = {"inner_race": "I", "outer_race": "O"}
FACTORS = (2, 4, 6, 8)


def detect(x: np.ndarray, k_mad: float = 10.0) -> np.ndarray:
    d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    mad = 1.4826 * np.median(np.abs(d))
    big = np.abs(d) > k_mad * mad
    dl = np.r_[0.0, d[:-1]]
    dr = np.r_[d[1:], 0.0]
    iso = (
        big
        & (np.sign(dl) == -np.sign(d))
        & (np.sign(dr) == -np.sign(d))
        & (np.abs(dl) < 0.8 * np.abs(d))
        & (np.abs(dr) < 0.8 * np.abs(d))
    )
    return np.flatnonzero(iso) + 1


def rayleigh(theta: np.ndarray, order: float) -> float:
    return float(np.abs(np.mean(np.exp(2j * np.pi * order * theta)))) if theta.size else np.nan


def work(task):
    name, seg, order = task
    rec = load(name + ".mat", RAW)
    spd = rec.speed_profile_hz(1024)
    x = rec.vibration[seg * SEG : (seg + 1) * SEG]
    sp = spd[seg * SEG : (seg + 1) * SEG]
    revs = phase_revolutions(sp, FS_HZ, x.size)
    full = set(detect(x).tolist())
    null_orders = np.random.default_rng(0).uniform(2.0, 15.0, 400)
    thf = revs[np.array(sorted(full), dtype=int)] if full else np.array([])
    out = []
    for m in FACTORS:
        idx = detect(x[::m]) * m
        other = np.array([i for i in idx.tolist() if i not in full], dtype=int)
        th = revs[other] if other.size else np.array([])
        r_null = np.array([rayleigh(th, o) for o in null_orders])
        out.append(
            {
                "record": name,
                "segment": seg,
                "factor": m,
                "n_replaced": int(idx.size),
                "n_not_fullrate_spike": int(other.size),
                "r_fault": round(rayleigh(th, order), 4),
                "null95": round(float(np.nanpercentile(r_null, 95)), 4) if other.size else "",
                "r_fullrate_spikes": round(rayleigh(thf, order), 4),
            }
        )
    return out


def main() -> int:
    cond = sys.argv[1] if len(sys.argv) > 1 else "inner_race"
    truth: dict[tuple[str, int], float] = {}
    with METRICS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["record"], int(r["segment"]))
            if r["truth_local_order"]:
                truth[key] = float(r["truth_local_order"])
            else:
                truth.setdefault(key, float(r["truth_order"]))
    c = CODE[cond]
    tasks = [
        (f"{c}-{s}-{t}", seg, truth[(f"{c}-{s}-{t}", seg)])
        for s in "ABCD"
        for t in (1, 2, 3)
        for seg in range(3)
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(work, tasks):
            rows += r
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"phaselock_{cond}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in FACTORS:
        rr = [r for r in rows if r["factor"] == m]
        lock = [r for r in rr if r["n_not_fullrate_spike"] >= 20]
        sig = sum(1 for r in lock if r["r_fault"] > r["null95"])
        print(
            f"{cond} {m}x: replaced/segment med {np.median([r['n_replaced'] for r in rr]):.0f}, "
            f"non-spike med {np.median([r['n_not_fullrate_spike'] for r in rr]):.0f} | "
            f"R at fault med {np.nanmedian([r['r_fault'] for r in lock]):.3f} vs null95 med "
            f"{np.nanmedian([float(r['null95']) for r in lock]):.3f} | above null95: "
            f"{sig}/{len(lock)} | R of full-rate spikes med "
            f"{np.nanmedian([r['r_fullrate_spikes'] for r in rr]):.3f}"
        )
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
