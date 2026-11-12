"""Piloto exploratório do E26 (2026-11-12, NÃO pré-registrado): seleção de banda guiada pelas ordens
de falha, no fluxo já decimado, no Ottawa. Ver o README.

    uv run python experiments/E26-fault-order-guided-selector/pilot.py [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from snq_data import undersample
from snq_data.ottawa import FS_HZ, load, record_names
from snq_recon.highres import envelope_order_spectrum
from snq_recon.selectors import _levels  # a partição do kurtograma

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RAW = ROOT / "datasets/raw/ottawa"
E20_METRICS = ROOT / "experiments/E20-fold-zone-rejection/results/metrics.csv"
SEG, N_SEG, PPR = 500_000, 3, 1024  # os segmentos do E20
FACTORS = [1, 2, 3, 4, 5, 6, 8]
CEILING_HZ, F_MIN_HZ, MAX_LEVEL = 25000.0, 500.0, 3
NOMINAL = {"BPFO": 4.5 * (1 - 0.206), "BPFI": 4.5 * (1 + 0.206)}  # 9 esferas, d/D cos = 0,206
HARMONICS = (1, 2)  # revisão de 2026-11-12: com ±12 %, 3·BPFO sobreporia 2·BPFI
W_SEARCH = W_LOCAL = 0.12  # o span de calibração do Ottawa no E07 e no E20
FALSE_ALARM = 0.05
TRUE_FAULT = {"I": "BPFI", "O": "BPFO", "H": "none"}


def bands(hi_hz: float) -> list[tuple[float, float]]:
    out = []
    for level in _levels(MAX_LEVEL):
        edges = np.linspace(F_MIN_HZ, hi_hz, int(round(2.0**level)) + 1)
        out += [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    return out


def prominence(orders: np.ndarray, mag: np.ndarray, o: float) -> float:
    s = (orders >= o * (1 - W_SEARCH)) & (orders <= o * (1 + W_SEARCH))
    loc = (orders >= o * (1 - W_LOCAL)) & (orders <= o * (1 + W_LOCAL))
    if not s.any() or not loc.any():
        return math.nan
    med = float(np.median(mag[loc]))
    return float(mag[s].max() / med) if med > 0 else math.nan


def scores(x: np.ndarray, v: np.ndarray, fs_hz: float) -> dict[str, tuple[float, tuple]]:
    best: dict[str, tuple[float, tuple]] = {c: (-math.inf, ()) for c in NOMINAL}
    for b in bands(min(CEILING_HZ, fs_hz / 2.0)):
        try:
            spec = envelope_order_spectrum(x, fs_hz, v, b, samples_per_rev=64)
        except ValueError:
            continue
        o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
        for c, oc in NOMINAL.items():
            s = float(np.nanmean([prominence(o, a, k * oc) for k in HARMONICS]))
            if np.isfinite(s) and s > best[c][0]:
                best[c] = (s, b)
    return best


def median3(x: np.ndarray) -> np.ndarray:
    """Mediana de 3 amostras (a do E20), aplicada ao fluxo já decimado."""
    y = x.copy()
    y[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


CLEAN = {"none": lambda x: x, "median3": median3}


def task(job) -> list[dict]:
    name, clean = job
    rec = load(name, RAW)
    speed = rec.speed_profile_hz(PPR)
    rows = []
    for seg in range(N_SEG):
        a, b = seg * SEG, (seg + 1) * SEG
        x, v = rec.vibration[a:b], speed[a:b]
        for m in FACTORS:
            sc = scores(CLEAN[clean](undersample(x, m)), undersample(v, m), FS_HZ / m)
            rows.append(
                {
                    "record": name.replace(".mat", ""),
                    "condition": name[0],
                    "segment": seg,
                    "factor": m,
                    **{f"score_{c}": round(s, 4) for c, (s, _b) in sc.items()},
                    **{
                        f"band_{c}": f"{bb[0]:.0f}-{bb[1]:.0f}" if bb else ""
                        for c, (_s, bb) in sc.items()
                    },
                }
            )
    return rows


def analyse(rows: list[dict]) -> dict:
    out: dict = {}
    for m in FACTORS:
        cell = [r for r in rows if r["factor"] == m]
        healthy = [max(r["score_BPFO"], r["score_BPFI"]) for r in cell if r["condition"] == "H"]
        tau = float(np.percentile(healthy, 100 * (1 - FALSE_ALARM)))
        acc: dict[str, list[int]] = defaultdict(list)
        for r in cell:
            top = max(NOMINAL, key=lambda c: r[f"score_{c}"])
            pred = top if r[f"score_{top}"] >= tau else "none"
            acc[r["condition"]].append(int(pred == TRUE_FAULT[r["condition"]]))
        out[f"{m}x"] = {
            "threshold": round(tau, 3),
            "outer_pct": round(100 * float(np.mean(acc["O"])), 1),
            "inner_pct": round(100 * float(np.mean(acc["I"])), 1),
            "balanced_pct": round(50 * (float(np.mean(acc["O"])) + float(np.mean(acc["I"]))), 1),
            "healthy_correct_pct": round(100 * float(np.mean(acc["H"])), 1),
        }
    return out


def e20_blind() -> dict:
    """Acerto do kurtograma cego do E20 (braço none, teto 25 kHz, régua aninhada)."""
    hits: dict[tuple, list[int]] = defaultdict(list)
    with E20_METRICS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["intervention"] == "none" and float(r["ceiling_hz"]) == CEILING_HZ and r["hit"]:
                hits[(int(r["factor"]), r["condition"])].append(int(r["hit"]))
    return {
        f"{m}x": {
            "outer_pct": round(100 * float(np.mean(hits[(m, "outer_race")])), 1),
            "inner_pct": round(100 * float(np.mean(hits[(m, "inner_race")])), 1),
        }
        for m in FACTORS
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", choices=sorted(CLEAN), default="none")
    args = ap.parse_args()
    names = record_names(("H", "I", "O"), ("A", "B", "C", "D"), (1, 2, 3))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(task, [(n, args.clean) for n in names]):
            rows += r
    out = EXP / ("results-pilot-w12" + ("" if args.clean == "none" else f"-{args.clean}"))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = {"guided": analyse(rows), "e20_blind_kurtogram": e20_blind()}
    (out / "summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for m in FACTORS:
        print(f"{m}x guiado {rep['guided'][f'{m}x']} | cego {rep['e20_blind_kurtogram'][f'{m}x']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
