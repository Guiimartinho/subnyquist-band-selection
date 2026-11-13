"""Exploratorio (revisao adversarial de 2026-11-13, fora do pre-registro): trem de picos do Ottawa.

Caracteriza, nos 36 registros brutos, os picos isolados de uma amostra pelo mesmo criterio do braco
``despike`` (desvio da media dos vizinhos acima de 10 MAD, vizinhos em sentido oposto e menores que
0,8 vez). Por registro grava:
- taxa de picos por segundo, fracao com desvio negativo e amplitude mediana em MAD;
- fracao dos intervalos entre picos iguais a exatamente 40 amostras;
- a raia dominante, entre 1 e 20 kHz, do espectro do indicador dos instantes de pico;
- a fracao de picos a ate 1 amostra de uma borda do encoder, e a mesma fracao para posicoes
  sorteadas (nulo).

Reproduz as medicoes feitas pelo reviewer-2 no scratchpad em 2026-11-13. Uso:
    uv run python experiments/E20-fold-zone-rejection/exploratory/spike_train.py
Grava ``results-post-despike/exploratory/spike_train.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
DATA = REPO / "datasets" / "raw" / "ottawa"
OUT = EXP / "results-post-despike" / "exploratory" / "spike_train.csv"
FS_HZ = 200_000.0
K_MAD = 10.0
SEED = 20261113


def spike_index(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dd = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    mad = 1.4826 * float(np.median(np.abs(dd)))
    big = np.abs(dd) > K_MAD * mad
    dl = np.r_[0.0, dd[:-1]]
    dr = np.r_[dd[1:], 0.0]
    iso = (
        big
        & (np.sign(dl) == -np.sign(dd))
        & (np.sign(dr) == -np.sign(dd))
        & (np.abs(dl) < 0.8 * np.abs(dd))
        & (np.abs(dr) < 0.8 * np.abs(dd))
    )
    idx = np.flatnonzero(iso)
    return idx + 1, dd[idx], mad


def nearest(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ref, a)
    lo = np.clip(pos - 1, 0, len(ref) - 1)
    hi = np.clip(pos, 0, len(ref) - 1)
    return np.minimum(np.abs(a - ref[lo]), np.abs(a - ref[hi]))


def record_stats(path: Path, rng: np.random.Generator) -> dict:
    m = loadmat(path)
    x = np.ravel(m["Channel_1"]).astype(float)
    enc = np.ravel(m["Channel_2"]).astype(float)
    sp, dev, mad = spike_index(x)
    isi = np.diff(sp)
    ind = np.zeros(len(x))
    ind[sp] = 1.0
    spec = np.abs(np.fft.rfft(ind - ind.mean()))
    f_hz = np.fft.rfftfreq(len(x), 1.0 / FS_HZ)
    band = (f_hz > 1000.0) & (f_hz < 20000.0)
    i = int(np.argmax(spec * band))
    thr = 0.5 * (np.percentile(enc, 5) + np.percentile(enc, 95))
    high = enc > thr
    edges = np.flatnonzero(high[1:] != high[:-1]) + 1
    rnd = np.sort(rng.integers(1, len(x) - 1, size=len(sp)))
    return {
        "record": path.stem,
        "condition": path.stem[0],
        "spikes_per_s": round(len(sp) / (len(x) / FS_HZ), 1),
        "frac_negative": round(float(np.mean(dev < 0)), 3),
        "median_amp_mad": round(float(np.median(np.abs(dev)) / mad), 1),
        "frac_isi_40": round(float(np.mean(isi == 40)), 3),
        "indicator_line_hz": round(float(f_hz[i]), 2),
        "indicator_line_over_median": round(float(spec[i] / np.median(spec[band])), 1),
        "frac_within1_encoder_edge": round(float(np.mean(nearest(sp, edges) <= 1)), 3),
        "frac_within1_null": round(float(np.mean(nearest(rnd, edges) <= 1)), 3),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = [record_stats(p, rng) for p in sorted(DATA.glob("*.mat"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for cond in "HIO":
        sub = [r for r in rows if r["condition"] == cond]
        rate = np.array([r["spikes_per_s"] for r in sub])
        lines = sorted({r["indicator_line_hz"] for r in sub})
        print(
            f"{cond}: n={len(sub)} picos/s mediana {np.median(rate):.0f} "
            f"[{rate.min():.0f}, {rate.max():.0f}] | negativos "
            f"{np.median([r['frac_negative'] for r in sub]):.2f} | amp "
            f"{np.median([r['median_amp_mad'] for r in sub]):.1f} MAD | isi=40 "
            f"{np.median([r['frac_isi_40'] for r in sub]):.2f} | raias {lines} | encoder "
            f"{np.median([r['frac_within1_encoder_edge'] for r in sub]):.3f} contra nulo "
            f"{np.median([r['frac_within1_null'] for r in sub]):.3f}"
        )


if __name__ == "__main__":
    main()
