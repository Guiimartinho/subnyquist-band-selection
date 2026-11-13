"""Seletor de banda guiado pelas ordens de falha e limpeza do fluxo decimado (E26).

Funções puras, usadas pela rodada do Paderborn e testadas em ``tests/test_e26.py``.
"""

from __future__ import annotations

import math

import numpy as np
from snq_recon.highres import envelope_order_spectrum
from snq_recon.selectors import _levels  # a partição do kurtograma


def median3(x: np.ndarray) -> np.ndarray:
    """Mediana de 3 amostras (a do E20), aplicada ao fluxo já decimado."""
    y = np.asarray(x, dtype=float).copy()
    if y.size >= 3:
        y[1:-1] = np.median(np.vstack([y[:-2], y[1:-1], y[2:]]), axis=0)
    return y


def windows(order: float, k: int, frac: float, exclusion: float) -> list[tuple[float, float]]:
    """Janela de busca [k·o(1−frac), k·o(1+frac)] sem as ordens inteiras ± ``exclusion``."""
    lo, hi = k * order * (1 - frac), k * order * (1 + frac)
    segs = [(lo, hi)]
    for n in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        a, b = n - exclusion, n + exclusion
        new = []
        for s, e in segs:
            if b <= s or a >= e:
                new.append((s, e))
                continue
            if s < a:
                new.append((s, a))
            if b < e:
                new.append((b, e))
        segs = new
    return segs


def prominence(orders: np.ndarray, mag: np.ndarray, segs: list[tuple[float, float]]) -> float:
    """Pico dividido pela mediana, na união dos intervalos; NaN se houver menos de 3 bins."""
    m = np.zeros(orders.size, dtype=bool)
    for s, e in segs:
        m |= (orders >= s) & (orders <= e)
    if int(m.sum()) < 3:
        return math.nan
    med = float(np.median(mag[m]))
    return float(mag[m].max() / med) if med > 0 else math.nan


def bands(fs_hz: float, f_min_hz: float, max_level: int) -> list[tuple[float, float]]:
    out = []
    for level in _levels(max_level):
        edges = np.linspace(f_min_hz, fs_hz / 2.0, int(round(2.0**level)) + 1)
        out += [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    return out


def guided_scores(
    x: np.ndarray,
    v: np.ndarray,
    fs_hz: float,
    orders: dict[str, float],
    f_min_hz: float,
    max_level: int,
    frac: float,
    exclusion: float,
    harmonics: tuple[int, ...],
    samples_per_rev: int,
) -> dict[str, tuple[float, tuple[float, float] | None]]:
    """Por falha candidata: o maior escore entre as bandas e a banda que o deu.

    O escore de uma banda é a média, nas harmônicas pedidas, da proeminência da raia na janela.
    """
    segs = {c: [windows(o, k, frac, exclusion) for k in harmonics] for c, o in orders.items()}
    best: dict[str, tuple[float, tuple[float, float] | None]] = {
        c: (-math.inf, None) for c in orders
    }
    for b in bands(fs_hz, f_min_hz, max_level):
        try:
            spec = envelope_order_spectrum(x, fs_hz, v, b, samples_per_rev=samples_per_rev)
        except ValueError:
            continue
        o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
        for c in orders:
            vals = [prominence(o, a, s) for s in segs[c]]
            score = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
            if np.isfinite(score) and score > best[c][0]:
                best[c] = (score, b)
    return best
