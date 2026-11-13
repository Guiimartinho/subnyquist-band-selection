"""Periodic sampling masks on the tick grid, and the E15 feasible set.

The feasible set, as the E15 README defines it:

- **grid:** the acquisition rate of the dataset is the tick grid;
- **periodic pattern** of ``L`` ticks with ``K`` active positions per period, mean rate ``K/L``
  of the grid;
- **minimum spacing** of ``g`` ticks between consecutive active positions, measured
  **circularly**: the last position of one period and the first of the next also have to be
  ``g`` ticks or more apart. It represents the conversion time of a single SAR.

Realization on the nRF54L15, as an architecture hypothesis and not as a measurement: a TIMER in
compare mode with a period of ``L`` ticks, one CC register per active position, each event
linked by DPPI to the SAMPLE task of the SAADC (see `timer_schedule`). **The number of CC
registers per TIMER instance of the nRF54L15 has not been checked** in this implementation. A
small ``K`` is a conservative choice, not a guarantee.

Pieces, and nothing beyond them:

``FeasibleSet``                 the ``(L, K, g)`` set and the feasibility check;
``project_to_feasible``         exact projection, by dynamic programming, of a score vector;
``fold_project``                exact projection of an unconstrained mask with horizon ``M·L``;
``uniform_positions``           the uniform pattern, which is decimation by ``L/K``;
``random_feasible_positions``   uniform draw over the feasible patterns;
``designed_positions``          multi-coset designed by a classical criterion, fixed, data-free;
``support_designed_positions``  multi-coset for a known spectral support
                                (`venkataramani2001optimal`), with ``band_support_cells``;
``FixedMask``                   any fixed pattern, as a module;
``GumbelTopKMask``              learned mask: Gumbel top-K with straight-through
                                (`huijben2020dps`, `vangorp2021adps`), with diagnostics;
``LoupeMask``                   LOUPE (`bahadir2020loupe`), unconstrained, and its projection;
``BlockAdaptiveMask``           block-wise adaptive policy, inspired by ADPS.

All mask modules receive the signal ``(B, N)`` and return the time-domain mask ``(B, N)``,
with ``N`` a multiple of ``L``. The product ``x·m`` is the masked, zero-filled signal that the
decoder receives.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

__all__ = [
    "BlockAdaptiveMask",
    "FeasibleSet",
    "FixedMask",
    "GumbelTopKMask",
    "LoupeMask",
    "alias_weights",
    "band_support_cells",
    "circular_gaps",
    "designed_positions",
    "fold_project",
    "noise_amplification",
    "positions_to_mask",
    "project_to_feasible",
    "random_feasible_positions",
    "shrink_band_to_support",
    "spacing_penalty",
    "support_designed_positions",
    "timer_schedule",
    "uniform_positions",
]


# ----------------------------------------------------------------------
# feasible set
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FeasibleSet:
    """Periodic patterns of ``period_ticks`` ticks with ``active_per_period`` active positions.

    ``min_gap_ticks = 1`` restricts nothing beyond distinct positions. The set is empty
    when ``K·g > L``, and that is a construction error, not a silently empty set.
    """

    period_ticks: int
    active_per_period: int
    min_gap_ticks: int = 1

    def __post_init__(self) -> None:
        period, k, g = self.period_ticks, self.active_per_period, self.min_gap_ticks
        if period < 1 or k < 1 or g < 1:
            raise ValueError("period_ticks, active_per_period and min_gap_ticks must be >= 1")
        if k > period:
            raise ValueError(f"{k} positions do not fit in a period of {period} ticks")
        if k * g > period:
            raise ValueError(
                f"empty set: {k} positions with spacing >= {g} require period >= {k * g}, "
                f"and the period is {period}"
            )

    @classmethod
    def for_budget(cls, budget: int, active_per_period: int, min_gap_ticks: int = 1) -> FeasibleSet:
        """Set with ``L = K·budget``: mean rate exactly ``1/budget`` of the grid."""
        return cls(int(budget) * int(active_per_period), int(active_per_period), int(min_gap_ticks))

    @property
    def rate_fraction(self) -> float:
        """Mean rate as a fraction of the grid, ``K/L``."""
        return self.active_per_period / self.period_ticks

    def violations(self, positions: ArrayLike) -> int:
        """How many circular intervals between consecutive positions fall below ``g``."""
        p = _as_positions(positions, self.period_ticks)
        return int(np.count_nonzero(circular_gaps(p, self.period_ticks) < self.min_gap_ticks))

    def is_feasible(self, positions: ArrayLike) -> bool:
        """``K`` distinct positions within the period and circular spacing >= ``g``."""
        p = np.asarray(positions).ravel()
        if p.size != self.active_per_period or not np.all(np.equal(np.mod(p, 1), 0)):
            return False
        p = p.astype(np.int64)
        if np.any(p < 0) or np.any(p >= self.period_ticks) or np.unique(p).size != p.size:
            return False
        return self.violations(p) == 0


def _as_positions(positions: ArrayLike, period_ticks: int) -> NDArray[np.int64]:
    p = np.unique(np.asarray(positions, dtype=np.int64).ravel())
    if p.size == 0:
        raise ValueError("pattern without active positions")
    if p[0] < 0 or p[-1] >= period_ticks:
        raise ValueError(f"positions outside [0, {period_ticks})")
    return p


def circular_gaps(positions: ArrayLike, period_ticks: int) -> NDArray[np.int64]:
    """Intervals between consecutive active positions, including the period wrap-around.

    >>> circular_gaps([0, 3, 5], 8).tolist()
    [3, 2, 3]
    """
    p = _as_positions(positions, period_ticks)
    return np.diff(np.concatenate([p, [p[0] + period_ticks]])).astype(np.int64)


def positions_to_mask(positions: ArrayLike, period_ticks: int) -> NDArray[np.float32]:
    """Binary mask of one period from the active positions."""
    m = np.zeros(int(period_ticks), dtype=np.float32)
    m[_as_positions(positions, period_ticks)] = 1.0
    return m


# ----------------------------------------------------------------------
# exact projection
# ----------------------------------------------------------------------
def _best_spaced_subset(w: NDArray[np.float64], k: int, gap: int) -> tuple[float, list[int]] | None:
    """Largest sum of ``k`` entries of ``w`` with indices spaced by ``gap`` or more (linear).

    Dynamic programming, ``O(len(w)·k)``. Returns ``None`` if ``k`` indices do not fit.
    """
    n = w.size
    if k == 0:
        return 0.0, []
    if n == 0 or (k - 1) * gap + 1 > n:
        return None
    neg = -np.inf
    f = np.full((n + 1, k + 1), neg)
    f[:, 0] = 0.0
    take = np.zeros((n + 1, k + 1), dtype=bool)
    for i in range(1, n + 1):
        prev = f[max(i - gap, 0), :-1] + w[i - 1]  # take element i-1
        skip = f[i - 1, 1:]
        better = prev > skip
        f[i, 1:] = np.where(better, prev, skip)
        take[i, 1:] = better
    if not np.isfinite(f[n, k]):
        return None
    out: list[int] = []
    i, j = n, k
    while j > 0:
        if take[i, j]:
            out.append(i - 1)
            i, j = max(i - gap, 0), j - 1
        else:
            i -= 1
    return float(f[n, k]), sorted(out)


def project_to_feasible(scores: ArrayLike, fs: FeasibleSet) -> NDArray[np.int64]:
    """**Exact** projection of a score vector onto the feasible set.

    Returns the ``K`` positions with the largest score sum and circular spacing >= ``g``. For
    binary masks of fixed cardinality, ``||m - w||² = K - 2<m, w> + ||w||²``, so maximizing
    the sum is the same as minimizing the Euclidean distance: it is the projection of ``w``.

    The outer loop fixes the smallest chosen position ``s``; the other ``K - 1`` lie in
    ``[s + g, min(L - 1, s + L - g)]``, which guarantees the wrap-around spacing. Cost
    ``O(L²·K)``. On a tie in the sum, the first ``s`` found wins (deterministic).
    """
    w = np.asarray(scores, dtype=np.float64).ravel()
    period, k, gap = fs.period_ticks, fs.active_per_period, fs.min_gap_ticks
    if w.size != period:
        raise ValueError(f"{w.size} scores for a period of {period} ticks")
    if not np.all(np.isfinite(w)):
        raise ValueError("non-finite scores")
    best_val, best = -np.inf, None
    for s in range(period):
        if k == 1:
            val, cand = float(w[s]), [s]
        else:
            lo, hi = s + gap, min(period - 1, s + period - gap)
            sub = _best_spaced_subset(w[lo : hi + 1], k - 1, gap) if hi >= lo else None
            if sub is None:
                continue
            val = float(w[s]) + sub[0]
            cand = [s] + [lo + i for i in sub[1]]
        if val > best_val + 1e-12:
            best_val, best = val, cand
    if best is None:  # unreachable with a valid FeasibleSet; kept as a safeguard
        raise ValueError("no feasible pattern found")
    out = np.asarray(sorted(best), dtype=np.int64)
    assert fs.is_feasible(out)
    return out


def fold_project(logits: ArrayLike, k_horizon: int, fs: FeasibleSet) -> NDArray[np.int64]:
    """Exact projection, onto the feasible set, of a mask learned **without** the constraint.

    The free mask has horizon ``P = M·L`` and ``k_horizon`` active positions, with no spacing;
    the deterministic hard version is the top-``k_horizon`` of the logits. Among the periodic
    masks of period ``L``, ``||m_P - tile(m_L)||² = const - 2·Σ_l c_l·m_L(l)``, with ``c_l`` the
    number of times position ``l`` of the period appears in the ``M`` copies: maximizing
    ``Σ c_l·m_L(l)`` with `project_to_feasible` is the exact Euclidean projection. Count ties:
    mean of the folded logits, with a weight too small to invert a count.
    """
    w = np.asarray(logits, dtype=np.float64).ravel()
    period = fs.period_ticks
    if w.size % period:
        raise ValueError(f"horizon of {w.size} ticks is not a multiple of the period {period}")
    if not 1 <= int(k_horizon) <= w.size:
        raise ValueError("k_horizon outside the horizon")
    hard = np.zeros(w.size)
    hard[np.argsort(-w, kind="stable")[: int(k_horizon)]] = 1.0
    counts = hard.reshape(-1, period).sum(axis=0)
    folded = w.reshape(-1, period).mean(axis=0)
    span = float(folded.max() - folded.min())
    tie = 1e-6 * (folded - folded.min()) / span if span > 0.0 else np.zeros(period)
    return project_to_feasible(counts + tie, fs)


# ----------------------------------------------------------------------
# fixed patterns
# ----------------------------------------------------------------------
def uniform_positions(fs: FeasibleSet) -> NDArray[np.int64]:
    """Uniform pattern: one conversion every ``L/K`` ticks, with zero phase.

    On the grid, it is exactly ``snq_data.undersample(x, L/K)``.
    """
    period, k = fs.period_ticks, fs.active_per_period
    if period % k:
        raise ValueError(f"uniform requires L to be a multiple of K (L={period}, K={k})")
    step = period // k
    if step < fs.min_gap_ticks:
        raise ValueError(
            f"uniform with step {step} violates the minimum spacing {fs.min_gap_ticks}"
        )
    return np.arange(k, dtype=np.int64) * step


def random_feasible_positions(fs: FeasibleSet, rng: np.random.Generator) -> NDArray[np.int64]:
    """Feasible pattern drawn **uniformly** among all feasible patterns.

    The ``K`` circular intervals are ``g + e_i`` with ``Σ e_i = L - K·g``; the composition
    ``e`` is drawn by stars and bars and the pattern is rotated by a uniform offset.
    Each pattern appears ``K`` times (one per choice of which interval is the first), all with
    the same probability, so the draw is uniform over the patterns.
    """
    period, k, gap = fs.period_ticks, fs.active_per_period, fs.min_gap_ticks
    free = period - k * gap
    if k > 1:
        bars = np.sort(rng.choice(free + k - 1, size=k - 1, replace=False))
        parts = np.diff(np.concatenate([[-1], bars, [free + k - 1]])) - 1
    else:
        parts = np.array([free])
    gaps = gap + parts
    offset = int(rng.integers(period))
    pos = (offset + np.concatenate([[0], np.cumsum(gaps[:-1])])) % period
    out = np.sort(pos.astype(np.int64))
    assert fs.is_feasible(out)
    return out


def alias_weights(positions: ArrayLike, period_ticks: int) -> NDArray[np.float64]:
    """``|M_l|``, ``l = 0..L-1``: replica weight at shift ``l·fs/L`` in the zero-filled signal.

    With ``z = x·m`` and periodic ``m``, ``Z(f) = Σ_l M_l·X(f - l·fs/L)`` and
    ``M_l = (1/L)·Σ_p exp(-j2π·l·p/L)``. ``M_0 = K/L``. The uniform pattern concentrates
    everything in ``K`` full-weight replicas (those of decimation); a non-uniform pattern
    spreads the weight.
    """
    p = _as_positions(positions, period_ticks)
    lags = np.arange(int(period_ticks))
    return np.abs(np.exp(-2j * np.pi * np.outer(lags, p) / period_ticks).sum(axis=1)) / period_ticks


def _candidate_patterns(fs: FeasibleSet, max_enumeration: int, seed: int) -> NDArray[np.int64]:
    """Feasible patterns with the first position at zero (the criterion is rotation-invariant)."""
    period, k = fs.period_ticks, fs.active_per_period
    n_comb = math.comb(period - 1, k - 1)
    if n_comb <= max_enumeration:
        rest = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(1, period), k - 1)),
            dtype=np.int64,
            count=n_comb * (k - 1),
        ).reshape(n_comb, k - 1)
    else:  # deterministic random search when the enumeration does not fit
        rng = np.random.default_rng([seed, period, k, fs.min_gap_ticks])
        n_draws = min(int(max_enumeration), 200_000)
        draws = np.stack([random_feasible_positions(fs, rng) for _ in range(n_draws)])
        draws = np.sort((draws - draws[:, :1]) % period, axis=1)
        rest = np.unique(draws[:, 1:], axis=0)
    cands = np.concatenate([np.zeros((rest.shape[0], 1), dtype=np.int64), rest], axis=1)
    gaps = np.diff(np.concatenate([cands, cands[:, :1] + period], axis=1), axis=1)
    return cands[np.all(gaps >= fs.min_gap_ticks, axis=1)]


def designed_positions(
    fs: FeasibleSet,
    criterion: str = "min_peak_alias",
    *,
    max_enumeration: int = 3_000_000,
    seed: int = 0,
) -> NDArray[np.int64]:
    """**Designed** multi-coset: a fixed pattern, chosen by a classical criterion, without data.

    ``min_peak_alias`` minimizes the largest replica weight away from the origin,
    ``max_{l≠0} |M_l|`` (`alias_weights`): it spreads the folded energy instead of concentrating
    it, as the uniform pattern does. Ties are broken by the coarray coverage (number of distinct
    circular differences) and then by lexicographic order. ``coarray`` inverts the priority of
    the two criteria.

    It is not the coprime pair of `zhao2018coprime`: with a rate of exactly ``1/B`` the coprime
    pair requires ``4B - 2`` events per period and 1-tick intervals, and does not fit in the
    same ``(L, K, g)``.
    """
    if criterion not in ("min_peak_alias", "coarray"):
        raise ValueError(f"unknown criterion: {criterion}")
    period, k = fs.period_ticks, fs.active_per_period
    if k == 1:
        return np.zeros(1, dtype=np.int64)
    cands = _candidate_patterns(fs, max_enumeration, seed)
    lags = np.arange(1, period)
    peaks = np.empty(cands.shape[0])
    cover = np.empty(cands.shape[0], dtype=np.int64)
    chunk = max(1, 2_000_000 // period)
    for a in range(0, cands.shape[0], chunk):
        c = cands[a : a + chunk]
        phase = np.exp(-2j * np.pi * lags[None, :, None] * c[:, None, :] / period)
        peaks[a : a + chunk] = np.abs(phase.sum(axis=2)).max(axis=1) / period
        diffs = (c[:, :, None] - c[:, None, :]) % period
        onehot = np.zeros((c.shape[0], period), dtype=bool)
        np.put_along_axis(onehot, diffs.reshape(c.shape[0], -1), True, axis=1)
        cover[a : a + chunk] = onehot[:, 1:].sum(axis=1)
    peaks = np.round(peaks, 12)
    keys = (-cover, peaks) if criterion == "min_peak_alias" else (peaks, -cover)
    # np.lexsort: the last key is the primary one; the pattern columns break ties last
    order = np.lexsort(tuple(cands[:, j] for j in range(k - 1, -1, -1)) + keys)
    out = cands[order[0]].astype(np.int64)
    assert fs.is_feasible(out)
    return out


def band_support_cells(
    band_hz: tuple[float, float], fs_hz: float, period_ticks: int, n_grid: int = 64
) -> list[tuple[float, tuple[int, ...]]]:
    """Active spectral cells of the multi-coset model for a known real band.

    ``[0, fs)`` is divided into ``L`` cells of width ``fs/L``. For each offset ``f``
    within a cell (grid of ``n_grid`` points), cell ``l`` is active when
    ``f + l·fs/L`` falls in the band ``[lo, hi)`` or in the mirror ``[fs - hi, fs - lo)``: the
    signal is real. Returns ``(weight, S)`` for each distinct set, with weight = fraction of the
    grid.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    if not 0.0 <= lo < hi <= fs_hz / 2.0:
        raise ValueError("band must lie in [0, fs/2]")
    width = fs_hz / period_ticks
    counts: Counter[tuple[int, ...]] = Counter()
    for f0 in (np.arange(n_grid) + 0.5) / n_grid * width:
        f = f0 + np.arange(period_ticks) * width
        active = ((f >= lo) & (f < hi)) | ((f >= fs_hz - hi) & (f < fs_hz - lo))
        cells = tuple(int(c) for c in np.flatnonzero(active))
        if cells:
            counts[cells] += 1
    return sorted(((c / n_grid, s) for s, c in counts.items()), key=lambda t: t[1])


def shrink_band_to_support(
    band_hz: tuple[float, float], fs_hz: float, period_ticks: int, k: int, n_grid: int = 64
) -> tuple[float, float]:
    """Shrinks the band around its center until no cell set exceeds ``K``.

    With ``K`` cosets, each offset ``f`` resolves at most ``K`` cells (Landau). A
    band wider than that has no exact multi-coset reconstruction at that rate; the design
    then targets its central part, and the caller records the shrunk band.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    for _ in range(200):
        band = (center - half, center + half)
        sets = band_support_cells(band, fs_hz, period_ticks, n_grid)
        if sets and max(len(s) for _w, s in sets) <= k:
            return band
        half *= 0.97
    raise ValueError("band did not fit in K cells")


def support_designed_positions(
    fs: FeasibleSet,
    supports: list[tuple[float, tuple[int, ...]]],
    *,
    max_enumeration: int = 3_000_000,
    seed: int = 0,
) -> NDArray[np.int64]:
    """Multi-coset designed for a **known spectral support**.

    In cell ``f``, the coset samples are ``y(f) = D(f)·A_S·x_S(f)``, with
    ``A_S[k, s] = exp(j2π·c_k·s/L)`` and unitary ``D(f)``. The chosen pattern minimizes the
    noise amplification of the least-squares reconstruction, ``Σ_S w_S·||A_S^+||_F²``,
    averaged over the sets of active cells (`band_support_cells`). A pattern that leaves
    some ``A_S`` singular is excluded. Exhaustive search over the feasible set. Tie: the
    smallest peak alias (`alias_weights`), a choice of this project.

    The criterion follows the spirit of `venkataramani2001optimal`, which optimizes the pattern
    for a known support through reconstruction error bounds. The exact form of the bound and
    the sequential selection of the paper **have not been checked** (the abstract was not
    available in the databases consulted on 2026-11-11). The noise amplification is invariant
    to rotation of the pattern.
    """
    period, k = fs.period_ticks, fs.active_per_period
    if not supports:
        raise ValueError("empty support")
    if max(len(s) for _w, s in supports) > k:
        raise ValueError(f"support with more than K={k} cells: shrink the band first")
    cands = _candidate_patterns(fs, max_enumeration, seed)
    score = np.zeros(cands.shape[0])
    for weight, cells in supports:
        s = np.asarray(cells, dtype=np.float64)
        chunk = 200_000
        for a in range(0, cands.shape[0], chunk):
            c = cands[a : a + chunk].astype(np.float64)
            mat = np.exp(2j * np.pi * c[:, :, None] * s[None, None, :] / period)
            sv = np.linalg.svd(mat, compute_uv=False)
            amp = np.where(sv.min(axis=1) < 1e-9, np.inf, (1.0 / sv**2).sum(axis=1))
            score[a : a + chunk] += weight * amp
    if not np.any(np.isfinite(score)):
        raise ValueError("no feasible pattern reconstructs this support")
    # Tie-break between patterns equally good for the support: smallest peak alias, which
    # concentrates less of the energy from outside the modeled support; then lexicographic order.
    best = np.isclose(score, np.nanmin(score), rtol=1e-9, atol=1e-12)
    cands, score = cands[best], score[best]
    lags = np.arange(1, period)
    peaks = np.empty(cands.shape[0])
    chunk = max(1, 2_000_000 // period)
    for a in range(0, cands.shape[0], chunk):
        c = cands[a : a + chunk]
        phase = np.exp(-2j * np.pi * lags[None, :, None] * c[:, None, :] / period)
        peaks[a : a + chunk] = np.abs(phase.sum(axis=2)).max(axis=1) / period
    keys = (np.round(peaks, 12),)
    order = np.lexsort(tuple(cands[:, j] for j in range(k - 1, -1, -1)) + keys)
    out = cands[order[0]].astype(np.int64)
    assert fs.is_feasible(out)
    return out


def noise_amplification(
    positions: ArrayLike, period_ticks: int, supports: list[tuple[float, tuple[int, ...]]]
) -> float:
    """``Σ_S w_S·||A_S^+||_F²`` of a pattern (infinite if some ``A_S`` is singular)."""
    c = _as_positions(positions, period_ticks).astype(np.float64)
    total = 0.0
    for weight, cells in supports:
        s = np.asarray(cells, dtype=np.float64)
        sv = np.linalg.svd(np.exp(2j * np.pi * np.outer(c, s) / period_ticks), compute_uv=False)
        if sv.min() < 1e-9:
            return math.inf
        total += weight * float((1.0 / sv**2).sum())
    return total


def timer_schedule(
    positions: ArrayLike, period_ticks: int, tick_hz: float, timer_clock_hz: float
) -> dict[str, object]:
    """How the pattern would be scheduled on a TIMER with DPPI, and whether the counts are integers.

    The pattern is rotated to have a position at tick 0. Then ``K`` registers suffice: the
    first ``K - 1`` at the remaining positions and the last one at the end of the period, with a
    COMPARE→CLEAR shortcut, so that the end-of-period event is position 0 of the next period.
    The counts are ``ticks·timer_clock_hz/tick_hz`` and are only realizable if they are integers.
    """
    p = _as_positions(positions, period_ticks)
    rot = (p - p[0]) % period_ticks
    events = np.concatenate([np.sort(rot)[1:], [period_ticks]])
    counts = events * (float(timer_clock_hz) / float(tick_hz))
    integer = bool(np.allclose(counts, np.round(counts), atol=1e-9))
    return {
        "rotation_ticks": int(p[0]),
        "compare_ticks": [int(e) for e in events],
        "compare_counts": [float(c) for c in counts],
        "counts_are_integer": integer,
        "compare_registers": int(p.size),
        "timer_clock_hz": float(timer_clock_hz),
    }


# ----------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------
def _tile(period_mask: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Repeats ``(B, L)`` or ``(L,)`` up to ``n_samples``, which must be a multiple of ``L``."""
    period = period_mask.shape[-1]
    if n_samples % period:
        raise ValueError(f"N={n_samples} is not a multiple of the period L={period}")
    reps = n_samples // period
    return period_mask.repeat(1, reps) if period_mask.dim() == 2 else period_mask.repeat(reps)


def spacing_penalty(soft: torch.Tensor, min_gap_ticks: int) -> torch.Tensor:
    """(Circular) pairs of positions less than ``g`` ticks apart, weighted by the relaxed mask.

    ``Σ_{d=1}^{g-1} Σ_l m_l·m_{(l+d) mod L}``, averaged over the batch. Zero when ``g = 1``.
    """
    total = soft.new_zeros(())
    for d in range(1, int(min_gap_ticks)):
        total = total + (soft * torch.roll(soft, shifts=-d, dims=-1)).sum(-1).mean()
    return total


def _gumbel_topk(
    logits: torch.Tensor, k: int, tau: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel top-K: hard mask and relaxation, ``(B, L)`` each.

    ``K`` draws without replacement: in each, ``softmax((w + G)/τ)`` over the positions still
    free, and the position with the largest perturbed score is blocked for the following ones.
    The sum of the ``K`` softmaxes is the relaxation; the sum of the ``K`` chosen positions is
    the hard mask. It is the probabilistic sampling construction with straight-through used in
    `vangorp2021adps`.
    """
    u = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator)
    gumbel = -torch.log(-torch.log(u.clamp(1e-10, 1.0 - 1e-10)))
    pert = logits + gumbel
    hard = torch.zeros_like(pert)
    soft = torch.zeros_like(pert)
    blocked = torch.zeros_like(pert, dtype=torch.bool)
    for _ in range(int(k)):
        masked = pert.masked_fill(blocked, float("-inf"))
        soft = soft + torch.softmax(masked / tau, dim=-1)
        idx = masked.detach().argmax(dim=-1, keepdim=True)
        hard = hard.scatter(-1, idx, 1.0)
        blocked = blocked.scatter(-1, idx, True)
    return hard, soft


class FixedMask(nn.Module):
    """Fixed pattern (uniform, seeded random, designed, or the final one of a learned mask)."""

    def __init__(self, positions: ArrayLike, period_ticks: int) -> None:
        super().__init__()
        self.period_ticks = int(period_ticks)
        self.register_buffer(
            "mask_period", torch.from_numpy(positions_to_mask(positions, period_ticks))
        )
        self.last_penalty = torch.zeros(())

    def positions(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.mask_period.detach().cpu().numpy() > 0.5).astype(np.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        return _tile(self.mask_period.to(x.dtype), n).unsqueeze(0).expand(b, n)


class GumbelTopKMask(nn.Module):
    """Learned mask: Gumbel top-K over the logits of one period of the set ``fs``.

    Training: hard mask in the forward pass and gradient through the relaxation
    (straight-through), as in `huijben2020dps` and `vangorp2021adps`, or pure relaxation
    (``straight_through=False``). The minimum spacing of ``fs`` enters through
    `spacing_penalty` on the relaxation.

    Evaluation: the mask is the **exact projection** of the logits onto ``fs``, deterministic.
    ``freeze()`` fixes that projection.

    With ``fs = FeasibleSet(M·L, M·K, 1)`` the same class is the mask **without the
    constraint**: a horizon of ``M`` periods, count only over the horizon and no spacing.
    ``audit_gap`` counts, during training, the hard masks that would violate the target spacing
    (``audit_*``), and `diagnostics` says whether the mask has left the noise: the margin
    between the ``K``-th and the ``(K+1)``-th logit against the standard deviation of the Gumbel
    noise (π/√6 ≈ 1.28), and the final positions against the projection of the initial logits,
    which the seed reproduces.
    """

    def __init__(
        self,
        fs: FeasibleSet,
        *,
        tau: float = 1.0,
        straight_through: bool = True,
        init: str = "flat",
        init_scale: float = 0.01,
        seed: int = 0,
        audit_gap: int | None = None,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.tau = float(tau)
        self.straight_through = bool(straight_through)
        gen = torch.Generator().manual_seed(int(seed))
        w = init_scale * torch.randn(fs.period_ticks, generator=gen)
        if init == "uniform":
            w[torch.from_numpy(uniform_positions(fs))] += 1.0
        elif init != "flat":
            raise ValueError(f"unknown init: {init}")
        self.logits = nn.Parameter(w)
        self.register_buffer("initial_logits", w.detach().clone(), persistent=False)
        self.register_buffer("frozen_period", torch.zeros(0))
        self.audit_gap = int(audit_gap) if audit_gap is not None else fs.min_gap_ticks
        self.audit_seen = 0
        self.audit_violating = 0
        self.last_penalty = torch.zeros(())
        self.last_hard_violations = 0
        self._projection: tuple[int, NDArray[np.int64]] | None = None

    def diagnostics(self) -> dict[str, object]:
        """Final logits, top-K margin, initial and final positions, audited violations."""
        w = self.logits.detach().cpu().numpy().astype(np.float64)
        w0 = self.initial_logits.detach().cpu().numpy().astype(np.float64)
        k = self.fs.active_per_period
        s = np.sort(w)[::-1]
        final = project_to_feasible(w, self.fs)
        initial = project_to_feasible(w0, self.fs)
        margin = float(s[k - 1] - s[k]) if k < s.size else math.inf
        gumbel_std = math.pi / math.sqrt(6.0)
        return {
            "logits_final": [round(float(v), 5) for v in w],
            "logits_final_std": float(w.std()),
            "logits_initial_std": float(w0.std()),
            "margin_k": margin,
            "gumbel_std": gumbel_std,
            # > 1: the deterministic top-K resists the Gumbel noise; << 1: the mask is still noise
            "margin_over_gumbel_std": margin / gumbel_std,
            "converged": bool(margin > gumbel_std),
            "positions_initial": initial.tolist(),
            "positions_final": final.tolist(),
            "moved": final.tolist() != initial.tolist(),
            "last_hard_violations": int(self.last_hard_violations),
            "audit_gap": self.audit_gap,
            "audit_masks_seen": int(self.audit_seen),
            "audit_violating_fraction": self.audit_violating / max(1, self.audit_seen),
            "final_mask_audit_violations": _gap_violations(
                final, self.fs.period_ticks, self.audit_gap
            ),
        }

    def set_temperature(self, tau: float) -> None:
        self.tau = float(tau)

    def _projected(self) -> NDArray[np.int64]:
        """Exact projection of the current logits, recomputed only when the logits change."""
        stamp = self.logits._version
        if self._projection is None or self._projection[0] != stamp:
            pos = project_to_feasible(self.logits.detach().cpu().numpy(), self.fs)
            self._projection = (stamp, pos)
        return self._projection[1]

    def positions(self) -> NDArray[np.int64]:
        if self.frozen_period.numel():
            return np.flatnonzero(self.frozen_period.cpu().numpy() > 0.5).astype(np.int64)
        return self._projected()

    def freeze(self) -> NDArray[np.int64]:
        """Fixes the current exact projection and freezes the logits."""
        self._projection = None
        pos = self._projected()
        self.frozen_period = torch.from_numpy(positions_to_mask(pos, self.fs.period_ticks)).to(
            self.logits.device
        )
        self.logits.requires_grad_(False)
        return pos

    def eval_period(self) -> torch.Tensor:
        if self.frozen_period.numel():
            return self.frozen_period
        mask = positions_to_mask(self._projected(), self.fs.period_ticks)
        return torch.from_numpy(mask).to(self.logits.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        if self.training and not self.frozen_period.numel():
            logits = self.logits.unsqueeze(0).expand(b, -1)
            hard, soft = _gumbel_topk(logits, self.fs.active_per_period, self.tau)
            m = hard + soft - soft.detach() if self.straight_through else soft
            self.last_penalty = spacing_penalty(soft, self.fs.min_gap_ticks)
            period = self.fs.period_ticks
            rows = [np.flatnonzero(r > 0.5) for r in hard.detach().cpu().numpy()]
            self.last_hard_violations = sum(
                _gap_violations(p, period, self.fs.min_gap_ticks) for p in rows
            )
            self.audit_seen += len(rows)
            self.audit_violating += sum(
                int(_gap_violations(p, period, self.audit_gap) > 0) for p in rows
            )
            return _tile(m.to(x.dtype), n)
        self.last_penalty = x.new_zeros(())
        return _tile(self.eval_period().to(x.dtype), n).unsqueeze(0).expand(b, n)


def _gap_violations(positions: ArrayLike, period_ticks: int, gap: int) -> int:
    """Circular intervals between consecutive positions that are smaller than ``gap``."""
    pos = np.asarray(positions, dtype=np.int64).ravel()
    if pos.size < 2 or gap <= 1:
        return 0
    return int(np.count_nonzero(circular_gaps(pos, period_ticks) < gap))


def _loupe_rescale(p: torch.Tensor, rate: float) -> torch.Tensor:
    """LOUPE renormalization: mean of the probability map equal to the target rate.

    If the mean ``x̄`` exceeds the rate ``α``, scales by ``α/x̄``; otherwise, scales ``1 - p`` by
    ``(1-α)/(1-x̄)``. The probabilities stay in ``[0, 1]`` in both branches.
    """
    xbar = p.mean()
    r = rate / xbar
    beta = (1.0 - rate) / (1.0 - xbar).clamp_min(1e-12)
    le = (r <= 1.0).to(p.dtype)
    return le * p * r + (1.0 - le) * (1.0 - (1.0 - p) * beta)


class LoupeMask(nn.Module):
    """LOUPE (`bahadir2020loupe`) on the tick grid, **without** the feasible-set constraint.

    One logit per position of a horizon of ``n_positions`` ticks, repeated periodically in
    time. Probability map ``p = σ(a·w)``, renormalized to the target rate. During training,
    each tick is an independent Bernoulli: **hard** forward pass, ``1[u < p]``, and gradient
    through the relaxation ``σ(b·(p - u))`` (straight-through). The count per period is not
    fixed (only the mean is ``α``) and there is no minimum spacing.

    Declared deviation from the original LOUPE, which also uses the relaxation in the forward
    pass: with it, the "rate" seen in training was 1.5 to 2.4 times the target rate (review of
    2026-11-11), and the model trained with more samples than the budget and was evaluated with
    fewer.

    "Free" evaluation: one Bernoulli realization per tick, with a fixed seed (``eval_seed``).
    **Not realizable** by the TIMER (count and positions vary from period to period).
    `project` takes the map to the feasible set: mean of the ``n_positions/L`` copies per
    position of the period, and exact projection. In E15 the H2 projection gap is measured with
    the Gumbel top-K without the constraint (`GumbelTopKMask` with a horizon and no spacing);
    LOUPE remains an optional arm.
    """

    def __init__(
        self,
        n_positions: int,
        rate: float,
        *,
        slope: float = 5.0,
        sample_slope: float = 12.0,
        seed: int = 0,
        eval_seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 < rate < 1.0:
            raise ValueError("rate must be in (0, 1)")
        self.n_positions = int(n_positions)
        self.rate = float(rate)
        self.slope = float(slope)
        self.sample_slope = float(sample_slope)
        self.eval_seed = int(eval_seed)
        gen = torch.Generator().manual_seed(int(seed))
        u = torch.rand(self.n_positions, generator=gen).clamp(1e-3, 1.0 - 1e-3)
        self.logits = nn.Parameter(-torch.log(1.0 / u - 1.0) / self.slope)
        self.last_penalty = torch.zeros(())
        self.last_realized_rate = float("nan")
        self.last_train_rate = float("nan")

    def probabilities(self) -> torch.Tensor:
        return _loupe_rescale(torch.sigmoid(self.slope * self.logits), self.rate)

    def folded(self, period_ticks: int) -> NDArray[np.float64]:
        """Mean probability per position of a period of ``period_ticks`` ticks."""
        p = self.probabilities().detach().cpu().numpy().astype(np.float64)
        if self.n_positions % period_ticks:
            raise ValueError("n_positions must be a multiple of the period")
        return p.reshape(-1, period_ticks).mean(axis=0)

    def project(self, fs: FeasibleSet) -> NDArray[np.int64]:
        return project_to_feasible(self.folded(fs.period_ticks), fs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        p = self.probabilities().to(x.dtype)
        pt = p.repeat(math.ceil(n / self.n_positions))[:n]
        self.last_penalty = x.new_zeros(())
        if self.training:
            u = torch.rand(b, n, device=x.device, dtype=x.dtype)
            soft = torch.sigmoid(self.sample_slope * (pt.unsqueeze(0) - u))
            hard = (u < pt.unsqueeze(0)).to(x.dtype)
            self.last_train_rate = float(hard.mean())
            return hard + soft - soft.detach()
        gen = torch.Generator().manual_seed(self.eval_seed)
        u = torch.rand(n, generator=gen).to(device=x.device, dtype=x.dtype)
        m = (u < pt).to(x.dtype)
        self.last_realized_rate = float(m.mean())
        return m.unsqueeze(0).expand(b, n)


class BlockAdaptiveMask(nn.Module):
    """Block-wise adaptive mask, inspired by ADPS (`vangorp2021adps`).

    The window is divided into ``n_blocks`` blocks (each a multiple of ``L``). Block ``j`` uses
    the logits ``w0 + Δθ(h_{<j})``, where ``h_{<j}`` is the mean of a summary of what **has
    already been acquired**: the relative log energy in ``n_features`` bands of the spectrum of
    the zero-filled signal of each previous block. ``Δθ`` is a small MLP with the last layer
    initialized at zero, so at the start the policy is the fixed mask ``w0``.

    Declared differences from ADPS: a policy per block of periods (not sample by
    sample) and no LSTM; the state is the mean of the summaries of the previous blocks.

    Training: Gumbel top-K with straight-through in each block, spacing penalty.
    Evaluation: exact projection per block and, at the seams between blocks, dropping the
    conversions less than ``g`` ticks after the previous one (``last_dropped`` counts how many).
    Realizable as reprogramming the TIMER CCs between blocks, with the CPU waking up at every
    block.
    """

    def __init__(
        self,
        fs: FeasibleSet,
        *,
        n_blocks: int = 4,
        n_features: int = 16,
        hidden: int = 32,
        tau: float = 1.0,
        init_scale: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.n_blocks = int(n_blocks)
        self.n_features = int(n_features)
        self.tau = float(tau)
        gen = torch.Generator().manual_seed(int(seed))
        self.logits0 = nn.Parameter(init_scale * torch.randn(fs.period_ticks, generator=gen))
        self.policy = nn.Sequential(
            nn.Linear(self.n_features, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), fs.period_ticks),
        )
        last = self.policy[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.last_penalty = torch.zeros(())
        self.last_dropped = 0
        self.last_block_positions: list[list[list[int]]] = []

    def set_temperature(self, tau: float) -> None:
        self.tau = float(tau)

    def _bounds(self, n: int) -> list[tuple[int, int]]:
        period = self.fs.period_ticks
        if n % period:
            raise ValueError(f"N={n} is not a multiple of the period L={period}")
        n_per = n // period
        if n_per < self.n_blocks:
            raise ValueError("window too short for the number of blocks")
        cuts = [round(i * n_per / self.n_blocks) * period for i in range(self.n_blocks + 1)]
        return list(itertools.pairwise(cuts))

    def _features(self, z_block: torch.Tensor) -> torch.Tensor:
        power = torch.fft.rfft(z_block, dim=-1).abs() ** 2
        bands = torch.tensor_split(power, self.n_features, dim=-1)
        e = torch.stack([band.sum(-1) for band in bands], dim=-1)
        rel = torch.log(e / e.sum(-1, keepdim=True).clamp_min(1e-20) + 1e-8)
        return rel - rel.mean(-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        period, k = self.fs.period_ticks, self.fs.active_per_period
        acc = x.new_zeros(b, self.n_features)
        masks: list[torch.Tensor] = []
        penalty = x.new_zeros(())
        positions: list[list[list[int]]] = [[] for _ in range(b)]
        for j, (lo, hi) in enumerate(self._bounds(n)):
            logits = self.logits0.unsqueeze(0).expand(b, -1)
            if j > 0:
                logits = logits + self.policy(acc / j)
            if self.training:
                hard, soft = _gumbel_topk(logits, k, self.tau)
                m_period = hard + soft - soft.detach()
                penalty = penalty + spacing_penalty(soft, self.fs.min_gap_ticks)
            else:
                rows = []
                for i, row in enumerate(logits.detach().cpu().numpy()):
                    pos = project_to_feasible(row, self.fs)
                    positions[i].append(pos.tolist())
                    rows.append(positions_to_mask(pos, period))
                m_period = torch.from_numpy(np.stack(rows)).to(device=x.device, dtype=x.dtype)
            m_block = _tile(m_period.to(x.dtype), hi - lo)
            acc = acc + self._features(x[:, lo:hi] * m_block)
            masks.append(m_block)
        m = torch.cat(masks, dim=1)
        self.last_penalty = penalty / len(masks)
        if not self.training:
            m, self.last_dropped = _enforce_global_gap(m, self.fs.min_gap_ticks)
            self.last_block_positions = positions
        return m


def _enforce_global_gap(m: torch.Tensor, gap: int) -> tuple[torch.Tensor, int]:
    """Drops conversions less than ``gap`` ticks after the previous kept one (block seams)."""
    if gap <= 1:
        return m, 0
    out = m.detach().cpu().numpy().copy()
    dropped = 0
    for row in out:
        last = -(10**12)
        for i in np.flatnonzero(row > 0.5):
            if i - last < gap:
                row[i] = 0.0
                dropped += 1
            else:
                last = i
    return torch.from_numpy(out).to(device=m.device, dtype=m.dtype), dropped
