"""Diagnosis from the envelope spectrum, against the geometric prior.

This is where the physical prior enters: the set of candidate frequencies is **finite and
comes from the bearing geometry**, not from a learned model. That is why the project can
speak of recovery rather than hallucination — the decision is a hypothesis test over
a discrete set known a priori.

The project's metric is **fault-frequency estimation error**, which is continuous, and not
classification accuracy. A class label is for classification; an error in hertz tells
how much of the information survived the spectral folding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from snq_core import BearingGeometry, alias_frequency_hz, fault_frequencies_hz

from snq_recon.envelope import EnvelopeSpectrum


@dataclass(frozen=True)
class FaultScore:
    """Score of one fault hypothesis."""

    fault: str
    expected_hz: float
    measured_hz: float
    error_hz: float
    fundamental_dominates: bool
    """Is the fundamental the largest harmonic of the series? If not, the candidate is suspected
    of being a subharmonic of the true line."""

    harmonic_energy: float
    """Sum of the magnitudes at the harmonics, **weighted by 1/h** and normalized by the
    local floor of the spectrum.

    **Only comparable within a single record.** The normalization is by the median of the
    spectrum, which in a clean signal is tiny and makes the ratio blow up: measured on real
    CWRU data, the normal record 97DE scored 6x more than the faulty record
    130DE. It serves to rank hypotheses within a record, not to detect a fault across
    records.
    """

    n_harmonics_found: int


@dataclass(frozen=True)
class Diagnosis:
    scores: dict[str, FaultScore]

    @property
    def best(self) -> FaultScore:
        return max(self.scores.values(), key=lambda s: s.harmonic_energy)

    def error_hz(self, fault: str) -> float:
        return self.scores[fault].error_hz

    def summary(self) -> str:
        rows = sorted(self.scores.values(), key=lambda s: -s.harmonic_energy)
        return "\n".join(
            f"  {s.fault:<5} expected {s.expected_hz:8.2f} Hz | measured {s.measured_hz:8.2f} Hz"
            f" | error {s.error_hz:6.2f} Hz | energy {s.harmonic_energy:6.2f}"
            f" | {s.n_harmonics_found} harmonics"
            for s in rows
        )


def _noise_floor(mag: np.ndarray) -> float:
    """Median as a robust estimator of the floor — immune to the few large lines."""
    m = float(np.median(mag))
    return m if m > 0.0 else float(np.mean(mag)) or 1.0


def score_fault(
    spec: EnvelopeSpectrum,
    expected_hz: float,
    *,
    n_harmonics: int = 5,
    tol_hz: float = 2.0,
    name: str = "",
    fold_harmonics: bool = True,
) -> FaultScore:
    """Scores a candidate frequency by the energy at its harmonics.

    Summing harmonics, instead of looking only at the fundamental, is what separates a real
    fault from an arbitrary line: a bearing fault produces an impact train, and an impact
    train is rich in harmonics.

    **The sum is weighted by 1/h, and the reason is not aesthetic.** A plain sum is
    biased in favor of subharmonics: the series of ``f/3`` contains ``f``, so it
    inherits the energy of the true line and can win. In the geometric prior of a
    bearing the decoy is guaranteed, because the FTF harmonics fall exactly on
    ``BPFO/3``. Measured on real CWRU data at factor 12: the harmonics of 35.77 Hz
    were worth 18.8 / 22.0 / 232.5 (dominated by the third, which is the true BPFO),
    against 232.5 / 18.1 / 5.3 for 107.30 Hz. The plain sum picked the subharmonic;
    the weighted sum picks the fundamental.

    ``fold_harmonics`` treats the harmonics above Nyquist as **folded
    measurements** instead of discarding them. It is the project's central mechanism applied
    here: under undersampling the fifth harmonic of BPFI does not vanish, it reappears
    at a predictable position — and that position is information, not loss.

    This only holds if the spectrum actually folds. When ``spec.folds_above_nyquist`` is false
    (order spectrum with angular anti-aliasing), there is no folded harmonic to look for, and
    looking for it would add whatever is at that position: the search stops at Nyquist.
    """
    floor = _noise_floor(spec.magnitude)
    total = 0.0
    found = 0
    measured = float("nan")
    nyquist = spec.freqs_hz[-1]
    per_harmonic: list[float] = []
    # getattr: an object with the same shape but without the field keeps the old behavior
    folds = fold_harmonics and bool(getattr(spec, "folds_above_nyquist", True))

    for h in range(1, n_harmonics + 1):
        target = h * expected_hz
        if target > nyquist:
            if not folds:
                break
            target = float(alias_frequency_hz(target, 2.0 * nyquist))
            if target < tol_hz:  # folded onto DC, where there is no information
                continue
        f_peak, mag = spec.peak_near(target, tol_hz)
        if mag > 2.0 * floor:
            total += mag / floor / h  # 1/h: keep the high harmonics from dominating
            found += 1
            per_harmonic.append(mag)
            if h == 1:
                measured = f_peak
        else:
            per_harmonic.append(0.0)

    if np.isnan(measured):
        measured, _ = spec.peak_near(expected_hz, tol_hz)

    err = abs(measured - expected_hz) if not np.isnan(measured) else float("inf")
    dominates = bool(per_harmonic) and per_harmonic[0] >= max(per_harmonic)
    return FaultScore(
        fault=name or f"{expected_hz:.1f}Hz",
        expected_hz=expected_hz,
        measured_hz=measured,
        error_hz=err,
        fundamental_dominates=dominates,
        harmonic_energy=total,
        n_harmonics_found=found,
    )


def diagnose(
    spec: EnvelopeSpectrum,
    geom: BearingGeometry,
    shaft_hz: float,
    *,
    n_harmonics: int = 5,
    tol_hz: float = 2.0,
) -> Diagnosis:
    """Tests the four bearing fault hypotheses against the envelope spectrum.

    >>> # usage: diagnose(envelope_spectrum(x, fs, band), geom, rpm/60)
    """
    expected = fault_frequencies_hz(geom, shaft_hz)
    return Diagnosis(
        {
            name: score_fault(spec, f, n_harmonics=n_harmonics, tol_hz=tol_hz, name=name)
            for name, f in expected.items()
        }
    )


def estimate_from_candidates(
    spec: EnvelopeSpectrum,
    candidates_hz: np.ndarray,
    *,
    n_harmonics: int = 3,
    tol_hz: float = 2.0,
) -> tuple[float, float]:
    """Picks, from a discrete set, the candidate best supported by the spectrum.

    It is the "with prior" mode of E02: instead of searching for the global peak, the choice
    is made within the set that the geometry admits. Returns
    ``(frequency, harmonic energy)``.

    **Candidates above the Nyquist frequency of the spectrum are discarded.** A
    line that does not fit in the observed band cannot be the fundamental measured there, and
    keeping it in the set degrades the estimate instead of helping — measured in E02, where
    the prior without this filter brought the hit rate down to zero at factor 12.
    """
    cands = np.asarray(candidates_hz, dtype=float).ravel()
    cands = cands[cands <= spec.freqs_hz[-1]]
    if cands.size == 0:
        raise ValueError("no candidate fits below the Nyquist frequency of the spectrum")
    scores = [score_fault(spec, float(f), n_harmonics=n_harmonics, tol_hz=tol_hz) for f in cands]
    best = max(scores, key=lambda s: s.harmonic_energy)
    return best.expected_hz, best.harmonic_energy


@dataclass(frozen=True)
class RefinedOrder:
    """Result of `refine_order_detail`: plateau center, width, and whether it touches the edge."""

    center: float
    plateau_width: float
    at_edge: bool


def refine_order_detail(
    spec: EnvelopeSpectrum,
    nominal: float,
    *,
    span: float,
    n_harmonics: int,
    tol: float,
    step: float = 0.005,
) -> RefinedOrder:
    """Like `refine_order`, but also reports the plateau width and whether it touches the edge.

    A plateau that touches ``nominal·(1 ± span)`` identifies no line at all: the maximum may
    lie outside the grid. Anyone using the result as ground truth must discard it (review of
    2026-11-11: O-A-2 calibrated exactly at the lower limit with ``samples_per_rev = 16``).
    """
    grid = np.arange(nominal * (1 - span), nominal * (1 + span), step)
    if grid.size == 0:
        raise ValueError("empty refinement grid: invalid span or step")
    scores = np.array(
        [
            score_fault(spec, float(f), n_harmonics=n_harmonics, tol_hz=tol).harmonic_energy
            for f in grid
        ]
    )
    top = float(np.max(scores))
    eps = 1e-12 * max(abs(top), 1.0)
    i0 = int(np.argmax(scores))
    j = i0
    while j + 1 < scores.size and scores[j + 1] >= top - eps:
        j += 1
    return RefinedOrder(
        center=float(0.5 * (grid[i0] + grid[j])),
        plateau_width=float(grid[j] - grid[i0]),
        at_edge=bool(i0 == 0 or j == grid.size - 1),
    )


def refine_order(
    spec: EnvelopeSpectrum,
    nominal: float,
    *,
    span: float,
    n_harmonics: int,
    tol: float,
    step: float = 0.005,
) -> float:
    """Candidate with the largest harmonic energy in a relative neighborhood of ``nominal``.

    Sweeps ``nominal·(1 ± span)`` in steps of ``step`` and scores each candidate with
    `score_fault`. The score is piecewise constant: ``peak_near`` takes the largest line within
    ±tol of each harmonic, so there is a **plateau** of candidates with the same score. The
    function returns the **center** of the contiguous plateau that contains the first maximum.

    Taking the first point of the plateau (which Python's ``max`` does) biases downward by half
    the plateau width. Measured on Ottawa (adversarial review of 2026-11-11): a plateau of 0.035
    order with ``samples_per_rev = 64`` and of 0.055–0.115 with 16, that is, a bias of 0.0175 to
    0.0575 order against a tolerance of 0.06. With the center, the two rulers agree in median.

    A plateau that touches the grid edge is truncated; use `refine_order_detail` to find out.
    """
    return refine_order_detail(
        spec, nominal, span=span, n_harmonics=n_harmonics, tol=tol, step=step
    ).center


def estimate_without_prior(spec: EnvelopeSpectrum, f_min_hz: float = 5.0) -> float:
    """Largest line of the envelope spectrum above ``f_min_hz``.

    It is the "without prior" mode (`no_prior` in E07): no restriction on where the fault can be. It
    serves as a contrast for measuring how much the geometric prior adds.
    """
    sel = spec.freqs_hz >= f_min_hz
    if not np.any(sel):
        return float("nan")
    idx = np.flatnonzero(sel)
    return float(spec.freqs_hz[idx[int(np.argmax(spec.magnitude[idx]))]])
