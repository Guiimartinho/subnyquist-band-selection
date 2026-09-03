"""Diagnóstico a partir do espectro de envelope, contra o prior geométrico.

É aqui que o prior físico entra: o conjunto de frequências candidatas é **finito e
sai da geometria do mancal**, não de um modelo aprendido. Por isso o projeto pode
falar em recuperação em vez de alucinação — a decisão é um teste de hipóteses sobre
um conjunto discreto conhecido a priori.

A métrica do projeto é **erro de estimação de frequência de falha**, contínuo, e não
acurácia de classificação. Rótulo de classe é para classificação; erro em hertz diz
quanto da informação sobreviveu à dobra espectral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from snq_core import BearingGeometry, fault_frequencies_hz

from snq_recon.envelope import EnvelopeSpectrum


@dataclass(frozen=True)
class FaultScore:
    """Pontuação de uma hipótese de falha."""

    fault: str
    expected_hz: float
    measured_hz: float
    error_hz: float
    harmonic_energy: float
    """Soma das magnitudes nos harmônicos, normalizada pelo piso local do espectro."""

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
            f"  {s.fault:<5} esperado {s.expected_hz:8.2f} Hz | medido {s.measured_hz:8.2f} Hz"
            f" | erro {s.error_hz:6.2f} Hz | energia {s.harmonic_energy:6.2f}"
            f" | {s.n_harmonics_found} harmônicos"
            for s in rows
        )


def _noise_floor(mag: np.ndarray) -> float:
    """Mediana como estimador robusto do piso — imune às poucas raias grandes."""
    m = float(np.median(mag))
    return m if m > 0.0 else float(np.mean(mag)) or 1.0


def score_fault(
    spec: EnvelopeSpectrum,
    expected_hz: float,
    *,
    n_harmonics: int = 5,
    tol_hz: float = 2.0,
    name: str = "",
) -> FaultScore:
    """Pontua uma frequência candidata pela energia nos seus harmônicos.

    Somar harmônicos, em vez de olhar só a fundamental, é o que separa uma falha real
    de uma raia qualquer: falha de rolamento produz trem de impactos, e trem de
    impactos é rico em harmônicos.
    """
    floor = _noise_floor(spec.magnitude)
    total = 0.0
    found = 0
    measured = float("nan")

    for h in range(1, n_harmonics + 1):
        target = h * expected_hz
        if target > spec.freqs_hz[-1]:
            break
        f_peak, mag = spec.peak_near(target, tol_hz)
        if mag > 2.0 * floor:
            total += mag / floor
            found += 1
            if h == 1:
                measured = f_peak

    if np.isnan(measured):
        measured, _ = spec.peak_near(expected_hz, tol_hz)

    err = abs(measured - expected_hz) if not np.isnan(measured) else float("inf")
    return FaultScore(name or f"{expected_hz:.1f}Hz", expected_hz, measured, err, total, found)


def diagnose(
    spec: EnvelopeSpectrum,
    geom: BearingGeometry,
    shaft_hz: float,
    *,
    n_harmonics: int = 5,
    tol_hz: float = 2.0,
) -> Diagnosis:
    """Testa as quatro hipóteses de falha do mancal contra o espectro de envelope.

    >>> # uso: diagnose(envelope_spectrum(x, fs, banda), geom, rpm/60)
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
    """Escolhe, num conjunto discreto, a candidata mais sustentada pelo espectro.

    É o modo "com prior" do E02: em vez de procurar o pico global, escolhe-se dentro
    do conjunto que a geometria admite. Devolve ``(frequência, energia harmônica)``.
    """
    cands = np.asarray(candidates_hz, dtype=float).ravel()
    if cands.size == 0:
        raise ValueError("conjunto de candidatas vazio")
    scores = [score_fault(spec, float(f), n_harmonics=n_harmonics, tol_hz=tol_hz) for f in cands]
    best = max(scores, key=lambda s: s.harmonic_energy)
    return best.expected_hz, best.harmonic_energy


def estimate_without_prior(spec: EnvelopeSpectrum, f_min_hz: float = 5.0) -> float:
    """Maior raia do espectro de envelope acima de ``f_min_hz``.

    É o modo "sem prior": nenhuma restrição sobre onde a falha pode estar. Serve de
    contraste para medir quanto o prior geométrico acrescenta.
    """
    sel = spec.freqs_hz >= f_min_hz
    if not np.any(sel):
        return float("nan")
    idx = np.flatnonzero(sel)
    return float(spec.freqs_hz[idx[int(np.argmax(spec.magnitude[idx]))]])
