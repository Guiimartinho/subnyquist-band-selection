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
from snq_core import BearingGeometry, alias_frequency_hz, fault_frequencies_hz

from snq_recon.envelope import EnvelopeSpectrum


@dataclass(frozen=True)
class FaultScore:
    """Pontuação de uma hipótese de falha."""

    fault: str
    expected_hz: float
    measured_hz: float
    error_hz: float
    fundamental_dominates: bool
    """A fundamental é o maior harmônico da série? Se não, a candidata é suspeita
    de ser subharmônica da raia verdadeira."""

    harmonic_energy: float
    """Soma das magnitudes nos harmônicos, **ponderada por 1/h** e normalizada pelo
    piso local do espectro.

    **Só comparável dentro de um mesmo registro.** A normalização é pela mediana do
    espectro, que num sinal limpo é minúscula e faz a razão explodir: medido em dado
    real do CWRU, o registro normal 97DE pontuou 6x mais que o registro com falha
    130DE. Serve para ordenar hipóteses num registro, não para detectar falha entre
    registros.
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
    fold_harmonics: bool = True,
) -> FaultScore:
    """Pontua uma frequência candidata pela energia nos seus harmônicos.

    Somar harmônicos, em vez de olhar só a fundamental, é o que separa uma falha real
    de uma raia qualquer: falha de rolamento produz trem de impactos, e trem de
    impactos é rico em harmônicos.

    **A soma é ponderada por 1/h, e o motivo não é estético.** Soma simples é
    enviesada a favor de subharmônicos: a série de ``f/3`` contém ``f``, então ela
    herda a energia da raia verdadeira e pode vencer. No prior geométrico de um
    rolamento o engodo é garantido, porque os harmônicos de FTF caem exatamente sobre
    ``BPFO/3``. Medido em dado real do CWRU no fator 12: os harmônicos de 35,77 Hz
    valiam 18,8 / 22,0 / 232,5 (dominados pelo terceiro, que é a BPFO verdadeira),
    contra 232,5 / 18,1 / 5,3 de 107,30 Hz. A soma simples elegia o subharmônico;
    a ponderada elege a fundamental.

    ``fold_harmonics`` trata os harmônicos acima de Nyquist como **medições
    dobradas** em vez de descartá-los. É o mecanismo central do projeto aplicado
    aqui: sob subamostragem o quinto harmônico de BPFI não some, ele reaparece
    numa posição previsível — e essa posição é informação, não perda.
    """
    floor = _noise_floor(spec.magnitude)
    total = 0.0
    found = 0
    measured = float("nan")
    nyquist = spec.freqs_hz[-1]
    per_harmonic: list[float] = []

    for h in range(1, n_harmonics + 1):
        target = h * expected_hz
        if target > nyquist:
            if not fold_harmonics:
                break
            target = float(alias_frequency_hz(target, 2.0 * nyquist))
            if target < tol_hz:  # dobrou em cima do DC, onde nao ha informacao
                continue
        f_peak, mag = spec.peak_near(target, tol_hz)
        if mag > 2.0 * floor:
            total += mag / floor / h  # 1/h: nao deixar harmonico alto dominar
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

    **Candidatas acima da frequência de Nyquist do espectro são descartadas.** Uma
    raia que não cabe na banda observada não pode ser a fundamental medida ali, e
    mantê-la no conjunto degrada a estimativa em vez de ajudar — medido no E02, onde
    o prior sem esse filtro derrubava o acerto a zero em fator 12.
    """
    cands = np.asarray(candidates_hz, dtype=float).ravel()
    cands = cands[cands <= spec.freqs_hz[-1]]
    if cands.size == 0:
        raise ValueError("nenhuma candidata cabe abaixo do Nyquist do espectro")
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
