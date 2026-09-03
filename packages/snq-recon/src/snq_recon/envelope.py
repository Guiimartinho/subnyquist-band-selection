"""Análise de envelope — o baseline oficial do projeto.

É o método que a indústria já usa, e o número que qualquer coisa aprendida tem de
bater. `CLAUDE.md`: o baseline a bater é amostragem passa-faixa clássica + envelope,
não uma CNN.

Uma armadilha descoberta em bancada de simulação e registrada no caderno de
2026-09-02: **tomar a magnitude de um sinal real dobra a largura de banda**. Se a
banda dobrada ocupa `[0, fs/2]` inteiro, `|y|` aliasa sobre si mesmo e o espectro de
envelope vira lixo. Por isso o envelope aqui é sempre calculado sobre o **sinal
analítico** (Hilbert), e `envelope_headroom_ok` existe para checar a folga antes de
confiar no resultado.

Referências: Randall & Antoni, *Rolling element bearing diagnostics — a tutorial*,
MSSP 2011; Antoni, *Fast computation of the kurtogram*, MSSP 2007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal


def bandpass_analytic(x: ArrayLike, fs_hz: float, f_lo_hz: float, f_hi_hz: float) -> NDArray:
    """Sinal analítico da banda ``[f_lo, f_hi]``, por filtragem no domínio da FFT.

    Zera tudo fora da banda e as frequências negativas de uma vez — o que produz o
    sinal analítico da banda sem precisar filtrar e depois aplicar Hilbert. Evita o
    transitório de filtro IIR e é exato para registro finito.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs_hz)

    keep = (freqs >= f_lo_hz) & (freqs < f_hi_hz)  # só positivas => analítico
    out = np.zeros(n, dtype=complex)
    out[keep] = 2.0 * spec[keep]
    return np.fft.ifft(out)


def envelope(x: ArrayLike, fs_hz: float, band_hz: tuple[float, float] | None = None) -> NDArray:
    """Envelope do sinal, opcionalmente restrito a uma banda.

    Sem banda, usa a transformada de Hilbert do sinal inteiro.
    """
    if band_hz is None:
        return np.abs(signal.hilbert(np.asarray(x, dtype=float).ravel()))
    return np.abs(bandpass_analytic(x, fs_hz, *band_hz))


@dataclass(frozen=True)
class EnvelopeSpectrum:
    freqs_hz: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    fs_hz: float
    band_hz: tuple[float, float] | None

    def peak_near(self, f_hz: float, tol_hz: float) -> tuple[float, float]:
        """Maior raia dentro de ``f_hz ± tol_hz``. Devolve ``(frequência, magnitude)``."""
        sel = np.abs(self.freqs_hz - f_hz) <= tol_hz
        if not np.any(sel):
            return (float("nan"), 0.0)
        idx = np.flatnonzero(sel)
        best = idx[int(np.argmax(self.magnitude[idx]))]
        return float(self.freqs_hz[best]), float(self.magnitude[best])


def envelope_spectrum(
    x: ArrayLike,
    fs_hz: float,
    band_hz: tuple[float, float] | None = None,
    *,
    detrend: bool = True,
) -> EnvelopeSpectrum:
    """Espectro do envelope — onde as frequências de falha aparecem.

    O valor médio do envelope é removido por padrão: ele é uma componente DC grande
    que domina o espectro e não carrega informação de falha.
    """
    env = envelope(x, fs_hz, band_hz)
    if detrend:
        env = env - float(np.mean(env))
    n = env.size
    mag = np.abs(np.fft.rfft(env * np.hanning(n))) * (2.0 / n)
    return EnvelopeSpectrum(
        freqs_hz=np.fft.rfftfreq(n, d=1.0 / fs_hz),
        magnitude=mag,
        fs_hz=fs_hz,
        band_hz=band_hz,
    )


def envelope_headroom_ok(band_hz: tuple[float, float], fs_hz: float, factor: float = 2.5) -> bool:
    """A taxa deixa folga para o envelope, ou ele vai aliasar?

    A largura de banda do envelope é ~2× a da banda do sinal. Amostrar em exatamente
    ``2·B`` acomoda a portadora dobrada mas **não** o envelope. A regra prática do
    projeto é ``fs >= 2.5·B``.

    >>> envelope_headroom_ok((140e3, 160e3), fs_hz=40e3)
    False
    >>> envelope_headroom_ok((140e3, 160e3), fs_hz=60e3)
    True
    """
    bandwidth = band_hz[1] - band_hz[0]
    return fs_hz >= factor * bandwidth


def squared_envelope_spectrum(
    x: ArrayLike, fs_hz: float, band_hz: tuple[float, float] | None = None
) -> EnvelopeSpectrum:
    """Espectro do envelope **ao quadrado**.

    Randall & Antoni argumentam que o envelope quadrático é preferível ao de
    magnitude: é o que sai naturalmente da teoria ciclo-estacionária e evita a
    distorção introduzida pela raiz quadrada implícita em ``|·|``.
    """
    env = (
        np.abs(bandpass_analytic(x, fs_hz, *band_hz)) ** 2
        if band_hz
        else np.abs(signal.hilbert(np.asarray(x, dtype=float).ravel())) ** 2
    )
    env = env - float(np.mean(env))
    n = env.size
    mag = np.abs(np.fft.rfft(env * np.hanning(n))) * (2.0 / n)
    return EnvelopeSpectrum(np.fft.rfftfreq(n, d=1.0 / fs_hz), mag, fs_hz, band_hz)
