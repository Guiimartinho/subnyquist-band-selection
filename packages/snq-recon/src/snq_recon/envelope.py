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


def _band_analytic_from_spectrum(
    spec: NDArray[np.complex128], freqs: NDArray[np.float64], f_lo_hz: float, f_hi_hz: float
) -> NDArray[np.complex128]:
    """Sinal analítico da banda a partir da FFT já calculada: ``2·X`` nas positivas, ``X`` no DC.

    O DC entra com peso 1, não 2: o sinal analítico é ``x + j·H{x}`` e ``H`` anula o DC,
    então ``X_a(0) = X(0)``. Com peso 2 a parte real ganhava a média de novo (corrigido em
    2026-11-11; só afeta bandas com ``f_lo_hz <= 0``). O bin de ``fs/2`` cai nas negativas de
    ``fftfreq`` e fica fora, como antes.
    """
    keep = (freqs >= f_lo_hz) & (freqs < f_hi_hz)  # só positivas => analítico
    out = np.zeros(spec.size, dtype=complex)
    out[keep] = 2.0 * spec[keep]
    if keep[0]:
        out[0] = spec[0]
    return np.fft.ifft(out)


def bandpass_analytic(x: ArrayLike, fs_hz: float, f_lo_hz: float, f_hi_hz: float) -> NDArray:
    """Sinal analítico da banda ``[f_lo, f_hi)``, por filtragem no domínio da FFT.

    Zera tudo fora da banda e as frequências negativas de uma vez — o que produz o
    sinal analítico da banda sem precisar filtrar e depois aplicar Hilbert. Com a banda
    inteira ``[0, fs/2)`` coincide com ``scipy.signal.hilbert`` (salvo o bin de ``fs/2``).

    **A filtragem é circular.** Ela é exata para a extensão periódica do registro, e
    essa extensão tem um salto na emenda ``x[-1] -> x[0]``: o salto aparece como um
    transiente nas duas bordas de ``z``. Nos espectros de envelope deste módulo a janela
    de Hann o apaga. Numa estatística sem janela, como a curtose, ele vira um falso
    transiente — por isso `fast_kurtogram` não usa esta função diretamente.
    """
    x = np.asarray(x, dtype=float).ravel()
    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0 / fs_hz)
    return _band_analytic_from_spectrum(spec, freqs, f_lo_hz, f_hi_hz)


def envelope(x: ArrayLike, fs_hz: float, band_hz: tuple[float, float] | None = None) -> NDArray:
    """Envelope do sinal, opcionalmente restrito a uma banda.

    Sem banda, usa a transformada de Hilbert do sinal inteiro.
    """
    if band_hz is None:
        return np.abs(signal.hilbert(np.asarray(x, dtype=float).ravel()))
    return np.abs(bandpass_analytic(x, fs_hz, *band_hz))


@dataclass(frozen=True)
class EnvelopeSpectrum:
    """Espectro de envelope em hertz, ou em ordens (ver `envelope_order_spectrum`).

    ``magnitude`` é ``|rfft(env·hann)|·2/n``: uma raia de amplitude ``A`` sai com ``A/2``,
    porque o ganho coerente da Hann não é compensado. Serve para argmax e razões, que é
    como o pacote a usa, e não como amplitude calibrada.
    """

    freqs_hz: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    fs_hz: float
    band_hz: tuple[float, float] | None
    folds_above_nyquist: bool = True
    """O conteúdo acima de ``fs_hz/2`` reaparece dobrado no espectro?

    Verdadeiro para envelope calculado sobre amostras uniformes (`envelope_spectrum`):
    os harmônicos acima de Nyquist estão lá, na posição do mapa de alias. Falso quando
    esse conteúdo foi removido antes, como no espectro de ordem com anti-alias angular.
    `score_fault` só procura harmônico dobrado quando este campo é verdadeiro.
    """

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
