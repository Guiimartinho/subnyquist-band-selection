"""Seletores de banda robustos a impulsos, na mesma partição do kurtograma rápido deste pacote.

- **Infograma** (Antoni, MSSP 74:73-94, 2016, seções 3 e 4, conferido no PDF em 2026-11-12).
  - O fluxo de energia da banda é ε(n) = |c(n)|², amostrado na taxa crítica.
  - Negentropia no tempo (eq. 10-11): ΔI_ε = <(ε²/<ε²>)·ln(ε²/<ε²>)>.
  - Negentropia no espectro do envelope ao quadrado (eq. 12): a mesma expressão sobre |E(α)|²,
    onde E é a DFT de ε, sem a componente α = 0, como recomenda a seção 4.2.
  - Média: ΔI_1/2 = (ΔI_ε + ΔI_E)/2 (eq. 13).
  - Pela seção 5.1.1, o infograma SES é insensível a ruído impulsivo; o SE e o kurtograma não.
  - Sob ruído gaussiano estacionário, os valores esperados são 1,1525 e 0,4228 (seção 4.3). Os
    testes conferem os dois.
- **Gini espectral** (Hebda-Sobkowicz, Zimroz e Wyłomańska, Appl. Sci. 10:2657, 2020, eq. 3):
  índice de Gini do envelope |c(n)| da banda, no lugar da curtose.

Desvios declarados.
- Os filtros são ideais por FFT, na partição de `fast_kurtogram`, e não o STFT da seção 4.2 do
  Antoni.
- Para o sinal analítico ideal de largura Δf, a taxa crítica é Δf.
- O seletor por variância condicional (CVB) não está aqui: a definição completa está em outro
  artigo (`hebdasobkowicz2020condvar`), que não foi lido.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import _band_analytic_from_spectrum
from snq_recon.kurtogram import KurtogramResult, _edge_guard, _levels, spectral_kurtosis

STATISTICS = ("sk", "se_negentropy", "ses_negentropy", "avg_negentropy", "gini")


def _plogp(p: NDArray[np.float64]) -> NDArray[np.float64]:
    safe = np.where(p > 0.0, p, 1.0)
    return np.where(p > 0.0, p * np.log(safe), 0.0)


def band_statistic(
    z: NDArray[np.complex128], fs_hz: float, bandwidth_hz: float, statistic: str
) -> float:
    """Estatística de uma banda a partir do seu sinal analítico ``z`` (já sem as bordas)."""
    if statistic == "sk":
        return float(spectral_kurtosis(z))
    if statistic not in STATISTICS:
        raise ValueError(f"estatística desconhecida: {statistic}")
    c = z[:: max(1, int(round(fs_hz / bandwidth_hz)))]  # taxa crítica do filtro ideal
    if c.size < 16:
        return -np.inf
    if statistic == "gini":
        a = np.sort(np.abs(c))
        s = float(a.sum())
        if s <= 0.0:
            return -np.inf
        n = a.size
        i = np.arange(1, n + 1)
        return float(1.0 - 2.0 * np.sum((a / s) * (n - i + 0.5) / n))
    e = np.abs(c) ** 2
    m2 = float(np.mean(e**2))
    if m2 <= 0.0:
        return -np.inf
    d_se = float(np.mean(_plogp(e**2 / m2)))
    spec = np.abs(np.fft.fft(e)[1:]) ** 2  # sem alpha = 0 (seção 4.2)
    m_spec = float(np.mean(spec))
    d_ses = float(np.mean(_plogp(spec / m_spec))) if m_spec > 0.0 else -np.inf
    if statistic == "se_negentropy":
        return d_se
    if statistic == "ses_negentropy":
        return d_ses
    return 0.5 * (d_se + d_ses)


def fast_band_selector(
    x: ArrayLike,
    fs_hz: float,
    statistic: str = "ses_negentropy",
    max_level: int = 6,
    f_min_hz: float = 0.0,
    f_max_hz: float | None = None,
    min_energy_frac: float = 1e-4,
) -> KurtogramResult:
    """A mesma varredura de `fast_kurtogram`, com outra estatística por banda.

    Com ``statistic="sk"`` reproduz `fast_kurtogram`. O campo ``kurtosis`` do resultado guarda o
    valor da estatística escolhida.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    nyq = fs_hz / 2.0
    hi = min(f_max_hz, nyq) if f_max_hz is not None else nyq
    if hi <= f_min_hz:
        raise ValueError("faixa de busca vazia")
    grid: list[tuple[float, float, float, float]] = []
    best = (f_min_hz, hi, -np.inf, 0.0)
    total_energy = float(np.sum(x**2))
    mirrored = np.concatenate([x, x[::-1]])
    spec = np.fft.fft(mirrored)
    freqs = np.fft.fftfreq(mirrored.size, d=1.0 / fs_hz)
    for level in _levels(max_level):
        n_bands = int(round(2.0**level))
        edges = np.linspace(f_min_hz, hi, n_bands + 1)
        for lo, up in zip(edges[:-1], edges[1:], strict=True):
            if up - lo <= 0.0:
                continue
            z = _band_analytic_from_spectrum(spec, freqs, float(lo), float(up))[:n]
            if (
                total_energy > 0.0
                and float(np.sum(np.abs(z) ** 2)) < min_energy_frac * total_energy
            ):
                continue
            guard = _edge_guard(n, fs_hz, float(up - lo))
            k = band_statistic(z[guard : n - guard], fs_hz, float(up - lo), statistic)
            grid.append((level, float(lo), float(up), k))
            if k > best[2]:
                best = (float(lo), float(up), k, level)
    return KurtogramResult(
        f_lo_hz=best[0], f_hi_hz=best[1], kurtosis=best[2], level=best[3], grid=grid
    )
