"""Estatística cíclica de segunda ordem de amostras não uniformes (co-arranjo de diferenças).

O diagnóstico de rolamento não precisa da forma de onda. Bastam a correlação cíclica nas
frequências cíclicas da falha e, por ela, o espectro do envelope ao quadrado de uma banda.

    R(τ, α) = (1/c_τ) Σ_n x[n] x[n−τ] m[n] m[n−τ] e^{−j2π α n/f_s}

- ``m`` é a máscara dos instantes amostrados na grade da taxa cheia.
- ``c_τ`` é o número de pares (n, n−τ) que existem na máscara.
- **Decimação uniforme por M:** só há atrasos múltiplos de M, e o eixo da portadora dobra.
- **Amostradores coprimos** (passos M e N, com mdc(M, N) = 1, rodando sem parar):
  - todo atraso inteiro aparece, pelo teorema de Bézout, com cerca de T/(MN) pares;
  - o eixo da portadora fica sem dobra até f_s/2, ao custo de mais variância por atraso.

O espectro do envelope ao quadrado de uma banda b sai da correlação cíclica:

    SES_b(α) ≈ Σ_τ r_b(τ) R(τ, α)

- ``r_b`` é a autocorrelação de uma banda ideal.
- A aproximação vale para α muito menor que a largura da banda, e para bandas mais largas que
  f_s/L, onde L é o maior atraso estimado.
- Para x real, R(−τ, α) = e^{j2π α τ/f_s} R(τ, α).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_mask(n: int, pattern: str, m: int = 1, n2: int | None = None) -> np.ndarray:
    """Máscara booleana dos instantes amostrados na grade da taxa cheia.

    ``full``: todos; ``uniform``: passo ``m``; ``coprime``: união dos passos ``m`` e ``n2``.
    """
    mask = np.zeros(n, dtype=bool)
    if pattern == "full":
        mask[:] = True
    elif pattern == "uniform":
        mask[::m] = True
    elif pattern == "coprime":
        if n2 is None or np.gcd(m, n2) != 1:
            raise ValueError("coprime exige dois passos coprimos")
        mask[::m] = True
        mask[::n2] = True
    else:
        raise ValueError(f"padrão desconhecido: {pattern}")
    return mask


@dataclass(frozen=True)
class CyclicCorrelation:
    """R(τ, α) para τ = 0..L e α na grade da FFT do registro, até ``alpha_max_hz``."""

    r: np.ndarray  # (L+1, K) complexo
    lags: np.ndarray  # (L+1,)
    alphas_hz: np.ndarray  # (K,)
    counts: np.ndarray  # (L+1,) pares por atraso
    fs_hz: float


def cyclic_autocorrelation(
    x: np.ndarray, mask: np.ndarray, max_lag: int, fs_hz: float, alpha_max_hz: float
) -> CyclicCorrelation:
    """Estima R(τ, α) usando só os pares de instantes amostrados (ver o docstring do módulo).

    Um atraso sem nenhum par fica com R = NaN.
    """
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x.shape != mask.shape:
        raise ValueError("x e mask precisam ter o mesmo tamanho")
    n = x.size
    xm = np.where(mask, x, 0.0)
    alphas = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    keep = alphas <= alpha_max_hz
    out = np.full((max_lag + 1, int(keep.sum())), np.nan, dtype=complex)
    counts = np.zeros(max_lag + 1, dtype=int)
    for tau in range(max_lag + 1):
        both = mask[tau:] & mask[: n - tau]
        c = int(both.sum())
        counts[tau] = c
        if c == 0:
            continue
        p = np.zeros(n)
        p[tau:] = xm[tau:] * xm[: n - tau]  # nulo onde algum dos dois não foi amostrado
        out[tau] = np.fft.rfft(p)[keep] / c
    return CyclicCorrelation(out, np.arange(max_lag + 1), alphas[keep], counts, fs_hz)


def band_lag_weights(lags: np.ndarray, fs_hz: float, f_lo_hz: float, f_hi_hz: float) -> np.ndarray:
    """Autocorrelação de uma banda ideal real [f_lo, f_hi] nos atrasos, igual a 1 em τ = 0."""
    tau = np.asarray(lags, dtype=float) / fs_hz
    bw = f_hi_hz - f_lo_hz
    fc = 0.5 * (f_lo_hz + f_hi_hz)
    return np.sinc(bw * tau) * np.cos(2.0 * np.pi * fc * tau)


def band_ses(cc: CyclicCorrelation, f_lo_hz: float, f_hi_hz: float) -> np.ndarray:
    """Espectro do envelope ao quadrado da banda, em módulo, pela correlação cíclica (α ≥ 0).

    Usa os atrasos de −L a L; os negativos saem de R(−τ, α) = e^{j2π α τ/f_s} R(τ, α).
    Atrasos sem par entram com peso zero.
    """
    w = band_lag_weights(cc.lags, cc.fs_hz, f_lo_hz, f_hi_hz)
    r = np.nan_to_num(cc.r, nan=0.0)
    phase = np.exp(2j * np.pi * np.outer(cc.lags, cc.alphas_hz) / cc.fs_hz)
    # τ = 0 uma vez; τ > 0 duas (o próprio e o simétrico negativo)
    total = w[0] * r[0] + np.sum(w[1:, None] * (r[1:] + phase[1:] * r[1:]), axis=0)
    return np.abs(total)
