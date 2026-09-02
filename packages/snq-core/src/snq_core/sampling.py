"""Matemática de amostragem estruturada.

Piso teórico do projeto: **Landau (1967)** — a taxa mínima de amostragem é a
medida do suporte espectral ocupado, e não o dobro da maior frequência presente.
A relação ``f_max = f_s/2`` é o corolário pessimista, válido só para sinal
arbitrário que ocupa toda a banda base.

Referências implementadas aqui:

- H. J. Landau, *Necessary density conditions for sampling and interpolation of
  certain entire functions*, Acta Mathematica 117:37-52, 1967.
- R. G. Vaughan, N. L. Scott, D. R. White, *The theory of bandpass sampling*,
  IEEE Trans. Signal Processing 39(9):1973-1984, 1991.

Convenção: todas as frequências em Hz, todos os tempos em segundos, e o nome do
parâmetro carrega a unidade (``fs_hz``, ``sigma_t_s``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Band = tuple[float, float]


# --------------------------------------------------------------------------
# Taxa mínima
# --------------------------------------------------------------------------
def landau_rate_hz(bands_hz: list[Band]) -> float:
    """Taxa de Landau para um sinal **real** ocupando ``bands_hz``.

    Para sinal real cada banda positiva tem a imagem negativa correspondente,
    então a medida total do suporte é o dobro da soma das larguras.

    >>> landau_rate_hz([(140e3, 160e3)])
    40000.0
    """
    if any(hi <= lo for lo, hi in bands_hz):
        raise ValueError("cada banda precisa satisfazer f_lo < f_hi")
    return 2.0 * float(sum(hi - lo for lo, hi in bands_hz))


def nyquist_rate_hz(bands_hz: list[Band]) -> float:
    """Taxa de Nyquist clássica: ``2 * f_max``. Existe aqui para comparação."""
    return 2.0 * float(max(hi for _, hi in bands_hz))


# --------------------------------------------------------------------------
# Zonas de Nyquist e dobra espectral
# --------------------------------------------------------------------------
def nyquist_zone(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.int_]:
    """Índice da zona de Nyquist que contém ``f_hz``. Zona ``k`` = ``[k·fs/2, (k+1)·fs/2]``.

    Zona 0 é a banda base. **Zona ímpar inverte o espectro** ao dobrar.
    """
    return np.floor(2.0 * np.asarray(f_hz, dtype=float) / fs_hz).astype(int)


def alias_frequency_hz(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.float64]:
    """Frequência aparente de ``f_hz`` após amostragem uniforme a ``fs_hz`` (sinal real).

    >>> float(alias_frequency_hz(150e3, 40e3))
    10000.0
    """
    r = np.mod(np.asarray(f_hz, dtype=float), fs_hz)
    out: NDArray[np.float64] = np.where(r > fs_hz / 2.0, fs_hz - r, r)
    return out


def alias_order(f_hz: ArrayLike, fs_hz: float) -> NDArray[np.int_]:
    """Ordem ``k`` da dobra: quantos múltiplos inteiros de ``fs`` foram removidos.

    É a incógnita que o projeto precisa identificar. Ela **não** é observável em
    uma única aquisição uniforme — apenas a frequência aliasada é. É recuperável
    a partir da fase relativa entre observações deslocadas (ver `multicoset`) ou
    de múltiplas taxas coprimas.
    """
    return np.rint(np.asarray(f_hz, dtype=float) / fs_hz).astype(int)


@dataclass(frozen=True)
class FoldedBand:
    """Resultado de dobrar uma banda para a base."""

    lo_hz: float
    hi_hz: float
    zone: int
    inverted: bool
    self_overlap: bool
    """``True`` se a banda cruza fronteira de zona e se dobra sobre si mesma —
    caso em que a informação é destruída de forma irrecuperável."""


def fold_band(f_lo_hz: float, f_hi_hz: float, fs_hz: float) -> FoldedBand:
    """Dobra ``[f_lo, f_hi]`` para a banda base sob amostragem a ``fs_hz``.

    Caso canônico do projeto — transdutor de emissão acústica ressonante:

    >>> fb = fold_band(140e3, 160e3, 40e3)
    >>> fb.lo_hz, fb.hi_hz, fb.inverted, fb.self_overlap
    (0.0, 20000.0, True, False)
    """
    if f_hi_hz <= f_lo_hz:
        raise ValueError("f_lo_hz precisa ser menor que f_hi_hz")

    k_lo = int(np.floor(2.0 * f_lo_hz / fs_hz))
    # A borda superior encosta na fronteira quando é múltiplo exato de fs/2;
    # nesse caso ela pertence à zona de baixo.
    k_hi_raw = 2.0 * f_hi_hz / fs_hz
    k_hi = int(np.ceil(k_hi_raw)) - 1 if float(k_hi_raw).is_integer() else int(np.floor(k_hi_raw))

    overlap = k_lo != k_hi
    inverted = (k_lo % 2) == 1

    a = float(alias_frequency_hz(f_lo_hz, fs_hz))
    b = float(alias_frequency_hz(f_hi_hz, fs_hz))
    lo, hi = (b, a) if inverted else (a, b)
    return FoldedBand(lo_hz=lo, hi_hz=hi, zone=k_lo, inverted=inverted, self_overlap=overlap)


# --------------------------------------------------------------------------
# Amostragem passa-faixa (Vaughan, Scott & White, 1991)
# --------------------------------------------------------------------------
def valid_bandpass_rates_hz(
    f_lo_hz: float, f_hi_hz: float, fs_max_hz: float | None = None
) -> list[tuple[float, float, int]]:
    """Intervalos de ``fs`` que amostram ``[f_lo, f_hi]`` sem sobreposição.

    Condição clássica, para ``n = 1 .. floor(f_hi / B)``::

        2·f_hi/n  <=  fs  <=  2·f_lo/(n-1)

    ``n = 1`` é o caso clássico (``fs >= 2·f_hi``, sem limite superior).
    ``n`` máximo dá a menor taxa possível.

    Retorna ``[(fs_min, fs_max, n)]`` em ordem decrescente de ``fs_min``.
    ``fs_max`` é ``inf`` para ``n = 1``. Se ``fs_max_hz`` for dado, intervalos
    inteiramente acima dele são descartados — é assim que se filtra pelo teto do
    conversor.

    >>> [(round(a), round(b), n) for a, b, n in
    ...  valid_bandpass_rates_hz(140e3, 160e3, fs_max_hz=250e3)][-1]
    (40000, 40000, 8)
    """
    if f_hi_hz <= f_lo_hz or f_lo_hz < 0:
        raise ValueError("banda inválida")
    bandwidth = f_hi_hz - f_lo_hz
    n_max = int(np.floor(f_hi_hz / bandwidth))

    out: list[tuple[float, float, int]] = []
    for n in range(1, n_max + 1):
        fs_min = 2.0 * f_hi_hz / n
        fs_max = float("inf") if n == 1 else 2.0 * f_lo_hz / (n - 1)
        if fs_max < fs_min:
            continue
        if fs_max_hz is not None and fs_min > fs_max_hz:
            continue
        out.append((fs_min, fs_max, n))
    return out


def min_bandpass_rate_hz(f_lo_hz: float, f_hi_hz: float) -> float:
    """Menor ``fs`` que amostra a banda sem sobreposição.

    Nunca é menor que a taxa de Landau; a diferença é o preço da amostragem
    uniforme em uma única fase.

    >>> min_bandpass_rate_hz(140e3, 160e3)
    40000.0
    """
    rates = valid_bandpass_rates_hz(f_lo_hz, f_hi_hz)
    return min(fs_min for fs_min, _, _ in rates)


# --------------------------------------------------------------------------
# Multi-coset: quantas fases são necessárias
# --------------------------------------------------------------------------
def multicoset_resolvable_orders(n_cosets: int) -> int:
    """Quantas ordens de alias distintas ``M`` cosets uniformes resolvem.

    Com deslocamentos ``delta_m = m·Ts/M``, a fase relativa entre canais
    distingue ``k mod M`` — extensão de banda de exatamente ``M×``.

    ``delta = Ts/2`` (M=2) distingue apenas a paridade de ``k``.
    Deslocamentos irracionais distinguem toda ordem sem ruído, mas o
    condicionamento degrada exponencialmente — ver `coset_matrix`.
    """
    if n_cosets < 1:
        raise ValueError("n_cosets >= 1")
    return n_cosets


def coset_matrix(offsets_s: ArrayLike, fs_hz: float, orders: ArrayLike) -> NDArray[np.complex128]:
    """Matriz que relaciona amplitudes por ordem de alias às observações deslocadas.

    Elemento ``(m, i) = exp(-j·2π·(f_a + k_i·fs)·delta_m)``, a menos do fator
    comum ``exp(-j·2π·f_a·delta_m)`` que é omitido por não depender de ``k``.
    Sobra a matriz de Vandermonde em ``exp(-j·2π·fs·delta_m)``.

    O número de condição desta matriz é o que decide a viabilidade prática:
    inversível não basta, é preciso ser bem condicionada sob 12 bits.
    """
    d = np.asarray(offsets_s, dtype=float)[:, None]
    k = np.asarray(orders, dtype=float)[None, :]
    out: NDArray[np.complex128] = np.exp(-2j * np.pi * fs_hz * k * d)
    return out


def uniform_coset_offsets_s(n_cosets: int, fs_hz: float) -> NDArray[np.float64]:
    """Deslocamentos ``m·Ts/M`` — a escolha que torna `coset_matrix` uma DFT.

    É o análogo 1D do microscanning: condicionamento ótimo, número de condição 1.
    """
    ts = 1.0 / fs_hz
    return np.arange(n_cosets, dtype=float) * ts / n_cosets
