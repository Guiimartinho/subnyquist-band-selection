"""Estimação de tom e resolução da ordem do alias por múltiplas taxas.

A ideia central do projeto, em forma operacional: a frequência aliasada sozinha
não determina ``f0``, mas o **padrão de dobra sob taxas distintas** determina.
Duas taxas ``fs1`` e ``fs2`` resolvem sem ambiguidade até
``lcm(fs1, fs2) / 2`` — que é o análogo, para sinal real, do Teorema Chinês do
Resto usado em radar com PRF escalonada.

A formulação aqui é de **teste de hipóteses sobre um conjunto discreto**, e não
de inversão. Isso é deliberado: é o que permite injetar o prior geométrico
(as raias que a máquina pode produzir) sem mudar o algoritmo.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray
from snq_core import alias_frequency_hz


def estimate_tone_hz(x: ArrayLike, fs_hz: float) -> float:
    """Frequência do tom dominante, por FFT com janela de Hann e interpolação parabólica.

    A interpolação de três pontos sobre a magnitude tira o estimador da grade da
    FFT; sem ela o erro fica travado em ``fs/(2N)`` e domina tudo o mais que este
    experimento quer medir.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    sp = np.abs(np.fft.rfft(x * np.hanning(n)))
    sp[0] = 0.0  # ignora DC
    k = int(np.argmax(sp))
    if k <= 0 or k >= sp.size - 1:
        return float(k * fs_hz / n)
    a, b, c = float(sp[k - 1]), float(sp[k]), float(sp[k + 1])
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    return float((k + delta) * fs_hz / n)


def unambiguous_range_hz(fs_hz: Sequence[float], quantum_hz: float = 1.0) -> float:
    """Maior ``f0`` resolvível sem ambiguidade por um conjunto de taxas.

    Vale ``lcm(fs) / 2``. As taxas são reduzidas a inteiros na unidade
    ``quantum_hz`` antes do lcm — escolha ``quantum_hz`` como o maior divisor
    comum pretendido do projeto de taxas.

    >>> unambiguous_range_hz([40e3, 42e3], quantum_hz=2e3)
    420000.0
    """
    units = [int(round(f / quantum_hz)) for f in fs_hz]
    lcm = units[0]
    for u in units[1:]:
        lcm = lcm * u // gcd(lcm, u)
    return float(lcm * quantum_hz / 2.0)


def alias_candidates_hz(f_alias_hz: float, fs_hz: float, f_max_hz: float) -> NDArray[np.float64]:
    """Todas as ``f0`` em ``[0, f_max]`` que se dobrariam para ``f_alias_hz`` a ``fs_hz``.

    São ``f0 = k·fs ± f_alias``. O conjunto é pequeno — cerca de
    ``2·f_max/fs`` elementos — e é ele que a segunda taxa desambigua.

    >>> len(alias_candidates_hz(10e3, 40e3, 100e3))
    5
    """
    ks = np.arange(0, int(np.ceil((f_max_hz + f_alias_hz) / fs_hz)) + 1)
    cand = np.concatenate([ks * fs_hz - f_alias_hz, ks * fs_hz + f_alias_hz])
    cand = cand[(cand >= 0.0) & (cand <= f_max_hz)]
    return np.unique(np.round(cand, 6))


def resolve_multirate_hz(
    alias_measured_hz: Sequence[float],
    fs_hz: Sequence[float],
    candidates_hz: ArrayLike,
) -> tuple[float, float]:
    """Escolhe a candidata cujo padrão de dobra melhor explica os aliases medidos.

    Para cada candidata ``c`` prevê-se ``alias(c, fs_i)`` e compara-se com o
    medido; o custo é o **pior** desvio entre as taxas (norma infinito), que é
    mais seletivo que a média quando uma das taxas está corrompida.

    Retorna ``(f0_estimado, residuo_hz)``. O resíduo é a métrica de confiança:
    resíduo grande significa que nenhuma candidata explica as observações, e
    nesse caso a resposta deve ser rejeitada em vez de aceita.
    """
    cand = np.asarray(candidates_hz, dtype=float).ravel()
    if cand.size == 0:
        raise ValueError("conjunto de candidatas vazio")

    # (n_candidatas, n_taxas)
    pred = np.stack([alias_frequency_hz(cand, fs) for fs in fs_hz], axis=1)
    meas = np.asarray(alias_measured_hz, dtype=float)[None, :]
    cost = np.max(np.abs(pred - meas), axis=1)
    i = int(np.argmin(cost))
    return float(cand[i]), float(cost[i])


def prior_grid_hz(f_lo_hz: float, f_hi_hz: float, spacing_hz: float) -> NDArray[np.float64]:
    """Conjunto discreto de frequências admissíveis — o prior físico, em 1D.

    No caso real este conjunto vem da geometria (ressonâncias conhecidas,
    harmônicos da rotação); ver ``snq_core.bearing.candidate_lines_hz``. Aqui ele
    é uma grade regular, para que o ganho do prior seja medido em função apenas
    da sua **cardinalidade**, sem confundir com a estrutura particular do
    conjunto.
    """
    return np.arange(f_lo_hz, f_hi_hz + spacing_hz, spacing_hz, dtype=float)
