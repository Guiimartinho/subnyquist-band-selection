"""Máscaras de amostragem periódicas sobre a grade de ticks, e o conjunto viável do E15.

O conjunto viável, como o README do E15 o define:

- **grade:** a taxa de aquisição do conjunto de dados é a grade de ticks;
- **padrão periódico** de ``L`` ticks com ``K`` posições ativas por período, taxa média ``K/L``
  da grade;
- **espaçamento mínimo** de ``g`` ticks entre posições ativas consecutivas, medido de forma
  **circular**: a última posição de um período e a primeira do seguinte também precisam estar a
  ``g`` ticks ou mais. Representa o tempo de conversão de um SAR único.

Realização no nRF54L15, como hipótese de arquitetura e não como medida: um TIMER em modo de
comparação com período de ``L`` ticks, um registrador CC por posição ativa, cada evento ligado
por DPPI à tarefa SAMPLE do SAADC (ver `timer_schedule`). **O número de registradores CC por
instância de TIMER do nRF54L15 não foi conferido** nesta implementação. ``K`` pequeno é escolha
conservadora, não garantia.

Peças, e nada além delas:

``FeasibleSet``                 o conjunto ``(L, K, g)`` e a checagem de viabilidade;
``project_to_feasible``         projeção exata, por programação dinâmica, de um vetor de escores;
``fold_project``                projeção exata de uma máscara de horizonte ``M·L`` sem restrição;
``uniform_positions``           o padrão uniforme, que é a decimação por ``L/K``;
``random_feasible_positions``   sorteio uniforme sobre os padrões viáveis;
``designed_positions``          multi-coset projetado por critério clássico, fixo e sem dados;
``support_designed_positions``  multi-coset para suporte espectral conhecido
                                (`venkataramani2001optimal`), com ``band_support_cells``;
``FixedMask``                   qualquer padrão fixo, como módulo;
``GumbelTopKMask``              máscara aprendida: top-K de Gumbel com straight-through
                                (`huijben2020dps`, `vangorp2021adps`), com diagnósticos;
``LoupeMask``                   LOUPE (`bahadir2020loupe`), sem restrição, e a projeção dele;
``BlockAdaptiveMask``           política adaptativa por blocos, inspirada no ADPS.

Todos os módulos de máscara recebem o sinal ``(B, N)`` e devolvem a máscara no tempo ``(B, N)``,
com ``N`` múltiplo de ``L``. O produto ``x·m`` é o sinal mascarado e preenchido com zeros que o
decodificador recebe.
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
# conjunto viável
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FeasibleSet:
    """Padrões periódicos de ``period_ticks`` ticks com ``active_per_period`` posições ativas.

    ``min_gap_ticks = 1`` não restringe nada além de posições distintas. O conjunto é vazio
    quando ``K·g > L``, e isso é erro de construção, não um conjunto vazio silencioso.
    """

    period_ticks: int
    active_per_period: int
    min_gap_ticks: int = 1

    def __post_init__(self) -> None:
        period, k, g = self.period_ticks, self.active_per_period, self.min_gap_ticks
        if period < 1 or k < 1 or g < 1:
            raise ValueError("period_ticks, active_per_period e min_gap_ticks precisam ser >= 1")
        if k > period:
            raise ValueError(f"{k} posicoes nao cabem num periodo de {period} ticks")
        if k * g > period:
            raise ValueError(
                f"conjunto vazio: {k} posicoes com espacamento >= {g} exigem periodo >= {k * g}, "
                f"e o periodo e {period}"
            )

    @classmethod
    def for_budget(cls, budget: int, active_per_period: int, min_gap_ticks: int = 1) -> FeasibleSet:
        """Conjunto com ``L = K·budget``: taxa média exatamente ``1/budget`` da grade."""
        return cls(int(budget) * int(active_per_period), int(active_per_period), int(min_gap_ticks))

    @property
    def rate_fraction(self) -> float:
        """Taxa média como fração da grade, ``K/L``."""
        return self.active_per_period / self.period_ticks

    def violations(self, positions: ArrayLike) -> int:
        """Quantos intervalos circulares entre posições consecutivas ficam abaixo de ``g``."""
        p = _as_positions(positions, self.period_ticks)
        return int(np.count_nonzero(circular_gaps(p, self.period_ticks) < self.min_gap_ticks))

    def is_feasible(self, positions: ArrayLike) -> bool:
        """``K`` posições distintas dentro do período e espaçamento circular >= ``g``."""
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
        raise ValueError("padrao sem posicoes ativas")
    if p[0] < 0 or p[-1] >= period_ticks:
        raise ValueError(f"posicoes fora de [0, {period_ticks})")
    return p


def circular_gaps(positions: ArrayLike, period_ticks: int) -> NDArray[np.int64]:
    """Intervalos entre posições ativas consecutivas, com a volta do período.

    >>> circular_gaps([0, 3, 5], 8).tolist()
    [3, 2, 3]
    """
    p = _as_positions(positions, period_ticks)
    return np.diff(np.concatenate([p, [p[0] + period_ticks]])).astype(np.int64)


def positions_to_mask(positions: ArrayLike, period_ticks: int) -> NDArray[np.float32]:
    """Máscara binária de um período a partir das posições ativas."""
    m = np.zeros(int(period_ticks), dtype=np.float32)
    m[_as_positions(positions, period_ticks)] = 1.0
    return m


# ----------------------------------------------------------------------
# projeção exata
# ----------------------------------------------------------------------
def _best_spaced_subset(w: NDArray[np.float64], k: int, gap: int) -> tuple[float, list[int]] | None:
    """Maior soma de ``k`` entradas de ``w`` com índices espaçados de ``gap`` ou mais (linear).

    Programação dinâmica ``O(len(w)·k)``. Devolve ``None`` se não couberem ``k`` índices.
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
        prev = f[max(i - gap, 0), :-1] + w[i - 1]  # escolher o elemento i-1
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
    """Projeção **exata** de um vetor de escores sobre o conjunto viável.

    Devolve as ``K`` posições de maior soma de escores com espaçamento circular >= ``g``. Para
    máscaras binárias de cardinalidade fixa, ``||m - w||² = K - 2<m, w> + ||w||²``, então
    maximizar a soma é o mesmo que minimizar a distância euclidiana: é a projeção de ``w``.

    O laço externo fixa a menor posição escolhida ``s``; as outras ``K - 1`` ficam em
    ``[s + g, min(L - 1, s + L - g)]``, o que garante o espaçamento da volta. Custo
    ``O(L²·K)``. Em empate de soma, vence o primeiro ``s`` encontrado (determinístico).
    """
    w = np.asarray(scores, dtype=np.float64).ravel()
    period, k, gap = fs.period_ticks, fs.active_per_period, fs.min_gap_ticks
    if w.size != period:
        raise ValueError(f"{w.size} escores para um periodo de {period} ticks")
    if not np.all(np.isfinite(w)):
        raise ValueError("escores nao finitos")
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
    if best is None:  # inalcançável com FeasibleSet válido; mantido como trava
        raise ValueError("nenhum padrao viavel encontrado")
    out = np.asarray(sorted(best), dtype=np.int64)
    assert fs.is_feasible(out)
    return out


def fold_project(logits: ArrayLike, k_horizon: int, fs: FeasibleSet) -> NDArray[np.int64]:
    """Projeção exata, no conjunto viável, de uma máscara aprendida **sem** a restrição.

    A máscara livre tem horizonte ``P = M·L`` e ``k_horizon`` posições ativas, sem espaçamento;
    a versão dura determinística é o top-``k_horizon`` dos logits. Entre as máscaras periódicas
    de período ``L``, ``||m_P - tile(m_L)||² = cte - 2·Σ_l c_l·m_L(l)``, com ``c_l`` o número de
    vezes que a posição ``l`` do período aparece nas ``M`` cópias: maximizar ``Σ c_l·m_L(l)`` com
    `project_to_feasible` é a projeção euclidiana exata. Empate de contagem: média dos logits
    dobrados, com peso pequeno demais para inverter uma contagem.
    """
    w = np.asarray(logits, dtype=np.float64).ravel()
    period = fs.period_ticks
    if w.size % period:
        raise ValueError(f"horizonte de {w.size} ticks nao e multiplo do periodo {period}")
    if not 1 <= int(k_horizon) <= w.size:
        raise ValueError("k_horizon fora do horizonte")
    hard = np.zeros(w.size)
    hard[np.argsort(-w, kind="stable")[: int(k_horizon)]] = 1.0
    counts = hard.reshape(-1, period).sum(axis=0)
    folded = w.reshape(-1, period).mean(axis=0)
    span = float(folded.max() - folded.min())
    tie = 1e-6 * (folded - folded.min()) / span if span > 0.0 else np.zeros(period)
    return project_to_feasible(counts + tie, fs)


# ----------------------------------------------------------------------
# padrões fixos
# ----------------------------------------------------------------------
def uniform_positions(fs: FeasibleSet) -> NDArray[np.int64]:
    """Padrão uniforme: uma conversão a cada ``L/K`` ticks, com fase zero.

    Na grade, é exatamente ``snq_data.undersample(x, L/K)``.
    """
    period, k = fs.period_ticks, fs.active_per_period
    if period % k:
        raise ValueError(f"uniforme exige L multiplo de K (L={period}, K={k})")
    step = period // k
    if step < fs.min_gap_ticks:
        raise ValueError(f"uniforme com passo {step} viola o espacamento minimo {fs.min_gap_ticks}")
    return np.arange(k, dtype=np.int64) * step


def random_feasible_positions(fs: FeasibleSet, rng: np.random.Generator) -> NDArray[np.int64]:
    """Padrão viável sorteado **uniformemente** entre todos os padrões viáveis.

    Os ``K`` intervalos circulares são ``g + e_i`` com ``Σ e_i = L - K·g``; a composição
    ``e`` é sorteada por estrelas e barras e o padrão é girado por um deslocamento uniforme.
    Cada padrão aparece ``K`` vezes (uma por escolha de qual intervalo é o primeiro), todas com
    a mesma probabilidade, então o sorteio é uniforme sobre os padrões.
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
    """``|M_l|``, ``l = 0..L-1``: peso da réplica deslocada de ``l·fs/L`` no sinal com zeros.

    Com ``z = x·m`` e ``m`` periódica, ``Z(f) = Σ_l M_l·X(f - l·fs/L)`` e
    ``M_l = (1/L)·Σ_p exp(-j2π·l·p/L)``. ``M_0 = K/L``. O uniforme concentra tudo em ``K``
    réplicas de peso cheio (as da decimação); um padrão não uniforme espalha o peso.
    """
    p = _as_positions(positions, period_ticks)
    lags = np.arange(int(period_ticks))
    return np.abs(np.exp(-2j * np.pi * np.outer(lags, p) / period_ticks).sum(axis=1)) / period_ticks


def _candidate_patterns(fs: FeasibleSet, max_enumeration: int, seed: int) -> NDArray[np.int64]:
    """Padrões viáveis com a primeira posição em zero (o critério é invariante a rotação)."""
    period, k = fs.period_ticks, fs.active_per_period
    n_comb = math.comb(period - 1, k - 1)
    if n_comb <= max_enumeration:
        rest = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(1, period), k - 1)),
            dtype=np.int64,
            count=n_comb * (k - 1),
        ).reshape(n_comb, k - 1)
    else:  # busca aleatória determinística quando a enumeração não cabe
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
    """Multi-coset **projetado**: padrão fixo, escolhido por critério clássico e sem dados.

    ``min_peak_alias`` minimiza o maior peso de réplica fora da origem,
    ``max_{l≠0} |M_l|`` (`alias_weights`): espalha a energia dobrada em vez de concentrá-la,
    como faz o uniforme. Desempate pela cobertura do co-arranjo (número de diferenças
    circulares distintas) e depois pela ordem lexicográfica. ``coarray`` inverte a prioridade
    dos dois critérios.

    Não é o par coprimo de `zhao2018coprime`: com taxa exatamente ``1/B`` o par coprimo exige
    ``4B - 2`` eventos por período e intervalos de 1 tick, e não cabe no mesmo ``(L, K, g)``.
    """
    if criterion not in ("min_peak_alias", "coarray"):
        raise ValueError(f"criterio desconhecido: {criterion}")
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
    # np.lexsort: a última chave é a primária; as colunas do padrão desempatam por último
    order = np.lexsort(tuple(cands[:, j] for j in range(k - 1, -1, -1)) + keys)
    out = cands[order[0]].astype(np.int64)
    assert fs.is_feasible(out)
    return out


def band_support_cells(
    band_hz: tuple[float, float], fs_hz: float, period_ticks: int, n_grid: int = 64
) -> list[tuple[float, tuple[int, ...]]]:
    """Células espectrais ativas do modelo multi-coset para uma banda real conhecida.

    ``[0, fs)`` é dividido em ``L`` células de largura ``fs/L``. Para cada deslocamento ``f``
    dentro de uma célula (grade de ``n_grid`` pontos), a célula ``l`` está ativa quando
    ``f + l·fs/L`` cai na banda ``[lo, hi)`` ou no espelho ``[fs - hi, fs - lo)``: o sinal é
    real. Devolve ``(peso, S)`` por conjunto distinto, com peso = fração da grade.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    if not 0.0 <= lo < hi <= fs_hz / 2.0:
        raise ValueError("banda precisa estar em [0, fs/2]")
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
    """Encolhe a banda em torno do centro até que nenhum conjunto de células passe de ``K``.

    Com ``K`` cosets, cada deslocamento ``f`` resolve no máximo ``K`` células (Landau). Uma
    banda mais larga que isso não tem reconstrução multi-coset exata nessa taxa; o desenho
    passa a mirar a parte central dela, e quem chama registra a banda encolhida.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    for _ in range(200):
        band = (center - half, center + half)
        sets = band_support_cells(band, fs_hz, period_ticks, n_grid)
        if sets and max(len(s) for _w, s in sets) <= k:
            return band
        half *= 0.97
    raise ValueError("banda nao coube em K celulas")


def support_designed_positions(
    fs: FeasibleSet,
    supports: list[tuple[float, tuple[int, ...]]],
    *,
    max_enumeration: int = 3_000_000,
    seed: int = 0,
) -> NDArray[np.int64]:
    """Multi-coset desenhado para um **suporte espectral conhecido**.

    Na célula ``f``, as amostras dos cosets são ``y(f) = D(f)·A_S·x_S(f)``, com
    ``A_S[k, s] = exp(j2π·c_k·s/L)`` e ``D(f)`` unitária. O padrão escolhido minimiza a
    amplificação de ruído da reconstrução por mínimos quadrados, ``Σ_S w_S·||A_S^+||_F²``,
    média sobre os conjuntos de células ativas (`band_support_cells`). Padrão que deixa
    algum ``A_S`` singular fica de fora. Busca exaustiva no conjunto viável. Empate: o menor
    alias de pico (`alias_weights`), escolha deste projeto.

    O critério segue o espírito de `venkataramani2001optimal`, que otimiza o padrão para um
    suporte conhecido por cotas de erro da reconstrução. A forma exata da cota e a seleção
    sequencial do artigo **não foram conferidas** (o resumo não estava disponível nas bases
    consultadas em 2026-11-11). A amplificação de ruído é invariante à rotação do padrão.
    """
    period, k = fs.period_ticks, fs.active_per_period
    if not supports:
        raise ValueError("suporte vazio")
    if max(len(s) for _w, s in supports) > k:
        raise ValueError(f"suporte com mais de K={k} celulas: encolha a banda antes")
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
        raise ValueError("nenhum padrao viavel reconstroi esse suporte")
    # Desempate entre padroes igualmente bons para o suporte: menor alias de pico, que concentra
    # menos a energia de fora do suporte modelado; depois, ordem lexicografica.
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
    """``Σ_S w_S·||A_S^+||_F²`` de um padrão (infinito se algum ``A_S`` for singular)."""
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
    """Como o padrão seria agendado num TIMER com DPPI, e se as contagens são inteiras.

    O padrão é girado para ter uma posição no tick 0. Então ``K`` registradores bastam: os
    ``K - 1`` primeiros nas posições restantes e o último no fim do período, com atalho
    COMPARE→CLEAR, de modo que o evento do fim do período é a posição 0 do período seguinte.
    As contagens são ``ticks·timer_clock_hz/tick_hz`` e só são realizáveis se inteiras.
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
# módulos
# ----------------------------------------------------------------------
def _tile(period_mask: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Repete ``(B, L)`` ou ``(L,)`` até ``n_samples``, exigindo ``n_samples`` múltiplo de ``L``."""
    period = period_mask.shape[-1]
    if n_samples % period:
        raise ValueError(f"N={n_samples} nao e multiplo do periodo L={period}")
    reps = n_samples // period
    return period_mask.repeat(1, reps) if period_mask.dim() == 2 else period_mask.repeat(reps)


def spacing_penalty(soft: torch.Tensor, min_gap_ticks: int) -> torch.Tensor:
    """Pares (circulares) de posições a menos de ``g`` ticks, ponderados pela máscara relaxada.

    ``Σ_{d=1}^{g-1} Σ_l m_l·m_{(l+d) mod L}``, média no lote. Zero quando ``g = 1``.
    """
    total = soft.new_zeros(())
    for d in range(1, int(min_gap_ticks)):
        total = total + (soft * torch.roll(soft, shifts=-d, dims=-1)).sum(-1).mean()
    return total


def _gumbel_topk(
    logits: torch.Tensor, k: int, tau: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K de Gumbel: máscara dura e relaxação, ``(B, L)`` cada.

    ``K`` sorteios sem reposição: em cada um, ``softmax((w + G)/τ)`` sobre as posições ainda
    livres, e a posição de maior escore perturbado é bloqueada para os seguintes. A soma das
    ``K`` softmaxes é a relaxação; a soma das ``K`` posições escolhidas é a máscara dura. É a
    construção de amostragem probabilística com straight-through usada em `vangorp2021adps`.
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
    """Padrão fixo (uniforme, aleatório semeado, projetado ou final de um aprendido)."""

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
    """Máscara aprendida: top-K de Gumbel sobre os logits de um período do conjunto ``fs``.

    Treino: máscara dura no forward e gradiente pela relaxação (straight-through), como em
    `huijben2020dps` e `vangorp2021adps`, ou relaxação pura (``straight_through=False``). O
    espaçamento mínimo de ``fs`` entra por `spacing_penalty` sobre a relaxação.

    Avaliação: a máscara é a **projeção exata** dos logits sobre ``fs``, determinística.
    ``freeze()`` fixa essa projeção.

    Com ``fs = FeasibleSet(M·L, M·K, 1)`` a mesma classe é a máscara **sem a restrição**:
    horizonte de ``M`` períodos, contagem só no horizonte e nenhum espaçamento. ``audit_gap``
    conta, no treino, as máscaras duras que violariam o espaçamento alvo (``audit_*``), e
    `diagnostics` diz se a máscara saiu do ruído: margem entre o ``K``-ésimo e o
    ``(K+1)``-ésimo logit contra o desvio do ruído de Gumbel (π/√6 ≈ 1,28), e posições finais
    contra a projeção dos logits iniciais, que a semente reproduz.
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
            raise ValueError(f"init desconhecido: {init}")
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
        """Logits finais, margem do top-K, posições iniciais e finais, violações auditadas."""
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
            # > 1: o top-K deterministico resiste ao ruido de Gumbel; << 1: a mascara ainda e ruido
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
        """Projeção exata dos logits atuais, refeita só quando os logits mudam."""
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
        """Fixa a projeção exata atual e congela os logits."""
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
    """Intervalos circulares entre posições consecutivas menores que ``gap``."""
    pos = np.asarray(positions, dtype=np.int64).ravel()
    if pos.size < 2 or gap <= 1:
        return 0
    return int(np.count_nonzero(circular_gaps(pos, period_ticks) < gap))


def _loupe_rescale(p: torch.Tensor, rate: float) -> torch.Tensor:
    """Renormalização do LOUPE: média do mapa de probabilidade igual à taxa alvo.

    Se a média ``x̄`` passa da taxa ``α``, escala por ``α/x̄``; senão, escala ``1 - p`` por
    ``(1-α)/(1-x̄)``. As probabilidades ficam em ``[0, 1]`` nos dois ramos.
    """
    xbar = p.mean()
    r = rate / xbar
    beta = (1.0 - rate) / (1.0 - xbar).clamp_min(1e-12)
    le = (r <= 1.0).to(p.dtype)
    return le * p * r + (1.0 - le) * (1.0 - (1.0 - p) * beta)


class LoupeMask(nn.Module):
    """LOUPE (`bahadir2020loupe`) sobre a grade de ticks, **sem** a restrição do conjunto viável.

    Um logit por posição de um horizonte de ``n_positions`` ticks, repetido periodicamente no
    tempo. Mapa de probabilidade ``p = σ(a·w)``, renormalizado para a taxa alvo. No treino,
    cada tick é um Bernoulli independente: forward **duro**, ``1[u < p]``, e gradiente pela
    relaxação ``σ(b·(p - u))`` (straight-through). A contagem por período não é fixa (só a
    média é ``α``) e não há espaçamento mínimo.

    Desvio declarado do LOUPE original, que usa a relaxação também no forward: com ela, a
    "taxa" vista no treino era 1,5 a 2,4 vezes a taxa alvo (revisão de 2026-11-11), e o modelo
    treinava com mais amostras do que o orçamento e era avaliado com menos.

    Avaliação "livre": uma realização de Bernoulli por tick, com semente fixa (``eval_seed``).
    **Não é realizável** pelo TIMER (contagem e posições variam de período a período).
    `project` leva o mapa ao conjunto viável: média das ``n_positions/L`` cópias por posição do
    período e projeção exata. No E15 o gap de projeção da H2 é medido com o top-K de Gumbel sem
    a restrição (`GumbelTopKMask` com horizonte e sem espaçamento); o LOUPE fica como braço
    opcional.
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
            raise ValueError("rate precisa estar em (0, 1)")
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
        """Probabilidade média por posição de um período de ``period_ticks`` ticks."""
        p = self.probabilities().detach().cpu().numpy().astype(np.float64)
        if self.n_positions % period_ticks:
            raise ValueError("n_positions precisa ser multiplo do periodo")
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
    """Máscara adaptativa por blocos, inspirada no ADPS (`vangorp2021adps`).

    A janela é dividida em ``n_blocks`` blocos (cada um múltiplo de ``L``). O bloco ``j`` usa
    os logits ``w0 + Δθ(h_{<j})``, onde ``h_{<j}`` é a média de um resumo do que **já foi
    adquirido**: a energia logarítmica relativa em ``n_features`` faixas do espectro do sinal
    preenchido com zeros de cada bloco anterior. ``Δθ`` é um MLP pequeno com a última camada
    iniciada em zero, então no início a política é a máscara fixa ``w0``.

    Diferenças declaradas em relação ao ADPS: política por blocos de períodos (não amostra a
    amostra) e sem LSTM; o estado é a média dos resumos dos blocos anteriores.

    Treino: top-K de Gumbel com straight-through em cada bloco, penalidade de espaçamento.
    Avaliação: projeção exata por bloco e, nas emendas entre blocos, descarte das conversões a
    menos de ``g`` ticks da anterior (``last_dropped`` conta quantas). Realizável como
    reprogramação dos CC do TIMER entre blocos, com a CPU acordando a cada bloco.
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
            raise ValueError(f"N={n} nao e multiplo do periodo L={period}")
        n_per = n // period
        if n_per < self.n_blocks:
            raise ValueError("janela curta demais para o numero de blocos")
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
    """Descarta conversões a menos de ``gap`` ticks da anterior mantida (emendas entre blocos)."""
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
