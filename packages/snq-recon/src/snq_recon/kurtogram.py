"""Kurtograma — escolhe a banda ressonante onde o transiente é mais impulsivo.

É a peça que decide **onde** a amostragem passa-faixa deve ser centrada. Sem ela a
escolha da banda é arbitrária, e é justamente essa ligação — kurtograma escolhe a
banda, teoria passa-faixa diz a que taxa amostrá-la — que sustenta a contribuição de
sistema do projeto.

Antoni (MSSP 21(1):108-124, 2007) mostra que o custo é **O(N log N)**, a mesma ordem
de uma FFT. Isso foi conferido na fonte e tem consequência de arquitetura: a varredura
**cabe no nó**, e não é preciso transmitir um registro em taxa cheia para o gateway só
para escolher a banda — que seria exatamente o gasto de rádio a evitar.

Implementação: grade 1/3-binária, filtragem ideal no domínio da FFT e curtose do
envelope complexo pela Eq. (7). Não é a recursão de banco de filtros do artigo. O que
coincide com a fonte e o que não coincide, conferido no PDF em 2026-11-11:

- **Coincide:** sem ``f_min_hz``/``f_max_hz``, o nível ``L`` tem ``2^L`` bandas de largura
  ``fs·2^(-L-1)``, com centros da Eq. (5) e larguras da Eq. (6); os níveis intermediários
  ``k + log2(3) - 1`` dão as larguras 1/6, 1/12, 1/24 da Fig. 5. Estimador da Eq. (7).
- **Desvio 1, partição da faixa de busca.** Com ``f_min_hz``/``f_max_hz`` as ``2^L`` bandas
  dividem ``[f_min, min(f_max, fs/2)]``, e não ``[0, fs/2]``. Os limites **mudam a
  partição**, não só mascaram candidatas. Consequência: a partição só depende de ``fs``
  quando ``fs/2 < f_max``. No E07 (500 Hz a 25 kHz sobre 200 kSPS) os fatores 1, 2 e 4
  veem as mesmas bandas e a 8x elas mudam; em Antoni mudariam a cada fator.
- **Desvio 2, nível 0 ausente.** A banda inteira não é candidata.
- **Desvio 3, filtros.** Máscaras ideais, sem sobreposição entre bandas vizinhas e sem
  decimação; os filtros quase-analíticos FIR do artigo se sobrepõem (Figs. 3 e 4) e a
  curtose é tomada na saída criticamente decimada.
- **Bordas, como na nota 3 do artigo.** A filtragem por FFT é circular e o salto na
  emenda do registro vira um transiente falso nas bordas. O registro é estendido por
  espelhamento (a extensão periódica fica contínua) e as ``EDGE_GUARD_PERIODS·fs/Δf``
  amostras de cada borda ficam fora da curtose. Antes da correção de 2026-11-11, ruído
  gaussiano estacionário com espectro de máquina dava SK acima de 100, contra a
  propriedade (ii) do artigo, que a exige nula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import _band_analytic_from_spectrum

EDGE_GUARD_PERIODS = 4.0
"""Amostras excluídas em cada borda, em unidades de ``fs/Δf`` (o inverso da largura da banda).

É a ordem do transitório do protótipo FIR de 8 coeficientes que Antoni usa em todos os
exemplos (nível ``k``: ``L·2^k = L·fs/(2·Δf)`` amostras, ``L = 8``) — essa conta é deste
projeto, não do artigo. Medido em ruído gaussiano colorido: com espelhamento, 4 basta
para a SK cair ao piso do estimador; sem espelhamento nem 32 bastava.
"""


def _edge_guard(n_samples: int, fs_hz: float, bandwidth_hz: float) -> int:
    """Amostras a descartar em cada borda; nunca mais que um quarto do registro."""
    return min(int(np.ceil(EDGE_GUARD_PERIODS * fs_hz / bandwidth_hz)), n_samples // 4)


def spectral_kurtosis(z: NDArray[np.complex128]) -> float:
    """Curtose do envelope de um sinal analítico de banda.

    Normalizada de modo que ruído gaussiano complexo circular dá ~0 e um trem de
    impulsos raros dá valor alto. É a definição usual em diagnóstico de máquinas.
    """
    e2 = np.abs(z) ** 2
    m2 = float(np.mean(e2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(e2**2) / (m2 * m2) - 2.0)


@dataclass(frozen=True)
class KurtogramResult:
    """Melhor par (banda, resolução) e o mapa completo."""

    f_lo_hz: float
    f_hi_hz: float
    kurtosis: float
    level: float
    grid: list[tuple[float, float, float, float]]
    """Todas as células avaliadas: ``(nível, f_lo, f_hi, curtose)``."""

    @property
    def band_hz(self) -> tuple[float, float]:
        return (self.f_lo_hz, self.f_hi_hz)

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.f_lo_hz + self.f_hi_hz)

    @property
    def bandwidth_hz(self) -> float:
        return self.f_hi_hz - self.f_lo_hz

    def summary(self) -> str:
        return (
            f"banda [{self.f_lo_hz / 1e3:.2f}, {self.f_hi_hz / 1e3:.2f}] kHz "
            f"(centro {self.center_hz / 1e3:.2f} kHz, largura {self.bandwidth_hz / 1e3:.2f} kHz) "
            f"| curtose {self.kurtosis:.2f} | nível {self.level:g}"
        )


def _levels(max_level: int) -> list[float]:
    """Níveis 1/3-binários de Antoni (2007), Fig. 5: ``k`` e ``k + log2(3) - 1``.

    O intermediário divide a banda em três, e não é um passo de 1/3 de oitava: está a
    0,585 oitava do nível ``k``. O nível 0 não entra (desvio 2 do docstring do módulo).
    """
    out: list[float] = []
    for k in range(1, max_level + 1):
        out += [float(k), k + np.log2(3.0) - 1.0]
    return sorted(v for v in out if v <= max_level)


def fast_kurtogram(
    x: ArrayLike,
    fs_hz: float,
    max_level: int = 6,
    f_min_hz: float = 0.0,
    f_max_hz: float | None = None,
    min_energy_frac: float = 1e-4,
) -> KurtogramResult:
    """Varre o plano (frequência central, largura) e devolve a banda mais impulsiva.

    ``min_energy_frac`` descarta bandas praticamente vazias: em sinal sintético sem
    ruído, a curtose de uma banda sem energia mede o arredondamento de ponto flutuante.
    Não confundir com o falso transiente da emenda do registro, que é tratado à parte
    (espelhamento e guarda de borda, ver o docstring do módulo) e que afetava bandas com
    energia real, acima de qualquer limiar razoável.

    ``max_level`` controla a resolução mais fina: no nível ``k`` a faixa de busca é
    dividida em ``2**k`` bandas. ``f_min_hz`` e ``f_max_hz`` **definem a faixa que é
    particionada**, não só mascaram candidatas: a partição de ``[f_min, min(f_max, fs/2)]``
    não depende de ``fs`` enquanto ``fs/2 >= f_max``. Com os dois no padrão, a partição é a
    de Antoni (2007).

    O resultado alimenta diretamente `snq_core.valid_bandpass_rates_hz`: a banda
    escolhida aqui é a que a amostragem passa-faixa vai centrar.
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

    # Extensão par: a extensão periódica de [x, x invertido] não tem salto na emenda,
    # só uma quina, cujo transiente decai como 1/t^2 e cabe na guarda de borda.
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
                continue  # banda vazia: curtose seria numericamente degenerada
            guard = _edge_guard(n, fs_hz, float(up - lo))
            k = spectral_kurtosis(z[guard : n - guard])  # nota 3 de Antoni (2007)
            grid.append((level, float(lo), float(up), k))
            if k > best[2]:
                best = (float(lo), float(up), k, level)

    return KurtogramResult(
        f_lo_hz=best[0], f_hi_hz=best[1], kurtosis=best[2], level=best[3], grid=grid
    )


def band_from_kurtogram(x: ArrayLike, fs_hz: float, **kw: object) -> tuple[float, float]:
    """Atalho: só a banda escolhida, para encadear com a análise de envelope."""
    return fast_kurtogram(x, fs_hz, **kw).band_hz  # type: ignore[arg-type]
