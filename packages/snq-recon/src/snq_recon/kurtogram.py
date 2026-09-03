"""Kurtograma — escolhe a banda ressonante onde o transiente é mais impulsivo.

É a peça que decide **onde** a amostragem passa-faixa deve ser centrada. Sem ela a
escolha da banda é arbitrária, e é justamente essa ligação — kurtograma escolhe a
banda, teoria passa-faixa diz a que taxa amostrá-la — que sustenta a contribuição de
sistema do projeto.

Antoni (MSSP 21(1):108-124, 2007) mostra que o custo é **O(N log N)**, a mesma ordem
de uma FFT. Isso foi conferido na fonte e tem consequência de arquitetura: a varredura
**cabe no nó**, e não é preciso transmitir um registro em taxa cheia para o gateway só
para escolher a banda — que seria exatamente o gasto de rádio a evitar.

Implementação: grade diádica com passos de 1/3 de oitava, filtragem no domínio da FFT
e curtose do envelope complexo. Não é a recursão de banco de filtros do artigo, e sim
a formulação em grade equivalente; a complexidade por nível é a mesma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snq_recon.envelope import bandpass_analytic


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
    """Níveis diádicos com passos de 1/3 de oitava, como em Antoni (2007)."""
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

    ``min_energy_frac`` descarta bandas praticamente vazias. Sem esse guarda, a
    curtose de uma banda sem sinal é dominada por vazamento numérico e pode ficar
    enorme — o kurtograma então "encontra" um transiente onde não há energia alguma.
    É um modo de falha real, observado em sinal sintético sem ruído.

    ``max_level`` controla a resolução mais fina: no nível ``k`` a banda base é
    dividida em ``2**k`` faixas. ``f_min_hz`` e ``f_max_hz`` restringem a varredura,
    útil quando a banda do transdutor já é conhecida.

    O resultado alimenta diretamente `snq_core.valid_bandpass_rates_hz`: a banda
    escolhida aqui é a que a amostragem passa-faixa vai centrar.
    """
    x = np.asarray(x, dtype=float).ravel()
    nyq = fs_hz / 2.0
    hi = min(f_max_hz, nyq) if f_max_hz is not None else nyq
    if hi <= f_min_hz:
        raise ValueError("faixa de busca vazia")

    grid: list[tuple[float, float, float, float]] = []
    best = (f_min_hz, hi, -np.inf, 0.0)
    total_energy = float(np.sum(x**2))

    for level in _levels(max_level):
        n_bands = int(round(2.0**level))
        edges = np.linspace(f_min_hz, hi, n_bands + 1)
        for lo, up in zip(edges[:-1], edges[1:], strict=True):
            if up - lo <= 0.0:
                continue
            z = bandpass_analytic(x, fs_hz, float(lo), float(up))
            if (
                total_energy > 0.0
                and float(np.sum(np.abs(z) ** 2)) < min_energy_frac * total_energy
            ):
                continue  # banda vazia: curtose seria numericamente degenerada
            k = spectral_kurtosis(z)
            grid.append((level, float(lo), float(up), k))
            if k > best[2]:
                best = (float(lo), float(up), k, level)

    return KurtogramResult(
        f_lo_hz=best[0], f_hi_hz=best[1], kurtosis=best[2], level=best[3], grid=grid
    )


def band_from_kurtogram(x: ArrayLike, fs_hz: float, **kw: object) -> tuple[float, float]:
    """Atalho: só a banda escolhida, para encadear com a análise de envelope."""
    return fast_kurtogram(x, fs_hz, **kw).band_hz  # type: ignore[arg-type]
