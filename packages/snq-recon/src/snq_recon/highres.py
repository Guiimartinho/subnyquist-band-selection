"""Estimação espectral de alta resolução, e análise de ordem.

Duas peças que a leitura acrescentou ao escopo do baseline.

**ESPRIT** existe aqui como controle honesto. Antes de afirmar que um estimador
aprendido melhora o E01 na região de −10 a −20 dB, é preciso saber se um método
clássico de alta resolução já fecha a lacuna. Se fechar, o ganho marginal do
aprendizado é pequeno — e isso é uma descoberta, não um atraso.

**Análise de ordem** vem de `sensors2025csvibration` (Kato & Otaka, Sensors 2025,
doi:10.3390/s25103167). Sob rotação variável, Fourier deixa de representar a vibração
de forma esparsa e a compressão despenca; reamostrar em incrementos angulares
constantes devolve a esparsidade, porque as frequências de falha viram **ordens
fixas** da rotação. É pré-requisito para XJTU-SY e FEMTO, que têm variação de rotação.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def esprit(x: ArrayLike, n_sources: int, fs_hz: float, subspace: int | None = None) -> NDArray:
    """Frequências por ESPRIT — resolução acima do limite da FFT.

    Explora a invariância rotacional do subespaço de sinal: duas submatrizes
    deslocadas de uma amostra se relacionam por uma rotação cujos autovalores são
    ``exp(j·2π·f/fs)``. Diferente do pico de FFT, não é limitado por ``fs/N``.

    ``n_sources`` é o número de senoides a estimar. Superestimar produz frequências
    espúrias; subestimar mistura componentes. É a fraqueza do método em relação a um
    estimador que detecta a ordem do modelo sozinho.

    >>> fs = 1000.0
    >>> t = np.arange(512) / fs
    >>> x = np.sin(2 * np.pi * 123.4 * t)
    >>> float(esprit(x, 1, fs)[0])  # doctest: +ELLIPSIS
    123.4...
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    m = subspace or max(2 * n_sources + 1, n // 3)
    if m >= n:
        raise ValueError(f"subespaço ({m}) precisa ser menor que o sinal ({n})")

    # Matriz de Hankel: cada coluna e uma janela deslizante do sinal.
    rows = n - m + 1
    h = np.lib.stride_tricks.sliding_window_view(x, m)[:rows].T

    u, _, _ = np.linalg.svd(h, full_matrices=False)
    signal_space = u[:, : 2 * n_sources]  # sinal real: 2 exponenciais por senoide

    # Invariancia rotacional entre as submatrizes deslocadas de uma linha.
    phi = np.linalg.pinv(signal_space[:-1]) @ signal_space[1:]
    eig = np.linalg.eigvals(phi)

    freqs = np.abs(np.angle(eig)) * fs_hz / (2.0 * np.pi)
    freqs = np.unique(np.round(freqs[freqs > 0], 9))
    return np.sort(freqs)[:n_sources] if freqs.size else np.array([np.nan])


def estimate_tone_esprit(x: ArrayLike, fs_hz: float) -> float:
    """Frequência do tom dominante por ESPRIT. Assinatura igual à do estimador de FFT."""
    f = esprit(x, n_sources=1, fs_hz=fs_hz)
    return float(f[0]) if f.size else float("nan")


# ----------------------------------------------------------------------
# análise de ordem
# ----------------------------------------------------------------------
def angular_resample(
    x: ArrayLike,
    fs_hz: float,
    shaft_hz: ArrayLike | float,
    samples_per_rev: int = 64,
) -> tuple[NDArray[np.float64], float]:
    """Reamostra o sinal em incrementos angulares constantes.

    ``shaft_hz`` pode ser um escalar (rotação constante) ou um perfil amostrado na
    mesma taxa de ``x`` (rotação variável, vindo de tacômetro).

    Devolve ``(sinal_em_angulo, samples_per_rev)``. Depois disso, uma frequência de
    falha que variava com a rotação vira uma **ordem fixa**: BPFO a 3,58× a rotação
    aparece sempre no mesmo lugar do espectro de ordem, independentemente do regime.

    É o mecanismo que `sensors2025csvibration` usa para devolver esparsidade sob
    rotação variável — e sem ele o E02 não se aplica a XJTU-SY nem a FEMTO.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    t = np.arange(n) / fs_hz

    speed = (
        np.full(n, float(shaft_hz))
        if np.isscalar(shaft_hz)
        else np.asarray(shaft_hz, dtype=float).ravel()
    )
    if speed.size != n:
        raise ValueError(f"perfil de rotação tem {speed.size} pontos, sinal tem {n}")
    if np.any(speed <= 0):
        raise ValueError("rotação precisa ser positiva em todo o registro")

    # Angulo acumulado em revolucoes: integral da rotacao.
    revs = np.concatenate([[0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(t))])
    total = float(revs[-1])
    if total < 1.0:
        raise ValueError(f"registro cobre só {total:.2f} revolução; insuficiente")

    grid = np.arange(0.0, total, 1.0 / samples_per_rev)
    return np.interp(grid, revs, x), float(samples_per_rev)


def order_spectrum(
    x_angular: ArrayLike, samples_per_rev: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Espectro em **ordens** da rotação, a partir do sinal já reamostrado em ângulo.

    Devolve ``(ordens, magnitude)``. A ordem 1 é a rotação do eixo; BPFO fica numa
    ordem fixa dada pela geometria, e não mais numa frequência que muda com o regime.
    """
    x = np.asarray(x_angular, dtype=float).ravel()
    n = x.size
    mag = np.abs(np.fft.rfft(x * np.hanning(n))) * (2.0 / n)
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)
    return orders, mag


def fault_orders(geom_ratio: float, n_elements: int) -> dict[str, float]:
    """Frequências de falha expressas em **ordens da rotação**, não em hertz.

    São constantes da geometria: não dependem do regime. É a forma correta de
    declarar o prior quando a rotação varia.
    """
    return {
        "BPFO": (n_elements / 2.0) * (1.0 - geom_ratio),
        "BPFI": (n_elements / 2.0) * (1.0 + geom_ratio),
        "BSF": (1.0 / (2.0 * geom_ratio)) * (1.0 - geom_ratio**2) if geom_ratio > 0 else np.inf,
        "FTF": 0.5 * (1.0 - geom_ratio),
    }
