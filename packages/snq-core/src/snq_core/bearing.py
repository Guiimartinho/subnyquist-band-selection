"""Frequências de falha de rolamento a partir da geometria.

Este módulo é a razão pela qual este projeto pode falar em **recuperação** e não
em alucinação: o conjunto de frequências candidatas é finito e sai da geometria
do mancal, não de um prior aprendido. Identificar qual componente super-Nyquist
gerou um alias observado deixa de ser inversão mal-posta e vira teste de
hipóteses sobre um conjunto discreto.

Referência: R. B. Randall & J. Antoni, *Rolling element bearing diagnostics —
a tutorial*, Mechanical Systems and Signal Processing 25(2):485-520, 2011.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BearingGeometry:
    """Geometria de um rolamento de elementos rolantes.

    ``n_elements``  número de esferas/roletes
    ``d_ball_m``    diâmetro do elemento rolante
    ``d_pitch_m``   diâmetro primitivo (centro a centro dos elementos)
    ``contact_angle_rad`` ângulo de contato (0 para rolamento radial puro)
    """

    n_elements: int
    d_ball_m: float
    d_pitch_m: float
    contact_angle_rad: float = 0.0
    name: str = ""

    @property
    def ratio(self) -> float:
        """``(d/D)·cos(phi)`` — aparece em todas as quatro fórmulas."""
        return (self.d_ball_m / self.d_pitch_m) * float(np.cos(self.contact_angle_rad))


def fault_frequencies_hz(geom: BearingGeometry, shaft_hz: float) -> dict[str, float]:
    """As quatro frequências características, em Hz, para rotação ``shaft_hz``.

    - ``BPFO`` ball pass frequency, outer race
    - ``BPFI`` ball pass frequency, inner race
    - ``BSF``  ball spin frequency (o *defeito* aparece em ``2·BSF``)
    - ``FTF``  fundamental train frequency (gaiola)

    São proporcionais à rotação: variar o regime **move as raias de forma
    conhecida**, e é isso que desambigua a ordem do alias.
    """
    r = geom.ratio
    n = geom.n_elements
    return {
        "BPFO": (n / 2.0) * shaft_hz * (1.0 - r),
        "BPFI": (n / 2.0) * shaft_hz * (1.0 + r),
        "BSF": (geom.d_pitch_m / (2.0 * geom.d_ball_m)) * shaft_hz * (1.0 - r * r),
        "FTF": 0.5 * shaft_hz * (1.0 - r),
    }


def candidate_lines_hz(
    geom: BearingGeometry, shaft_hz: float, n_harmonics: int = 8, n_sidebands: int = 2
) -> NDArray[np.float64]:
    """Conjunto discreto de raias esperadas — o prior físico do projeto.

    Harmônicos das frequências de falha, com bandas laterais espaçadas de
    ``shaft_hz`` (modulação por rotação). É este conjunto, e não uma grade
    contínua, que o decodificador precisa selecionar.
    """
    base = fault_frequencies_hz(geom, shaft_hz)
    lines: list[float] = []
    for f in base.values():
        for h in range(1, n_harmonics + 1):
            centre = h * f
            for s in range(-n_sidebands, n_sidebands + 1):
                v = centre + s * shaft_hz
                if v > 0.0:
                    lines.append(v)
    return np.unique(np.asarray(lines, dtype=float))


# Geometrias documentadas dos datasets públicos usados no projeto.
# Conferir contra a documentação de cada dataset antes de usar em paper.
CWRU_6205_2RS_JEM_SKF = BearingGeometry(
    n_elements=9,
    d_ball_m=7.94e-3,
    d_pitch_m=39.04e-3,
    contact_angle_rad=0.0,
    name="CWRU drive-end SKF 6205-2RS JEM",
)
