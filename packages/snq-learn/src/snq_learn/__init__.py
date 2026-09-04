"""snq-learn - metodos aprendidos, com modelo por baixo.

Nao comecar por CNN generica. A referencia e model-based deep learning:
algorithm unrolling (Monga, Li & Eldar, IEEE SPM 2021) da uma rede pequena,
interpretavel e eficiente em dados - e que cabe em 256 kB de RAM.

A fronteira perseguida e aquisicao aprendida orientada a tarefa
(Shlezinger et al., IEEE TSP 2022; Bahadir et al., IEEE TCI 2020;
van Gorp et al., ICML 2021).

Restricao dura: a politica de amostragem aprendida precisa ser realizavel pelo
DPPI/TIMER do nRF54L15. Mascara que o hardware nao agenda nao e resultado.
"""

from snq_learn.distillation import (
    DistillConfig,
    SubNyquistStudent,
    distill,
    evaluate,
    teacher_distribution,
)

__all__ = [
    "DistillConfig",
    "SubNyquistStudent",
    "distill",
    "evaluate",
    "teacher_distribution",
]
