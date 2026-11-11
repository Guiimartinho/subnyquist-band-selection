"""snq-learn - metodos aprendidos, com modelo por baixo.

Nao comecar por CNN generica. A referencia e model-based deep learning:
algorithm unrolling (Monga, Li & Eldar, IEEE SPM 2021) da uma rede pequena,
interpretavel e eficiente em dados - e que cabe em 256 kB de RAM.

A fronteira perseguida e aquisicao aprendida orientada a tarefa
(Shlezinger et al., IEEE TSP 2022; Bahadir et al., IEEE TCI 2020;
van Gorp et al., ICML 2021).

Restricao dura: a politica de amostragem aprendida precisa ser realizavel pelo
DPPI/TIMER do nRF54L15. Mascara que o hardware nao agenda nao e resultado.

Modulos:

  distillation   destilacao Nyquist -> sub-Nyquist (E04/E05)
  masks          conjunto viavel, projecao exata; mascaras fixa, aprendida, LOUPE, adaptativa (E15)
  order_decoder  decodificador desdobrado: banco de filtros -> envelope -> ordem (E15)
  wdcnn          WDCNN de zhang2017wdcnn, so como baseline (E15)
  training       laco com teto de epocas, paciencia e melhor checkpoint (E15)

Este pacote nao importa snq-data nem snq-recon: recebe tensores. Quem carrega dados e
calcula a fase do tacometro e o runner do experimento.
"""

from snq_learn.distillation import (
    DistillConfig,
    SubNyquistStudent,
    distill,
    evaluate,
    teacher_distribution,
)
from snq_learn.masks import (
    BlockAdaptiveMask,
    FeasibleSet,
    FixedMask,
    GumbelTopKMask,
    LoupeMask,
    alias_weights,
    band_support_cells,
    circular_gaps,
    designed_positions,
    fold_project,
    noise_amplification,
    positions_to_mask,
    project_to_feasible,
    random_feasible_positions,
    shrink_band_to_support,
    spacing_penalty,
    support_designed_positions,
    timer_schedule,
    uniform_positions,
)
from snq_learn.order_decoder import (
    AngularPlan,
    ComplexFilterBank,
    OrderAxis,
    OrderDecoder,
    SampledOrderModel,
    angular_plan,
    envelope_sample_positions,
    kurtogram_bands_hz,
    order_cross_entropy,
    soft_order_target,
    stack_plans,
)
from snq_learn.training import FitConfig, FitResult, fit
from snq_learn.wdcnn import WDCNN

__all__ = [
    "WDCNN",
    "AngularPlan",
    "BlockAdaptiveMask",
    "ComplexFilterBank",
    "DistillConfig",
    "FeasibleSet",
    "FitConfig",
    "FitResult",
    "FixedMask",
    "GumbelTopKMask",
    "LoupeMask",
    "OrderAxis",
    "OrderDecoder",
    "SampledOrderModel",
    "SubNyquistStudent",
    "alias_weights",
    "angular_plan",
    "band_support_cells",
    "circular_gaps",
    "designed_positions",
    "distill",
    "envelope_sample_positions",
    "evaluate",
    "fit",
    "fold_project",
    "kurtogram_bands_hz",
    "noise_amplification",
    "order_cross_entropy",
    "positions_to_mask",
    "project_to_feasible",
    "random_feasible_positions",
    "shrink_band_to_support",
    "soft_order_target",
    "spacing_penalty",
    "stack_plans",
    "support_designed_positions",
    "teacher_distribution",
    "timer_schedule",
    "uniform_positions",
]
