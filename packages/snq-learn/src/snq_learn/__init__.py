"""snq-learn - learned methods, with a model underneath.

Do not start from a generic CNN. The reference is model-based deep learning:
algorithm unrolling (Monga, Li & Eldar, IEEE SPM 2021) gives a small network that is
interpretable and data-efficient - and that fits in 256 kB of RAM.

The frontier pursued is task-oriented learned acquisition
(Shlezinger et al., IEEE TSP 2022; Bahadir et al., IEEE TCI 2020;
van Gorp et al., ICML 2021).

Hard constraint: the learned sampling policy must be realizable by the
DPPI/TIMER of the nRF54L15. A mask that the hardware does not schedule is not a result.

Modules:

  distillation   Nyquist -> sub-Nyquist distillation (E04/E05)
  masks          feasible set, exact projection; fixed, learned, LOUPE and adaptive masks (E15)
  order_decoder  unrolled decoder: filter bank -> envelope -> order (E15)
  wdcnn          WDCNN from zhang2017wdcnn, only as a baseline (E15)
  training       loop with an epoch cap, patience and best checkpoint (E15)

This package does not import snq-data or snq-recon: it receives tensors. The experiment
runner is what loads the data and computes the tachometer phase.
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
