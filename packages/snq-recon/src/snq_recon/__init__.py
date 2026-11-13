"""snq-recon - classical reconstruction and estimation algorithms.

The project's official baseline: bandpass sampling + envelope analysis.
If a learned method does not beat it, it does not exist.

Implemented, and this docstring lists **only** what actually exists. The previous version
advertised `robust_crt` and `omp / fista` as if they existed; neither of them existed, and
`omp` only came into existence with E13. Advertising a nonexistent function is a defect,
not a plan.

  envelope            baseline; bandpass_analytic, envelope_spectrum, headroom
  kurtogram           selection of the resonant band to undersample (Antoni 2007)
  diagnosis           score against the geometric prior, estimation with and without prior
  estimation          alias candidates, multi-rate, unambiguous range
  highres             esprit, angular resampling, order spectrum
  sparse              sampling at random instants, order basis, OMP

Does not exist, and must not be cited as if it existed:
  robust_crt          alias-order disambiguation by coprime rates
  fista               sparse reconstruction by l1
  prony               spectral line estimation (ESPRIT is available and solves the same problem)
"""

from snq_recon.diagnosis import (
    Diagnosis,
    FaultScore,
    RefinedOrder,
    diagnose,
    estimate_from_candidates,
    estimate_without_prior,
    refine_order,
    refine_order_detail,
    score_fault,
)
from snq_recon.envelope import (
    EnvelopeSpectrum,
    bandpass_analytic,
    envelope,
    envelope_headroom_ok,
    envelope_spectrum,
    squared_envelope_spectrum,
)
from snq_recon.estimation import (
    alias_candidates_hz,
    estimate_tone_hz,
    prior_grid_hz,
    resolve_multirate_hz,
    unambiguous_range_hz,
)
from snq_recon.highres import (
    angular_resample,
    envelope_order_spectrum,
    esprit,
    estimate_tone_esprit,
    fault_orders,
    order_spectrum,
    phase_revolutions,
)
from snq_recon.kurtogram import (
    KurtogramResult,
    band_from_kurtogram,
    fast_kurtogram,
    spectral_kurtosis,
)
from snq_recon.sparse import (
    SparseSolution,
    interpolated_phase_rev,
    interpolated_speed_hz,
    omp,
    order_basis,
    order_of_column,
    random_time_indices,
    reconstruct_from_orders,
    sparse_order_spectrum,
    uniform_time_indices,
)

__all__ = [
    "Diagnosis",
    "EnvelopeSpectrum",
    "FaultScore",
    "KurtogramResult",
    "SparseSolution",
    "alias_candidates_hz",
    "angular_resample",
    "envelope_order_spectrum",
    "esprit",
    "estimate_tone_esprit",
    "fault_orders",
    "order_spectrum",
    "band_from_kurtogram",
    "bandpass_analytic",
    "diagnose",
    "envelope",
    "envelope_headroom_ok",
    "envelope_spectrum",
    "estimate_from_candidates",
    "estimate_without_prior",
    "fast_kurtogram",
    "interpolated_phase_rev",
    "interpolated_speed_hz",
    "omp",
    "order_basis",
    "order_of_column",
    "phase_revolutions",
    "random_time_indices",
    "reconstruct_from_orders",
    "RefinedOrder",
    "refine_order",
    "refine_order_detail",
    "score_fault",
    "sparse_order_spectrum",
    "spectral_kurtosis",
    "squared_envelope_spectrum",
    "estimate_tone_hz",
    "prior_grid_hz",
    "resolve_multirate_hz",
    "unambiguous_range_hz",
    "uniform_time_indices",
]
