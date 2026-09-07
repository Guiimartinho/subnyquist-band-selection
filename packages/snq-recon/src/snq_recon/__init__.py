"""snq-recon - algoritmos classicos de reconstrucao e estimacao.

Baseline oficial do projeto: amostragem passa-faixa + analise de envelope.
Se um metodo aprendido nao bate isso, ele nao existe.

Implementado, e este docstring lista **so** o que existe de fato. A versao anterior
anunciava `robust_crt` e `omp / fista` como se existissem; nenhum dos dois existia, e
`omp` so passou a existir com o E13. Anunciar funcao inexistente e defeito, nao plano.

  envelope            baseline; bandpass_analytic, envelope_spectrum, headroom
  kurtogram           selecao da banda ressonante a subamostrar (Antoni 2007)
  diagnosis           escore contra o prior geometrico, estimacao com e sem prior
  estimation          candidatas de alias, multi-taxa, faixa nao-ambigua
  highres             esprit, reamostragem angular, espectro de ordem
  sparse              amostragem em instantes aleatorios, base de ordem, OMP

Nao existe, e nao deve ser citado como se existisse:
  robust_crt          desambiguacao da ordem do alias por taxas coprimas
  fista               reconstrucao esparsa por l1
  prony               estimacao de linhas espectrais (ha ESPRIT, que resolve o mesmo)
"""

from snq_recon.diagnosis import (
    Diagnosis,
    FaultScore,
    diagnose,
    estimate_from_candidates,
    estimate_without_prior,
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
