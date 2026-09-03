"""snq-data - loaders de datasets publicos e subamostragem artificial.

Regra: dados brutos nunca sao versionados. O repositorio versiona apenas os
manifests em datasets/manifests/ e o codigo que baixa, valida e prepara.

Regra de subamostragem: NAO decimar com filtro anti-aliasing. Decimacao com AAF
destroi exatamente o efeito em estudo. Usar x[::M] puro, ou reamostrar nos
instantes nao-uniformes do padrao em teste.

Ver a skill dataset-ingest para o procedimento de ingestao.
"""

from snq_data.datasets import (
    AntiAliasFilterUsedError,
    Manifest,
    effective_rate_hz,
    load_all_manifests,
    sha256,
    undersample,
    undersample_cosets,
    undersampling_report,
)

__all__ = [
    "AntiAliasFilterUsedError",
    "Manifest",
    "effective_rate_hz",
    "load_all_manifests",
    "sha256",
    "undersample",
    "undersample_cosets",
    "undersampling_report",
]
