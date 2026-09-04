"""snq-data - loaders de datasets publicos e subamostragem artificial.

Regra: dados brutos nunca sao versionados. O repositorio versiona apenas os
manifests em datasets/manifests/ e o codigo que baixa, valida e prepara.

Regra de subamostragem: NAO decimar com filtro anti-aliasing. Decimacao com AAF
destroi exatamente o efeito em estudo. Usar x[::M] puro, ou reamostrar nos
instantes nao-uniformes do padrao em teste.

Ver a skill dataset-ingest para o procedimento de ingestao.
"""

from snq_data.cwru import (
    CwruCatalog,
    CwruRecord,
    download_record,
    load_record,
    parse_record_id,
)
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
from snq_data.ottawa import (
    FAULT_ORDERS as OTTAWA_FAULT_ORDERS,
)
from snq_data.ottawa import (
    OttawaRecord,
)

__all__ = [
    "AntiAliasFilterUsedError",
    "CwruCatalog",
    "CwruRecord",
    "download_record",
    "load_record",
    "parse_record_id",
    "OttawaRecord",
    "OTTAWA_FAULT_ORDERS",
    "Manifest",
    "effective_rate_hz",
    "load_all_manifests",
    "sha256",
    "undersample",
    "undersample_cosets",
    "undersampling_report",
]
