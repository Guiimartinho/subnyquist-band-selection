"""snq-data - public dataset loaders and artificial undersampling.

Rule: raw data is never versioned. The repository versions only the
manifests in datasets/manifests/ and the code that downloads, validates and prepares them.

Undersampling rule: do NOT decimate with an anti-aliasing filter. Decimation with an AAF
destroys exactly the effect under study. Use plain x[::M], or resample at the
non-uniform instants of the pattern under test.

See the dataset-ingest skill for the ingestion procedure.
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
