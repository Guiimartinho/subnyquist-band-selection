"""snq-data - loaders de datasets publicos e subamostragem artificial.

Regra: dados brutos nunca sao versionados. O repositorio versiona apenas os
manifests em datasets/manifests/ e o codigo que baixa, valida e prepara.

Regra de subamostragem: NAO decimar com filtro anti-aliasing. Decimacao com AAF
destroi exatamente o efeito em estudo. Usar x[::M] puro, ou reamostrar nos
instantes nao-uniformes do padrao em teste.

A ser implementado: load_manifest, fetch, verify_checksum, undersample,
make_split. Ver a skill dataset-ingest.
"""

__all__: list[str] = []
