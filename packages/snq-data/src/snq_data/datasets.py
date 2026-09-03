"""Manifests de dataset, subamostragem artificial e verdade fundamental.

Regra do projeto: **dados brutos nunca sao versionados.** O repositorio guarda
`datasets/manifests/*.yaml` e este codigo; `raw/` e `processed/` estao no
`.gitignore`.

O presente que os datasets publicos dao a este projeto e que foram adquiridos
**acima** do Nyquist necessario. Isso permite subamostrar artificialmente e ter
verdade fundamental exata - que e o unico jeito de medir erro de estimacao de
frequencia de falha, em vez de acuracia de classificacao.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from snq_core import BearingGeometry, fault_frequencies_hz
from snq_core import bearing as _bearing


class AntiAliasFilterUsedError(RuntimeError):
    """Levantado quando se tenta decimar com filtro anti-aliasing.

    Decimacao com AAF destroi exatamente o efeito que este projeto estuda. Se um
    resultado parecer bom demais, este e o primeiro ponto a conferir.
    """


@dataclass(frozen=True)
class Manifest:
    """Descricao versionada de um dataset publico."""

    name: str
    full_name: str
    url: str
    license: str
    fs_hz: list[float]
    notes: str = ""
    channels: list[str] = field(default_factory=list)
    bearing: dict[str, str] = field(default_factory=dict)
    undersampling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def geometry(self, channel: str | None = None) -> BearingGeometry:
        """Geometria do rolamento declarada no manifest, resolvida por nome.

        O nome aponta para uma constante de ``snq_core.bearing`` - assim a
        geometria mora num lugar so e nao e redigitada por dataset.
        """
        if not self.bearing:
            raise ValueError(f"manifest {self.name!r} nao declara geometria de rolamento")
        key = channel or next(iter(self.bearing))
        const = self.bearing[key]
        geom = getattr(_bearing, const, None)
        if not isinstance(geom, BearingGeometry):
            raise ValueError(f"{const!r} nao e uma BearingGeometry em snq_core.bearing")
        return geom

    def fault_truth_hz(self, shaft_hz: float, channel: str | None = None) -> dict[str, float]:
        """Verdade fundamental **continua**: as frequencias de falha da geometria.

        E isto que o projeto mede o erro contra, e nao o rotulo de classe.
        """
        return fault_frequencies_hz(self.geometry(channel), shaft_hz)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def undersample(
    x: ArrayLike, factor: int, *, anti_alias: bool = False, offset: int = 0
) -> NDArray[np.float64]:
    """Decimacao **pura** por ``factor``, sem filtro anti-aliasing.

    ``anti_alias=True`` levanta ``AntiAliasFilterUsedError`` de proposito: a
    assinatura existe para que quem tentar filtrar antes de decimar receba um
    erro explicativo em vez de um resultado silenciosamente errado.

    >>> undersample(np.arange(10.0), 3).tolist()
    [0.0, 3.0, 6.0, 9.0]
    """
    if anti_alias:
        raise AntiAliasFilterUsedError(
            "Decimar com AAF destroi a informacao super-Nyquist em estudo. "
            "Use decimacao pura; se precisa selecionar banda, use um AAF "
            "PASSA-FAIXA na cadeia analogica (snq_sim.AntiAliasFilter)."
        )
    if factor < 1:
        raise ValueError("factor >= 1")
    if not 0 <= offset < factor:
        raise ValueError("offset precisa estar em [0, factor)")
    return np.asarray(x, dtype=float).ravel()[offset::factor]


def undersample_cosets(x: ArrayLike, factor: int, offsets: ArrayLike) -> list[NDArray[np.float64]]:
    """Multi-coset: um fluxo por deslocamento inteiro de amostra.

    E a versao discreta do que o DPPI/TIMER do nRF54L15 faz em hardware. Cada
    coset ve o mesmo sinal com fase de amostragem diferente, e e a diferenca
    entre eles que carrega a ordem do alias.
    """
    return [undersample(x, factor, offset=int(o)) for o in np.asarray(offsets).ravel()]


def effective_rate_hz(fs_original_hz: float, factor: int) -> float:
    return float(fs_original_hz) / int(factor)


def undersampling_report(fs_original_hz: float, factor: int, band_hz: tuple[float, float]) -> dict:
    """Resumo do que a subamostragem faz com uma banda de interesse.

    Devolve a taxa efetiva, a dobra resultante e se a banda colide consigo mesma.
    Chamar isto antes de gerar um split evita gastar horas num fator que destroi
    a informacao.
    """
    from snq_core import fold_band  # import local: evita ciclo na leitura do modulo

    fs = effective_rate_hz(fs_original_hz, factor)
    fb = fold_band(band_hz[0], band_hz[1], fs)
    return {
        "fs_effective_hz": fs,
        "factor": int(factor),
        "alias_lo_hz": fb.lo_hz,
        "alias_hi_hz": fb.hi_hz,
        "zone": fb.zone,
        "inverted": fb.inverted,
        "self_overlap": fb.self_overlap,
        "usable": not fb.self_overlap,
    }


def load_all_manifests(manifest_dir: str | Path) -> dict[str, Manifest]:
    """Carrega todos os manifests de um diretorio, indexados por nome.

    Apenas manifests de dataset. Catalogos de triagem (por exemplo a classificacao
    de registros do CWRU) vivem em ``datasets/catalogs/`` e tem outro esquema.
    """
    out: dict[str, Manifest] = {}
    for p in sorted(Path(manifest_dir).glob("*.yaml")):
        m = Manifest.load(p)
        out[m.name] = m
    return out
