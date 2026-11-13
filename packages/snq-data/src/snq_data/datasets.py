"""Dataset manifests, artificial undersampling and ground truth.

Project rule: **raw data is never versioned.** The repository keeps
`datasets/manifests/*.yaml` and this code; `raw/` and `processed/` are in
`.gitignore`.

The gift that public datasets give this project is that they were acquired
**above** the required Nyquist rate. This makes it possible to undersample artificially
and have exact ground truth - which is the only way to measure fault-frequency
estimation error, instead of classification accuracy.
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
    """Raised when one tries to decimate with an anti-aliasing filter.

    Decimation with an AAF destroys exactly the effect this project studies. If a
    result looks too good, this is the first place to check.
    """


@dataclass(frozen=True)
class Manifest:
    """Versioned description of a public dataset."""

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
        """Bearing geometry declared in the manifest, resolved by name.

        The name points to a constant in ``snq_core.bearing`` - this way the
        geometry lives in a single place and is not retyped for each dataset.
        """
        if not self.bearing:
            raise ValueError(f"manifest {self.name!r} does not declare a bearing geometry")
        key = channel or next(iter(self.bearing))
        const = self.bearing[key]
        geom = getattr(_bearing, const, None)
        if not isinstance(geom, BearingGeometry):
            raise ValueError(f"{const!r} is not a BearingGeometry in snq_core.bearing")
        return geom

    def fault_truth_hz(self, shaft_hz: float, channel: str | None = None) -> dict[str, float]:
        """**Continuous** ground truth: the fault frequencies from the geometry.

        This is what the project measures the error against, not the class label.
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
    """**Plain** decimation by ``factor``, without an anti-aliasing filter.

    ``anti_alias=True`` raises ``AntiAliasFilterUsedError`` on purpose: the
    signature exists so that whoever tries to filter before decimating gets an
    explanatory error instead of a silently wrong result.

    >>> undersample(np.arange(10.0), 3).tolist()
    [0.0, 3.0, 6.0, 9.0]
    """
    if anti_alias:
        raise AntiAliasFilterUsedError(
            "Decimating with an AAF destroys the super-Nyquist information under study. "
            "Use plain decimation; if you need to select a band, use a BAND-PASS "
            "AAF in the analog chain (snq_sim.AntiAliasFilter)."
        )
    if factor < 1:
        raise ValueError("factor >= 1")
    if not 0 <= offset < factor:
        raise ValueError("offset must be in [0, factor)")
    return np.asarray(x, dtype=float).ravel()[offset::factor]


def undersample_cosets(x: ArrayLike, factor: int, offsets: ArrayLike) -> list[NDArray[np.float64]]:
    """Multi-coset: one stream per integer sample offset.

    It is the discrete version of what the DPPI/TIMER of the nRF54L15 does in hardware.
    Each coset sees the same signal with a different sampling phase, and it is the
    difference between them that carries the alias order.
    """
    return [undersample(x, factor, offset=int(o)) for o in np.asarray(offsets).ravel()]


def effective_rate_hz(fs_original_hz: float, factor: int) -> float:
    return float(fs_original_hz) / int(factor)


def undersampling_report(fs_original_hz: float, factor: int, band_hz: tuple[float, float]) -> dict:
    """Summary of what undersampling does to a band of interest.

    Returns the effective rate, the resulting fold and whether the band collides with
    itself. Calling this before generating a split avoids spending hours on a factor
    that destroys the information.
    """
    from snq_core import fold_band  # local import: avoids a cycle when the module is loaded

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
    """Loads all manifests in a directory, indexed by name.

    Dataset manifests only. Screening catalogs (for example the classification
    of CWRU records) live in ``datasets/catalogs/`` and have a different schema.
    """
    out: dict[str, Manifest] = {}
    for p in sorted(Path(manifest_dir).glob("*.yaml")):
        m = Manifest.load(p)
        out[m.name] = m
    return out
