"""Ottawa 2018 dataset — bearing vibration under variable speed.

200 kSPS, accelerometer on channel 1 and **encoder on channel 2**. It is the only one of the
candidates with an angular reference, which Smith & Randall call *"essential"* for variable
speed — and without which order analysis does not work.

Three facts that **are not on the dataset page** and were determined by a dry run
on 2026-09-03:

- **The encoder has 1024 pulses per revolution.** No value is documented; 1024 is
  the only candidate that makes order 1 (shaft) appear in the order spectrum together with
  the fault frequency.
- **The informative band is at 6.8–10 kHz**, with a kurtosis of 34. The 97% of the energy
  below 50 Hz is structural content, not fault.
- **The nominal headroom of 200 kSPS is misleading.** Only 0.1% of the energy is above
  20 kHz, so undersampling only starts to test anything from the effective Nyquist rate
  of the informative band onwards, around 20 kSPS.

License CC BY 4.0. Programmatic access through the public Mendeley API, without registration.
"""

from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FS_HZ = 200_000.0
PULSES_PER_REV = 1024
"""Determined empirically; see the module docstring."""

API = "https://data.mendeley.com/public-api/datasets/v43hmbwxpm/files?folder_id=root&version=1"
USER_AGENT = "subnyquist-thesis/0.1 (academic use)"

CONDITIONS = {"H": "healthy", "I": "inner_race", "O": "outer_race"}
SCENARIOS = ("A", "B", "C", "D")
"""A: accelerating · B: decelerating · C: accelerates then decelerates · D: the reverse."""

# Fault orders of this test rig, derived from the dry run: the BPFI measured at 5.43 implies
# 9 elements and a ratio of ~0.2067, and the BPFO follows from that. Check against the
# dataset documentation before publishing.
N_ELEMENTS = 9
GEOM_RATIO = 0.2067
FAULT_ORDERS = {
    "inner_race": (N_ELEMENTS / 2) * (1 + GEOM_RATIO),
    "outer_race": (N_ELEMENTS / 2) * (1 - GEOM_RATIO),
    "cage": 0.5 * (1 - GEOM_RATIO),
}


@dataclass(frozen=True)
class OttawaRecord:
    """A record, always carrying its own provenance."""

    name: str
    condition: str
    scenario: str
    trial: int
    vibration: NDArray[np.float64]
    encoder: NDArray[np.float64]
    fs_hz: float = FS_HZ

    @property
    def duration_s(self) -> float:
        return self.vibration.size / self.fs_hz

    @property
    def fault_order(self) -> float | None:
        return FAULT_ORDERS.get(self.condition)

    def cite(self) -> str:
        return f"Ottawa {self.name} ({self.condition}, scenario {self.scenario})"

    def speed_profile_hz(self, ppr: int = PULSES_PER_REV) -> NDArray[np.float64]:
        """Instantaneous rotation in Hz, sampled on the same grid as the vibration.

        Extracted from the rising edges of the encoder and interpolated. It is what feeds
        `snq_recon.angular_resample`.
        """
        enc = self.encoder
        thr = 0.5 * (float(enc.max()) + float(enc.min()))
        edges = np.flatnonzero((enc[:-1] < thr) & (enc[1:] >= thr)) / self.fs_hz
        if edges.size < 3:
            raise ValueError(f"{self.name}: encoder without enough pulses")
        rot = 1.0 / (np.diff(edges) * ppr)
        t = np.arange(self.vibration.size) / self.fs_hz
        return np.interp(t, edges[:-1], rot)

    def speed_range_rpm(self, ppr: int = PULSES_PER_REV) -> tuple[float, float]:
        r = self.speed_profile_hz(ppr) * 60.0
        return float(r.min()), float(r.max())


def _file_index() -> dict[str, str]:
    """File name → download URL, through the public Mendeley API."""
    req = urllib.request.Request(API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310
        data = json.loads(r.read())
    return {f["filename"]: f["content_details"]["download_url"] for f in data}


def download(name: str, dest_dir: str | Path, *, tries: int = 4) -> Path:
    """Downloads a record by name, for example ``"I-A-1.mat"``."""
    dest = Path(dest_dir) / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _file_index()[name]

    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
                expected = r.headers.get("Content-Length")
                blob = r.read()
            if expected and len(blob) != int(expected):
                raise OSError(f"truncated: {len(blob)} of {expected}")
            if not blob.startswith(b"MATLAB"):
                raise OSError("response is not a MAT file")
            dest.write_bytes(blob)
            return dest
        # IncompleteRead inherits from HTTPException, not from OSError (see snq_data.cwru).
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(3.0 * (attempt + 1))
    raise OSError(f"{name}: failed after {tries} attempts ({last})")


def load(name: str, data_dir: str | Path) -> OttawaRecord:
    """Loads an already downloaded record."""
    from scipy.io import loadmat

    path = Path(data_dir) / name
    mat = loadmat(io.BytesIO(path.read_bytes()))
    stem = name.replace(".mat", "")
    cond, scen, trial = stem.split("-")
    return OttawaRecord(
        name=stem,
        condition=CONDITIONS[cond],
        scenario=scen,
        trial=int(trial),
        vibration=np.ravel(mat["Channel_1"]).astype(float),
        encoder=np.ravel(mat["Channel_2"]).astype(float),
    )


def record_names(
    conditions: tuple[str, ...] = ("H", "I", "O"),
    scenarios: tuple[str, ...] = SCENARIOS,
    trials: tuple[int, ...] = (1, 2, 3),
) -> list[str]:
    """Record names for the requested subset, in deterministic order."""
    return [f"{c}-{s}-{t}.mat" for c in conditions for s in scenarios for t in trials]
