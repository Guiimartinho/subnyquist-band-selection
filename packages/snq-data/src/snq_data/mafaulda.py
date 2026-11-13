"""MAFAULDA - Machinery Fault Database (UFRJ), bearing subsets.

Facts checked on the official page on 2026-11-11
(https://www02.smt.ufrj.br/~offshore/mfs/page_01.html):

- Sequences in CSV, 8 columns, 250,000 samples. The page declares **50 kHz** and 5 s, and says
  that the NI 9234 modules sample at 51.2 kHz.

**The actual rate is 51.2 kHz, not 50 kHz.** This was inferred on 2026-11-11 over the 1120
sequences (``FS_HZ``; the declared one is kept in ``DECLARED_FS_HZ``), from two independent
pieces of evidence:

1. **Tachometer against the file name**, which is the nominal rotation. The median ratio is
   0.9769 assuming 50 kHz (bias of -2.3 %) and 1.0004 assuming 51.2 kHz (IQR 0.9988 to
   1.0022). The names are multiples of 1.024 = 51.2/50.
2. **Mains line (120 Hz, 2nd harmonic of 60 Hz) in the microphone.** It falls at 117.18 Hz on
   the 50 kHz axis, exactly 120 x 50/51.2. This holds for 863 of the 1120 sequences within
   +-0.15 Hz, against 3 within +-0.15 Hz of 120 Hz.

Consequences: each sequence lasts 4.883 s. Frequencies in Hz computed with 50 kHz come out
2.3 % low. Order analysis from the tachometer is not affected, because tachometer and
vibration share the clock.
- Columns: tachometer; *underhang* bearing accelerometer (axial, radial, tangential);
  *overhang* bearing accelerometer (axial, radial, tangential); microphone.
- Bearing: 8 balls, ball diameter 0.7145 cm, cage diameter 2.8519 cm, contact angle
  not declared. The documented orders (FTF 0.3750; BPFO 2.9980; BPFI 5.0020;
  BSF 1.8710) match this geometry with zero angle (``computed_orders``).
- There is no inner-race fault: the bearing faults are ``outer_race``,
  ``ball_fault`` and ``cage_fault``, in the ``underhang`` and ``overhang`` bearings, with
  additional unbalance masses of 0, 6, 20 and 35 g.
- License and terms of use: **not declared** on the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FS_HZ = 51_200.0
"""Actual rate, inferred from the tachometer and the mains line (see the module docstring)."""
DECLARED_FS_HZ = 50_000.0
"""Rate declared on the official page; use only to compare with works that adopted it."""
BASE_URL = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda"
ARCHIVES = ("normal.tgz", "underhang.tgz", "overhang.tgz")

N_ELEMENTS = 8
BALL_DIAMETER_CM = 0.7145
PITCH_DIAMETER_CM = 2.8519
GEOM_RATIO = BALL_DIAMETER_CM / PITCH_DIAMETER_CM
DOCUMENTED_ORDERS = {"FTF": 0.3750, "BPFO": 2.9980, "BPFI": 5.0020, "BSF": 1.8710}

COLUMNS = (
    "tachometer",
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
    "microphone",
)
BEARINGS = ("underhang", "overhang")
MAX_ROTATION_HZ = 70.0
"""Above the highest rotation in the dataset (61.44 Hz); sets the tachometer hold-off."""
FAULT_FAMILY = {"outer_race": "BPFO", "ball_fault": "BSF", "cage_fault": "FTF"}

_LOAD = re.compile(r"^(\d+)g$")


def computed_orders(geom_ratio: float = GEOM_RATIO, n: int = N_ELEMENTS) -> dict[str, float]:
    """Fault orders from the geometry, with zero contact angle."""
    return {
        "FTF": 0.5 * (1 - geom_ratio),
        "BPFO": 0.5 * n * (1 - geom_ratio),
        "BPFI": 0.5 * n * (1 + geom_ratio),
        "BSF": (1 - geom_ratio**2) / (2 * geom_ratio),
    }


def parse_path(rel_path: str) -> tuple[str | None, str, int | None, float]:
    """``(bearing, fault, load_g, nominal_rotation_hz)`` from the relative path.

    Examples: ``normal/12.288.csv`` and ``underhang/outer_race/6g/30.1056.csv``. The file
    name is the nominal rotation in Hz.
    """
    parts = Path(rel_path).parts
    rotation = float(Path(rel_path).stem)
    if parts[0] == "normal":
        return None, "normal", None, rotation
    if parts[0] not in BEARINGS or len(parts) < 4:
        raise ValueError(f"path outside the MAFAULDA scheme: {rel_path}")
    m = _LOAD.match(parts[2])
    if parts[1] not in FAULT_FAMILY or m is None:
        raise ValueError(f"path outside the MAFAULDA scheme: {rel_path}")
    return parts[0], parts[1], int(m.group(1)), rotation


@dataclass(frozen=True)
class MafauldaRecord:
    """One sequence, always carrying its own provenance."""

    path: str
    bearing: str | None
    fault: str
    load_g: int | None
    nominal_rotation_hz: float
    data: NDArray[np.float32]
    fs_hz: float = FS_HZ

    def channel(self, name: str) -> NDArray[np.float64]:
        return self.data[:, COLUMNS.index(name)].astype(np.float64)

    def vibration(self, axis: str = "radial", side: str | None = None) -> NDArray[np.float64]:
        """Accelerometer of the faulty bearing (or ``side``), on the requested axis."""
        bearing = side or self.bearing or "underhang"
        return self.channel(f"{bearing}_{axis}")

    @property
    def fault_order(self) -> float | None:
        family = FAULT_FAMILY.get(self.fault)
        return DOCUMENTED_ORDERS[family] if family else None

    def tachometer_edges_s(self) -> NDArray[np.float64]:
        """Instants of the tachometer rising edges (one pulse per revolution).

        Uses hysteresis (rises above 60 % of the swing and only re-arms below 40 %) and a
        physical hold-off time: after each edge, it ignores new edges for
        ``1/(2*MAX_ROTATION_HZ)``. Without this, tachometer noise produces spurious edges: one
        sequence went as far as 6.3 x the nominal rotation with the simple threshold. A filter
        on the median of the intervals fails when there is one glitch per pulse, because the
        median becomes the glitch interval.
        """
        tach = self.channel("tachometer")
        lo_v, hi_v = float(tach.min()), float(tach.max())
        hi = lo_v + 0.6 * (hi_v - lo_v)
        lo = lo_v + 0.4 * (hi_v - lo_v)
        holdoff = int(self.fs_hz / (2.0 * MAX_ROTATION_HZ))
        edges: list[int] = []
        armed = bool(tach[0] < lo)
        for i in np.flatnonzero((tach >= hi) | (tach <= lo)):
            if edges and i - edges[-1] < holdoff:
                continue  # inside the hold-off nothing counts: neither edge nor re-arm (glitch)
            if armed and tach[i] >= hi:
                edges.append(int(i))
                armed = False
            elif not armed and tach[i] <= lo:
                armed = True
        e = np.asarray(edges, dtype=np.int64)
        if e.size < 3:
            raise ValueError(f"{self.path}: tachometer without enough pulses")
        return e / self.fs_hz

    def tachometer_rotation_hz(self) -> float:
        """Mean rotation from the pulses, to check against ``nominal_rotation_hz``."""
        edges = self.tachometer_edges_s()
        return float((edges.size - 1) / (edges[-1] - edges[0]))

    def speed_profile_hz(self) -> NDArray[np.float64]:
        """Instantaneous rotation in Hz, on the vibration grid (interpolated between pulses)."""
        edges = self.tachometer_edges_s()
        rot = 1.0 / np.diff(edges)
        t = np.arange(self.data.shape[0]) / self.fs_hz
        return np.interp(t, edges[:-1], rot)


def list_records(root: str | Path, subsets: tuple[str, ...] = ("normal", *BEARINGS)) -> list[str]:
    """Relative paths of all sequences in the subsets, in deterministic order."""
    root = Path(root)
    out: list[str] = []
    for sub in subsets:
        out.extend(p.relative_to(root).as_posix() for p in (root / sub).rglob("*.csv"))
    return sorted(out)


def load(rel_path: str, root: str | Path, cache_dir: str | Path | None = None) -> MafauldaRecord:
    """Loads one sequence. With ``cache_dir``, stores/reads a ``.npy`` copy (float32)."""
    bearing, fault, load_g, rotation = parse_path(rel_path)
    cache = Path(cache_dir) / Path(rel_path).with_suffix(".npy") if cache_dir else None
    if cache is not None and cache.exists():
        data = np.load(cache)
    else:
        data = np.loadtxt(Path(root) / rel_path, delimiter=",", dtype=np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, data)
    if data.ndim != 2 or data.shape[1] != len(COLUMNS):
        raise ValueError(f"{rel_path}: expected {len(COLUMNS)} columns, got {data.shape}")
    return MafauldaRecord(
        path=rel_path,
        bearing=bearing,
        fault=fault,
        load_g=load_g,
        nominal_rotation_hz=rotation,
        data=data,
    )
