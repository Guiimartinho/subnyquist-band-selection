"""Paderborn - KAt-DataCenter (Paderborn University), 6203 bearings.

Facts checked on 2026-11-12 on the official pages and in the PHME16 paper (Lessmeier, Kimotho,
Zimmer and Sextro, 2016). The details are in ``datasets/manifests/paderborn.yaml``.

- **License:** CC BY-NC 4.0, with mandatory citation. The raw data is not versioned.
- **Bearings:** 32, of which 6 are healthy (K001-K006), 12 have artificial damage and 14 have
  real damage from accelerated lifetime tests (tables 4 and 5 of PHME16).
- **Measurements:** 4 operating settings (``SETTINGS``), with 20 measurements of 4 s each.
- **File:** ``<code>/<setting>_<code>_<n>.mat``, in MATLAB 5. The single variable is named
  after the file and is a struct with ``X`` (time axes) and ``Y`` (signals).
  - ``vibration_1`` and the two phase currents are at 64 kHz (256001 samples).
  - ``speed`` (rpm), ``force`` and ``torque`` are at 4 kHz (16001 samples).
- **Geometry:** table 1 of PHME16 (FAG), with 8 balls, d = 6.75 mm, D = 28.55 mm and contact
  angle 0°.
  - According to PHME16, across the manufacturers used (FAG, MTK, IBU/IBB) "the characteristic
    kinematic frequencies do not vary more than 1-2%".
  - BPFO (3.054) and BPFI (4.946) lie 0.054 order away from shaft harmonics 3 and 5, so the
    order tolerance has to stay below ~0.027.

**Time base (checked on 2026-11-12, over the 2560 files).**
- The number of 64 kHz samples varies: 256001 in 1042 files, up to 262062 in the longest ones.
- The ``HostService`` axis always runs from 0 to 4 s, which would suggest a variable rate.
- **The rate is a fixed 64 kHz.** The PMSM has 4 pole pairs, so the current rotates at exactly
  4× the rotation measured on the 4 kHz channel. With a fixed 64 kHz, the ratio between the
  two has median 0.99998 (P1 0.99936; P99 1.00057), and only 1 file exceeds 0.5 %. Under the
  time-axis hypothesis, 185 files exceed 0.5 %, up to 8 %.
- The longer records are just longer. The loader cuts the vibration to the first 256001
  samples (4 s), aligned with the rotation channel.
- **One file does not open in scipy:** ``KA08/N15_M01_F10_KA08_2.mat`` ("Expecting matrix here").
  The loader raises ``ValueError``, and the caller skips the measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_HZ = 64_000.0
SPEED_FS_HZ = 4_000.0
N_ELEMENTS = 8
BALL_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
GEOM_RATIO = BALL_DIAMETER_MM / PITCH_DIAMETER_MM

SETTINGS: dict[str, dict[str, float]] = {
    "N15_M07_F10": {"rpm": 1500.0, "torque_nm": 0.7, "radial_force_n": 1000.0},
    "N09_M07_F10": {"rpm": 900.0, "torque_nm": 0.7, "radial_force_n": 1000.0},
    "N15_M01_F10": {"rpm": 1500.0, "torque_nm": 0.1, "radial_force_n": 1000.0},
    "N15_M07_F04": {"rpm": 1500.0, "torque_nm": 0.7, "radial_force_n": 400.0},
}
MEASUREMENTS = tuple(range(1, 21))

# code -> (origin, component, damage extent level). PHME16, tables 4 and 5.
BEARINGS: dict[str, tuple[str, str, int]] = {
    **{f"K00{i}": ("healthy", "none", 0) for i in range(1, 7)},
    "KA01": ("artificial", "OR", 1),
    "KA03": ("artificial", "OR", 2),
    "KA05": ("artificial", "OR", 1),
    "KA06": ("artificial", "OR", 2),
    "KA07": ("artificial", "OR", 1),
    "KA08": ("artificial", "OR", 2),
    "KA09": ("artificial", "OR", 2),
    "KI01": ("artificial", "IR", 1),
    "KI03": ("artificial", "IR", 1),
    "KI05": ("artificial", "IR", 1),
    "KI07": ("artificial", "IR", 2),
    "KI08": ("artificial", "IR", 2),
    "KA04": ("real", "OR", 1),
    "KA15": ("real", "OR", 1),
    "KA16": ("real", "OR", 2),
    "KA22": ("real", "OR", 1),
    "KA30": ("real", "OR", 1),
    "KB23": ("real", "IR+OR", 2),
    "KB24": ("real", "IR+OR", 3),
    "KB27": ("real", "OR+IR", 1),
    "KI04": ("real", "IR", 1),
    "KI14": ("real", "IR", 1),
    "KI16": ("real", "IR", 3),
    "KI17": ("real", "IR", 1),
    "KI18": ("real", "IR", 2),
    "KI21": ("real", "IR", 1),
}

_NAME = re.compile(r"^(N\d\d_M\d\d_F\d\d)_(K[0-9A-Z]{3})_(\d{1,2})$")


def computed_orders(geom_ratio: float = GEOM_RATIO, n: int = N_ELEMENTS) -> dict[str, float]:
    """Fault orders from the geometry, with zero contact angle."""
    return {
        "FTF": 0.5 * (1 - geom_ratio),
        "BPFO": 0.5 * n * (1 - geom_ratio),
        "BPFI": 0.5 * n * (1 + geom_ratio),
        "BSF": (1 - geom_ratio**2) / (2 * geom_ratio),
    }


def parse_name(stem: str) -> tuple[str, str, int]:
    """``N15_M07_F10_KA01_3`` -> (setting, code, measurement)."""
    m = _NAME.match(stem)
    if not m or m.group(1) not in SETTINGS or m.group(2) not in BEARINGS:
        raise ValueError(f"name outside the KAt-DataCenter scheme: {stem}")
    return m.group(1), m.group(2), int(m.group(3))


def measurement_path(root: str | Path, code: str, setting: str, n: int) -> Path:
    if code not in BEARINGS or setting not in SETTINGS or n not in MEASUREMENTS:
        raise ValueError(f"nonexistent measurement: {code} {setting} {n}")
    return Path(root) / code / f"{setting}_{code}_{n}.mat"


def _flat(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.ravel()[0]
    return v


@dataclass(frozen=True)
class Measurement:
    code: str
    setting: str
    index: int
    vibration: np.ndarray
    speed_rpm: np.ndarray
    fs_hz: float = FS_HZ
    speed_fs_hz: float = SPEED_FS_HZ

    def speed_profile_hz(self) -> np.ndarray:
        """Rotation in Hz on the vibration time axis (linear interpolation of the 4 kHz channel)."""
        t_v = np.arange(self.vibration.size) / self.fs_hz
        t_s = np.arange(self.speed_rpm.size) / self.speed_fs_hz
        return np.interp(t_v, t_s, self.speed_rpm) / 60.0


N_VIB = 256_001  # 4 s at 64 kHz, aligned with the 16001 rotation samples at 4 kHz


def load_measurement(
    root: str | Path,
    code: str,
    setting: str,
    n: int,
    n_samples: int = N_VIB,
    min_samples: int | None = None,
) -> Measurement:
    """Reads a measurement, with the vibration cut to 4 s at a fixed 64 kHz (see module docstring).

    Checks the variable name, the rasters, the duration of the rotation channel and the minimum
    length of the vibration. An unreadable file becomes a ``ValueError``.
    """
    from scipy.io import loadmat

    path = measurement_path(root, code, setting, n)
    try:
        m = loadmat(path, squeeze_me=False, struct_as_record=False)
    except (TypeError, ValueError) as exc:  # KA08/N15_M01_F10_KA08_2.mat
        raise ValueError(f"{path.name}: unreadable by scipy ({exc})") from exc
    keys = [k for k in m if not k.startswith("__")]
    if keys != [path.stem]:
        raise ValueError(f"{path.name}: unexpected variables {keys}")
    s = m[keys[0]][0, 0]
    sig = {}
    for i in range(s.Y.shape[1]):
        e = s.Y[0, i]
        sig[str(_flat(e.Name))] = (str(_flat(e.Raster)), np.asarray(e.Data, dtype=float).ravel())
    for name, raster in (("vibration_1", "HostService"), ("speed", "Mech_4kHz")):
        if name not in sig or sig[name][0] != raster:
            raise ValueError(f"{path.name}: channel {name} missing or outside raster {raster}")
    vib, spd = sig["vibration_1"][1], sig["speed"][1]
    dur_s = (spd.size - 1) / SPEED_FS_HZ
    if not 3.99 < dur_s < 4.01:
        raise ValueError(f"{path.name}: rotation signal of {dur_s:.4f} s, expected 4 s")
    lo = N_VIB - 2 if min_samples is None else int(min_samples)
    if vib.size < lo:
        raise ValueError(f"{path.name}: vibration with {vib.size} samples, fewer than {lo}")
    return Measurement(code, setting, n, vib[:n_samples], spd)
