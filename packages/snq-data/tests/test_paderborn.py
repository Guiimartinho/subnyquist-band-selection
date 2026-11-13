"""Paderborn: geometry, naming scheme and reading a synthetic .mat with the same structure."""

from __future__ import annotations

import numpy as np
import pytest
from snq_data import paderborn as P


def test_computed_orders_and_distance_to_shaft_harmonics():
    o = P.computed_orders()
    assert o["BPFO"] == pytest.approx(3.0543, abs=1e-4)
    assert o["BPFI"] == pytest.approx(4.9457, abs=1e-4)
    # the reason for the tight tolerance: both lie ~0.054 order away from a shaft harmonic
    assert abs(o["BPFO"] - 3) == pytest.approx(0.0543, abs=1e-4)
    assert abs(o["BPFI"] - 5) == pytest.approx(0.0543, abs=1e-4)


def test_bearing_tables_match_the_official_counts():
    kinds = [v[0] for v in P.BEARINGS.values()]
    assert (kinds.count("healthy"), kinds.count("artificial"), kinds.count("real")) == (6, 12, 14)


def test_parse_name_scheme():
    assert P.parse_name("N15_M07_F10_KA01_3") == ("N15_M07_F10", "KA01", 3)
    with pytest.raises(ValueError):
        P.parse_name("N12_M07_F10_KA01_3")
    with pytest.raises(ValueError):
        P.parse_name("N15_M07_F10_KX99_3")


def _fake_mat(path, stem: str, n_vib: int = 256001, n_spd: int = 16001) -> None:
    from scipy.io import savemat

    fields = [("Name", "O"), ("Type", "O"), ("Data", "O"), ("Unit", "O"), ("Raster", "O")]
    y = np.empty((1, 2), dtype=fields)
    rng = np.random.default_rng(0)
    for i, (name, raster, n, mean) in enumerate(
        (("vibration_1", "HostService", n_vib, 0.0), ("speed", "Mech_4kHz", n_spd, 1500.0))
    ):
        y[0, i] = (name, np.int32(4), (mean + rng.normal(size=n)).reshape(1, -1), "", raster)
    x = np.empty((1, 1), dtype=fields)
    x[0, 0] = ("", np.int32(4), np.zeros((1, 3)), "", "HostService")
    savemat(path, {stem: {"Info": {}, "X": x, "Y": y, "Description": ""}})


def test_load_measurement_reads_the_struct(tmp_path):
    stem = "N15_M07_F10_KA01_1"
    (tmp_path / "KA01").mkdir()
    _fake_mat(tmp_path / "KA01" / f"{stem}.mat", stem)
    m = P.load_measurement(tmp_path, "KA01", "N15_M07_F10", 1)
    assert m.vibration.size == 256001 and m.speed_rpm.size == 16001
    v = m.speed_profile_hz()
    assert v.size == m.vibration.size
    assert float(np.mean(v)) == pytest.approx(25.0, abs=0.05)


def test_load_measurement_trims_longer_records_to_4s(tmp_path):
    """Fixed 64 kHz rate: a longer record is just longer, and it is cut to 4 s."""
    stem = "N15_M07_F10_KA01_3"
    (tmp_path / "KA01").mkdir()
    _fake_mat(tmp_path / "KA01" / f"{stem}.mat", stem, n_vib=262062)
    m = P.load_measurement(tmp_path, "KA01", "N15_M07_F10", 3)
    assert m.vibration.size == P.N_VIB


@pytest.mark.parametrize("n_vib,n_spd", [(256001, 8001), (200000, 16001)])
def test_load_measurement_rejects_short_channels(tmp_path, n_vib, n_spd):
    stem = "N15_M07_F10_KA01_2"
    (tmp_path / "KA01").mkdir()
    _fake_mat(tmp_path / "KA01" / f"{stem}.mat", stem, n_vib=n_vib, n_spd=n_spd)
    with pytest.raises(ValueError):
        P.load_measurement(tmp_path, "KA01", "N15_M07_F10", 2)
