"""The E20 interventions remove what they claim to remove, and only that."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "e20_run", Path(__file__).resolve().parents[1] / "run.py"
)
e20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e20)

FS = 200_000.0
N = 200_000  # 1 s: 1 Hz resolution, tones at the bin centre


def tone(f_hz: float) -> np.ndarray:
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * f_hz * t)


def amp(x: np.ndarray, f_hz: float) -> float:
    xf = np.abs(np.fft.rfft(x)) * 2 / x.size
    return float(xf[int(round(f_hz))])


def test_none_returns_input_unchanged():
    x = tone(80_000.0)
    assert e20.apply_intervention(x, FS, {"kind": "none"}, 4) is x


def test_stop_removes_only_its_zone():
    x = tone(80_000.0) + tone(3_000.0) + tone(30_000.0)
    y = e20.apply_intervention(x, FS, {"kind": "stop", "bands_hz": [[75_000.0, 87_250.0]]}, 4)
    assert amp(y, 80_000.0) < 1e-9
    assert abs(amp(y, 3_000.0) - 1.0) < 1e-9
    assert abs(amp(y, 30_000.0) - 1.0) < 1e-9


def test_aaf_ideal_kills_everything_above_effective_nyquist():
    x = tone(80_000.0) + tone(30_000.0) + tone(10_000.0)
    y = e20.apply_intervention(x, FS, {"kind": "aaf_ideal"}, 4)  # effective Nyquist = 25 kHz
    assert amp(y, 80_000.0) < 1e-9
    assert amp(y, 30_000.0) < 1e-9
    assert abs(amp(y, 10_000.0) - 1.0) < 1e-9


def test_zone_z3_folds_onto_search_band_at_4x_and_2x():
    """The arithmetic that motivated E20: 80 kHz shows up at 20 kHz at both 4x and 2x."""
    x = tone(80_000.0)
    for factor in (2, 4):
        d = x[::factor]
        xf = np.abs(np.fft.rfft(d))
        f = np.fft.rfftfreq(d.size, d=factor / FS)
        assert abs(f[int(np.argmax(xf))] - 20_000.0) < 1.0


def test_mcnemar_exact_p_matches_known_values():
    assert e20.mcnemar_exact_p(0, 0) == 1.0
    assert abs(e20.mcnemar_exact_p(19, 1) - 2 * 21 / 2**20) < 1e-12
    assert e20.mcnemar_exact_p(5, 5) == 1.0


def test_despike_and_median3_remove_an_isolated_spike_without_a_band_mask():
    import numpy as np

    fs = 200_000.0
    t = np.arange(20_000) / fs
    x = np.sin(2 * np.pi * 3000.0 * t)
    y = x.copy()
    y[5_000] -= 5.0  # isolated single-sample spike
    for kind in ("despike", "median3"):
        z = e20.apply_intervention(y, fs, {"kind": kind}, factor=4)
        # despike interpolates; the median of 3 returns one of the neighbours (no interpolation)
        lo, hi = sorted((x[4_999], x[5_001]))
        assert lo - 1e-9 <= z[5_000] <= hi + 1e-9, kind
        assert np.max(np.abs(np.delete(z - x, [4_999, 5_000, 5_001]))) < 0.05, kind


def test_analyse_selector_default_is_the_plain_kurtogram_and_robust_ones_run():
    """Per-arm selector (2026-11-12): kurtogram by default; robust ones run on the same search."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "base.yaml").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(0)
    fs, n = 50_000.0, 50_000
    x = rng.normal(size=n)
    x[::997] += 30.0  # isolated spikes
    spd = np.full(n, 20.0)
    k0, _ = e20.analyse(x, fs, spd, cfg, 25_000.0)
    k1, _ = e20.analyse(x, fs, spd, cfg, 25_000.0, "sk")
    assert (k0.f_lo_hz, k0.f_hi_hz) == (k1.f_lo_hz, k1.f_hi_hz)
    for sel in ("ses_negentropy", "gini"):
        k, spec = e20.analyse(x, fs, spd, cfg, 25_000.0, sel)
        assert k.f_hi_hz <= 25_000.0 and spec.magnitude.size > 0


def test_post_despike_config_keeps_the_signal_before_decimation():
    """The after-decimation arm leaves the full-rate signal untouched (kind none)."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "post_despike.yaml").read_text(
            encoding="utf-8"
        )
    )
    spec = cfg["interventions"]["despike_decimado"]
    x = np.random.default_rng(0).normal(size=1000)
    assert np.array_equal(e20.apply_intervention(x, FS, spec, 4), x)
    assert spec["post"] == "despike"
