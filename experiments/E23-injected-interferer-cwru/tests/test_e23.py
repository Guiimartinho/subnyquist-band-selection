"""E23: interferer and control are what they claim to be; the prediction is the fold arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from snq_core.sampling import alias_frequency_hz

_spec = importlib.util.spec_from_file_location(
    "e23_run", Path(__file__).resolve().parents[1] / "run.py"
)
e23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e23)

FS = 48_000.0
TAU = 0.0005
CFG = {
    "injection": {"burst_rate_hz": 97.7, "burst_tau_s": TAU},
    "prediction": {"capture_share": 0.5, "clear_share": 0.05},
}
N = 1 << 16


def kurt(x: np.ndarray) -> float:
    return float(np.mean(x**4) / np.mean(x**2) ** 2)


def test_interferer_has_unit_power_and_lives_near_its_carrier():
    for kind in ("impulsive", "surrogate"):
        s = e23.interferer(N, FS, kind, 16_000.0, CFG, [1, 2, 3])
        assert abs(np.mean(s**2) - 1.0) < 1e-9
        spec = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(s.size, 1 / FS)
        near = np.sum(spec[np.abs(f - 16_000.0) <= 4 * e23.burst_bandwidth_hz(TAU)])
        assert near / np.sum(spec) > 0.9, kind


def test_surrogate_keeps_the_power_spectrum_and_loses_the_impulsiveness():
    imp = e23.interferer(N, FS, "impulsive", 16_000.0, CFG, [7, 8, 9])
    sur = e23.interferer(N, FS, "surrogate", 16_000.0, CFG, [7, 8, 9])
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(sur)), np.abs(np.fft.rfft(imp)), rtol=1e-6, atol=1e-6
    )
    assert kurt(imp) > 6.0
    # the spectrum is a line spectrum (periodic train): one surrogate realization is a repeated
    # random waveform, and its temporal kurtosis fluctuates; the mean over realizations goes to ~3
    ks = [kurt(e23.interferer(N, FS, "surrogate", 16_000.0, CFG, [s, 8, 9])) for s in range(12)]
    assert max(ks) < 4.0
    assert abs(np.mean(ks) - 3.0) < 0.25


def test_fold_matches_snq_core_alias_map():
    f = np.array([100.0, 7_000.0, 13_000.0, 16_000.0, 19_000.0, 22_000.0])
    for fs_eff in (48_000.0, 24_000.0, 16_000.0, 9_600.0, 4_000.0):
        np.testing.assert_allclose(
            e23.fold_to_baseband_hz(f, fs_eff), alias_frequency_hz(f, fs_eff), atol=1e-9
        )


def lorentzian_share(fc_hz, factor, f_min_hz, f_max_hz):
    """Analytic reference: Lorentzian PSD of the burst, folded by the snq_core map."""
    f = np.arange(0.0, FS / 2.0, 1.0)
    gamma = 1.0 / (2 * np.pi * TAU)
    psd = 1.0 / (1.0 + ((f - fc_hz) / gamma) ** 2) + 1.0 / (1.0 + ((f + fc_hz) / gamma) ** 2)
    img = alias_frequency_hz(f, FS / factor)
    inside = (img >= f_min_hz) & (img <= f_max_hz)
    return float(psd[inside].sum() / psd.sum())


def test_folded_share_matches_the_analytic_lorentzian_fold():
    s = e23.interferer(N, FS, "impulsive", 16_000.0, CFG, [4, 5, 6])
    cases = [(1, 10_000.0), (2, 10_000.0), (2, 5_000.0), (3, 8_000.0), (4, 2_500.0)]
    for factor, ceiling in cases:
        got = e23.folded_share_in_search(s, FS, factor, 500.0, ceiling)
        ref = lorentzian_share(16_000.0, factor, 500.0, ceiling)
        assert abs(got - ref) < 0.05, (factor, ceiling, got, ref)
    # 1x: no folding, only the tail leaks; 2x with a 10 kHz ceiling: image at 8 kHz, inside
    assert e23.folded_share_in_search(s, FS, 1, 500.0, 10_000.0) < 0.05
    assert e23.folded_share_in_search(s, FS, 2, 500.0, 10_000.0) > 0.5
    # 2x with a 5 kHz ceiling: the same image (8 kHz) lies above the ceiling -> no predicted capture
    assert e23.folded_share_in_search(s, FS, 2, 500.0, 5_000.0) < 0.05


def test_interferer_line_flags_burst_rate_harmonics_only():
    cfg = {"injection": {"burst_rate_hz": 97.7}}
    assert e23.on_interferer_line(97.0, cfg, 2.0)
    assert e23.on_interferer_line(195.4, cfg, 2.0)
    assert not e23.on_interferer_line(107.4, cfg, 2.0)  # CWRU BPFO at ~1797 rpm


def test_prediction_class_has_a_borderline_band():
    assert e23.prediction_class(0.7, CFG) == 1
    assert e23.prediction_class(0.01, CFG) == 0
    assert e23.prediction_class(0.2, CFG) == -1


def test_spike_train_is_isolated_single_samples_at_the_requested_rate():
    rng = np.random.default_rng(3)
    s = e23.spike_train(1_000_000, FS, 1300.0, -1.0, rng)
    idx = np.flatnonzero(s)
    assert set(np.unique(s[idx])) == {-1.0}
    assert np.all(np.diff(idx) > 1)  # never two adjacent spikes
    rate = idx.size / (1_000_000 / FS)
    assert abs(rate - 1300.0) / 1300.0 < 0.1


def test_despike_removes_most_injected_spikes_and_leaves_smooth_signal():
    rng = np.random.default_rng(4)
    t = np.arange(200_000) / FS
    x = np.sin(2 * np.pi * 3000.0 * t) + 0.05 * rng.standard_normal(t.size)
    mad = e23.residual_mad(x)
    spikes = e23.spike_train(t.size, FS, 1300.0, -1.0, rng)
    y = e23.despike(x + 16 * mad * spikes)
    at = np.flatnonzero(spikes)
    removed = np.mean(np.abs(y[at] - x[at]) < 8 * mad)
    # spikes two samples away from another one escape the criterion (the middle neighbor deviates
    # as much as they do); at 1300/s and 48 kHz they are about 3 %
    assert removed > 0.9
    assert np.mean(np.abs(e23.despike(x) - x) > 0) < 0.002  # the clean signal is barely touched
