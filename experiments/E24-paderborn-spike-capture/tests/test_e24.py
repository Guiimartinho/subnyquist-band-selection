"""E24: spike removal, guard, per-bearing test and Holm."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "e24_run", Path(__file__).resolve().parents[1] / "run.py"
)
e24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e24)


def test_despike_removes_isolated_spikes_and_keeps_the_rest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    idx = np.arange(500, 19_500, 997)
    x[idx] += 25.0
    y = e24.despike(x)
    assert np.all(np.abs(y[idx]) < 6.0)
    changed = np.flatnonzero(y != x)
    assert set(idx.tolist()) <= set(changed.tolist())
    assert changed.size <= idx.size + 5  # almost nothing besides the spikes


def test_cluster_test_detects_a_consistent_gain_and_not_a_null():
    a = {("OR", f"A{i}"): 0.8 for i in range(6)} | {("IR", f"B{i}"): 0.7 for i in range(6)}
    b = {k: v - 0.2 for k, v in a.items()}
    res = e24.cluster_test(a, b, 500, 0)
    assert res["diff_points"] == pytest.approx(20.0)
    assert res["p_one_sided"] < 0.01
    null = e24.cluster_test(a, a, 500, 0)
    assert null["p_one_sided"] == 1.0  # Δ* = 0 counts against, as in E15


def test_holm():
    assert e24.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_post_decimation_despike_cleans_spikes_that_survive_decimation():
    """H3: a spike that falls on a kept sample stays isolated in the decimated stream."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=40_000)
    x[np.arange(400, 39_600, 1600)] += 30.0  # all at multiples of 4: they survive 4x
    xd = x[::4]
    y = e24.despike(xd)
    assert np.max(np.abs(y)) < 8.0
    assert np.max(np.abs(xd)) > 25.0


def test_notch_zeroes_only_its_band_and_the_ceiling_caps_the_search():
    """E24b: the notch removes 16 kHz and keeps 3 kHz; the arm's ceiling caps the chosen band."""
    fs = e24.FS
    t = np.arange(64000) / fs
    x = np.sin(2 * np.pi * 16000 * t) + np.sin(2 * np.pi * 3000 * t)
    y = e24.notch(x, fs, [[15000.0, 17000.0]])
    f = np.fft.rfftfreq(y.size, 1 / fs)
    sy, sx = np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x))
    assert sy[np.argmin(np.abs(f - 16000))] < 1e-6 * sx.max()
    assert sy[np.argmin(np.abs(f - 3000))] > 0.99 * sx[np.argmin(np.abs(f - 3000))]
    cfg = {"kurtogram": {"max_level": 2, "f_min_hz": 500.0}}
    noise = np.random.default_rng(0).normal(size=64000)
    assert e24.select_band(noise, fs, "sk", cfg, 8000.0).band_hz[1] <= 8000.0 + 1e-6
    assert e24.select_band(noise, fs, "sk", cfg).band_hz[1] <= fs / 2 + 1e-6
