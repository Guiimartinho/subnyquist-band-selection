"""High resolution and order analysis."""

import numpy as np
import pytest
from snq_recon import (
    angular_resample,
    esprit,
    estimate_tone_esprit,
    estimate_tone_hz,
    fault_orders,
    order_spectrum,
)

FS = 1000.0


def _tone(f, n=512, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS + phase)


def test_esprit_supera_a_grade_da_fft():
    """Frequency between bins: the FFT is stuck on the grid, ESPRIT is not.

    With N=512 and fs=1000 the FFT resolution is 1.95 Hz. A line at 123.4 Hz falls between
    bins, and that is exactly where subspace methods win.
    """
    x = _tone(123.4)
    assert abs(estimate_tone_esprit(x, FS) - 123.4) < 0.05


def test_esprit_separa_duas_raias_proximas():
    """Two sinusoids less than one bin apart — the FFT does not separate them."""
    x = _tone(100.0) + _tone(101.0)
    f = esprit(x, n_sources=2, fs_hz=FS)
    assert f.size == 2
    assert abs(f[0] - 100.0) < 0.3
    assert abs(f[1] - 101.0) < 0.3


def test_esprit_com_subespaco_grande_demais_e_erro():
    """Negative test: a subspace larger than the signal makes no sense."""
    with pytest.raises(ValueError, match="subspace"):
        esprit(_tone(100.0, n=64), n_sources=1, fs_hz=FS, subspace=64)


def test_ordens_de_falha_nao_dependem_do_regime():
    """It is the reason order analysis exists: the order is a constant of the geometry."""
    o = fault_orders(geom_ratio=0.2034, n_elements=9)
    assert o["BPFO"] + o["BPFI"] == pytest.approx(9.0)
    assert o["BPFO"] < o["BPFI"]
    assert o["FTF"] == pytest.approx(o["BPFO"] / 9.0)


def test_reamostragem_angular_fixa_a_ordem_sob_rotacao_variavel():
    """The result that justifies the module.

    A signal whose frequency follows the shaft speed spreads energy over the frequency
    spectrum when the speed varies. In the **order** spectrum it becomes a single
    line again.
    """
    n, fs = 20_000, 5_000.0
    t = np.arange(n) / fs
    # shaft speed rising 30% over the record
    rot = 20.0 * (1.0 + 0.3 * t / t[-1])
    fase = 2 * np.pi * np.cumsum(rot) / fs
    x = np.sin(3.5 * fase)  # component at order 3.5

    ang, spr = angular_resample(x, fs, rot, samples_per_rev=64)
    ordens, mag = order_spectrum(ang, spr)
    pico = ordens[np.argmax(mag[1:]) + 1]
    assert abs(pico - 3.5) < 0.05, f"peak at {pico:.3f}, expected 3.5"


def test_reamostragem_recusa_registro_curto_demais():
    """Negative test: less than one revolution defines no order at all."""
    with pytest.raises(ValueError, match="revolution"):
        angular_resample(np.zeros(100), 1000.0, shaft_hz=1.0)


def test_reamostragem_recusa_rotacao_nula():
    with pytest.raises(ValueError, match="positive"):
        angular_resample(np.zeros(1000), 1000.0, shaft_hz=np.zeros(1000))


def test_esprit_e_fft_concordam_em_alto_snr():
    """Sanity check: with high SNR and a line on a bin, the two have to give the same result."""
    x = _tone(125.0)  # 125 Hz falls exactly on a bin with N=512, fs=1000
    assert abs(estimate_tone_esprit(x, FS) - estimate_tone_hz(x, FS)) < 0.5
