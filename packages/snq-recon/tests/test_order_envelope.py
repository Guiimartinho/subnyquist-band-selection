"""Test of the envelope in the order domain.

The property of interest is the one E07 exploits: under **variable** shaft speed, the
fault frequency in hertz moves and the order does not. A test that uses a constant
speed cannot tell the right code from the wrong one, so here the speed sweeps.
"""

from __future__ import annotations

import numpy as np
import pytest
from snq_recon import envelope_order_spectrum


def _impactos_modulados(fs_hz=50_000.0, seconds=2.0, order=5.43, f_res=8000.0):
    """Impact train locked to an order, under a shaft speed that sweeps 15→30 Hz."""
    n = int(fs_hz * seconds)
    rot = np.linspace(15.0, 30.0, n)  # variable shaft speed: the whole point of the test
    revs = np.cumsum(rot) / fs_hz

    # One impact every 1/order of a revolution, not every fixed time interval.
    fase = revs * order
    impacto = np.zeros(n)
    impacto[np.flatnonzero(np.diff(np.floor(fase)) > 0)] = 1.0

    # Resonance fixed in HERTZ - it does not follow the shaft speed. That is why the envelope
    # has to be taken in time, before the angular resampling.
    decay = np.exp(-np.arange(int(0.002 * fs_hz)) * 900.0 / fs_hz)
    ring = decay * np.sin(2 * np.pi * f_res * np.arange(decay.size) / fs_hz)
    return np.convolve(impacto, ring)[:n], rot


def test_ordem_de_falha_aparece_apesar_da_rotacao_variar():
    x, rot = _impactos_modulados(order=5.43)
    spec = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    sel = spec.freqs_hz >= 2.0
    pico = float(spec.freqs_hz[sel][np.argmax(spec.magnitude[sel])])
    assert pico == pytest.approx(5.43, abs=0.05)


def test_a_mesma_falha_em_hertz_se_espalha():
    """Control: in hertz the line smears because the shaft speed sweeps 15→30 Hz.

    Without this contrast the test above does not prove that the angular resampling did
    anything. The metric is the concentration in a **relative** window around the peak, and
    not max/mean, because the two spectra have very different numbers of lines and
    max/mean rewards the one with fewer lines.
    """
    from snq_recon import envelope_spectrum

    x, rot = _impactos_modulados(order=5.43)
    ordem = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    hertz = envelope_spectrum(x, 50_000.0, (6000.0, 11000.0))

    def concentracao(eixo, mag, lo):
        sel = eixo >= lo
        eixo, mag = eixo[sel], mag[sel]
        pico = float(eixo[np.argmax(mag)])
        perto = np.abs(eixo - pico) <= 0.02 * pico
        largo = (eixo >= 0.5 * pico) & (eixo <= 2.0 * pico)
        return float(mag[perto].sum() / mag[largo].sum())

    # In hertz the fault occupies 81-163 Hz (5.43 x 15 to 5.43 x 30); in orders, a single line.
    assert concentracao(ordem.freqs_hz, ordem.magnitude, 2.0) > 3.0 * concentracao(
        hertz.freqs_hz, hertz.magnitude, 20.0
    )


def test_unidades_do_objeto_devolvido_sao_ordens():
    x, rot = _impactos_modulados()
    spec = envelope_order_spectrum(x, 50_000.0, rot, (6000.0, 11000.0), samples_per_rev=64)
    # fs_hz carries samples_per_rev, and the largest "frequency" is the Nyquist order.
    assert spec.fs_hz == pytest.approx(64.0)
    assert spec.freqs_hz[-1] == pytest.approx(32.0, rel=0.01)
