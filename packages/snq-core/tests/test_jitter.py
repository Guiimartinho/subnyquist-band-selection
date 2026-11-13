"""Jitter: the limit that decides the practical undersampling ceiling."""

import pytest
from snq_core import jitter as J


def test_erro_de_fase_cresce_com_a_frequencia_de_entrada():
    """Undersampling does not protect against jitter: what counts is the input frequency."""
    sigma = 100e-12
    assert float(J.phase_error_rad(150e3, sigma)) == pytest.approx(
        10.0 * float(J.phase_error_rad(15e3, sigma))
    )


def test_teto_de_snr_por_jitter():
    """1 ns of jitter at 150 kHz already limits below the 12-bit floor."""
    snr = float(J.snr_limit_db(150e3, 1e-9))
    assert snr < J.quantization_snr_db(12)


def test_jitter_pequeno_nao_limita_um_adc_de_12_bits():
    """Negative test: with 10 ps the converter is the limiting factor again."""
    snr = float(J.snr_limit_db(150e3, 10e-12))
    assert snr > J.quantization_snr_db(12)


def test_frequencia_maxima_decresce_com_o_jitter():
    assert J.max_input_frequency_hz(12, 1e-9) < J.max_input_frequency_hz(12, 100e-12)
