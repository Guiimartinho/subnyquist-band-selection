"""Jitter: o limite que decide o teto prático de subamostragem."""

import pytest
from snq_core import jitter as J


def test_erro_de_fase_cresce_com_a_frequencia_de_entrada():
    """Subamostrar não protege do jitter: o que conta é a frequência de entrada."""
    sigma = 100e-12
    assert float(J.phase_error_rad(150e3, sigma)) == pytest.approx(
        10.0 * float(J.phase_error_rad(15e3, sigma))
    )


def test_teto_de_snr_por_jitter():
    """1 ns de jitter a 150 kHz já limita abaixo do piso de 12 bits."""
    snr = float(J.snr_limit_db(150e3, 1e-9))
    assert snr < J.quantization_snr_db(12)


def test_jitter_pequeno_nao_limita_um_adc_de_12_bits():
    """Teste negativo: com 10 ps o conversor volta a ser o limitante."""
    snr = float(J.snr_limit_db(150e3, 10e-12))
    assert snr > J.quantization_snr_db(12)


def test_frequencia_maxima_decresce_com_o_jitter():
    assert J.max_input_frequency_hz(12, 1e-9) < J.max_input_frequency_hz(12, 100e-12)
