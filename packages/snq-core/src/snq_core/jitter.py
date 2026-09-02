"""Jitter do relógio de amostragem — o inimigo real deste projeto.

Em subamostragem o erro de fase é proporcional à frequência **de entrada**, não à
de conversão. Subamostrar ``N×`` amplifica o efeito do jitter ``N×``. Este módulo
existe para que nenhuma promessa de fator de subamostragem seja feita sem passar
por aqui primeiro.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def phase_error_rad(f_in_hz: ArrayLike, sigma_t_s: float) -> NDArray[np.float64]:
    """Desvio padrão do erro de fase induzido por jitter ``sigma_t_s``.

    ``sigma_phi = 2·pi·f_in·sigma_t``. Depende da frequência **de entrada**.
    """
    return 2.0 * np.pi * np.asarray(f_in_hz, dtype=float) * sigma_t_s


def snr_limit_db(f_in_hz: ArrayLike, sigma_t_s: float) -> NDArray[np.float64]:
    """Teto de SNR imposto só pelo jitter: ``-20·log10(2·pi·f_in·sigma_t)``.

    Compare com o SNR de quantização de um ADC de ``B`` bits (~``6.02·B + 1.76``).
    Quando este teto cai abaixo daquele, **o conversor deixou de ser o limitante**
    e o oscilador passou a ser — o ponto em que o projeto precisa de referência
    externa.
    """
    return -20.0 * np.log10(phase_error_rad(f_in_hz, sigma_t_s))


def quantization_snr_db(n_bits: int) -> float:
    """SNR de quantização ideal de onda cheia, em dB."""
    return 6.02 * n_bits + 1.76


def max_input_frequency_hz(n_bits: int, sigma_t_s: float) -> float:
    """Frequência de entrada em que o jitter iguala o piso de quantização.

    É o teto prático de subamostragem do hardware. Acima dela, aumentar a
    resolução do ADC não compra nada.
    """
    target = quantization_snr_db(n_bits)
    return float(10.0 ** (-target / 20.0) / (2.0 * np.pi * sigma_t_s))
