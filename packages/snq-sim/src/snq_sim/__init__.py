"""snq-sim - cadeia de aquisicao fisica honesta.

As tres paredes do projeto (banda do transdutor, banda do track-and-hold, taxa do
SAR) aparecem explicitamente aqui. Simulacao que assume amostrador ideal nao diz
nada sobre o hardware.
"""

from snq_sim.chain import (
    AcquisitionChain,
    AntiAliasFilter,
    Quantizer,
    Sampler,
    Transducer,
    impulse_train,
)
from snq_sim.timeencoding import (
    IntegrateFireTEM,
    crossing_times,
    multichannel_tem,
)

__all__ = [
    "AcquisitionChain",
    "IntegrateFireTEM",
    "crossing_times",
    "multichannel_tem",
    "AntiAliasFilter",
    "Quantizer",
    "Sampler",
    "Transducer",
    "impulse_train",
]
