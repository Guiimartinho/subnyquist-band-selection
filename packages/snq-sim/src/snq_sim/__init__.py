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

__all__ = [
    "AcquisitionChain",
    "AntiAliasFilter",
    "Quantizer",
    "Sampler",
    "Transducer",
    "impulse_train",
]
