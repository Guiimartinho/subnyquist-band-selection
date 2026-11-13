"""snq-sim - honest physical acquisition chain.

The project's three walls (transducer bandwidth, track-and-hold bandwidth, SAR
rate) appear explicitly here. A simulation that assumes an ideal sampler says
nothing about the hardware.
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
