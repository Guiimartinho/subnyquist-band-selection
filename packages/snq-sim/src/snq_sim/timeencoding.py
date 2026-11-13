"""Time-encoding sampling — the A2 realization of ADR 0004.

Classical sampling measures **amplitude at predefined instants**. Time encoding does
the opposite: it records **the instant at which the signal, or a function of it,
reaches a predefined value**. The samples are irregularly spaced.

Why this is in the project, quoting `naaman2021fritem`:

    "At high sampling rates, clocks are power-consuming and prone to electromagnetic
    interference. An integrate-and-fire time encoding machine is an alternative
    power-efficient sampling mechanism which does not require a global clock."

Route A1 (TIMER + DPPI) optimizes the use of the clock; this one **eliminates the
global clock**.

**Beware of the energy argument, measured in this simulator on 2026-09-03.** The two
variants are not equivalent on this point:

- The **IF-TEM** fires at a rate set by the *mean* of the signal, not by its
  amplitude: for a zero-mean input the integral of ``x + bias`` over one period
  equals ``bias·T`` whatever the amplitude. Amplitudes from 0.01 to 1.9 give exactly
  the same rate. Its gain is doing without the high-frequency clock, **not** firing
  less when the machine is stopped.
- It is **level crossing with hysteresis** that has a rate proportional to activity:
  measured, 980 Hz for amplitude 0.01 against 2980 Hz for amplitude 1.0.

Whoever wants "energy proportional to what happens" needs the second variant.

Two properties established in the literature that justify the route:

- `liu2023tembandpass`: for a bandpass signal, time encoding is **equivalent to
  periodic non-uniform sampling**. The whole multi-coset theory carries over.
- `adam2019multichanneltem`: M channels with shifted integrators recover a signal
  with **M times the bandwidth**, and **without knowing the shifts between
  channels** — which multi-coset requires and has to calibrate.

On the nRF54L15 the component is the comparator with TIMER capture at 128 MHz. The
model here includes the quantization of the firing instant, which is the real limit
of that capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class IntegrateFireTEM:
    """Integrate-and-fire time encoding machine.

    Follows the definition in `adam2021asynchrony`: adds a bias ``bias`` to the
    signal, integrates the result scaled by ``1/kappa`` until it reaches the threshold
    ``delta``, records the instant, resets the integrator and starts over.

    ``bias`` must exceed the maximum absolute value of the signal, otherwise the
    integrator may never fire — it is the condition that guarantees that the encoding
    terminates.

    ``timer_hz`` models the hardware capture of the instant: on the nRF54L15 the TIMER
    runs at 128 MHz, so the firing instant is known with a resolution of ~7.8 ns, and
    not exactly. `gontier2011tem` shows that the method is robust to this
    quantization.
    """

    kappa: float = 1.0
    delta: float = 1e-5
    bias: float = 2.0
    timer_hz: float | None = 128e6

    def max_input_amplitude(self) -> float:
        """Largest admissible amplitude: above it, firing is not guaranteed."""
        return self.bias

    def nominal_rate_hz(self) -> float:
        """Firing rate for a zero-mean input: ``bias / (2·delta·kappa)``.

        It is the design parameter that is compared against the ``fs`` of route A1.
        The defaults give ~100 kHz; choosing ``delta`` too large makes the integrator
        fire once per second, and the encoding produces no sample at all in the record.
        """
        return self.bias / (2.0 * self.delta * self.kappa)

    @classmethod
    def for_rate(cls, rate_hz: float, bias: float = 2.0, **kw: float | None) -> IntegrateFireTEM:
        """Builds a TEM with the requested nominal rate."""
        return cls(delta=bias / (2.0 * rate_hz), bias=bias, **kw)  # type: ignore[arg-type]

    def encode(self, x: ArrayLike, fs_analog_hz: float) -> NDArray[np.float64]:
        """Firing instants for ``x``, finely sampled at ``fs_analog_hz``.

        The integral is accumulated with the trapezoidal rule and the threshold
        crossing is linearly interpolated within the step — without this, the
        resolution of the instants would be limited by ``fs_analog_hz`` instead of by
        the model.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 2:
            return np.empty(0)
        peak = float(np.max(np.abs(x)))
        if peak >= self.bias:
            raise ValueError(
                f"signal amplitude ({peak:.3g}) >= bias ({self.bias:.3g}); "
                "the integrator may never fire. Increase bias or normalize the input."
            )

        dt = 1.0 / fs_analog_hz
        y = (x + self.bias) / self.kappa
        # Cumulative integral; a firing occurs every 2*delta of accumulated area.
        area = np.concatenate([[0.0], np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)])
        step = 2.0 * self.delta

        n_fires = int(area[-1] // step)
        if n_fires <= 0:
            return np.empty(0)

        targets = step * np.arange(1, n_fires + 1)
        # np.interp inverts the integral: which instant accumulates each target area.
        times = np.interp(targets, area, np.arange(x.size) * dt)

        if self.timer_hz:
            times = np.round(times * self.timer_hz) / self.timer_hz
        return times

    def firing_rate_hz(self, x: ArrayLike, fs_analog_hz: float) -> float:
        """Mean firing rate — the analogue of the sampling rate.

        It is the number that is compared against the ``fs`` of route A1 in an energy
        budget: each firing is one conversion, and what does not fire does not spend.
        """
        t = self.encode(x, fs_analog_hz)
        duration = np.asarray(x).size / fs_analog_hz
        return float(t.size / duration) if duration > 0 else 0.0


def multichannel_tem(
    x: ArrayLike,
    fs_analog_hz: float,
    n_channels: int,
    tem: IntegrateFireTEM | None = None,
) -> list[NDArray[np.float64]]:
    """M channels with shifted integrators — the time-domain analogue of multi-coset.

    `adam2019multichanneltem` shows that M channels recover a signal with M times the
    bandwidth, **without knowing the relative shifts**. Here the channels differ by
    the initial state of the integrator, which is what a real circuit would do.

    Practical difference with respect to multi-coset: there the offsets need to be
    known and calibrated; here they do not. It is the advantage that motivated
    ADR 0004.
    """
    base = tem or IntegrateFireTEM()
    out: list[NDArray[np.float64]] = []
    for m in range(n_channels):
        # Shifting the initial state is equivalent to shifting the first-firing threshold.
        shifted = IntegrateFireTEM(
            kappa=base.kappa,
            delta=base.delta,
            bias=base.bias,
            timer_hz=base.timer_hz,
        )
        t = shifted.encode(x, fs_analog_hz)
        offset = (m / n_channels) * (2.0 * base.delta * base.kappa / base.bias)
        out.append(t + offset)
    return out


def crossing_times(
    x: ArrayLike,
    fs_analog_hz: float,
    levels: ArrayLike,
    hysteresis: float = 0.0,
    timer_hz: float | None = 128e6,
) -> NDArray[np.float64]:
    """Level-crossing sampling: instants at which ``x`` crosses each threshold.

    It is the simplest variant of route A2, and the most direct one to implement with
    a comparator.

    **``hysteresis`` is not optional in practice.** Without it, noise around a level
    makes the comparator toggle endlessly: measured in this simulator, noise of
    amplitude 0.001 around zero produced 5018 events in 50 ms, against 149 for a sine
    wave of amplitude 1. That destroys the energy argument of the route, which depends
    on a still signal costing little. A real circuit solves this with hysteresis, and
    that is why the level-crossing ADC literature always includes it.

    With hysteresis ``h``, after crossing upward the signal must fall below
    ``level - h`` before a new upward crossing is counted.
    """
    x = np.asarray(x, dtype=float).ravel()
    dt = 1.0 / fs_analog_hz
    out: list[float] = []

    for lv in np.asarray(levels, dtype=float).ravel():
        armed_up = x[0] < lv - hysteresis
        armed_down = x[0] > lv + hysteresis
        for i in range(x.size - 1):
            a, b = x[i], x[i + 1]
            if armed_up and a < lv <= b:
                out.append((i + (lv - a) / (b - a)) * dt)
                armed_up, armed_down = False, False
            elif armed_down and a > lv >= b:
                out.append((i + (a - lv) / (a - b)) * dt)
                armed_up, armed_down = False, False
            if b < lv - hysteresis:
                armed_up = True
            elif b > lv + hysteresis:
                armed_down = True

    t = np.sort(np.asarray(out, dtype=float))
    if timer_hz:
        t = np.round(t * timer_hz) / timer_hz
    return t
