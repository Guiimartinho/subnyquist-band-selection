"""Rolling-bearing fault frequencies from the geometry.

This module is the reason this project can speak of **recovery** and not of
hallucination: the set of candidate frequencies is finite and comes from the
bearing geometry, not from a learned prior. Identifying which super-Nyquist
component produced an observed alias stops being an ill-posed inversion and
becomes a hypothesis test over a discrete set.

Reference: R. B. Randall & J. Antoni, *Rolling element bearing diagnostics —
a tutorial*, Mechanical Systems and Signal Processing 25(2):485-520, 2011.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of a rolling-element bearing.

    ``n_elements``  number of balls/rollers
    ``d_ball_m``    diameter of the rolling element
    ``d_pitch_m``   pitch diameter (center to center of the elements)
    ``contact_angle_rad`` contact angle (0 for a pure radial bearing)
    """

    n_elements: int
    d_ball_m: float
    d_pitch_m: float
    contact_angle_rad: float = 0.0
    name: str = ""

    @property
    def ratio(self) -> float:
        """``(d/D)·cos(phi)`` — appears in all four formulas."""
        return (self.d_ball_m / self.d_pitch_m) * float(np.cos(self.contact_angle_rad))


def fault_frequencies_hz(geom: BearingGeometry, shaft_hz: float) -> dict[str, float]:
    """The four characteristic frequencies, in Hz, for shaft speed ``shaft_hz``.

    - ``BPFO`` ball pass frequency, outer race
    - ``BPFI`` ball pass frequency, inner race
    - ``BSF``  ball spin frequency (the *defect* shows up at ``2·BSF``)
    - ``FTF``  fundamental train frequency (cage)

    They are proportional to the shaft speed: changing the operating speed **moves the
    lines in a known way**, and that is what disambiguates the alias order.
    """
    r = geom.ratio
    n = geom.n_elements
    return {
        "BPFO": (n / 2.0) * shaft_hz * (1.0 - r),
        "BPFI": (n / 2.0) * shaft_hz * (1.0 + r),
        "BSF": (geom.d_pitch_m / (2.0 * geom.d_ball_m)) * shaft_hz * (1.0 - r * r),
        "FTF": 0.5 * shaft_hz * (1.0 - r),
    }


def candidate_lines_hz(
    geom: BearingGeometry, shaft_hz: float, n_harmonics: int = 8, n_sidebands: int = 2
) -> NDArray[np.float64]:
    """Discrete set of expected spectral lines — the project's physical prior.

    Harmonics of the fault frequencies, with sidebands spaced by ``shaft_hz``
    (modulation by the shaft rotation). It is this set, and not a continuous grid,
    that the decoder needs to select from.
    """
    base = fault_frequencies_hz(geom, shaft_hz)
    lines: list[float] = []
    for f in base.values():
        for h in range(1, n_harmonics + 1):
            centre = h * f
            for s in range(-n_sidebands, n_sidebands + 1):
                v = centre + s * shaft_hz
                if v > 0.0:
                    lines.append(v)
    return np.unique(np.asarray(lines, dtype=float))


# Documented geometries of the public datasets used in the project.
# Check against each dataset's documentation before using in a paper.
CWRU_6205_2RS_JEM_SKF = BearingGeometry(
    n_elements=9,
    d_ball_m=7.94e-3,
    d_pitch_m=39.04e-3,
    contact_angle_rad=0.0,
    name="CWRU drive-end SKF 6205-2RS JEM",
)
