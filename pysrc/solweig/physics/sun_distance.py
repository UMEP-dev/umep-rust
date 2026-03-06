from __future__ import annotations

import numpy as np


def sun_distance(jday: int) -> float:
    """Relative earth-sun distance for a given day of year.

    Reference: Partridge and Platt (1975).
    """
    b = 2.0 * np.pi * jday / 365.0
    D = np.sqrt(
        1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b) + 0.000719 * np.cos(2.0 * b) + 0.000077 * np.sin(2.0 * b)
    )
    return D
