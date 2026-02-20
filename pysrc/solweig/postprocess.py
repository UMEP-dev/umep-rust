"""Thermal comfort index computation: UTCI and PET grid functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .models import HumanParams
from .rustalgos import pet as pet_rust
from .rustalgos import utci as utci_rust

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Post-Processing: Thermal Comfort Indices
# =============================================================================


def compute_utci_grid(
    tmrt: NDArray[np.floating],
    ta: float,
    rh: float,
    wind: float,
) -> NDArray[np.floating]:
    """
    Compute UTCI (Universal Thermal Climate Index) for a single grid.

    Thin wrapper around the Rust UTCI polynomial implementation.

    Args:
        tmrt: Mean Radiant Temperature grid (°C).
        ta: Air temperature (°C).
        rh: Relative humidity (%).
        wind: Wind speed at 10m height (m/s).

    Returns:
        UTCI grid (°C).

    Example:
        # Compute UTCI for a single result
        utci = compute_utci_grid(
            tmrt=result.tmrt,
            ta=25.0,
            rh=60.0,
            wind=2.0,
        )
    """

    wind_grid = np.full_like(tmrt, wind, dtype=np.float32)
    return utci_rust.utci_grid(ta, rh, tmrt, wind_grid)


def compute_pet_grid(
    tmrt: NDArray[np.floating],
    ta: float,
    rh: float,
    wind: float,
    human: HumanParams | None = None,
) -> NDArray[np.floating]:
    """
    Compute PET (Physiological Equivalent Temperature) for a single grid.

    Thin wrapper around the Rust PET iterative solver.

    Args:
        tmrt: Mean Radiant Temperature grid (°C).
        ta: Air temperature (°C).
        rh: Relative humidity (%).
        wind: Wind speed at 10m height (m/s).
        human: Human body parameters. Uses defaults if not provided.

    Returns:
        PET grid (°C).

    Example:
        # Compute PET for a single result
        pet = compute_pet_grid(
            tmrt=result.tmrt,
            ta=25.0,
            rh=60.0,
            wind=2.0,
            human=HumanParams(weight=75, height=1.75),
        )
    """

    if human is None:
        human = HumanParams()

    wind_grid = np.full_like(tmrt, wind, dtype=np.float32)
    return pet_rust.pet_grid(
        ta,
        rh,
        tmrt,
        wind_grid,
        human.weight,
        float(human.age),
        human.height,
        human.activity,
        human.clothing,
        human.sex,
    )
