"""
Mean Radiant Temperature (Tmrt) computation component.

Computes Tmrt from radiation budget using human body geometry and absorptivities.
The human body is modeled as a standing or sitting cylinder with specific view factors.

Reference:
- Lindberg et al. (2008, 2016) - SOLWEIG Tmrt calculation
- Höppe (1992) - Human body radiation model
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import HumanParams
    from ..bundles import RadiationBundle

# Stefan-Boltzmann constant (W/m²/K⁴)
SBC = 5.67e-8


def compute_tmrt(
    radiation: RadiationBundle,
    human: HumanParams,
    use_anisotropic_sky: bool,
    use_legacy_kelvin_offset: bool = False,
) -> NDArray[np.floating]:
    """
    Compute Mean Radiant Temperature from radiation budget.

    Tmrt represents the uniform temperature of an imaginary enclosure where
    the radiant heat exchange with the human body equals that in the actual
    non-uniform radiant environment.

    Args:
        radiation: Complete radiation budget from all directions
        human: Human parameters (height, posture, absorptivities)
        use_anisotropic_sky: Whether anisotropic sky model was used
        use_legacy_kelvin_offset: If True, use -273.2 (legacy) instead of -273.15
            for Kelvin to Celsius conversion (backwards compatibility)

    Returns:
        Tmrt array in degrees Celsius, clipped to [-50, 80]

    Formula:
        Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
        where Sstr = absorbed shortwave + absorbed longwave

    Reference:
        Lindberg et al. (2008): "SOLWEIG 1.0 - modelling spatial variations
        of 3D radiant fluxes and mean radiant temperature in complex urban settings"
    """
    # View factors based on posture (from SOLWEIG parameters)
    cyl = human.posture == "standing"
    if cyl:
        f_up = 0.06  # View factor to sky/ground from top/bottom
        f_side = 0.22  # View factor from 4 cardinal directions
        f_cyl = 0.28  # Cylindrical projection factor for direct beam
    else:
        f_up = 0.166666
        f_side = 0.166666
        f_cyl = 0.2

    # Extract radiation components
    kdown = radiation.kdown
    kup = radiation.kup
    ldown = radiation.ldown
    lup = radiation.lup
    kside_n = radiation.kside.north
    kside_e = radiation.kside.east
    kside_s = radiation.kside.south
    kside_w = radiation.kside.west
    lside_n = radiation.lside.north
    lside_e = radiation.lside.east
    lside_s = radiation.lside.south
    lside_w = radiation.lside.west
    kside_total = radiation.kside_total
    lside_total = radiation.lside_total

    # Compute absorbed radiation
    if use_anisotropic_sky:
        # Anisotropic model formula (cyl=1, aniso=1)
        # Uses full directional radiation with cylindrical projection
        k_absorbed = human.abs_k * (
            kside_total * f_cyl  # Anisotropic shortwave on vertical body surface
            + (kdown + kup) * f_up  # Downwelling + upwelling on top/bottom
            + (kside_n + kside_e + kside_s + kside_w) * f_side  # Directional from 4 sides
        )

        l_absorbed = human.abs_l * (
            (ldown + lup) * f_up
            + lside_total * f_cyl  # Anisotropic longwave on vertical surface
            + (lside_n + lside_e + lside_s + lside_w) * f_side
        )
    else:
        # Isotropic model: use only direct beam on vertical (kside_total = kside_i)
        k_absorbed = human.abs_k * (
            kside_total * f_cyl  # Direct beam on vertical body surface
            + (kdown + kup) * f_up  # Downwelling + upwelling on top/bottom
            + (kside_n + kside_e + kside_s + kside_w) * f_side  # Diffuse from 4 sides
        )

        # Isotropic longwave: no lside_total term (only directional components)
        l_absorbed = human.abs_l * ((ldown + lup) * f_up + (lside_n + lside_e + lside_s + lside_w) * f_side)

    # Total absorbed radiation (Sstr)
    sstr = k_absorbed + l_absorbed

    # Convert to Tmrt using Stefan-Boltzmann law
    # Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
    # Kelvin to Celsius conversion: -273.15 is scientifically correct,
    # -273.2 is legacy for backwards compatibility
    kelvin_offset = 273.2 if use_legacy_kelvin_offset else 273.15
    tmrt = np.sqrt(np.sqrt(sstr / (human.abs_l * SBC))) - kelvin_offset

    # Clip to physically reasonable range
    tmrt = np.clip(tmrt, -50.0, 80.0)

    return tmrt.astype(np.float32)
