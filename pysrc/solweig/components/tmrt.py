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

from .. import rustalgos
from ..buffers import as_float32

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import HumanParams
    from ..bundles import RadiationBundle


def compute_tmrt(
    radiation: RadiationBundle,
    human: HumanParams,
    use_anisotropic_sky: bool,
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

    Returns:
        Tmrt array in degrees Celsius, clipped to [-50, 80]

    Formula:
        Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
        where Sstr = absorbed shortwave + absorbed longwave

    Reference:
        Lindberg et al. (2008): "SOLWEIG 1.0 - modelling spatial variations
        of 3D radiant fluxes and mean radiant temperature in complex urban settings"
    """
    # Extract radiation components (use as_float32 to avoid unnecessary copies)
    kdown = as_float32(radiation.kdown)
    kup = as_float32(radiation.kup)
    ldown = as_float32(radiation.ldown)
    lup = as_float32(radiation.lup)
    kside_n = as_float32(radiation.kside.north)
    kside_e = as_float32(radiation.kside.east)
    kside_s = as_float32(radiation.kside.south)
    kside_w = as_float32(radiation.kside.west)
    lside_n = as_float32(radiation.lside.north)
    lside_e = as_float32(radiation.lside.east)
    lside_s = as_float32(radiation.lside.south)
    lside_w = as_float32(radiation.lside.west)
    kside_total = as_float32(radiation.kside_total)
    lside_total = as_float32(radiation.lside_total)

    # Posture flag (True for standing, False for sitting)
    is_standing = human.posture == "standing"

    # Create parameter struct (reduces 18 params to 15)
    tmrt_params = rustalgos.tmrt.TmrtParams(
        abs_k=human.abs_k,
        abs_l=human.abs_l,
        is_standing=is_standing,
        use_anisotropic_sky=use_anisotropic_sky,
    )

    # Call Rust implementation with parameter struct
    tmrt = rustalgos.tmrt.compute_tmrt(
        kdown,
        kup,
        ldown,
        lup,
        kside_n,
        kside_e,
        kside_s,
        kside_w,
        lside_n,
        lside_e,
        lside_s,
        lside_w,
        kside_total,
        lside_total,
        tmrt_params,
    )

    return tmrt
