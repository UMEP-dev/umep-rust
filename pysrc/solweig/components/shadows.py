"""
Shadow computation component.

Handles:
- Ray tracing for building and vegetation shadows
- Vegetation transmissivity (seasonal leaf on/off)
- Combined shadow accounting for light penetration through vegetation
- Wall sun exposure for thermal calculations

Returns a ShadowBundle with all shadow components.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ..bundles import ShadowBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import Weather


def compute_transmissivity(
    doy: int,
    physics: SimpleNamespace | None = None,
    conifer: bool = False,
) -> float:
    """
    Compute vegetation transmissivity based on day of year and leaf status.

    This implements seasonal leaf on/off logic from configs.py EnvironData.
    During leaf-on season, vegetation transmits less light (low psi ~0.03).
    During leaf-off season (winter), bare branches transmit more light (psi ~0.5).

    Args:
        doy: Day of year (1-366)
        physics: Physics params from load_physics() containing Tree_settings.
            If provided, reads Transmissivity, First_day_leaf, Last_day_leaf.
        conifer: Override to treat vegetation as conifer (always leaf-on).

    Returns:
        Transmissivity value:
        - 0.03 (default) during leaf-on period
        - 0.5 during leaf-off period (deciduous trees in winter)

    Reference:
        configs.py EnvironData.leafon computation and psi assignment
    """
    # Default values matching configs.py
    transmissivity = 0.03
    transmissivity_leafoff = 0.5
    first_day = 100  # ~April 10
    last_day = 300  # ~October 27
    is_conifer = conifer

    # Override from physics params if provided
    if physics is not None and hasattr(physics, "Tree_settings"):
        ts = physics.Tree_settings.Value
        transmissivity = getattr(ts, "Transmissivity", 0.03)
        transmissivity_leafoff = getattr(ts, "Transmissivity_leafoff", 0.5)
        first_day = int(getattr(ts, "First_day_leaf", 100))
        last_day = int(getattr(ts, "Last_day_leaf", 300))
        # Note: Conifer flag may not be in all params files
        is_conifer = conifer or getattr(ts, "Conifer", False)

    # Determine leaf on/off
    if is_conifer:
        leaf_on = True
    elif first_day > last_day:
        # Wraps around year end (southern hemisphere or unusual dates)
        leaf_on = doy > first_day or doy < last_day
    else:
        # Normal case: leaves on between first_day and last_day
        leaf_on = first_day < doy < last_day

    return transmissivity if leaf_on else transmissivity_leafoff


def compute_shadows(
    weather: Weather,
    dsm: NDArray[np.floating],
    pixel_size: float,
    max_height: float,
    use_veg: bool,
    physics: SimpleNamespace | None,
    conifer: bool,
    cdsm: NDArray[np.floating] | None = None,
    tdsm: NDArray[np.floating] | None = None,
    bush: NDArray[np.floating] | None = None,
    wall_ht: NDArray[np.floating] | None = None,
    wall_asp_rad: NDArray[np.floating] | None = None,
) -> ShadowBundle:
    """
    Compute shadows from buildings and vegetation.

    Uses ray tracing to determine shadowed areas based on sun position.
    Accounts for vegetation transmissivity (light passing through canopy).

    Args:
        weather: Weather data including sun position (azimuth, altitude)
        dsm: Digital Surface Model (building heights)
        pixel_size: Grid resolution in meters
        max_height: Maximum building height for shadow computation
        use_veg: Whether to include vegetation shadows
        physics: Physics parameters (for transmissivity calculation)
        conifer: Whether vegetation is coniferous (always leaf-on)
        cdsm: Canopy Digital Surface Model (optional, for vegetation)
        tdsm: Trunk Digital Surface Model (optional, for vegetation)
        bush: Bush/shrub layer (optional, for vegetation)
        wall_ht: Wall heights (optional, for wall sun exposure)
        wall_asp_rad: Wall aspects in radians (optional, for wall sun exposure)

    Returns:
        ShadowBundle containing:
            - shadow: Combined shadow fraction (1=sunlit, 0=shaded)
            - bldg_sh: Building shadow only
            - veg_sh: Vegetation shadow only
            - wallsun: Wall sun exposure (for wall temperature)
            - psi: Vegetation transmissivity used

    Reference:
        Lindberg et al. (2008) - SOLWEIG shadow model
        Formula: shadow = bldg_sh - (1 - veg_sh) * (1 - psi)
    """
    # Import here to avoid circular dependency
    from ..rustalgos import shadowing

    has_walls = wall_ht is not None and wall_asp_rad is not None

    # Call Rust shadow calculation
    shadow_result = shadowing.calculate_shadows_wall_ht_25(
        weather.sun_azimuth,
        weather.sun_altitude,
        pixel_size,
        max_height,
        dsm,
        cdsm if use_veg else None,
        tdsm if use_veg else None,
        bush if use_veg else None,
        wall_ht if has_walls else None,
        wall_asp_rad if has_walls else None,
        None,  # walls_scheme
        None,  # aspect_scheme
        3.0,  # min_sun_altitude
    )

    # Vegetation transmissivity - compute dynamically based on season
    doy = weather.datetime.timetuple().tm_yday
    psi = compute_transmissivity(doy, physics, conifer)

    # Extract shadow arrays
    bldg_sh = np.array(shadow_result.bldg_sh)

    # Compute combined shadow accounting for vegetation transmissivity
    # This matches the reference: shadow = bldg_sh - (1 - veg_sh) * (1 - psi)
    # where psi is vegetation transmissivity (fraction of light that passes through)
    if use_veg:
        veg_sh = np.array(shadow_result.veg_sh)
        shadow = bldg_sh - (1 - veg_sh) * (1 - psi)
        # Note: No clipping here to match reference exactly. In practice, shadow
        # should stay in [0,1] because veg_sh is constrained by bldg_sh.
    else:
        veg_sh = np.zeros_like(bldg_sh)
        shadow = bldg_sh

    # Wall sun exposure (for wall temperature calculation)
    wallsun = np.array(shadow_result.wall_sun) if has_walls else np.zeros_like(dsm)

    return ShadowBundle(
        shadow=shadow.astype(np.float32),
        bldg_sh=bldg_sh.astype(np.float32),
        veg_sh=veg_sh.astype(np.float32),
        wallsun=wallsun.astype(np.float32),
        psi=psi,
    )
