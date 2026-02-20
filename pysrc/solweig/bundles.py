"""
Data bundle classes for SOLWEIG computation components.

These dataclasses group related arrays and values to reduce parameter passing
and make data flow clearer through the computation pipeline.

Each bundle represents the output of a distinct computation stage:
- DirectionalArrays: N/E/S/W directional components (used by SVF and radiation)
- SvfBundle: All sky view factor arrays
- ShadowBundle: Shadow computation results
- GroundBundle: Ground temperature model outputs
- RadiationBundle: Radiation calculation results
- GvfBundle: Ground view factor results
- LupBundle: Upwelling longwave with thermal state

This modular design enables:
1. Easier testing of individual components
2. Clearer boundaries for Rust migration
3. Reduced parameter counts (bundles instead of 10+ arrays)
4. Better code organization and maintainability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from .models import ThermalState


@dataclass
class DirectionalArrays:
    """
    Directional arrays for N, E, S, W components.

    Used for:
    - SVF directional components (svf_north, svf_east, svf_south, svf_west)
    - Radiation directional components (kside_n, kside_e, kside_s, kside_w)
    - Longwave directional components (lside_n, lside_e, lside_s, lside_w)

    Attributes:
        north: North-facing component
        east: East-facing component
        south: South-facing component
        west: West-facing component
    """

    north: NDArray[np.floating]
    east: NDArray[np.floating]
    south: NDArray[np.floating]
    west: NDArray[np.floating]


@dataclass
class SvfBundle:
    """
    Sky View Factor computation results.

    Groups all SVF-related arrays to simplify passing to radiation calculations.

    Attributes:
        svf: Total sky view factor (0-1)
        svf_directional: Directional SVF components (N, E, S, W)
        svf_veg: Vegetation-only SVF
        svf_veg_directional: Directional vegetation SVF (N, E, S, W)
        svf_aveg: SVF above vegetation (building shadow on veg)
        svf_aveg_directional: Directional SVF above vegetation (N, E, S, W)
        svfbuveg: Combined SVF accounting for vegetation transmissivity
        svfalfa: Angular factor from SVF (for anisotropic calculations)
    """

    svf: NDArray[np.floating]
    svf_directional: DirectionalArrays
    svf_veg: NDArray[np.floating]
    svf_veg_directional: DirectionalArrays
    svf_aveg: NDArray[np.floating]
    svf_aveg_directional: DirectionalArrays
    svfbuveg: NDArray[np.floating]
    svfalfa: NDArray[np.floating]


@dataclass
class ShadowBundle:
    """
    Shadow computation results.

    Attributes:
        shadow: Combined shadow fraction (1=sunlit, 0=shaded)
        bldg_sh: Building shadow only
        veg_sh: Vegetation shadow only
        wallsun: Wall sun exposure (for wall temperature)
        psi: Vegetation transmissivity used (for reference)
    """

    shadow: NDArray[np.floating]
    bldg_sh: NDArray[np.floating]
    veg_sh: NDArray[np.floating]
    wallsun: NDArray[np.floating]
    psi: float


@dataclass
class GroundBundle:
    """
    Ground temperature model outputs.

    Results from the ground temperature computation, including
    spatially-varying surface properties.

    Attributes:
        tg: Ground temperature deviation from air temperature (K or °C)
        tg_wall: Wall temperature deviation from air temperature
        ci_tg: Clearness index correction factor
        alb_grid: Albedo per pixel (0-1)
        emis_grid: Emissivity per pixel (0-1)
    """

    tg: NDArray[np.floating]
    tg_wall: float
    ci_tg: float
    alb_grid: NDArray[np.floating]
    emis_grid: NDArray[np.floating]


@dataclass
class GvfBundle:
    """
    Ground View Factor computation results.

    Includes upwelling longwave radiation components before thermal delay
    and albedo view factors for reflected shortwave radiation.

    Attributes:
        lup: Upwelling longwave radiation (W/m²)
        lup_e: Upwelling longwave from east
        lup_s: Upwelling longwave from south
        lup_w: Upwelling longwave from west
        lup_n: Upwelling longwave from north
        gvfalb: Ground view factor × albedo (for Kup calculation)
        gvfalb_e: GVF × albedo from east
        gvfalb_s: GVF × albedo from south
        gvfalb_w: GVF × albedo from west
        gvfalb_n: GVF × albedo from north
        gvfalbnosh: GVF × albedo without shadow (for anisotropic)
        gvfalbnosh_e: GVF × albedo (no shadow) from east
        gvfalbnosh_s: GVF × albedo (no shadow) from south
        gvfalbnosh_w: GVF × albedo (no shadow) from west
        gvfalbnosh_n: GVF × albedo (no shadow) from north
    """

    lup: NDArray[np.floating]
    lup_e: NDArray[np.floating]
    lup_s: NDArray[np.floating]
    lup_w: NDArray[np.floating]
    lup_n: NDArray[np.floating]
    gvfalb: NDArray[np.floating]
    gvfalb_e: NDArray[np.floating]
    gvfalb_s: NDArray[np.floating]
    gvfalb_w: NDArray[np.floating]
    gvfalb_n: NDArray[np.floating]
    gvfalbnosh: NDArray[np.floating]
    gvfalbnosh_e: NDArray[np.floating]
    gvfalbnosh_s: NDArray[np.floating]
    gvfalbnosh_w: NDArray[np.floating]
    gvfalbnosh_n: NDArray[np.floating]


@dataclass
class LupBundle:
    """
    Upwelling longwave radiation with thermal state.

    Results after applying TsWaveDelay thermal inertia model.
    Includes updated thermal state for next timestep.

    Attributes:
        lup: Final upwelling longwave (center view) after thermal delay
        lup_e: Final upwelling longwave from east
        lup_s: Final upwelling longwave from south
        lup_w: Final upwelling longwave from west
        lup_n: Final upwelling longwave from north
        state: Updated thermal state to carry forward to next timestep
    """

    lup: NDArray[np.floating]
    lup_e: NDArray[np.floating]
    lup_s: NDArray[np.floating]
    lup_w: NDArray[np.floating]
    lup_n: NDArray[np.floating]
    state: ThermalState | None  # Forward reference to avoid circular import


@dataclass
class RadiationBundle:
    """
    Radiation calculation outputs.

    Complete radiation budget including shortwave and longwave components.

    Attributes:
        kdown: Downwelling shortwave radiation (W/m²)
        kup: Upwelling shortwave radiation (W/m²)
        ldown: Downwelling longwave radiation (W/m²)
        lup: Upwelling longwave radiation (W/m²)
        kside: Shortwave radiation from 4 directions (W/m²)
        lside: Longwave radiation from 4 directions (W/m²)
        kside_total: Total shortwave on vertical surface (for anisotropic Tmrt)
        lside_total: Total longwave on vertical surface (for anisotropic Tmrt)
        drad: Diffuse radiation term (for Tmrt calculation)
    """

    kdown: NDArray[np.floating]
    kup: NDArray[np.floating]
    ldown: NDArray[np.floating]
    lup: NDArray[np.floating]
    kside: DirectionalArrays
    lside: DirectionalArrays
    kside_total: NDArray[np.floating]
    lside_total: NDArray[np.floating]
    drad: NDArray[np.floating]


@dataclass
class WallBundle:
    """
    Wall geometry data.

    Wall heights and aspects needed for shadow calculation and wall temperature.

    Attributes:
        wall_height: Wall height at each pixel (meters)
        wall_aspect: Wall orientation at each pixel (degrees, 0=North)
    """

    wall_height: NDArray[np.floating]
    wall_aspect: NDArray[np.floating]


@dataclass
class VegetationBundle:
    """
    Vegetation geometry data.

    Vegetation heights needed for shadow and SVF calculations.

    Attributes:
        cdsm: Canopy Digital Surface Model (vegetation heights)
        tdsm: Trunk Digital Surface Model (trunk zone heights)
        bush: Bush/shrub layer (boolean or height)
    """

    cdsm: NDArray[np.floating] | None
    tdsm: NDArray[np.floating] | None
    bush: NDArray[np.floating] | None


__all__ = [
    "DirectionalArrays",
    "SvfBundle",
    "ShadowBundle",
    "GroundBundle",
    "GvfBundle",
    "LupBundle",
    "RadiationBundle",
    "WallBundle",
    "VegetationBundle",
]
