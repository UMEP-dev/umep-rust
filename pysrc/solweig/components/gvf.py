"""
Ground View Factor (GVF) computation component.

Computes upwelling longwave radiation from surrounding surfaces (ground + walls)
and albedo view factors for reflected shortwave radiation.

The GVF represents how much a person at a given height "sees" the ground and walls
versus the sky. This determines the thermal radiation received from below and sides.

Reference:
- Lindberg et al. (2008) - SOLWEIG GVF model with wall radiation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..buffers import as_float32
from ..bundles import GvfBundle
from ..constants import KELVIN_OFFSET, SBC
from ..physics.morphology import generate_binary_structure

try:
    from ..rustalgos import morphology as _rust_morph

    def _binary_dilation(input_array, structure, iterations):
        return _rust_morph.binary_dilation(
            input_array.astype(np.uint8),
            structure.astype(np.uint8),
            iterations,
        ).astype(bool)
except ImportError:
    from ..physics.morphology import binary_dilation as _binary_dilation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import HumanParams, SurfaceData, Weather


def detect_building_mask(
    dsm: NDArray[np.floating],
    land_cover: NDArray[np.integer] | None,
    wall_height: NDArray[np.floating] | None,
    pixel_size: float,
) -> NDArray[np.floating]:
    """
    Create a building mask for GVF calculation.

    GVF (Ground View Factor) expects: 0=building, 1=ground.
    This is used to normalize GVF values over buildings where GVF doesn't apply.

    Args:
        dsm: Digital Surface Model array.
        land_cover: Optional land cover grid (UMEP standard: ID 2 = buildings).
        wall_height: Optional wall height grid.
        pixel_size: Pixel size in meters.

    Returns:
        Building mask where 0=building pixels, 1=ground pixels.

    Detection strategy:
        1. If land_cover provided: Use ID 2 (buildings) directly
        2. Elif wall_height provided: Dilate wall pixels + detect elevated areas
        3. Else: Assume all ground (no buildings)
    """
    if land_cover is not None:
        # Use land cover directly: ID 2 = buildings
        buildings = np.ones_like(dsm, dtype=np.float32)
        buildings[land_cover == 2] = 0.0
        return buildings

    if wall_height is not None:
        # Approximate building footprints from wall heights
        # Wall pixels mark building edges; dilate to capture interiors
        wall_mask = wall_height > 0

        # Dilate to capture building interiors (typical building width up to 50m)
        struct = generate_binary_structure(2, 2)  # 8-connectivity
        iterations = int(25 / pixel_size) + 1
        dilated = _binary_dilation(wall_mask, struct, iterations=iterations)

        # Also detect elevated areas (building roofs)
        ground_level = np.nanpercentile(dsm[~wall_mask], 10) if np.any(~wall_mask) else np.nanmin(dsm)
        elevated = dsm > (ground_level + 2.0)  # At least 2m above ground

        # Combine: building pixels where either dilated walls OR elevated flat areas
        is_building = dilated | (elevated & ~np.isnan(dsm))

        # Invert: 0=building, 1=ground
        return (~is_building).astype(np.float32)

    # No building info available - assume all ground
    return np.ones_like(dsm, dtype=np.float32)


def compute_gvf(
    surface: SurfaceData,
    weather: Weather,
    human: HumanParams,
    tg: NDArray[np.floating],
    tg_wall: float,
    shadow: NDArray[np.floating],
    wallsun: NDArray[np.floating],
    alb_grid: NDArray[np.floating],
    emis_grid: NDArray[np.floating],
    svf: NDArray[np.floating],
    pixel_size: float,
    wall_ht: NDArray[np.floating] | None = None,
    wall_asp: NDArray[np.floating] | None = None,
) -> GvfBundle:
    """
    Compute Ground View Factor for upwelling longwave and albedo components.

    GVF represents how much a person "sees" the ground and walls from a given height.
    This determines thermal radiation received from surrounding surfaces.

    Args:
        surface: Surface data (DSM, land cover)
        weather: Weather data (temperature)
        human: Human parameters (height, posture)
        tg: Ground temperature deviation from air temperature (K)
        tg_wall: Wall temperature deviation from air temperature (K)
        shadow: Combined shadow fraction (1=sunlit, 0=shaded)
        wallsun: Wall sun exposure (for wall temperature)
        alb_grid: Albedo per pixel (0-1)
        emis_grid: Emissivity per pixel (0-1)
        svf: Sky view factor (for simplified GVF when no walls)
        pixel_size: Grid resolution in meters
        wall_ht: Wall heights (optional, for full GVF with walls)
        wall_asp: Wall aspects in degrees (optional, for full GVF with walls)

    Returns:
        GvfBundle containing:
            - lup_*: Upwelling longwave from 5 directions (center, N, E, S, W)
            - gvfalb_*: Ground view factor × albedo (for reflected shortwave)
            - gvfalbnosh_*: GVF × albedo without shadow (for anisotropic)

    Reference:
        Lindberg et al. (2008) - SOLWEIG model equations for GVF calculation
    """
    # Import here to avoid circular dependency
    from ..rustalgos import gvf as gvf_module

    has_walls = wall_ht is not None and wall_asp is not None

    # Human height parameters for GVF (matching runner: first=round(height), second=round(height*20))
    first = np.round(human.height)
    if first == 0.0:
        first = 1.0
    second = np.round(human.height * 20.0)

    # Building mask for GVF calculation
    buildings = detect_building_mask(
        surface.dsm,
        surface.land_cover,
        wall_ht if has_walls else None,
        pixel_size,
    )

    # Wall properties (from SOLWEIG parameters)
    albedo_wall = 0.20
    emis_wall = 0.90

    # Land cover settings for gvf_calc
    use_landcover = surface.land_cover is not None
    lc_grid = surface.land_cover.astype(np.float32) if surface.land_cover is not None else None

    if has_walls:
        # Type narrowing - wall_ht and wall_asp are not None when has_walls is True
        assert wall_ht is not None
        assert wall_asp is not None
        # Use full GVF calculation with wall radiation
        # Create parameter struct (reduces 20 params to 11)
        gvf_params = gvf_module.GvfScalarParams(
            scale=pixel_size,
            first=first,
            second=second,
            tgwall=tg_wall,
            ta=weather.ta,
            ewall=emis_wall,
            sbc=SBC,
            albedo_b=albedo_wall,
            twater=weather.ta,  # Twater = Ta (approximation for water temperature)
            landcover=use_landcover,
        )
        gvf_result = gvf_module.gvf_calc(
            as_float32(wallsun),
            as_float32(wall_ht),
            as_float32(buildings),
            as_float32(shadow),
            as_float32(wall_asp),
            as_float32(tg),
            as_float32(emis_grid),
            as_float32(alb_grid),
            lc_grid,
            gvf_params,
        )

        # Extract GVF results
        lup = np.array(gvf_result.gvf_lup)
        lup_e = np.array(gvf_result.gvf_lup_e)
        lup_s = np.array(gvf_result.gvf_lup_s)
        lup_w = np.array(gvf_result.gvf_lup_w)
        lup_n = np.array(gvf_result.gvf_lup_n)
        gvfalb = np.array(gvf_result.gvfalb)
        gvfalb_e = np.array(gvf_result.gvfalb_e)
        gvfalb_s = np.array(gvf_result.gvfalb_s)
        gvfalb_w = np.array(gvf_result.gvfalb_w)
        gvfalb_n = np.array(gvf_result.gvfalb_n)
        gvfalbnosh = np.array(gvf_result.gvfalbnosh)
        gvfalbnosh_e = np.array(gvf_result.gvfalbnosh_e)
        gvfalbnosh_s = np.array(gvf_result.gvfalbnosh_s)
        gvfalbnosh_w = np.array(gvf_result.gvfalbnosh_w)
        gvfalbnosh_n = np.array(gvf_result.gvfalbnosh_n)
    else:
        # Simplified GVF (no walls)
        # Ground view factor is complement of sky view factor
        gvf_simple = 1.0 - svf

        # Ground temperature with shadow effect
        # Convention: shadow=1 for sunlit, shadow=0 for shaded
        # Sunlit areas get full ground temperature deviation; shaded areas get none
        tg_with_shadow = tg * shadow

        # Upwelling longwave: Stefan-Boltzmann law for ground emission
        # Lup = emissivity × SBC × T^4
        lup = emis_grid * SBC * np.power(weather.ta + tg_with_shadow + KELVIN_OFFSET, 4)

        # Simplified: assume isotropic (all directions same)
        lup_e = lup
        lup_s = lup
        lup_w = lup
        lup_n = lup

        # Albedo view factors for reflected shortwave
        gvfalb = alb_grid * gvf_simple
        gvfalb_e = gvfalb
        gvfalb_s = gvfalb
        gvfalb_w = gvfalb
        gvfalb_n = gvfalb

        # Without shadow (for anisotropic calculations)
        gvfalbnosh = alb_grid
        gvfalbnosh_e = alb_grid
        gvfalbnosh_s = alb_grid
        gvfalbnosh_w = alb_grid
        gvfalbnosh_n = alb_grid

    return GvfBundle(
        lup=as_float32(lup),
        lup_e=as_float32(lup_e),
        lup_s=as_float32(lup_s),
        lup_w=as_float32(lup_w),
        lup_n=as_float32(lup_n),
        gvfalb=as_float32(gvfalb),
        gvfalb_e=as_float32(gvfalb_e),
        gvfalb_s=as_float32(gvfalb_s),
        gvfalb_w=as_float32(gvfalb_w),
        gvfalb_n=as_float32(gvfalb_n),
        gvfalbnosh=as_float32(gvfalbnosh),
        gvfalbnosh_e=as_float32(gvfalbnosh_e),
        gvfalbnosh_s=as_float32(gvfalbnosh_s),
        gvfalbnosh_w=as_float32(gvfalbnosh_w),
        gvfalbnosh_n=as_float32(gvfalbnosh_n),
    )
