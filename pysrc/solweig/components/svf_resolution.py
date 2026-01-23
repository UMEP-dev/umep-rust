"""
SVF (Sky View Factor) resolution component.

Handles three sources of SVF data:
1. Cached SVF from surface preparation (surface.svf)
2. Pre-computed SVF (precomputed.svf)
3. On-the-fly computation using Rust skyview module

Returns a complete SvfBundle with all directional components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..bundles import DirectionalArrays, SvfBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import PrecomputedData, SurfaceData


def resolve_svf(
    surface: SurfaceData,
    precomputed: PrecomputedData | None,
    dsm: NDArray[np.floating],
    cdsm: NDArray[np.floating] | None,
    tdsm: NDArray[np.floating] | None,
    pixel_size: float,
    use_veg: bool,
    max_height: float,
    psi: float | None = None,
) -> tuple[SvfBundle, bool]:
    """
    Resolve SVF data from available sources or compute on-the-fly.

    Checks three sources in priority order:
    1. surface.svf (cached/prepared) - fastest
    2. precomputed.svf (legacy) - fast
    3. Compute fresh using skyview.calculate_svf - slower

    Args:
        surface: Surface data (may contain cached SVF)
        precomputed: Pre-computed data (may contain SVF)
        dsm: Digital Surface Model (for fresh computation)
        cdsm: Canopy DSM (for fresh computation if use_veg=True)
        tdsm: Trunk DSM (for fresh computation if use_veg=True)
        pixel_size: Grid resolution in meters
        use_veg: Whether to include vegetation in SVF calculation
        max_height: Maximum building height for SVF computation
        psi: Vegetation transmissivity (optional, for svfbuveg calculation)
             If None, uses preliminary calculation that needs later adjustment

    Returns:
        Tuple of (SvfBundle, needs_psi_adjustment):
            - SvfBundle: Complete SVF data with all directional components
            - needs_psi_adjustment: True if svfbuveg needs recalculation with psi

    Note:
        When loading from cached/precomputed sources, svfbuveg already includes
        transmissivity adjustment, so needs_psi_adjustment=False.
        When computing fresh, svfbuveg is preliminary and needs_psi_adjustment=True.
    """
    # Import here to avoid circular dependency
    from ..rustalgos import skyview

    needs_psi_adjustment = False

    # Priority 1: Check surface.svf (from prepare/cache)
    if surface.svf is not None:
        svf_data = surface.svf
        svf = svf_data.svf
        svf_directional = DirectionalArrays(
            north=svf_data.svf_north,
            east=svf_data.svf_east,
            south=svf_data.svf_south,
            west=svf_data.svf_west,
        )
        svf_veg = svf_data.svf_veg
        svf_veg_directional = DirectionalArrays(
            north=svf_data.svf_veg_north,
            east=svf_data.svf_veg_east,
            south=svf_data.svf_veg_south,
            west=svf_data.svf_veg_west,
        )
        svf_aveg = svf_data.svf_aveg
        svf_aveg_directional = DirectionalArrays(
            north=svf_data.svf_aveg_north,
            east=svf_data.svf_aveg_east,
            south=svf_data.svf_aveg_south,
            west=svf_data.svf_aveg_west,
        )
        svfbuveg = svf_data.svfbuveg
        # Cached svfbuveg already includes transmissivity adjustment
        needs_psi_adjustment = False

    # Priority 2: Check precomputed.svf (legacy)
    elif precomputed is not None and precomputed.svf is not None:
        svf_data = precomputed.svf
        svf = svf_data.svf
        svf_directional = DirectionalArrays(
            north=svf_data.svf_north,
            east=svf_data.svf_east,
            south=svf_data.svf_south,
            west=svf_data.svf_west,
        )
        svf_veg = svf_data.svf_veg
        svf_veg_directional = DirectionalArrays(
            north=svf_data.svf_veg_north,
            east=svf_data.svf_veg_east,
            south=svf_data.svf_veg_south,
            west=svf_data.svf_veg_west,
        )
        svf_aveg = svf_data.svf_aveg
        svf_aveg_directional = DirectionalArrays(
            north=svf_data.svf_aveg_north,
            east=svf_data.svf_aveg_east,
            south=svf_data.svf_aveg_south,
            west=svf_data.svf_aveg_west,
        )
        svfbuveg = svf_data.svfbuveg
        # Precomputed svfbuveg already includes transmissivity adjustment
        needs_psi_adjustment = False

    # Priority 3: Compute fresh
    else:
        # Compute SVF with directional components
        svf_result = skyview.calculate_svf(
            dsm,
            cdsm if use_veg else np.zeros_like(dsm),
            tdsm if use_veg else np.zeros_like(dsm),
            pixel_size,
            use_veg,
            max_height,
            2,  # patch_option (153 patches)
            3.0,  # min_sun_elev_deg
            None,
        )

        svf = np.array(svf_result.svf)
        svf_directional = DirectionalArrays(
            north=np.array(svf_result.svf_north),
            east=np.array(svf_result.svf_east),
            south=np.array(svf_result.svf_south),
            west=np.array(svf_result.svf_west),
        )
        svf_veg = np.array(svf_result.svf_veg) if use_veg else np.zeros_like(svf)
        svf_veg_directional = DirectionalArrays(
            north=np.array(svf_result.svf_veg_north) if use_veg else np.zeros_like(svf),
            east=np.array(svf_result.svf_veg_east) if use_veg else np.zeros_like(svf),
            south=np.array(svf_result.svf_veg_south) if use_veg else np.zeros_like(svf),
            west=np.array(svf_result.svf_veg_west) if use_veg else np.zeros_like(svf),
        )
        svf_aveg = np.array(svf_result.svf_veg_blocks_bldg_sh) if use_veg else np.zeros_like(svf)
        svf_aveg_directional = DirectionalArrays(
            north=np.array(svf_result.svf_veg_blocks_bldg_sh_north) if use_veg else np.zeros_like(svf),
            east=np.array(svf_result.svf_veg_blocks_bldg_sh_east) if use_veg else np.zeros_like(svf),
            south=np.array(svf_result.svf_veg_blocks_bldg_sh_south) if use_veg else np.zeros_like(svf),
            west=np.array(svf_result.svf_veg_blocks_bldg_sh_west) if use_veg else np.zeros_like(svf),
        )

        # Combined SVF - use psi if provided, otherwise preliminary calculation
        if psi is not None and use_veg:
            # Formula: svfbuveg = svf - (1 - svf_veg) * (1 - psi)
            svfbuveg = svf - (1.0 - svf_veg) * (1.0 - psi)
            svfbuveg = np.clip(svfbuveg, 0.0, 1.0)
            needs_psi_adjustment = False
        else:
            # Preliminary calculation (will need adjustment with psi later)
            svfbuveg = svf + svf_veg - 1.0
            svfbuveg = np.clip(svfbuveg, 0.0, 1.0)
            needs_psi_adjustment = True

    # Compute svfalfa (SVF angle) from SVF values
    # Formula: svfalfa = arcsin(exp(log(1 - (svf + svf_veg - 1)) / 2))
    # Used in anisotropic sky calculations
    tmp = np.clip(svf + svf_veg - 1.0, 0.0, 1.0)
    eps = np.finfo(np.float32).tiny
    safe_term = np.clip(1.0 - tmp, eps, 1.0)
    svfalfa = np.arcsin(np.exp(np.log(safe_term) / 2.0))

    # Construct bundle
    bundle = SvfBundle(
        svf=svf,
        svf_directional=svf_directional,
        svf_veg=svf_veg,
        svf_veg_directional=svf_veg_directional,
        svf_aveg=svf_aveg,
        svf_aveg_directional=svf_aveg_directional,
        svfbuveg=svfbuveg,
        svfalfa=svfalfa,
    )

    return bundle, needs_psi_adjustment


def adjust_svfbuveg_with_psi(
    svf: NDArray[np.floating],
    svf_veg: NDArray[np.floating],
    psi: float,
    use_veg: bool,
) -> NDArray[np.floating]:
    """
    Adjust svfbuveg with vegetation transmissivity.

    This is needed when SVF was computed fresh without knowledge of psi.

    Args:
        svf: Total sky view factor
        svf_veg: Vegetation-only SVF
        psi: Vegetation transmissivity (0.03 typical for deciduous trees)
        use_veg: Whether vegetation is active

    Returns:
        Adjusted svfbuveg array

    Formula:
        svfbuveg = svf - (1 - svf_veg) * (1 - psi)
    """
    if use_veg:
        svfbuveg = svf - (1.0 - svf_veg) * (1.0 - psi)
        return np.clip(svfbuveg, 0.0, 1.0).astype(np.float32)
    else:
        return svf.astype(np.float32)
