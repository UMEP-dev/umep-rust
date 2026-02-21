"""
Internal SVF resolution.

Looks up SVF arrays from ``surface.svf`` or ``precomputed.svf`` at
calculation time. Not part of the public API.
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
    Look up SVF from surface or precomputed data.

    Returns (SvfBundle, needs_psi_adjustment). Raises
    MissingPrecomputedData if SVF is not available from either source.
    """
    # Import here to avoid circular dependency

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
        # Geometric svfbuveg without psi — adjusted at calculation time
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
        # Geometric svfbuveg without psi — adjusted at calculation time
        needs_psi_adjustment = False

    # No SVF available — require explicit computation
    else:
        from ..errors import MissingPrecomputedData

        raise MissingPrecomputedData(
            "Sky View Factor (SVF) data is required but not available.",
            "Call surface.compute_svf() before calculate(), or use SurfaceData.prepare() "
            "which computes SVF automatically.",
        )

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
