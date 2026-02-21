"""
Ground View Factor (GVF) — building mask detection.

Provides ``detect_building_mask()`` used by ``calculate_core_fused()`` to
identify building footprint pixels before the fused Rust GVF computation.

Reference:
- Lindberg et al. (2008) - SOLWEIG GVF model with wall radiation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        1. If land_cover provided: Use ID 2 (buildings) directly.
        2. Elif wall_height provided: Dilate wall pixels to fill building
           interiors, then combine with pixels elevated >2 m above the
           10th-percentile ground level (catches rooftops).
        3. Else: Assume all ground (no buildings).
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
