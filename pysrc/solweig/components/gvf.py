"""
Ground View Factor (GVF) — building mask detection.

Provides ``detect_building_mask()`` used by ``calculate_core_fused()`` to
identify building footprint pixels before the fused Rust GVF computation.

Reference:
- Lindberg et al. (2008) - SOLWEIG GVF model with wall radiation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..errors import InvalidSurfaceData
from ..physics.morphology import generate_binary_structure

logger = logging.getLogger(__name__)

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
    if dsm is None or dsm.size == 0:
        raise InvalidSurfaceData(
            "DSM is required for building mask detection but is missing or empty.",
            field="dsm",
        )

    if pixel_size <= 0:
        raise InvalidSurfaceData(
            f"pixel_size must be positive, got {pixel_size}.",
            field="pixel_size",
            expected="> 0",
            got=str(pixel_size),
        )

    if land_cover is not None:
        if land_cover.shape != dsm.shape:
            raise InvalidSurfaceData(
                f"land_cover shape {land_cover.shape} does not match DSM shape {dsm.shape}.",
                field="land_cover",
                expected=str(dsm.shape),
                got=str(land_cover.shape),
            )
        # Use land cover directly: ID 2 = buildings
        buildings = np.ones_like(dsm, dtype=np.float32)
        buildings[land_cover == 2] = 0.0
        return buildings

    if wall_height is not None:
        if wall_height.shape != dsm.shape:
            raise InvalidSurfaceData(
                f"wall_height shape {wall_height.shape} does not match DSM shape {dsm.shape}.",
                field="wall_height",
                expected=str(dsm.shape),
                got=str(wall_height.shape),
            )
        # Approximate building footprints from wall heights
        # Wall pixels mark building edges; dilate to capture interiors
        wall_mask = wall_height > 0

        # Dilate building mask by ~25m to capture nearby ground level.
        # Assumes max building footprint radius ~25m; scale iterations by pixel size.
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
    logger.debug("No land_cover or wall_height provided; assuming all-ground building mask.")
    return np.ones_like(dsm, dtype=np.float32)
