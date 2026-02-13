"""Geospatial backend detection — single source of truth.

Determines whether to use rasterio or GDAL for raster I/O and geometric
utilities.  In QGIS / OSGeo4W environments rasterio is never attempted
(it causes numpy binary-incompatibility crashes).

Exported flags
--------------
GDAL_ENV : bool
    True  → use GDAL for raster operations.
    False → use rasterio.
RASTERIO_AVAILABLE : bool
    True when rasterio was successfully imported.
GDAL_AVAILABLE : bool
    True when GDAL (osgeo) was successfully imported.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def in_osgeo_environment() -> bool:
    """Return True when running inside QGIS or OSGeo4W."""
    if "qgis" in sys.modules or "qgis.core" in sys.modules:
        return True
    if any(k in os.environ for k in ("QGIS_PREFIX_PATH", "QGIS_DEBUG", "OSGEO4W_ROOT")):
        return True
    exe = sys.executable.lower()
    return any(m in exe for m in ("osgeo4w", "qgis"))


# ---------------------------------------------------------------------------
# Import probes
# ---------------------------------------------------------------------------


def _try_import_rasterio() -> bool:
    try:
        import pyproj  # noqa: F401
        import rasterio  # noqa: F401
        from rasterio.features import rasterize  # noqa: F401
        from rasterio.mask import mask  # noqa: F401
        from rasterio.transform import Affine, from_origin  # noqa: F401
        from rasterio.windows import Window  # noqa: F401
        from shapely import geometry  # noqa: F401

        return True
    except (ImportError, OSError, RuntimeError) as e:
        logger.debug("Rasterio import failed: %s", e)
        return False


def _try_import_gdal() -> bool:
    try:
        from osgeo import gdal, osr  # noqa: F401

        return True
    except (ImportError, OSError) as e:
        logger.debug("GDAL import failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Backend selection  (runs once at first import)
# ---------------------------------------------------------------------------


def _setup_geospatial_backend() -> tuple[bool, bool, bool]:
    """Choose the geospatial backend.

    Returns (gdal_env, rasterio_available, gdal_available).
    """
    # 1. Forced via env-var
    if os.environ.get("UMEP_USE_GDAL", "").lower() in ("1", "true", "yes"):
        if _try_import_gdal():
            logger.info("Using GDAL for raster operations (forced via UMEP_USE_GDAL).")
            return True, False, True
        raise ImportError("UMEP_USE_GDAL is set but GDAL could not be imported. Install GDAL or unset UMEP_USE_GDAL.")

    # 2. QGIS / OSGeo4W — prefer GDAL, never try rasterio first
    if in_osgeo_environment():
        logger.debug("Detected OSGeo4W/QGIS environment, preferring GDAL backend.")
        if _try_import_gdal():
            logger.info("Using GDAL for raster operations (OSGeo4W/QGIS environment).")
            return True, False, True
        # Unexpected — GDAL should always be present here
        logger.warning("GDAL import failed in OSGeo4W/QGIS environment, trying rasterio...")
        if _try_import_rasterio():
            logger.info("Using rasterio for raster operations.")
            return False, True, False
        raise ImportError(
            "Failed to import both GDAL and rasterio in OSGeo4W/QGIS environment.\n"
            "This is unexpected — GDAL should be available. Check your installation."
        )

    # 3. Standard environment — prefer rasterio, fall back to GDAL
    if _try_import_rasterio():
        logger.info("Using rasterio for raster operations.")
        return False, True, False

    logger.warning("Rasterio import failed, trying GDAL...")
    if _try_import_gdal():
        logger.info("Using GDAL for raster operations.")
        return True, False, True

    raise ImportError(
        "Neither rasterio nor GDAL could be imported.\n"
        "Install with: pip install rasterio\n"
        "Or for QGIS/OSGeo4W environments, ensure GDAL is properly configured."
    )


GDAL_ENV, RASTERIO_AVAILABLE, GDAL_AVAILABLE = _setup_geospatial_backend()
