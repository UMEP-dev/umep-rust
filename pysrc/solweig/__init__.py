"""
SOLWEIG - High-performance urban microclimate model.

A Python package with Rust-accelerated algorithms for computing mean radiant
temperature (Tmrt) and other urban climate parameters.

## Simplified API (Recommended)

    import solweig
    from datetime import datetime

    result = solweig.calculate(
        surface=solweig.SurfaceData(dsm=my_dsm_array),
        location=solweig.Location(latitude=57.7, longitude=12.0),
        weather=solweig.Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25, rh=50, global_rad=800
        ),
    )
    print(f"Tmrt: {result.tmrt.mean():.1f}°C")

## Low-level API

    import solweig

    # Load raster data
    dsm, transform, crs, nodata = solweig.io.load_raster("dsm.tif")

    # Calculate sky view factor
    solweig.svf.generate_svf(dsm_path, bbox, out_dir)

    # Generate shadows
    solweig.shadows.generate_shadows(dsm_path, date, wall_ht, wall_aspect, bbox, out_dir)

    # Generate wall heights and aspects
    solweig.walls.generate_wall_hts(dsm_path, bbox, out_dir)
"""

import logging

logger = logging.getLogger(__name__)

# Version
__version__ = "0.0.1a1"

# Import simplified API (Phase 2 modernization)
from .api import (
    SurfaceData,
    Location,
    Weather,
    HumanParams,
    SolweigResult,
    calculate,
    calculate_timeseries,
    calculate_tiled,
    load_params,
    # Tiled processing helpers
    calculate_buffer_distance,
    TileSpec,
    generate_tiles,
)

# Import I/O module
from . import io
from . import configs
from . import tiles
from . import svf
from . import shadows
from . import walls

# Import runner classes (after configs to avoid circular imports)
try:
    from .runner import SolweigRunCore
    from .solweig_runner_rust import SolweigRunRust
except ImportError as e:
    logger.debug(f"Runner imports deferred: {e}")
    SolweigRunCore = None
    SolweigRunRust = None

# Try to import Rust algorithms
try:
    from .rustalgos import GPU_ENABLED, shadowing, skyview, gvf, sky, vegetation, utci, pet

    # Enable GPU by default if available
    if GPU_ENABLED:
        shadowing.enable_gpu()
        logger.info("GPU acceleration enabled by default")
    else:
        logger.debug("GPU support not compiled in this build")

except ImportError as e:
    logger.warning(f"Failed to import Rust algorithms: {e}")
    GPU_ENABLED = False
    shadowing = None
    skyview = None
    gvf = None
    sky = None
    vegetation = None
    utci = None
    pet = None

__all__ = [
    # Version
    "__version__",
    # Simplified API (recommended)
    "SurfaceData",
    "Location",
    "Weather",
    "HumanParams",
    "SolweigResult",
    "calculate",
    "calculate_timeseries",
    "calculate_tiled",
    "load_params",
    # Tiled processing helpers
    "calculate_buffer_distance",
    "TileSpec",
    "generate_tiles",
    # Modules
    "io",
    "configs",
    "tiles",
    "svf",
    "shadows",
    "walls",
    # Runner classes
    "SolweigRunCore",
    "SolweigRunRust",
    # Rust modules
    "GPU_ENABLED",
    "shadowing",
    "skyview",
    "gvf",
    "sky",
    "vegetation",
    "utci",
    "pet",
]
