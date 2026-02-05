"""
SOLWEIG - High-performance urban microclimate model.

A Python package with Rust-accelerated algorithms for computing mean radiant
temperature (Tmrt) and other urban climate parameters.

## Modern API

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

## Time Series

    results = solweig.calculate_timeseries(
        surface=surface,
        weather_series=[weather1, weather2, weather3],
        location=location,
    )

## Utilities

    # Load raster data
    dsm, transform, crs, nodata = solweig.io.load_raster("dsm.tif")

    # Generate wall heights and aspects
    solweig.walls.generate_wall_hts(dsm_path, bbox, out_dir)
"""

import contextlib
import logging

logger = logging.getLogger(__name__)

# Version
__version__ = "0.0.1a1"

# Import simplified API (Phase 2 modernization)
# Import utility modules
from . import io, progress, walls  # noqa: E402
from .api import (  # noqa: E402
    HumanParams,
    Location,
    ModelConfig,
    PrecomputedData,
    SolweigResult,
    SurfaceData,
    TileSpec,
    Weather,
    calculate,
    # Tiled processing helpers
    calculate_buffer_distance,
    calculate_tiled,
    calculate_timeseries,
    compute_pet,
    compute_pet_grid,
    # Post-processing: Thermal comfort indices
    compute_utci,
    compute_utci_grid,
    # Run metadata/provenance
    create_run_metadata,
    # I/O
    download_epw,
    generate_tiles,
    load_materials,
    load_params,
    load_physics,
    load_run_metadata,
    save_run_metadata,
    # Validation
    validate_inputs,
)
from .errors import SolweigError  # noqa: E402

# Try to import Rust algorithms
try:
    from .rustalgos import GPU_ENABLED, gvf, pet, shadowing, sky, skyview, utci, vegetation

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


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available at runtime.

    Returns True if:
    - GPU support was compiled into the Rust extension
    - A GPU device was successfully detected and initialized

    Use this to check GPU status before running compute-intensive operations.

    Returns:
        True if GPU acceleration is available, False otherwise.
    """
    if not GPU_ENABLED:
        return False
    if shadowing is None:
        return False
    try:
        return shadowing.is_gpu_enabled()
    except (AttributeError, RuntimeError):
        return False


def get_compute_backend() -> str:
    """
    Get the current compute backend.

    Returns:
        "gpu" if GPU acceleration is available and enabled, "cpu" otherwise.
    """
    return "gpu" if is_gpu_available() else "cpu"


def disable_gpu() -> None:
    """
    Disable GPU acceleration, falling back to CPU.

    This can be useful for debugging or if GPU results differ from expected.
    The change takes effect immediately for subsequent calculations.
    """
    if shadowing is not None:
        with contextlib.suppress(AttributeError):
            shadowing.disable_gpu()


__all__ = [
    # Version
    "__version__",
    # Core API
    "SurfaceData",
    "PrecomputedData",
    "Location",
    "Weather",
    "HumanParams",
    "ModelConfig",
    "SolweigResult",
    "SolweigError",
    "calculate",
    "calculate_timeseries",
    "calculate_tiled",
    "validate_inputs",
    "load_params",
    "load_physics",
    "load_materials",
    # Tiled processing
    "calculate_buffer_distance",
    "TileSpec",
    "generate_tiles",
    # Post-processing: Thermal comfort
    "compute_utci",
    "compute_pet",
    "compute_utci_grid",
    "compute_pet_grid",
    # Run metadata/provenance
    "create_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # I/O
    "download_epw",
    # Utility modules
    "io",
    "walls",
    "progress",
    # GPU utilities
    "is_gpu_available",
    "get_compute_backend",
    "disable_gpu",
    "GPU_ENABLED",
]
