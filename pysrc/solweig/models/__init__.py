"""Data models for SOLWEIG calculations.

Modules
-------
state
    ``ThermalState`` (thermal inertia carry-forward) and ``TileSpec``
    (tile geometry for large-raster processing).
surface
    ``SurfaceData`` — DSM, CDSM, DEM, land cover, walls, and SVF.
weather
    ``Location`` and ``Weather`` dataclasses.
precomputed
    ``SvfArrays``, ``ShadowArrays``, ``PrecomputedData`` — cached
    preprocessing results loaded from disk.
config
    ``ModelConfig`` and ``HumanParams`` — run-time settings.
results
    ``SolweigResult`` — output grids (Tmrt, radiation, shadow).
"""

from .config import HumanParams, ModelConfig
from .precomputed import PrecomputedData, ShadowArrays, SvfArrays
from .results import SolweigResult
from .state import ThermalState, TileSpec
from .surface import SurfaceData
from .weather import Location, Weather

__all__ = [
    # State management
    "ThermalState",
    "TileSpec",
    # Surface data
    "SurfaceData",
    # Weather and location
    "Location",
    "Weather",
    # Precomputed data
    "SvfArrays",
    "ShadowArrays",
    "PrecomputedData",
    # Configuration
    "ModelConfig",
    "HumanParams",
    # Results
    "SolweigResult",
]
