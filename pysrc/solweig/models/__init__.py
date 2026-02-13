"""Data models for SOLWEIG calculations.

This package contains all data model classes organized by domain:
- state: ThermalState, TileSpec
- surface: SurfaceData
- weather: Location, Weather
- precomputed: SvfArrays, ShadowArrays, PrecomputedData
- config: ModelConfig, HumanParams
- results: SolweigResult
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
