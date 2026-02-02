"""
Simplified SOLWEIG API

This module provides a clean, minimal API for SOLWEIG calculations.
It wraps the complex internal machinery with simple dataclasses that:
- Take minimal user input
- Auto-compute derived values (sun position, diffuse fraction, etc.)
- Provide sensible defaults

Example:
    import solweig
    from datetime import datetime

    result = solweig.calculate(
        surface=solweig.SurfaceData(dsm=my_dsm_array),
        location=solweig.Location(latitude=57.7, longitude=12.0),
        weather=solweig.Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0, rh=50.0, global_rad=800.0
        ),
    )
    print(f"Tmrt: {result.tmrt.mean():.1f}°C")
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from .computation import calculate_core

# Import from extracted modules
from .config import load_materials, load_params, load_physics
from .metadata import create_run_metadata, load_run_metadata, save_run_metadata
from .models import (
    HumanParams,
    Location,
    ModelConfig,
    PrecomputedData,
    ShadowArrays,
    SolweigResult,
    SurfaceData,
    SvfArrays,
    ThermalState,
    TileSpec,
    Weather,
)
from .postprocess import (
    compute_pet,
    compute_pet_grid,
    compute_utci,
    compute_utci_grid,
)
from .tiling import (
    calculate_buffer_distance,
    calculate_tiled,
    generate_tiles,
    validate_tile_size,
)
from .timeseries import calculate_timeseries
from .utils import dict_to_namespace, extract_bounds, intersect_bounds, namespace_to_dict, resample_to_grid

# Version for cache validation
__version__ = "0.0.1a1"

if TYPE_CHECKING:
    pass


def calculate(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    config: ModelConfig | None = None,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool = False,
    conifer: bool = False,
    poi_coords: list[tuple[int, int]] | None = None,
    state: ThermalState | None = None,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
) -> SolweigResult:
    """
    Calculate mean radiant temperature (Tmrt).

    This is the main entry point for SOLWEIG calculations.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        location: Geographic location (lat, lon, UTC offset).
        weather: Weather data (datetime, temperature, humidity, radiation).
        config: Model configuration object. If provided, overrides individual parameters
            (use_anisotropic_sky, human, physics, materials).
        human: Human body parameters (absorption, posture, weight, height, etc.).
            If None, uses HumanParams defaults. Overridden by config.human if config provided.
        precomputed: Pre-computed preprocessing data (walls, SVF, shadow matrices). Optional.
            When provided, skips expensive preprocessing computations.
            Use PrecomputedData.load() to load from directories.
        use_anisotropic_sky: Use anisotropic sky model for radiation. Default False.
            Requires precomputed.shadow_matrices to be provided.
            Uses Perez diffuse model and patch-based longwave calculation.
            Overridden by config.use_anisotropic_sky if config provided.
        conifer: Treat vegetation as evergreen conifers (always leaf-on). Default False.
            When False, uses seasonal leaf on/off logic (deciduous trees).
            When True, vegetation always has leaves (transmissivity constant).
            Only relevant when CDSM (canopy) data is provided in surface.
        poi_coords: Optional list of (row, col) coordinates for POI mode.
            If provided, only computes at these points (much faster).
        state: Thermal state from previous timestep. Optional.
            When provided, enables accurate multi-timestep simulation with
            thermal inertia modeling (TsWaveDelay). The returned result
            will include updated state for the next timestep.
        physics: Physics parameters (Tree_settings, Posture geometry) from load_physics().
            Site-independent scientific constants. If None, uses bundled defaults.
            Overridden by config.physics if config provided.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
            Overridden by config.materials if config provided.

    Returns:
        SolweigResult with Tmrt and optionally UTCI/PET grids.
        When state parameter is provided, result.state contains the
        updated thermal state for the next timestep.

    Example:
        # Single timestep with all defaults
        result = calculate(
            surface=SurfaceData(dsm=my_dsm),
            location=Location(latitude=57.7, longitude=12.0),
            weather=Weather(datetime=dt, ta=25, rh=50, global_rad=800),
        )

        # Multi-timestep with state management
        state = ThermalState.initial(dsm.shape)
        for weather in weather_list:
            result = calculate(surface, location, weather, state=state)
            state = result.state  # Carry forward to next timestep

        # With custom human parameters
        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=HumanParams(abs_k=0.65, weight=70, height=1.65),
        )

        # With custom physics (e.g., different tree transmissivity)
        physics = load_physics("custom_trees.json")
        result = calculate(surface, location, weather, physics=physics)

        # With landcover materials (requires land_cover grid in surface)
        materials = load_materials("site_materials.json")
        result = calculate(surface, location, weather, materials=materials)
    """
    # Apply config if provided (overrides individual parameters)
    if config is not None:
        use_anisotropic_sky = config.use_anisotropic_sky
        if config.human is not None:
            human = config.human
        if config.physics is not None:
            physics = config.physics
        if config.materials is not None:
            materials = config.materials

    # Use default human params if not provided
    if human is None:
        human = HumanParams()

    # Load default physics if not provided
    if physics is None:
        physics = load_physics()

    # Compute derived weather values (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Note: poi_coords parameter exists but POI mode not yet implemented
    if poi_coords is not None:
        raise NotImplementedError("POI mode (point-of-interest calculation) is planned for Phase 4")

    # Call the new modular core calculation (Phase 5 refactoring)
    return calculate_core(
        surface=surface,
        location=location,
        weather=weather,
        human=human,
        precomputed=precomputed,
        use_anisotropic_sky=use_anisotropic_sky,
        state=state,
        physics=physics,
        materials=materials,
        conifer=conifer,
    )


# =============================================================================
# Public API - All exports
# =============================================================================

__all__ = [
    # Main calculation functions
    "calculate",
    "calculate_timeseries",
    "calculate_tiled",
    # Dataclasses - Core inputs
    "SurfaceData",
    "Location",
    "Weather",
    "HumanParams",
    # Dataclasses - Configuration
    "ModelConfig",
    "PrecomputedData",
    "ThermalState",
    "TileSpec",
    # Dataclasses - Internal (for advanced use)
    "SvfArrays",
    "ShadowArrays",
    # Results
    "SolweigResult",
    # Post-processing
    "compute_utci",
    "compute_pet",
    "compute_utci_grid",
    "compute_pet_grid",
    # Configuration loading
    "load_params",
    "load_physics",
    "load_materials",
    # Metadata
    "create_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # Tiling utilities
    "calculate_buffer_distance",
    "validate_tile_size",
    "generate_tiles",
    # Utilities
    "dict_to_namespace",
    "namespace_to_dict",
    "extract_bounds",
    "intersect_bounds",
    "resample_to_grid",
]
