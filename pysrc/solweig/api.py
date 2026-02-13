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

import numpy as np

from .computation import calculate_core, calculate_core_fused  # noqa: F401 (calculate_core kept for direct use)
from .errors import (
    ConfigurationError,
    GridShapeMismatch,
    InvalidSurfaceData,
    MissingPrecomputedData,
    SolweigError,
    WeatherDataError,
)
from .io import download_epw

# Import from extracted modules
from .loaders import load_materials, load_params, load_physics, resolve_wall_params
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
    calculate_timeseries_tiled,
    generate_tiles,
    validate_tile_size,
)
from .timeseries import calculate_timeseries
from .utils import dict_to_namespace, extract_bounds, intersect_bounds, namespace_to_dict, resample_to_grid

if TYPE_CHECKING:
    pass


def validate_inputs(
    surface: SurfaceData,
    location: Location | None = None,
    weather: Weather | list[Weather] | None = None,
    use_anisotropic_sky: bool = False,
    precomputed: PrecomputedData | None = None,
) -> list[str]:
    """
    Validate inputs before calculation (preflight check).

    Call this before expensive operations to catch errors early.
    Raises exceptions for fatal errors, returns warnings for potential issues.

    Args:
        surface: Surface data to validate.
        location: Location to validate (optional).
        weather: Weather data to validate (optional, can be single or list).
        use_anisotropic_sky: Whether anisotropic sky will be used.
        precomputed: Precomputed data to validate.

    Returns:
        List of warning messages (empty if all valid).

    Raises:
        GridShapeMismatch: If surface grid shapes don't match DSM.
        MissingPrecomputedData: If required precomputed data is missing.
        WeatherDataError: If weather data is invalid.

    Example:
        try:
            warnings = solweig.validate_inputs(surface, location, weather)
            for w in warnings:
                print(f"Warning: {w}")
            result = solweig.calculate(surface, location, weather)
        except solweig.GridShapeMismatch as e:
            print(f"Grid mismatch: {e.field} expected {e.expected}, got {e.got}")
        except solweig.MissingPrecomputedData as e:
            print(f"Missing data: {e}")
    """
    warnings = []
    dsm_shape = surface.dsm.shape

    # Check grid shapes match DSM
    grids_to_check = [
        ("cdsm", surface.cdsm),
        ("dem", surface.dem),
        ("tdsm", surface.tdsm),
        ("wall_height", surface.wall_height),
        ("wall_aspect", surface.wall_aspect),
        ("land_cover", surface.land_cover),
        ("albedo", surface.albedo),
        ("emissivity", surface.emissivity),
    ]
    for name, grid in grids_to_check:
        if grid is not None and grid.shape != dsm_shape:
            raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF arrays if present
    if surface.svf is not None:
        svf_grids = [
            ("svf.svf", surface.svf.svf),
            ("svf.svf_north", surface.svf.svf_north),
            ("svf.svf_east", surface.svf.svf_east),
            ("svf.svf_south", surface.svf.svf_south),
            ("svf.svf_west", surface.svf.svf_west),
        ]
        for name, grid in svf_grids:
            if grid is not None and grid.shape != dsm_shape:
                raise GridShapeMismatch(name, dsm_shape, grid.shape)

    # Check SVF is available (required for all calculations)
    if surface.svf is None and (precomputed is None or precomputed.svf is None):
        raise MissingPrecomputedData(
            "Sky View Factor (SVF) data is required but not available.",
            "Call surface.compute_svf() before calculate(), or use SurfaceData.prepare() "
            "which computes SVF automatically.",
        )

    # Check anisotropic sky requirements
    if use_anisotropic_sky:
        has_shadow_matrices = (precomputed is not None and precomputed.shadow_matrices is not None) or (
            surface.shadow_matrices is not None
        )
        if not has_shadow_matrices:
            raise MissingPrecomputedData(
                "shadow_matrices required for anisotropic sky model",
                "Either set use_anisotropic_sky=False, or provide shadow matrices via "
                "precomputed=PrecomputedData(shadow_matrices=...) or surface.shadow_matrices",
            )

    # Check for potential issues (warnings, not errors)
    if surface.cdsm is not None and not surface._preprocessed and surface.cdsm_relative:
        warnings.append(
            "CDSM provided with cdsm_relative=True but preprocess() not called. "
            "Vegetation heights may be incorrect. Call surface.preprocess() first."
        )
    if surface.tdsm is not None and not surface._preprocessed and surface.tdsm_relative:
        warnings.append(
            "TDSM provided with tdsm_relative=True but preprocess() not called. "
            "Trunk heights may be incorrect. Call surface.preprocess() first."
        )

    # DSM height sanity checks
    dsm_max = float(np.nanmax(surface.dsm))
    dsm_min = float(np.nanmin(surface.dsm))
    height_range = dsm_max - dsm_min

    if height_range > 500:
        warnings.append(
            f"DSM height range is {height_range:.0f}m (max={dsm_max:.0f}m, min={dsm_min:.0f}m). "
            "This exceeds typical urban areas. If your DSM contains terrain elevation, "
            "provide a DEM to separate ground from building heights."
        )

    if surface.dem is None and dsm_min > 100:
        warnings.append(
            f"DSM minimum value is {dsm_min:.0f}m with no DEM provided. "
            "If this is above-sea-level elevation, provide a DEM so SOLWEIG can "
            "compute building heights correctly."
        )

    # Per-layer relative height mismatch detection
    for grid_name, grid, is_relative in [
        ("CDSM", surface.cdsm, surface.cdsm_relative),
        ("TDSM", surface.tdsm, surface.tdsm_relative),
    ]:
        if grid is not None and is_relative:
            nonzero = grid[grid > 0]
            if nonzero.size > 0:
                grid_min_nz = float(np.nanmin(nonzero))
                if grid_min_nz > 50:
                    flag = f"{grid_name.lower()}_relative"
                    warnings.append(
                        f"{grid_name} minimum non-zero value is {grid_min_nz:.0f}m with "
                        f"{flag}=True. Relative vegetation heights are typically "
                        f"0-50m. If it contains absolute elevations, set {flag}=False."
                    )

    if surface.cdsm is not None and not surface.cdsm_relative and surface._looks_like_relative_heights():
        cdsm_max = float(np.nanmax(surface.cdsm))
        warnings.append(
            f"CDSM values (max={cdsm_max:.1f}m) are much smaller than DSM "
            f"(min={dsm_min:.1f}m) with cdsm_relative=False. "
            "If CDSM contains height-above-ground, set cdsm_relative=True "
            "and call surface.preprocess()."
        )

    # Validate weather if provided
    if weather is not None:
        weather_list = weather if isinstance(weather, list) else [weather]
        for i, w in enumerate(weather_list):
            # Basic range checks (Weather.__post_init__ catches some, but we add more)
            if w.ta < -100 or w.ta > 60:
                warnings.append(
                    f"Weather[{i}].ta={w.ta}°C is outside typical range [-100, 60]. Verify this is correct."
                )
            if w.global_rad > 1400:
                warnings.append(
                    f"Weather[{i}].global_rad={w.global_rad} W/m² exceeds solar constant. Verify this is correct."
                )

    return warnings


def calculate(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    config: ModelConfig | None = None,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool | None = None,
    conifer: bool = False,
    poi_coords: list[tuple[int, int]] | None = None,
    state: ThermalState | None = None,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    wall_material: str | None = None,
    max_shadow_distance_m: float | None = None,
    return_state_copy: bool = True,
    _requested_outputs: set[str] | None = None,
) -> SolweigResult:
    """
    Calculate mean radiant temperature (Tmrt).

    This is the main entry point for SOLWEIG calculations.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        location: Geographic location (lat, lon, UTC offset).
        weather: Weather data (datetime, temperature, humidity, radiation).
        config: Model configuration object providing base settings.
            Explicit parameters (human, use_anisotropic_sky, etc.) override
            config values when provided.
        human: Human body parameters (absorption, posture, weight, height, etc.).
            If None, uses config.human or HumanParams defaults.
        precomputed: Pre-computed preprocessing data (walls, SVF, shadow matrices). Optional.
            When provided, skips expensive preprocessing computations.
            Use PrecomputedData.load() to load from directories.
        use_anisotropic_sky: Use anisotropic sky model for radiation.
            If None, uses config.use_anisotropic_sky (default True).
            Requires precomputed.shadow_matrices to be provided.
            Uses Perez diffuse model and patch-based longwave calculation.
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
            Site-independent scientific constants. If None, uses config.physics or bundled defaults.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
            If None, uses config.materials.
        wall_material: Wall material type for temperature model.
            One of "brick", "concrete", "wood", "cobblestone" (case-insensitive).
            If None (default), uses generic wall params from materials JSON.
        return_state_copy: If True (default), return a deep-copied thermal state.
            Set False in internal time-series loops to avoid per-step state copies.

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

        # With config as base, explicit param override
        config = ModelConfig(use_anisotropic_sky=True)
        result = calculate(
            surface, location, weather,
            config=config,
            use_anisotropic_sky=False,  # Explicit param wins
        )
    """
    import logging

    logger = logging.getLogger(__name__)

    # Build effective configuration: explicit params override config
    # Config provides base values, explicit params take precedence
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_max_shadow = max_shadow_distance_m

    if config is not None:
        # Use config values as fallback for None parameters
        if effective_aniso is None:
            effective_aniso = config.use_anisotropic_sky
        if effective_human is None:
            effective_human = config.human
        if effective_physics is None:
            effective_physics = config.physics
        if effective_materials is None:
            effective_materials = config.materials
        if effective_max_shadow is None:
            effective_max_shadow = config.max_shadow_distance_m

        # Debug log when explicit params override config
        overrides = []
        if use_anisotropic_sky is not None and use_anisotropic_sky != config.use_anisotropic_sky:
            overrides.append(f"use_anisotropic_sky={use_anisotropic_sky}")
        if human is not None and config.human is not None:
            overrides.append("human")
        if physics is not None and config.physics is not None:
            overrides.append("physics")
        if materials is not None and config.materials is not None:
            overrides.append("materials")
        if overrides:
            logger.debug(f"Explicit params override config: {', '.join(overrides)}")

    # Apply defaults for anything still None
    if effective_aniso is None:
        effective_aniso = True
    if effective_human is None:
        effective_human = HumanParams()
    # Auto-load bundled UMEP JSON as default materials (single source of truth)
    if effective_materials is None:
        effective_materials = load_params()

    # Assign back to use in the rest of the function
    use_anisotropic_sky = effective_aniso
    human = effective_human
    physics = effective_physics
    materials = effective_materials

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

    # Fill NaN in surface layers (idempotent — skipped if already done)
    surface.fill_nan()

    # Fused Rust pipeline — single FFI call per daytime timestep.
    # Both isotropic and anisotropic sky models are supported.
    return calculate_core_fused(
        surface=surface,
        location=location,
        weather=weather,
        human=human,
        precomputed=precomputed,
        state=state,
        physics=physics,
        materials=materials,
        conifer=conifer,
        wall_material=wall_material,
        use_anisotropic_sky=use_anisotropic_sky,
        max_shadow_distance_m=effective_max_shadow,
        return_state_copy=return_state_copy,
        requested_outputs=_requested_outputs,
    )


# =============================================================================
# Public API - All exports
# =============================================================================

__all__ = [
    # Main calculation functions
    "calculate",
    "calculate_timeseries",
    "calculate_tiled",
    "calculate_timeseries_tiled",
    "validate_inputs",
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
    # Errors
    "SolweigError",
    "InvalidSurfaceData",
    "GridShapeMismatch",
    "MissingPrecomputedData",
    "WeatherDataError",
    "ConfigurationError",
    # Post-processing
    "compute_utci",
    "compute_pet",
    "compute_utci_grid",
    "compute_pet_grid",
    # Configuration loading
    "load_params",
    "load_physics",
    "load_materials",
    "resolve_wall_params",
    # Metadata
    "create_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # Tiling utilities
    "calculate_buffer_distance",
    "validate_tile_size",
    "generate_tiles",
    # I/O
    "download_epw",
    # Utilities
    "dict_to_namespace",
    "namespace_to_dict",
    "extract_bounds",
    "intersect_bounds",
    "resample_to_grid",
]
