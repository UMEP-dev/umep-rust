"""Orchestration layer for a single SOLWEIG timestep.

The public entry point is :func:`calculate_core_fused`, which hands off
the full pipeline (shadows, ground temperature, GVF, thermal delay,
radiation, Tmrt) to a single fused Rust FFI call.  The older
:func:`calculate_core` is retained for reference but is no longer
invoked by the public API.

Pipeline::

    SVF resolution → Shadows → Ground temp → GVF
        → Thermal delay → Radiation → Tmrt
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .bundles import LupBundle
from .components.ground import compute_ground_temperature
from .components.gvf import compute_gvf
from .components.radiation import compute_radiation
from .components.shadows import compute_shadows
from .components.svf_resolution import resolve_svf
from .components.tmrt import compute_tmrt
from .constants import KELVIN_OFFSET, SBC
from .rustalgos import ground as ground_rust

_OUT_SHADOW = 1 << 0
_OUT_KDOWN = 1 << 1
_OUT_KUP = 1 << 2
_OUT_LDOWN = 1 << 3
_OUT_LUP = 1 << 4
_OUT_ALL = _OUT_SHADOW | _OUT_KDOWN | _OUT_KUP | _OUT_LDOWN | _OUT_LUP

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .api import HumanParams, Location, PrecomputedData, SolweigResult, SurfaceData, ThermalState, Weather
    from .bundles import GvfBundle


def _nighttime_result(
    surface: SurfaceData,
    weather: Weather,
    state: ThermalState | None,
    materials: SimpleNamespace | None,
    copy_state: bool = True,
) -> SolweigResult:
    """
    Compute simplified nighttime result when sun is below horizon.

    At night:
    - Tmrt ≈ Ta (no solar radiation)
    - Shadow = 0 (everything shaded)
    - Kdown = Kup = 0 (no shortwave)
    - Ldown/Lup from atmospheric and surface emission only

    Args:
        surface: Surface data (for emissivity grid)
        weather: Weather data (for air temperature)
        state: Optional thermal state (resets for morning)
        materials: Material properties (for emissivity)
        copy_state: If True, copy state before mutation. If False, mutate in-place.

    Returns:
        SolweigResult with nighttime values
    """
    # Import here to avoid circular dependency
    from .api import SolweigResult

    rows, cols = surface.dsm.shape

    # Get emissivity grid for nighttime longwave
    _, emis_grid, _, _, _ = surface.get_land_cover_properties(materials)

    # Nighttime: Tmrt ≈ Ta (simplified, no solar heating)
    # Preserve NaN from DSM to mark invalid pixels (consistent with daytime path)
    nan_mask = np.isnan(surface.dsm)
    tmrt = np.full((rows, cols), weather.ta, dtype=np.float32)
    tmrt[nan_mask] = np.nan
    shadow = np.zeros((rows, cols), dtype=np.float32)  # 0 = shaded (night)

    # Nighttime longwave: Lup = SBC × emis × Ta⁴
    ta_k = weather.ta + KELVIN_OFFSET
    lup_night = SBC * emis_grid * np.power(ta_k, 4)
    # Ldown from sky with typical nighttime emissivity ~0.95
    ldown_night = np.full((rows, cols), SBC * 0.95 * np.power(ta_k, 4), dtype=np.float32)

    # Update thermal state for nighttime (copy first, then mutate)
    output_state = None
    if state is not None:
        output_state = state.copy() if copy_state else state
        output_state.firstdaytime = 1.0  # Reset for morning
        output_state.timeadd = 0.0  # Reset time accumulator

    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow,
        kdown=np.zeros((rows, cols), dtype=np.float32),
        kup=np.zeros((rows, cols), dtype=np.float32),
        ldown=ldown_night.astype(np.float32),
        lup=lup_night.astype(np.float32),
        utci=None,
        pet=None,
        state=output_state,
    )


def _apply_thermal_delay(
    gvf_bundle: GvfBundle,
    ground_tg: NDArray[np.floating],
    shadow: NDArray[np.floating],
    weather: Weather,
    state: ThermalState | None,
) -> LupBundle:
    """
    Apply thermal inertia (TsWaveDelay) to upwelling longwave radiation.

    This models the thermal mass of ground and walls, smoothing rapid temperature
    changes throughout the day. Essential for accurate time-series simulations.

    Uses batched Rust function to reduce FFI overhead from 6 calls to 1.

    Args:
        gvf_bundle: Ground view factor results (raw Lup before delay)
        ground_tg: Ground temperature deviation from air temperature (K)
        shadow: Shadow fraction (for ground temperature with shadow effect)
        weather: Weather data (for air temperature and daytime flag)
        state: Thermal state carrying forward surface temperature history

    Returns:
        LupBundle with thermally-delayed upwelling longwave and updated state
    """
    from .buffers import as_float32

    output_state = None

    if state is not None:
        # Compute ground temperature with shadow effect
        tg_temp = (ground_tg * shadow + weather.ta).astype(np.float32)

        # Apply TsWaveDelay for thermal mass effect (batched - 6 calls → 1)
        firstdaytime_int = int(state.firstdaytime)
        result = ground_rust.ts_wave_delay_batch(
            as_float32(gvf_bundle.lup),
            as_float32(gvf_bundle.lup_e),
            as_float32(gvf_bundle.lup_s),
            as_float32(gvf_bundle.lup_w),
            as_float32(gvf_bundle.lup_n),
            tg_temp,
            firstdaytime_int,
            state.timeadd,
            state.timestep_dec,
            as_float32(state.tgmap1),
            as_float32(state.tgmap1_e),
            as_float32(state.tgmap1_s),
            as_float32(state.tgmap1_w),
            as_float32(state.tgmap1_n),
            as_float32(state.tgout1),
        )

        # Extract delayed outputs
        lup = np.asarray(result.lup)
        lup_e = np.asarray(result.lup_e)
        lup_s = np.asarray(result.lup_s)
        lup_w = np.asarray(result.lup_w)
        lup_n = np.asarray(result.lup_n)

        # Build output state from result (copy first, then mutate the copy)
        output_state = state.copy()
        output_state.timeadd = result.timeadd
        output_state.tgmap1 = np.asarray(result.tgmap1)
        output_state.tgmap1_e = np.asarray(result.tgmap1_e)
        output_state.tgmap1_s = np.asarray(result.tgmap1_s)
        output_state.tgmap1_w = np.asarray(result.tgmap1_w)
        output_state.tgmap1_n = np.asarray(result.tgmap1_n)
        output_state.tgout1 = np.asarray(result.tgout1)

        # Update firstdaytime flag for next timestep
        if weather.is_daytime:
            output_state.firstdaytime = 0.0
        else:
            output_state.firstdaytime = 1.0
            output_state.timeadd = 0.0
    else:
        # Single timestep: use raw GVF values (no thermal delay)
        lup = gvf_bundle.lup
        lup_e = gvf_bundle.lup_e
        lup_s = gvf_bundle.lup_s
        lup_w = gvf_bundle.lup_w
        lup_n = gvf_bundle.lup_n

    return LupBundle(
        lup=lup.astype(np.float32),
        lup_e=lup_e.astype(np.float32),
        lup_s=lup_s.astype(np.float32),
        lup_w=lup_w.astype(np.float32),
        lup_n=lup_n.astype(np.float32),
        state=output_state,
    )


def calculate_core(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    use_anisotropic_sky: bool,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    conifer: bool = False,
    wall_material: str | None = None,
    max_shadow_distance_m: float | None = None,
) -> SolweigResult:
    """
    Core SOLWEIG calculation orchestrating all components.

    .. deprecated::
        This non-fused path is no longer used by the public API.
        Use ``calculate_core_fused()`` instead, which is called by ``calculate()``.
        This function is retained for reference but may diverge from the fused path.

    This is the clean orchestration layer that wires together all extracted
    components in a linear flow:

    1. Early exit for nighttime (sun below horizon)
    2. SVF resolution (from cache/precomputed or fresh computation)
    3. Shadow computation (buildings + vegetation with transmissivity)
    4. Ground temperature model (TgMaps with land cover parameterization)
    5. Ground View Factor (upwelling radiation from surfaces)
    6. Thermal delay (TsWaveDelay for thermal inertia)
    7. Radiation calculation (shortwave + longwave from all directions)
    8. Tmrt calculation (absorbed radiation → mean radiant temperature)

    Args:
        surface: Surface/terrain data (DSM, vegetation, walls, land cover)
        location: Geographic location (latitude, longitude)
        weather: Weather conditions (temperature, humidity, radiation, sun position)
        human: Human parameters (height, posture, absorptivities)
        precomputed: Optional pre-computed data (SVF, shadow matrices)
        use_anisotropic_sky: Use anisotropic (Perez) diffuse sky model
        state: Optional thermal state for time-series (carries forward temperatures)
        physics: Optional physics parameters (vegetation transmissivity, etc.)
        materials: Optional material properties (albedo, emissivity by land cover)
        conifer: Treat vegetation as evergreen conifers (always leaf-on)

    Returns:
        SolweigResult with Tmrt, shadow, radiation components, and updated state

    Reference:
        Lindberg et al. (2008, 2016) - SOLWEIG model description
    """
    # Import here to avoid circular dependency
    from .api import SolweigResult

    # Early exit for nighttime
    if weather.sun_altitude <= 0:
        return _nighttime_result(surface, weather, state, materials)

    # === Daytime calculation ===

    # Extract grid dimensions
    rows, cols = surface.dsm.shape
    pixel_size = surface.pixel_size

    # Get surface properties: albedo, emissivity, and ground temp parameters
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = surface.get_land_cover_properties(materials)

    # Prepare vegetation inputs
    use_veg = surface.cdsm is not None
    cdsm = surface.cdsm if use_veg else None
    tdsm = surface.tdsm if use_veg else None
    # Bush layer: Create empty mask when vegetation present (no bushes assumed)
    # Rust validation requires all three vegetation inputs (cdsm, tdsm, bush) or none
    if use_veg:
        pool = surface.get_buffer_pool()
        bush = pool.get_zeros("bush")
    else:
        bush = None

    # Prepare wall inputs
    has_walls = surface.wall_height is not None and surface.wall_aspect is not None
    wall_ht = surface.wall_height if has_walls else None
    wall_asp = surface.wall_aspect if has_walls else None

    # Maximum casting height above local ground for shadow/SVF computation.
    # Use SurfaceData.max_height for consistency with tiled/SVF paths.
    max_height = surface.max_height

    # Cap shadow reach if max_shadow_distance_m is set
    if max_shadow_distance_m is not None:
        from .tiling import MIN_SUN_ELEVATION_DEG

        height_cap = max_shadow_distance_m * np.tan(np.radians(MIN_SUN_ELEVATION_DEG))
        max_height = min(max_height, height_cap)

    # Step 1: SVF Resolution (sky view factors from cached/precomputed sources)
    svf_bundle, _needs_psi_adjustment = resolve_svf(
        surface=surface,
        precomputed=precomputed,
        dsm=surface.dsm,
        cdsm=cdsm,
        tdsm=tdsm,
        pixel_size=pixel_size,
        use_veg=use_veg,
        max_height=max_height,
    )

    # Step 2: Shadow Computation (with vegetation transmissivity)
    shadow_bundle = compute_shadows(
        weather=weather,
        dsm=surface.dsm,
        pixel_size=pixel_size,
        max_height=max_height,
        use_veg=use_veg,
        physics=physics,
        conifer=conifer,
        cdsm=cdsm,
        tdsm=tdsm,
        bush=bush,
        wall_ht=wall_ht,
        wall_asp_rad=wall_asp * (np.pi / 180.0) if wall_asp is not None else None,
    )

    # Step 3: Ground Temperature Model (TgMaps with land cover parameterization)
    # Resolve wall params: explicit wall_material wins, then materials JSON, then Rust defaults
    tgk_wall = None
    tstart_wall = None
    tmaxlst_wall = None
    if wall_material is not None:
        from .loaders import resolve_wall_params

        tgk_wall, tstart_wall, tmaxlst_wall = resolve_wall_params(wall_material, materials)
    elif materials is not None:
        tgk_wall = getattr(getattr(getattr(materials, "Ts_deg", None), "Value", None), "Walls", None)
        tstart_wall = getattr(getattr(getattr(materials, "Tstart", None), "Value", None), "Walls", None)
        tmaxlst_wall = getattr(getattr(getattr(materials, "TmaxLST", None), "Value", None), "Walls", None)

    ground_bundle = compute_ground_temperature(
        weather=weather,
        location=location,
        alb_grid=alb_grid,
        emis_grid=emis_grid,
        tgk_grid=tgk_grid,
        tstart_grid=tstart_grid,
        tmaxlst_grid=tmaxlst_grid,
        tgk_wall=tgk_wall,
        tstart_wall=tstart_wall,
        tmaxlst_wall=tmaxlst_wall,
    )

    # Step 4: Ground View Factor (upwelling radiation from ground + walls)
    gvf_bundle = compute_gvf(
        surface=surface,
        weather=weather,
        human=human,
        tg=ground_bundle.tg,
        tg_wall=ground_bundle.tg_wall,
        shadow=shadow_bundle.shadow,
        wallsun=shadow_bundle.wallsun,
        alb_grid=ground_bundle.alb_grid,
        emis_grid=ground_bundle.emis_grid,
        svf=svf_bundle.svf,
        pixel_size=pixel_size,
        wall_ht=wall_ht,
        wall_asp=wall_asp,
    )

    # Step 5: Apply Thermal Delay (TsWaveDelay for thermal inertia)
    lup_bundle = _apply_thermal_delay(
        gvf_bundle=gvf_bundle,
        ground_tg=ground_bundle.tg,
        shadow=shadow_bundle.shadow,
        weather=weather,
        state=state,
    )

    # Step 6: Radiation Calculation (complete shortwave + longwave budget)
    radiation_bundle = compute_radiation(
        weather=weather,
        svf_bundle=svf_bundle,
        shadow_bundle=shadow_bundle,
        gvf_bundle=gvf_bundle,
        lup_bundle=lup_bundle,
        human=human,
        use_anisotropic_sky=use_anisotropic_sky,
        precomputed=precomputed,
        albedo_wall=0.20,  # Default cobblestone
        emis_wall=0.90,  # Default brick/concrete
        tg_wall=ground_bundle.tg_wall,
    )

    # Step 7: Tmrt Calculation (absorbed radiation → mean radiant temperature)
    tmrt = compute_tmrt(
        radiation=radiation_bundle,
        human=human,
        use_anisotropic_sky=use_anisotropic_sky,
    )

    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow_bundle.shadow,
        kdown=radiation_bundle.kdown,
        kup=radiation_bundle.kup,
        ldown=radiation_bundle.ldown,
        lup=lup_bundle.lup,
        utci=None,  # Computed separately via post-processing
        pet=None,  # Computed separately via post-processing
        state=lup_bundle.state,
    )


def calculate_core_fused(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    conifer: bool = False,
    wall_material: str | None = None,
    use_anisotropic_sky: bool = False,
    max_shadow_distance_m: float | None = None,
    return_state_copy: bool = True,
    requested_outputs: set[str] | None = None,
) -> SolweigResult:
    """
    Fused SOLWEIG calculation — single Rust FFI call per daytime timestep.

    Functionally identical to calculate_core() but orchestrates shadows, ground
    temperature, GVF, thermal delay, radiation, and Tmrt entirely within Rust,
    eliminating intermediate numpy allocations and FFI round-trips.

    This is the primary compute path used by ``calculate()``.
    Supports both isotropic and anisotropic (Perez) sky models.

    Args:
        surface: Surface/terrain data (DSM, vegetation, walls, land cover).
        location: Geographic location (latitude, longitude).
        weather: Weather conditions with derived sun position.
        human: Human parameters (height, posture, absorptivities).
        precomputed: Optional pre-computed data (SVF, shadow matrices).
        state: Optional thermal state for time-series (carries forward temperatures).
        physics: Optional physics parameters (vegetation transmissivity, etc.).
        materials: Optional material properties (albedo, emissivity by land cover).
        conifer: Treat vegetation as evergreen conifers (always leaf-on).
        wall_material: Wall material type ("brick", "concrete", "wood", "cobblestone").
        use_anisotropic_sky: Use anisotropic (Perez) diffuse sky model.
        max_shadow_distance_m: Maximum shadow reach in metres.
        return_state_copy: If True, return a deep-copied thermal state.
        requested_outputs: Set of output names to materialize (None = all).

    Returns:
        SolweigResult with Tmrt, shadow, radiation components, and updated state.
    """
    from .api import SolweigResult
    from .buffers import as_float32
    from .components.gvf import detect_building_mask
    from .components.shadows import compute_transmissivity
    from .components.svf_resolution import resolve_svf
    from .models.state import ThermalState
    from .physics.clearnessindex_2013b import clearnessindex_2013b
    from .physics.daylen import daylen
    from .physics.diffusefraction import diffusefraction
    from .rustalgos import pipeline

    # Ensure derived weather fields are computed (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Early exit for nighttime
    if weather.sun_altitude <= 0:
        return _nighttime_result(surface, weather, state, materials, copy_state=return_state_copy)

    # === Precompute (stays in Python) ===

    rows, cols = surface.dsm.shape
    pixel_size = surface.pixel_size

    # Valid pixel mask (True where all layers have finite data)
    # Computed once by SurfaceData.prepare(), or derived from DSM on-the-fly
    valid_mask = surface.valid_mask
    valid_source = valid_mask if valid_mask is not None else surface.dsm
    valid_mask_key = (id(valid_source), valid_source.shape)
    valid_mask_cache = getattr(surface, "_valid_mask_u8_cache", None)
    if valid_mask_cache is not None and valid_mask_cache[0] == valid_mask_key:
        valid_mask_u8 = valid_mask_cache[1]
    else:
        if valid_mask is None:
            valid_mask = np.isfinite(surface.dsm)
        valid_mask_u8 = np.ascontiguousarray(valid_mask, dtype=np.uint8)
        surface._valid_mask_u8_cache = valid_mask_key, valid_mask_u8

    # Valid-bounds crop (ported from old main implementation):
    # trim heavy per-timestep compute to the minimal bounding rectangle of valid pixels.
    bbox_cache = getattr(surface, "_valid_bbox_cache", None)
    if bbox_cache is not None and bbox_cache[0] == valid_mask_key:
        r0, r1, c0, c1 = bbox_cache[1]
    else:
        rows_any = np.any(valid_mask_u8 != 0, axis=1)
        cols_any = np.any(valid_mask_u8 != 0, axis=0)
        if not rows_any.any() or not cols_any.any():
            r0, r1, c0, c1 = 0, rows, 0, cols
        else:
            r_idx = np.flatnonzero(rows_any)
            c_idx = np.flatnonzero(cols_any)
            r0, r1 = int(r_idx[0]), int(r_idx[-1]) + 1
            c0, c1 = int(c_idx[0]), int(c_idx[-1]) + 1
        surface._valid_bbox_cache = valid_mask_key, (r0, r1, c0, c1)

    full_area = rows * cols
    crop_area = (r1 - r0) * (c1 - c0)
    use_crop = (r0, r1, c0, c1) != (0, rows, 0, cols) and crop_area < int(full_area * 0.98)
    crop_slice = (slice(r0, r1), slice(c0, c1))

    # Select which non-Tmrt outputs to materialize from Rust.
    if requested_outputs is None:
        output_mask = _OUT_ALL
    else:
        output_mask = 0
        if "shadow" in requested_outputs:
            output_mask |= _OUT_SHADOW
        if "kdown" in requested_outputs:
            output_mask |= _OUT_KDOWN
        if "kup" in requested_outputs:
            output_mask |= _OUT_KUP
        if "ldown" in requested_outputs:
            output_mask |= _OUT_LDOWN
        if "lup" in requested_outputs:
            output_mask |= _OUT_LUP

    # Land cover properties
    lc_props_key = (id(surface.land_cover), id(surface.albedo), id(surface.emissivity), id(materials))
    lc_props_cache = getattr(surface, "_land_cover_props_cache", None)
    if lc_props_cache is not None and lc_props_cache[0] == lc_props_key:
        alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = lc_props_cache[1]
    else:
        alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = surface.get_land_cover_properties(materials)
        surface._land_cover_props_cache = lc_props_key, (alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid)

    # Vegetation inputs
    use_veg = surface.cdsm is not None
    cdsm = surface.cdsm if use_veg else None
    tdsm = surface.tdsm if use_veg else None
    if use_veg:
        pool = surface.get_buffer_pool()
        bush = pool.get_zeros("bush")
    else:
        bush = None

    # Wall inputs
    has_walls = surface.wall_height is not None and surface.wall_aspect is not None
    wall_ht = surface.wall_height if has_walls else None
    wall_asp = surface.wall_aspect if has_walls else None

    # Keep height semantics aligned with daytime path and tiled/SVF flows.
    max_height = surface.max_height

    # Cap shadow reach if max_shadow_distance_m is set
    if max_shadow_distance_m is not None:
        from .tiling import MIN_SUN_ELEVATION_DEG

        height_cap = max_shadow_distance_m * np.tan(np.radians(MIN_SUN_ELEVATION_DEG))
        max_height = min(max_height, height_cap)

    # SVF resolution (cached between timesteps)
    svf_bundle, _needs_psi_adjustment = resolve_svf(
        surface=surface,
        precomputed=precomputed,
        dsm=surface.dsm,
        cdsm=cdsm,
        tdsm=tdsm,
        pixel_size=pixel_size,
        use_veg=use_veg,
        max_height=max_height,
    )

    # Vegetation transmissivity
    doy = weather.datetime.timetuple().tm_yday
    psi = compute_transmissivity(doy, physics, conifer)

    # Adjust svfbuveg for vegetation transmissivity (shortwave sees through canopy)
    # Without this, isotropic diffuse (drad), Kup, and Kdown treat vegetation as
    # fully opaque.  The anisotropic path already applies psi per sky patch via
    # diffsh(psi), and kside_veg applies psi per direction, but the scalar svfbuveg
    # used for isotropic diffuse and wall reflection was unadjusted.
    from .components.svf_resolution import adjust_svfbuveg_with_psi

    svf_bundle.svfbuveg = adjust_svfbuveg_with_psi(svf_bundle.svf, svf_bundle.svf_veg, psi, use_veg)

    # Wall material resolution
    tgk_wall = 0.37
    tstart_wall = -3.41
    tmaxlst_wall = 15.0
    albedo_wall = 0.20
    emis_wall = 0.90
    if wall_material is not None:
        from .loaders import resolve_wall_params

        tgk_wall, tstart_wall, tmaxlst_wall = resolve_wall_params(wall_material, materials)
    elif materials is not None:
        tgk_w = getattr(getattr(getattr(materials, "Ts_deg", None), "Value", None), "Walls", None)
        tstart_w = getattr(getattr(getattr(materials, "Tstart", None), "Value", None), "Walls", None)
        tmaxlst_w = getattr(getattr(getattr(materials, "TmaxLST", None), "Value", None), "Walls", None)
        if tgk_w is not None:
            tgk_wall = tgk_w
        if tstart_w is not None:
            tstart_wall = tstart_w
        if tmaxlst_w is not None:
            tmaxlst_wall = tmaxlst_w

    # Weather-derived scalars for ground temperature model
    _, _, _, snup = daylen(doy, location.latitude)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0
    zen_deg = 90.0 - weather.sun_altitude

    # Clear-sky radiation for ground temperature CI correction
    zen_rad = zen_deg * (np.pi / 180.0)
    location_dict = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "altitude": 0.0,
    }
    i0, _, _, _, _ = clearnessindex_2013b(
        zen_rad,
        doy,
        weather.ta,
        weather.rh / 100.0,
        weather.global_rad,
        location_dict,
        -999.0,
    )
    if i0 > 0 and weather.sun_altitude > 0:
        rad_i0, rad_d0 = diffusefraction(i0, weather.sun_altitude, 1.0, weather.ta, weather.rh)
        rad_g0 = rad_i0 * np.sin(weather.sun_altitude * np.pi / 180.0) + rad_d0
    else:
        rad_g0 = 0.0

    # === Build Rust input structs ===

    ws = pipeline.WeatherScalars(
        sun_azimuth=float(weather.sun_azimuth),
        sun_altitude=float(weather.sun_altitude),
        sun_zenith=float(weather.sun_zenith),
        ta=float(weather.ta),
        rh=float(weather.rh),
        global_rad=float(weather.global_rad),
        direct_rad=float(weather.direct_rad),
        diffuse_rad=float(weather.diffuse_rad),
        altmax=float(weather.altmax),
        clearness_index=float(weather.clearness_index),
        dectime=float(dectime),
        snup=float(snup),
        rad_g0=float(rad_g0),
        zen_deg=float(zen_deg),
        psi=float(psi),
        is_daytime=weather.sun_altitude > 0,
        jday=int(weather.datetime.timetuple().tm_yday) if weather.datetime is not None else 180,
        patch_option=0,  # Set below if anisotropic
    )

    hs = pipeline.HumanScalars(
        height=float(human.height),
        abs_k=float(human.abs_k),
        abs_l=float(human.abs_l),
        is_standing=human.posture == "standing",
    )

    cs = pipeline.ConfigScalars(
        pixel_size=float(pixel_size),
        max_height=float(max_height),
        albedo_wall=float(albedo_wall),
        emis_wall=float(emis_wall),
        tgk_wall=float(tgk_wall),
        tstart_wall=float(tstart_wall),
        tmaxlst_wall=float(tmaxlst_wall),
        use_veg=use_veg,
        has_walls=has_walls,
        conifer=conifer,
        use_anisotropic=use_anisotropic_sky,
    )

    # Buildings mask for GVF (computed from DSM/land_cover/walls)
    buildings_key = (id(surface.dsm), id(surface.land_cover), id(wall_ht), float(pixel_size))
    buildings_cache = getattr(surface, "_buildings_mask_cache", None)
    if buildings_cache is not None and buildings_cache[0] == buildings_key:
        buildings = buildings_cache[1]
    else:
        buildings = detect_building_mask(
            surface.dsm,
            surface.land_cover,
            wall_ht,
            pixel_size,
        )
        surface._buildings_mask_cache = buildings_key, buildings

    if surface.land_cover is not None:
        lc_grid_key = id(surface.land_cover)
        lc_grid_cache = getattr(surface, "_lc_grid_f32_cache", None)
        if lc_grid_cache is not None and lc_grid_cache[0] == lc_grid_key:
            lc_grid = lc_grid_cache[1]
        else:
            lc_grid = surface.land_cover.astype(np.float32)
            surface._lc_grid_f32_cache = lc_grid_key, lc_grid
    else:
        lc_grid = None

    # GVF geometry cache: precompute on first daytime call, reuse on subsequent.
    # Keep separate caches for full-grid and cropped-grid execution.
    gvf_cache = None
    if has_walls:
        assert wall_asp is not None  # guaranteed by has_walls
        assert wall_ht is not None
        if use_crop:
            gvf_crop_key = (
                id(buildings),
                id(wall_asp),
                id(wall_ht),
                id(alb_grid),
                r0,
                r1,
                c0,
                c1,
                float(pixel_size),
                float(human.height),
                float(albedo_wall),
            )
            gvf_crop_cache = getattr(surface, "_gvf_geometry_cache_crop", None)
            if gvf_crop_cache is not None and gvf_crop_cache[0] == gvf_crop_key:
                gvf_cache = gvf_crop_cache[1]
            else:
                gvf_cache = pipeline.precompute_gvf_cache(
                    as_float32(buildings[crop_slice]),
                    as_float32(wall_asp[crop_slice]),
                    as_float32(wall_ht[crop_slice]),
                    as_float32(alb_grid[crop_slice]),
                    float(pixel_size),
                    float(human.height),
                    float(albedo_wall),
                )
                surface._gvf_geometry_cache_crop = gvf_crop_key, gvf_cache
        else:
            gvf_cache = getattr(surface, "_gvf_geometry_cache", None)
            if gvf_cache is None:
                gvf_cache = pipeline.precompute_gvf_cache(
                    as_float32(buildings),
                    as_float32(wall_asp),
                    as_float32(wall_ht),
                    as_float32(alb_grid),
                    float(pixel_size),
                    float(human.height),
                    float(albedo_wall),
                )
                surface._gvf_geometry_cache = gvf_cache

    # Anisotropic sky: Perez luminance, steradians, ASVF, and esky are now
    # computed inside the Rust pipeline (no Python round-trip). We only need
    # the shadow matrices and the patch_option.
    aniso_shmat = None
    aniso_vegshmat = None
    aniso_vbshmat = None

    if use_anisotropic_sky:
        shadow_mats = None
        if precomputed is not None and precomputed.shadow_matrices is not None:
            shadow_mats = precomputed.shadow_matrices
        elif surface.shadow_matrices is not None:
            shadow_mats = surface.shadow_matrices

        if shadow_mats is not None:
            ws.patch_option = shadow_mats.patch_option
            if use_crop:
                aniso_crop_key = (
                    id(shadow_mats._shmat_u8),
                    id(shadow_mats._vegshmat_u8),
                    id(shadow_mats._vbshmat_u8),
                    r0,
                    r1,
                    c0,
                    c1,
                )
                aniso_crop_cache = getattr(surface, "_aniso_shadow_crop_cache", None)
                if aniso_crop_cache is not None and aniso_crop_cache[0] == aniso_crop_key:
                    aniso_shmat, aniso_vegshmat, aniso_vbshmat = aniso_crop_cache[1]
                else:
                    aniso_shmat = np.ascontiguousarray(shadow_mats._shmat_u8[crop_slice])
                    aniso_vegshmat = np.ascontiguousarray(shadow_mats._vegshmat_u8[crop_slice])
                    aniso_vbshmat = np.ascontiguousarray(shadow_mats._vbshmat_u8[crop_slice])
                    surface._aniso_shadow_crop_cache = aniso_crop_key, (aniso_shmat, aniso_vegshmat, aniso_vbshmat)
            else:
                # Keep original arrays to preserve stable pointers across timesteps.
                aniso_shmat = shadow_mats._shmat_u8
                aniso_vegshmat = shadow_mats._vegshmat_u8
                aniso_vbshmat = shadow_mats._vbshmat_u8

    # Thermal state (create initial if None)
    if state is None:
        state = ThermalState.initial((rows, cols))

    firstdaytime_int = int(state.firstdaytime)

    def _sel(arr):
        if arr is None:
            return None
        return arr[crop_slice] if use_crop else arr

    dsm_call = _sel(surface.dsm)
    cdsm_call = _sel(cdsm)
    tdsm_call = _sel(tdsm)
    bush_call = _sel(bush)
    wall_ht_call = _sel(wall_ht)
    wall_asp_call = _sel(wall_asp)
    svf_call = _sel(svf_bundle.svf)
    svf_n_call = _sel(svf_bundle.svf_directional.north)
    svf_e_call = _sel(svf_bundle.svf_directional.east)
    svf_s_call = _sel(svf_bundle.svf_directional.south)
    svf_w_call = _sel(svf_bundle.svf_directional.west)
    svf_veg_call = _sel(svf_bundle.svf_veg)
    svf_veg_n_call = _sel(svf_bundle.svf_veg_directional.north)
    svf_veg_e_call = _sel(svf_bundle.svf_veg_directional.east)
    svf_veg_s_call = _sel(svf_bundle.svf_veg_directional.south)
    svf_veg_w_call = _sel(svf_bundle.svf_veg_directional.west)
    svf_aveg_call = _sel(svf_bundle.svf_aveg)
    svf_aveg_n_call = _sel(svf_bundle.svf_aveg_directional.north)
    svf_aveg_e_call = _sel(svf_bundle.svf_aveg_directional.east)
    svf_aveg_s_call = _sel(svf_bundle.svf_aveg_directional.south)
    svf_aveg_w_call = _sel(svf_bundle.svf_aveg_directional.west)
    svfbuveg_call = _sel(svf_bundle.svfbuveg)
    svfalfa_call = _sel(svf_bundle.svfalfa)
    alb_call = _sel(alb_grid)
    emis_call = _sel(emis_grid)
    tgk_call = _sel(tgk_grid)
    tstart_call = _sel(tstart_grid)
    tmaxlst_call = _sel(tmaxlst_grid)
    buildings_call = _sel(buildings)
    lc_grid_call = _sel(lc_grid)
    valid_mask_call = _sel(valid_mask_u8)
    tgmap1_call = _sel(state.tgmap1)
    tgmap1_e_call = _sel(state.tgmap1_e)
    tgmap1_s_call = _sel(state.tgmap1_s)
    tgmap1_w_call = _sel(state.tgmap1_w)
    tgmap1_n_call = _sel(state.tgmap1_n)
    tgout1_call = _sel(state.tgout1)

    # === Call fused Rust pipeline ===

    result = pipeline.compute_timestep(
        # Scalar structs
        ws,
        hs,
        cs,
        # GVF geometry cache (None on first call triggers full GVF, then cached)
        gvf_cache,
        # Surface arrays
        as_float32(dsm_call),
        as_float32(cdsm_call) if cdsm_call is not None else None,
        as_float32(tdsm_call) if tdsm_call is not None else None,
        as_float32(bush_call) if bush_call is not None else None,
        as_float32(wall_ht_call) if wall_ht_call is not None else None,
        as_float32(wall_asp_call) if wall_asp_call is not None else None,
        # SVF arrays
        as_float32(svf_call),
        as_float32(svf_n_call),
        as_float32(svf_e_call),
        as_float32(svf_s_call),
        as_float32(svf_w_call),
        as_float32(svf_veg_call),
        as_float32(svf_veg_n_call),
        as_float32(svf_veg_e_call),
        as_float32(svf_veg_s_call),
        as_float32(svf_veg_w_call),
        as_float32(svf_aveg_call),
        as_float32(svf_aveg_n_call),
        as_float32(svf_aveg_e_call),
        as_float32(svf_aveg_s_call),
        as_float32(svf_aveg_w_call),
        as_float32(svfbuveg_call),
        as_float32(svfalfa_call),
        # Land cover property grids
        as_float32(alb_call),
        as_float32(emis_call),
        as_float32(tgk_call),
        as_float32(tstart_call),
        as_float32(tmaxlst_call),
        # Buildings mask + land cover
        as_float32(buildings_call),
        as_float32(lc_grid_call) if lc_grid_call is not None else None,
        # Anisotropic sky inputs (None for isotropic; Perez computed in Rust)
        aniso_shmat,
        aniso_vegshmat,
        aniso_vbshmat,
        # Thermal state
        firstdaytime_int,
        float(state.timeadd),
        float(state.timestep_dec),
        as_float32(tgmap1_call),
        as_float32(tgmap1_e_call),
        as_float32(tgmap1_s_call),
        as_float32(tgmap1_w_call),
        as_float32(tgmap1_n_call),
        as_float32(tgout1_call),
        # Valid pixel mask for early NaN exit
        valid_mask_call,
        output_mask,
    )

    # === Unpack result and update thermal state ===

    state.timeadd = result.timeadd
    if use_crop:
        state.tgmap1[crop_slice] = np.asarray(result.tgmap1)
        state.tgmap1_e[crop_slice] = np.asarray(result.tgmap1_e)
        state.tgmap1_s[crop_slice] = np.asarray(result.tgmap1_s)
        state.tgmap1_w[crop_slice] = np.asarray(result.tgmap1_w)
        state.tgmap1_n[crop_slice] = np.asarray(result.tgmap1_n)
        state.tgout1[crop_slice] = np.asarray(result.tgout1)
    else:
        state.tgmap1 = np.asarray(result.tgmap1)
        state.tgmap1_e = np.asarray(result.tgmap1_e)
        state.tgmap1_s = np.asarray(result.tgmap1_s)
        state.tgmap1_w = np.asarray(result.tgmap1_w)
        state.tgmap1_n = np.asarray(result.tgmap1_n)
        state.tgout1 = np.asarray(result.tgout1)

    if weather.is_daytime:
        state.firstdaytime = 0.0
    else:
        state.firstdaytime = 1.0
        state.timeadd = 0.0

    output_state = state.copy() if return_state_copy else state

    tmrt = np.asarray(result.tmrt)
    shadow = np.asarray(result.shadow) if result.shadow is not None else None
    kdown = np.asarray(result.kdown) if result.kdown is not None else None
    kup = np.asarray(result.kup) if result.kup is not None else None
    ldown = np.asarray(result.ldown) if result.ldown is not None else None
    lup = np.asarray(result.lup) if result.lup is not None else None

    if use_crop:

        def _uncrop(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            full = np.full((rows, cols), np.nan, dtype=np.float32)
            full[crop_slice] = arr
            return full

        tmrt = _uncrop(tmrt)
        shadow = _uncrop(shadow)
        kdown = _uncrop(kdown)
        kup = _uncrop(kup)
        ldown = _uncrop(ldown)
        lup = _uncrop(lup)

    assert tmrt is not None  # tmrt is always computed
    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow,
        kdown=kdown,
        kup=kup,
        ldown=ldown,
        lup=lup,
        utci=None,
        pet=None,
        state=output_state,
    )
