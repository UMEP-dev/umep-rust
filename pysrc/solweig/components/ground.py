"""
Ground temperature model component.

Implements the SOLWEIG TgMaps ground temperature model with:
- Parameterization from land cover properties
- Diurnal temperature cycle based on sun altitude
- Clearness index correction for cloudy conditions

Reference:
- Lindberg et al. (2008, 2016) - SOLWEIG ground temperature parameterization
- Reindl et al. (1990) - Clearness index approach
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..algorithms.clearnessindex_2013b import clearnessindex_2013b
from ..algorithms.daylen import daylen
from ..algorithms.diffusefraction import diffusefraction
from ..bundles import GroundBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..api import Location, Weather


def compute_ground_temperature(
    weather: Weather,
    location: Location,
    alb_grid: NDArray[np.floating],
    emis_grid: NDArray[np.floating],
    tgk_grid: NDArray[np.floating],
    tstart_grid: NDArray[np.floating],
    tmaxlst_grid: NDArray[np.floating],
) -> GroundBundle:
    """
    Compute ground and wall temperature deviations from air temperature.

    Uses the SOLWEIG TgMaps model with land-cover-specific parameterization.
    Temperature amplitude depends on max sun altitude and land cover type.
    Clearness index correction accounts for reduced heating under cloudy skies.

    Args:
        weather: Weather data including temperature, radiation, sun position
        location: Geographic location (latitude, longitude) for sunrise calculation
        alb_grid: Albedo per pixel (0-1) from land cover properties
        emis_grid: Emissivity per pixel (0-1) from land cover properties
        tgk_grid: TgK parameter per pixel (temperature gain coefficient)
        tstart_grid: Tstart parameter per pixel (temperature baseline offset)
        tmaxlst_grid: TmaxLST parameter per pixel (hour of maximum temperature)

    Returns:
        GroundBundle containing:
            - tg: Ground temperature deviation from air temperature (K)
            - tg_wall: Wall temperature deviation from air temperature (K)
            - ci_tg: Clearness index correction factor (0-1)
            - alb_grid: Albedo grid (passed through for convenience)
            - emis_grid: Emissivity grid (passed through for convenience)

    Reference:
        Lindberg et al. (2008): "Urban Multi-scale Environmental Predictor (UMEP)"
        Formula: Tgamp = TgK * altmax + Tstart
                 Tg = Tgamp * sin(phase * pi/2) * CI_TgG
    """
    # Air temperature in Kelvin
    ta_k = weather.ta + 273.15

    # Day of year and sunrise time
    jday = weather.datetime.timetuple().tm_yday
    _, _, _, snup = daylen(jday, location.latitude)

    # Wall parameters (scalar, use default cobblestone values from parametersforsolweig.json)
    tstart_wall = -3.41  # Wall baseline offset
    tmaxlst_wall = 15.0  # Wall max surface temp hour (hour of day)
    tgk_wall = 0.37  # Wall TgK parameter

    # Maximum sun altitude for the day (computed in Weather.compute_derived())
    altmax = weather.altmax

    # Temperature amplitude based on max sun altitude (per-pixel from land cover)
    # tgk_grid contains TgK values: asphalt ~0.58, grass ~0.21, water ~0.0
    # tstart_grid contains Tstart values: asphalt ~-9.78, grass ~-3.38, water ~0.0
    # Formula: Tgamp = TgK * altmax + Tstart (Lindberg et al. 2008, 2016)
    tgamp = tgk_grid * altmax + tstart_grid

    # Wall temperature amplitude
    tgamp_wall = tgk_wall * altmax + tstart_wall

    # Decimal time (fraction of day)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0

    # Phase calculation matching reference (per-pixel for ground):
    # phase = ((dectime - SNUP/24) / (TmaxLST/24 - SNUP/24))
    # Tg = Tgamp * sin(phase * pi/2)
    snup_frac = snup / 24.0
    tmaxlst_frac = tmaxlst_grid / 24.0  # Per-pixel from land cover
    tmaxlst_wall_frac = tmaxlst_wall / 24.0

    # Per-pixel phase calculation for ground
    # tmaxlst_grid varies by land cover: 15h for paved/asphalt, 14h for grass, 12h for water
    denom = tmaxlst_frac - snup_frac
    denom = np.where(denom > 0, denom, 1.0)  # Avoid division by zero
    phase = (dectime - snup_frac) / denom
    phase = np.clip(phase, 0.0, 1.0)

    # Ground temperature: only positive when after sunrise
    tg = np.where(dectime > snup_frac, tgamp * np.sin(phase * np.pi / 2.0), 0.0)

    # Wall phase (scalar)
    if dectime > snup_frac and tmaxlst_wall_frac > snup_frac:
        phase_wall = (dectime - snup_frac) / (tmaxlst_wall_frac - snup_frac)
        phase_wall = min(max(phase_wall, 0.0), 1.0)
        tg_wall = tgamp_wall * np.sin(phase_wall * np.pi / 2.0)
    else:
        tg_wall = 0.0

    # Clamp negative Tg values (morning transition, can happen with negative Tstart)
    tg = np.maximum(tg, 0.0)
    tg_wall = max(tg_wall, 0.0) if isinstance(tg_wall, (int, float)) else np.maximum(tg_wall, 0.0)

    # CI_TgG correction for non-clear conditions (Lindberg et al. 2008, Reindl et al. 1990)
    # This accounts for reduced ground heating under cloudy skies
    # Full formula from solweig.py: CI_TgG = (radG / radG0) + (1 - corr)
    zen = (90.0 - weather.sun_altitude) * (np.pi / 180.0)  # zenith in radians
    deg2rad = np.pi / 180.0

    # Get clear sky radiation (I0) from clearnessindex function
    location_dict = {"latitude": location.latitude, "longitude": location.longitude, "altitude": 0.0}
    i0, _, _, _, _ = clearnessindex_2013b(
        zen, jday, weather.ta, weather.rh / 100.0, weather.global_rad, location_dict, -999.0
    )

    # Calculate clear sky direct and diffuse components
    if i0 > 0 and weather.sun_altitude > 0:
        rad_i0, rad_d0 = diffusefraction(i0, weather.sun_altitude, 1.0, weather.ta, weather.rh)
        # Clear sky global horizontal radiation
        rad_g0 = rad_i0 * np.sin(weather.sun_altitude * deg2rad) + rad_d0
        # Zenith correction (Lindberg et al. 2008)
        zen_deg = 90.0 - weather.sun_altitude
        if zen_deg > 0 and zen_deg < 90:
            corr = 0.1473 * np.log(90.0 - zen_deg) + 0.3454
        else:
            corr = 0.3454
        # CI_TgG calculation
        if rad_g0 > 0:
            ci_tg = (weather.global_rad / rad_g0) + (1.0 - corr)
            ci_tg = min(ci_tg, 1.0)  # Clamp to max 1
            if np.isinf(ci_tg) or np.isnan(ci_tg):
                ci_tg = 1.0
        else:
            ci_tg = weather.clearness_index
    else:
        ci_tg = weather.clearness_index

    # Apply clearness correction
    tg = tg * ci_tg
    tg_wall = tg_wall * ci_tg

    # Clamp negative values after CI correction
    tg = np.maximum(tg, 0.0)

    return GroundBundle(
        tg=tg.astype(np.float32),
        tg_wall=float(tg_wall),
        ci_tg=float(ci_tg),
        alb_grid=alb_grid,
        emis_grid=emis_grid,
    )
