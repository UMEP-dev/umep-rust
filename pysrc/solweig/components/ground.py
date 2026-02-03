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
    from ..rustalgos import ground as ground_rust

    # Day of year and sunrise time
    jday = weather.datetime.timetuple().tm_yday
    _, _, _, snup = daylen(jday, location.latitude)

    # Maximum sun altitude for the day (computed in Weather.compute_derived())
    altmax = weather.altmax

    # Decimal time (fraction of day)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0

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
    else:
        rad_g0 = 0.0

    # Zenith angle in degrees
    zen_deg = 90.0 - weather.sun_altitude

    # Call Rust implementation (positional args to match Rust signature)
    tg, tg_wall, ci_tg, alb_grid_out, emis_grid_out = ground_rust.compute_ground_temperature(
        weather.ta,
        weather.sun_altitude,
        altmax,
        dectime,
        snup,
        weather.global_rad,
        rad_g0,
        zen_deg,
        alb_grid.astype(np.float32),
        emis_grid.astype(np.float32),
        tgk_grid.astype(np.float32),
        tstart_grid.astype(np.float32),
        tmaxlst_grid.astype(np.float32),
    )

    return GroundBundle(
        tg=tg,
        tg_wall=float(tg_wall),
        ci_tg=float(ci_tg),
        alb_grid=alb_grid_out,
        emis_grid=emis_grid_out,
    )
