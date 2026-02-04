"""Weather and location data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..algorithms import sun_position as sp
from ..algorithms.clearnessindex_2013b import clearnessindex_2013b
from ..algorithms.diffusefraction import diffusefraction
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from .surface import SurfaceData

logger = get_logger(__name__)


@dataclass
class Location:
    """
    Geographic location for sun position calculations.

    Attributes:
        latitude: Latitude in degrees (north positive).
        longitude: Longitude in degrees (east positive).
        altitude: Altitude above sea level in meters. Default 0.
        utc_offset: UTC offset in hours. Default 0.
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    utc_offset: int = 0

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")

    @classmethod
    def from_dsm_crs(cls, dsm_path: str | Path, utc_offset: int = 0, altitude: float = 0.0) -> Location:
        """
        Extract location from DSM raster's CRS by converting center point to WGS84.

        Args:
            dsm_path: Path to DSM GeoTIFF file with valid CRS.
            utc_offset: UTC offset in hours. Must be provided by user.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If DSM has no CRS or CRS conversion fails.

        Example:
            location = Location.from_dsm_crs("dsm.tif", utc_offset=2)
        """
        from .. import io

        try:
            from pyproj import Transformer
        except ImportError as err:
            raise ImportError("pyproj is required for CRS extraction. Install with: pip install pyproj") from err

        # Load DSM to get CRS and bounds
        _, transform, crs_wkt, _ = io.load_raster(str(dsm_path))

        if not crs_wkt:
            raise ValueError(
                f"DSM has no CRS metadata: {dsm_path}\n"
                f"Either:\n"
                f"  1. Add CRS to GeoTIFF: gdal_edit.py -a_srs EPSG:XXXXX {dsm_path}\n"
                f"  2. Provide location manually: Location(latitude=X, longitude=Y, utc_offset={utc_offset})"
            )

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        # We need the raster dimensions to find center - load again to get shape
        dsm_array, _, _, _ = io.load_raster(str(dsm_path))
        rows, cols = dsm_array.shape

        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)

        logger.info(f"Extracted location from DSM CRS: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+d})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    @classmethod
    def from_surface(cls, surface: SurfaceData, utc_offset: int | None = None, altitude: float = 0.0) -> Location:
        """
        Extract location from SurfaceData's CRS by converting center point to WGS84.

        This avoids reloading the DSM raster when you already have loaded SurfaceData.

        Args:
            surface: SurfaceData instance loaded from GeoTIFF.
            utc_offset: UTC offset in hours. If not provided, defaults to 0 with a warning.
                Always provide this explicitly for correct sun position calculations.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If surface has no CRS metadata.
            ImportError: If pyproj is not installed.

        Example:
            surface = SurfaceData.from_geotiff("dsm.tif")
            location = Location.from_surface(surface, utc_offset=2)  # Athens: UTC+2
        """
        import warnings

        try:
            from pyproj import Transformer
        except ImportError as err:
            raise ImportError("pyproj is required for CRS extraction. Install with: pip install pyproj") from err

        # Check if geotransform and CRS are available
        if not hasattr(surface, "_geotransform") or surface._geotransform is None:
            raise ValueError(
                "Surface data has no geotransform metadata.\n"
                "Load surface with SurfaceData.from_geotiff() or provide location manually."
            )
        if not hasattr(surface, "_crs_wkt") or surface._crs_wkt is None:
            raise ValueError(
                "Surface data has no CRS metadata.\n"
                "Provide location manually: Location(latitude=X, longitude=Y, utc_offset=0)"
            )

        transform = surface._geotransform
        crs_wkt = surface._crs_wkt
        rows, cols = surface.dsm.shape

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)

        # Warn if utc_offset not explicitly provided
        if utc_offset is None:
            warnings.warn(
                f"UTC offset not specified for auto-extracted location ({lat:.4f}°N, {lon:.4f}°E).\n"
                f"Defaulting to UTC+0, which may cause incorrect sun positions.\n"
                f"Fix: Location.from_surface(surface, utc_offset=YOUR_OFFSET) or\n"
                f"     Location(latitude={lat:.4f}, longitude={lon:.4f}, utc_offset=YOUR_OFFSET)",
                UserWarning,
                stacklevel=2,
            )
            utc_offset = 0

        logger.debug(f"Auto-extracted location: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+d})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    def to_sun_position_dict(self) -> dict:
        """Convert to dict format expected by sun_position module."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }


@dataclass
class Weather:
    """
    Weather/meteorological data for a single timestep.

    Only basic measurements are required. Derived values (sun position,
    direct/diffuse radiation split) are computed automatically.

    Attributes:
        datetime: Date and time of measurement (end of interval).
        ta: Air temperature in °C.
        rh: Relative humidity in % (0-100).
        global_rad: Global solar radiation in W/m².
        ws: Wind speed in m/s. Default 1.0.
        pressure: Atmospheric pressure in hPa. Default 1013.25.
        timestep_minutes: Data timestep in minutes. Default 60.0.
            Sun position is computed at datetime - timestep/2 to represent
            the center of the measurement interval.
        measured_direct_rad: Optional measured direct beam radiation in W/m².
            If provided with measured_diffuse_rad, these override the computed values.
        measured_diffuse_rad: Optional measured diffuse radiation in W/m².
            If provided with measured_direct_rad, these override the computed values.

    Auto-computed (after calling compute_derived()):
        sun_altitude: Sun altitude angle in degrees.
        sun_azimuth: Sun azimuth angle in degrees.
        direct_rad: Direct beam radiation in W/m² (from measured or computed).
        diffuse_rad: Diffuse radiation in W/m² (from measured or computed).
    """

    datetime: dt
    ta: float
    rh: float
    global_rad: float
    ws: float = 1.0
    pressure: float = 1013.25
    timestep_minutes: float = 60.0  # Timestep in minutes (for half-timestep sun position offset)
    measured_direct_rad: float | None = None  # Optional measured direct beam radiation
    measured_diffuse_rad: float | None = None  # Optional measured diffuse radiation
    precomputed_sun_altitude: float | None = None  # Optional pre-computed sun altitude
    precomputed_sun_azimuth: float | None = None  # Optional pre-computed sun azimuth
    precomputed_altmax: float | None = None  # Optional pre-computed max sun altitude for day

    # Auto-computed values (set by compute_derived)
    sun_altitude: float = field(default=0.0, init=False)
    sun_azimuth: float = field(default=0.0, init=False)
    sun_zenith: float = field(default=90.0, init=False)
    direct_rad: float = field(default=0.0, init=False)
    diffuse_rad: float = field(default=0.0, init=False)
    clearness_index: float = field(default=1.0, init=False)
    altmax: float = field(default=45.0, init=False)  # Maximum sun altitude for the day

    _derived_computed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not 0 <= self.rh <= 100:
            raise ValueError(f"Relative humidity must be in [0, 100], got {self.rh}")
        if self.global_rad < 0:
            raise ValueError(f"Global radiation must be >= 0, got {self.global_rad}")

    def compute_derived(self, location: Location) -> None:
        """
        Compute derived values: sun position and radiation split.

        Must be called before using sun_altitude, sun_azimuth, direct_rad,
        or diffuse_rad.

        Sun position is calculated at the center of the measurement interval
        (datetime - timestep/2), which is standard for meteorological data
        where measurements are averaged over the interval.

        Args:
            location: Geographic location for sun position calculation.
        """
        # Always create location_dict (needed for clearness index calculation)
        location_dict = location.to_sun_position_dict()

        # Use pre-computed sun position if provided, otherwise compute
        if self.precomputed_sun_altitude is not None and self.precomputed_sun_azimuth is not None:
            self.sun_altitude = self.precomputed_sun_altitude
            self.sun_azimuth = self.precomputed_sun_azimuth
            self.sun_zenith = 90.0 - self.sun_altitude
            self.altmax = self.precomputed_altmax if self.precomputed_altmax is not None else self.sun_altitude
        else:
            # Apply half-timestep offset for sun position
            # Meteorological data timestamps typically represent the end of an interval,
            # so we compute sun position at the center of the interval to match SOLWEIG runner
            from datetime import timedelta

            half_timestep = timedelta(minutes=self.timestep_minutes / 2.0)
            sun_time = self.datetime - half_timestep

            # Compute sun position using NREL algorithm
            time_dict = {
                "year": sun_time.year,
                "month": sun_time.month,
                "day": sun_time.day,
                "hour": sun_time.hour,
                "min": sun_time.minute,
                "sec": sun_time.second,
                "UTC": location.utc_offset,
            }
            location_dict = location.to_sun_position_dict()

            sun = sp.sun_position(time_dict, location_dict)

            # Extract scalar values (sun_position may return 0-d arrays)
            zenith = sun["zenith"]
            azimuth = sun["azimuth"]
            self.sun_zenith = float(np.asarray(zenith).flat[0]) if hasattr(zenith, "__iter__") else float(zenith)
            self.sun_azimuth = float(np.asarray(azimuth).flat[0]) if hasattr(azimuth, "__iter__") else float(azimuth)
            self.sun_altitude = 90.0 - self.sun_zenith

            # Use pre-computed altmax if available (avoids expensive 96-iteration loop)
            if self.precomputed_altmax is not None:
                self.altmax = self.precomputed_altmax
            else:
                # Calculate maximum sun altitude for the day (iterate in 15-min intervals)
                # This matches the method in configs.py:EnvironData
                from datetime import timedelta

                ymd = self.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                sunmaximum = -90.0
                fifteen_min = 15.0 / 1440.0  # 15 minutes as fraction of day

                for step in range(96):  # 24 hours * 4 (15-min intervals)
                    step_time = ymd + timedelta(days=step * fifteen_min)
                    time_dict_step = {
                        "year": step_time.year,
                        "month": step_time.month,
                        "day": step_time.day,
                        "hour": step_time.hour,
                        "min": step_time.minute,
                        "sec": 0,
                        "UTC": location.utc_offset,
                    }
                    sun_step = sp.sun_position(time_dict_step, location_dict)
                    zenith_step = sun_step["zenith"]
                    zenith_val = (
                        float(np.asarray(zenith_step).flat[0])
                        if hasattr(zenith_step, "__iter__")
                        else float(zenith_step)
                    )
                    alt_step = 90.0 - zenith_val
                    if alt_step > sunmaximum:
                        sunmaximum = alt_step

                self.altmax = max(sunmaximum, 0.0)  # Ensure non-negative

        # Use measured radiation values if provided, otherwise compute
        if self.measured_direct_rad is not None and self.measured_diffuse_rad is not None:
            # Use pre-measured direct and diffuse radiation
            self.direct_rad = self.measured_direct_rad
            self.diffuse_rad = self.measured_diffuse_rad
            self.clearness_index = 1.0  # Not computed when using measured values
        elif self.sun_altitude > 0 and self.global_rad > 0:
            # Compute clearness index
            zen_rad = self.sun_zenith * (np.pi / 180.0)
            result = clearnessindex_2013b(
                zen_rad,
                self.datetime.timetuple().tm_yday,
                self.ta,
                self.rh / 100.0,
                self.global_rad,
                location_dict,
                self.pressure,
            )
            # clearnessindex_2013b returns: (I0, CI, Kt, I0_et, diff_et)
            _, self.clearness_index, kt, _, _ = result

            # Use Reindl model for diffuse fraction
            self.direct_rad, self.diffuse_rad = diffusefraction(
                self.global_rad, self.sun_altitude, kt, self.ta, self.rh
            )
        else:
            # Night or no radiation
            self.direct_rad = 0.0
            self.diffuse_rad = self.global_rad
            self.clearness_index = 1.0

        self._derived_computed = True

    @property
    def is_daytime(self) -> bool:
        """Check if sun is above horizon."""
        return self.sun_altitude > 0

    @classmethod
    def from_values(
        cls,
        ta: float,
        rh: float,
        global_rad: float,
        datetime: dt | None = None,
        ws: float = 1.0,
        **kwargs,
    ) -> Weather:
        """
        Quick factory for creating Weather with minimal required values.

        Useful for testing and single-timestep calculations where you
        just need to specify the essential parameters.

        Args:
            ta: Air temperature in °C.
            rh: Relative humidity in % (0-100).
            global_rad: Global solar radiation in W/m².
            datetime: Date and time. If None, uses current time.
            ws: Wind speed in m/s. Default 1.0.
            **kwargs: Additional Weather parameters (pressure, etc.)

        Returns:
            Weather object ready for calculation.

        Example:
            # Quick weather for testing
            weather = Weather.from_values(ta=25, rh=50, global_rad=800)

            # With specific datetime
            weather = Weather.from_values(
                ta=30, rh=60, global_rad=900,
                datetime=datetime(2024, 7, 15, 14, 0)
            )
        """
        if datetime is None:
            datetime = dt.now()
        return cls(datetime=datetime, ta=ta, rh=rh, global_rad=global_rad, ws=ws, **kwargs)

    @classmethod
    def from_epw(
        cls,
        path: str | Path,
        start: str | dt | None = None,
        end: str | dt | None = None,
        hours: list[int] | None = None,
        year: int | None = None,
    ) -> list[Weather]:
        """
        Load weather data from an EnergyPlus Weather (EPW) file.

        Args:
            path: Path to the EPW file.
            start: Start date/datetime. Can be:
                   - ISO date string "YYYY-MM-DD" or "MM-DD" (for TMY with year=None)
                   - datetime object
                   If None, uses first date in file.
            end: End date/datetime (inclusive). Same format as start.
                 If None, uses same as start (single day).
            hours: List of hours to include (0-23). If None, includes all hours.
            year: Year override for TMY files. If None and start/end use MM-DD format,
                  matches any year in the file.

        Returns:
            List of Weather objects for each timestep in the requested range.

        Raises:
            FileNotFoundError: If the EPW file doesn't exist.
            ValueError: If requested dates are outside the EPW file's date range.

        Example:
            # Load a single day
            weather_list = Weather.from_epw("weather.epw", start="2023-07-15", end="2023-07-15")

            # Load with specific hours only (daylight hours)
            weather_list = Weather.from_epw(
                "weather.epw",
                start="2023-07-15",
                end="2023-07-16",
                hours=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            )

            # TMY file (year-agnostic)
            weather_list = Weather.from_epw("tmy.epw", start="07-15", end="07-15")
        """
        from .. import io as common

        # Parse EPW file
        df, epw_info = common.read_epw(path)

        # Parse start/end dates
        def parse_date(date_val, is_tmy: bool, default_year: int):
            if date_val is None:
                return None
            if isinstance(date_val, dt):
                return date_val
            # String parsing
            date_str = str(date_val)
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 2:
                    # MM-DD format (TMY)
                    month, day = int(parts[0]), int(parts[1])
                    return dt(default_year, month, day)
                elif len(parts) == 3:
                    # YYYY-MM-DD format
                    return dt.fromisoformat(date_str)
            raise ValueError(f"Cannot parse date: {date_val}. Use 'YYYY-MM-DD' or 'MM-DD' format.")

        # Determine if using TMY mode (year-agnostic)
        is_tmy = year is None and start is not None and isinstance(start, str) and len(start.split("-")) == 2

        # Get default year from EPW data
        if df.index.empty:
            raise ValueError("EPW file contains no data")
        default_year = df.index[0].year if year is None else year

        # Parse dates
        start_dt = parse_date(start, is_tmy, default_year)
        end_dt = parse_date(end, is_tmy, default_year)

        if start_dt is None:
            start_dt = df.index[0].replace(tzinfo=None)
        if end_dt is None:
            end_dt = start_dt

        # Make end_dt inclusive of the full day
        if end_dt.hour == 0 and end_dt.minute == 0:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Filter by date range
        # Remove timezone from index for comparison if needed
        df_idx = df.index.tz_localize(None) if df.index.tz is not None else df.index

        if is_tmy:
            # TMY mode: match month and day, ignore year
            mask = (
                (df_idx.month > start_dt.month) | ((df_idx.month == start_dt.month) & (df_idx.day >= start_dt.day))
            ) & ((df_idx.month < end_dt.month) | ((df_idx.month == end_dt.month) & (df_idx.day <= end_dt.day)))
        else:
            # Normal mode: match full datetime
            mask = (df_idx >= start_dt) & (df_idx <= end_dt)

        df_filtered = df[mask]

        if df_filtered.empty:
            # Build helpful error message
            avail_start = df_idx.min()
            avail_end = df_idx.max()
            raise ValueError(
                f"Requested dates {start_dt.date()} to {end_dt.date()} not found in EPW file.\n"
                f"EPW file '{path}' contains data for: {avail_start.date()} to {avail_end.date()}\n"
                "Suggestions:\n"
                "  - Use dates within the available range\n"
                "  - For TMY files, use 'MM-DD' format (e.g., '07-15') to match any year"
            )

        # Filter by hours if specified
        if hours is not None:
            hours_set = set(hours)
            hour_mask = df_filtered.index.hour.isin(hours_set)
            df_filtered = df_filtered[hour_mask]

        # Create Weather objects
        weather_list = []
        for timestamp, row in df_filtered.iterrows():
            # Create Weather object with available data
            # EPW has dni/dhi which we can use as measured values
            w = cls(
                datetime=timestamp.to_pydatetime().replace(tzinfo=None),
                ta=float(row["temp_air"]) if not np.isnan(row["temp_air"]) else 20.0,
                rh=float(row["relative_humidity"]) if not np.isnan(row["relative_humidity"]) else 50.0,
                global_rad=float(row["ghi"]) if not np.isnan(row["ghi"]) else 0.0,
                ws=float(row["wind_speed"]) if not np.isnan(row["wind_speed"]) else 1.0,
                pressure=(float(row["atmospheric_pressure"]) / 100.0)
                if not np.isnan(row["atmospheric_pressure"])
                else 1013.25,  # Convert Pa to hPa
                measured_direct_rad=float(row["dni"]) if not np.isnan(row["dni"]) else None,
                measured_diffuse_rad=float(row["dhi"]) if not np.isnan(row["dhi"]) else None,
            )
            weather_list.append(w)

        if weather_list:
            logger.info(
                f"Loaded {len(weather_list)} timesteps from EPW: "
                f"{weather_list[0].datetime.strftime('%Y-%m-%d %H:%M')} → "
                f"{weather_list[-1].datetime.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            logger.warning(f"No timesteps found in EPW file for date range {start_dt} to {end_dt}")

        return weather_list
