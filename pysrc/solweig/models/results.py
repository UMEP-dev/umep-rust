"""Result data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models import HumanParams
    from .state import ThermalState
    from .surface import SurfaceData
    from .weather import Weather

logger = get_logger(__name__)


@dataclass
class SolweigResult:
    """
    Results from a SOLWEIG calculation.

    All output grids have the same shape as the input DSM.

    Attributes:
        tmrt: Mean Radiant Temperature grid (°C).
        utci: Universal Thermal Climate Index grid (°C). Optional.
        pet: Physiological Equivalent Temperature grid (°C). Optional.
        shadow: Shadow mask (1=shadow, 0=sunlit).
        kdown: Downwelling shortwave radiation (W/m²).
        kup: Upwelling shortwave radiation (W/m²).
        ldown: Downwelling longwave radiation (W/m²).
        lup: Upwelling longwave radiation (W/m²).
        state: Thermal state for multi-timestep chaining. Optional.
            When state parameter was passed to calculate(), this contains
            the updated state for the next timestep.
    """

    tmrt: NDArray[np.floating]
    shadow: NDArray[np.floating] | None = None
    kdown: NDArray[np.floating] | None = None
    kup: NDArray[np.floating] | None = None
    ldown: NDArray[np.floating] | None = None
    lup: NDArray[np.floating] | None = None
    utci: NDArray[np.floating] | None = None
    pet: NDArray[np.floating] | None = None
    state: ThermalState | None = None

    def to_geotiff(
        self,
        output_dir: str | Path,
        timestamp: dt | None = None,
        outputs: list[str] | None = None,
        surface: SurfaceData | None = None,
        transform: list[float] | None = None,
        crs_wkt: str | None = None,
    ) -> None:
        """
        Save results to GeoTIFF files.

        Creates one GeoTIFF file per output variable per timestep.
        Filename pattern: {output}_{YYYYMMDD}_{HHMM}.tif

        Args:
            output_dir: Directory to write GeoTIFF files.
            timestamp: Timestamp for filename. If None, uses current time.
            outputs: List of outputs to save. Options: "tmrt", "utci", "pet",
                "shadow", "kdown", "kup", "ldown", "lup".
                Default: ["tmrt"] (only save Mean Radiant Temperature).
            surface: SurfaceData object (if loaded via from_geotiff, contains CRS/transform).
                If provided and transform/crs_wkt not specified, uses surface metadata.
            transform: GDAL-style geotransform [x_origin, pixel_width, 0,
                y_origin, 0, -pixel_height]. If None, attempts to use surface metadata,
                otherwise uses identity transform.
            crs_wkt: Coordinate reference system in WKT format. If None, attempts to use
                surface metadata, otherwise no CRS set.

        Example:
            # With surface metadata (recommended when using from_geotiff)
            >>> surface, precomputed = SurfaceData.from_geotiff("dsm.tif", svf_dir="svf/")
            >>> result = solweig.calculate(surface, location, weather, precomputed=precomputed)
            >>> result.to_geotiff("output/", timestamp=weather.dt, surface=surface)

            # Without surface metadata (explicit transform/CRS)
            >>> result.to_geotiff(
            ...     "output/",
            ...     timestamp=datetime(2023, 7, 15, 12, 0),
            ...     outputs=["tmrt", "utci", "pet"],
            ...     transform=[0, 1, 0, 0, 0, -1],
            ...     crs_wkt="EPSG:32633",
            ... )
        """
        from .. import io

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default outputs: just tmrt
        if outputs is None:
            outputs = ["tmrt"]

        # Default timestamp: current time
        if timestamp is None:
            timestamp = dt.now()

        # Format timestamp for filename
        ts_str = timestamp.strftime("%Y%m%d_%H%M")

        # Use surface metadata if available and not overridden
        if surface is not None:
            if transform is None and surface._geotransform is not None:
                transform = surface._geotransform
            if crs_wkt is None and surface._crs_wkt is not None:
                crs_wkt = surface._crs_wkt

        # Default transform: identity (top-left origin, 1m pixels)
        if transform is None:
            height, width = self.tmrt.shape
            transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

        # Default CRS: empty string (no CRS)
        if crs_wkt is None:
            crs_wkt = ""

        # Map output names to arrays
        available_outputs = {
            "tmrt": self.tmrt,
            "utci": self.utci,
            "pet": self.pet,
            "shadow": self.shadow,
            "kdown": self.kdown,
            "kup": self.kup,
            "ldown": self.ldown,
            "lup": self.lup,
        }

        # Save each requested output
        for name in outputs:
            if name not in available_outputs:
                logger.warning(f"Unknown output '{name}', skipping. Valid: {list(available_outputs.keys())}")
                continue

            array = available_outputs[name]
            if array is None:
                logger.warning(f"Output '{name}' is None (not computed), skipping.")
                continue

            # Write to GeoTIFF in component subdirectory
            comp_dir = output_dir / name
            comp_dir.mkdir(parents=True, exist_ok=True)
            filepath = comp_dir / f"{name}_{ts_str}.tif"
            io.save_raster(
                out_path_str=str(filepath),
                data_arr=array,
                trf_arr=transform,
                crs_wkt=crs_wkt,
                no_data_val=np.nan,
            )
            logger.debug(f"Saved {name} to {filepath}")

    def compute_utci(
        self,
        weather_or_ta: Weather | float,
        rh: float | None = None,
        wind: float | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute UTCI (Universal Thermal Climate Index) from this result's Tmrt.

        Can be called with either a Weather object or individual values:
            utci = result.compute_utci(weather)
            utci = result.compute_utci(ta=25.0, rh=50.0, wind=2.0)

        Args:
            weather_or_ta: Either a Weather object, or air temperature in °C.
            rh: Relative humidity in % (required if weather_or_ta is float).
            wind: Wind speed at 10m height in m/s. Default 1.0 if not provided.

        Returns:
            UTCI grid (°C) with same shape as tmrt.

        Example:
            result = solweig.calculate(surface, location, weather)

            # Pattern A: Pass weather object (convenient)
            utci = result.compute_utci(weather)

            # Pattern B: Pass individual values (explicit)
            utci = result.compute_utci(25.0, rh=50.0, wind=2.0)
        """
        from ..postprocess import compute_utci_grid
        from .weather import Weather as WeatherClass

        # Check if first argument is a Weather object
        if isinstance(weather_or_ta, WeatherClass):
            return compute_utci_grid(self.tmrt, weather_or_ta.ta, weather_or_ta.rh, weather_or_ta.ws)
        else:
            # Individual values
            ta = float(weather_or_ta)
            if rh is None:
                raise ValueError("rh is required when ta is provided as a float")
            return compute_utci_grid(self.tmrt, ta, rh, wind if wind is not None else 1.0)

    def compute_pet(
        self,
        weather_or_ta: Weather | float,
        rh: float | None = None,
        wind: float | None = None,
        human: HumanParams | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute PET (Physiological Equivalent Temperature) from this result's Tmrt.

        Can be called with either a Weather object or individual values:
            pet = result.compute_pet(weather)
            pet = result.compute_pet(ta=25.0, rh=50.0, wind=2.0)

        Args:
            weather_or_ta: Either a Weather object, or air temperature in °C.
            rh: Relative humidity in % (required if weather_or_ta is float).
            wind: Wind speed at 10m height in m/s. Default 1.0 if not provided.
            human: Human body parameters. Uses defaults if not provided.

        Returns:
            PET grid (°C) with same shape as tmrt.

        Note:
            PET uses an iterative solver and is ~50× slower than UTCI.

        Example:
            result = solweig.calculate(surface, location, weather)

            # Pattern A: Pass weather object (convenient)
            pet = result.compute_pet(weather)

            # Pattern B: Pass individual values with custom human params
            pet = result.compute_pet(
                25.0, rh=50.0, wind=2.0,
                human=HumanParams(weight=70, height=1.65)
            )
        """
        from ..postprocess import compute_pet_grid
        from .weather import Weather as WeatherClass

        # Check if first argument is a Weather object
        if isinstance(weather_or_ta, WeatherClass):
            return compute_pet_grid(self.tmrt, weather_or_ta.ta, weather_or_ta.rh, weather_or_ta.ws, human)
        else:
            # Individual values
            ta = float(weather_or_ta)
            if rh is None:
                raise ValueError("rh is required when ta is provided as a float")
            return compute_pet_grid(self.tmrt, ta, rh, wind if wind is not None else 1.0, human)


@dataclass
class TileSpec:
    """
    Specification for a single tile with overlap regions.

    Attributes:
        row_start, row_end: Core tile row bounds (without overlap).
        col_start, col_end: Core tile column bounds (without overlap).
        row_start_full, row_end_full: Full tile row bounds (with overlap).
        col_start_full, col_end_full: Full tile column bounds (with overlap).
        overlap_top, overlap_bottom: Vertical overlap in pixels.
        overlap_left, overlap_right: Horizontal overlap in pixels.
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_start_full: int
    row_end_full: int
    col_start_full: int
    col_end_full: int
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int

    @property
    def core_shape(self) -> tuple[int, int]:
        """Shape of core tile (without overlap)."""
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def full_shape(self) -> tuple[int, int]:
        """Shape of full tile (with overlap)."""
        return (self.row_end_full - self.row_start_full, self.col_end_full - self.col_start_full)

    @property
    def core_slice(self) -> tuple[slice, slice]:
        """Slices for extracting core from full tile result."""
        return (
            slice(self.overlap_top, self.overlap_top + self.core_shape[0]),
            slice(self.overlap_left, self.overlap_left + self.core_shape[1]),
        )

    @property
    def write_slice(self) -> tuple[slice, slice]:
        """Slices for writing core to global output."""
        return (
            slice(self.row_start, self.row_end),
            slice(self.col_start, self.col_end),
        )

    @property
    def read_slice(self) -> tuple[slice, slice]:
        """Slices for reading full tile from global input."""
        return (
            slice(self.row_start_full, self.row_end_full),
            slice(self.col_start_full, self.col_end_full),
        )
