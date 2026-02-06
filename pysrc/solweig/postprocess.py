"""Post-processing: UTCI and PET thermal comfort indices."""

from __future__ import annotations

import logging
import time
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import HumanParams
from .rustalgos import pet as pet_rust
from .rustalgos import utci as utci_rust

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models import Location, Weather


# =============================================================================
# Post-Processing: Thermal Comfort Indices
# =============================================================================


def compute_utci_grid(
    tmrt: NDArray[np.floating],
    ta: float,
    rh: float,
    wind: float,
) -> NDArray[np.floating]:
    """
    Compute UTCI (Universal Thermal Climate Index) for a single grid.

    This is a thin wrapper around the Rust UTCI implementation for in-memory processing.
    For batch processing of saved Tmrt files, use compute_utci() instead.

    Args:
        tmrt: Mean Radiant Temperature grid (°C).
        ta: Air temperature (°C).
        rh: Relative humidity (%).
        wind: Wind speed at 10m height (m/s).

    Returns:
        UTCI grid (°C).

    Example:
        # Compute UTCI for a single result
        utci = compute_utci_grid(
            tmrt=result.tmrt,
            ta=25.0,
            rh=60.0,
            wind=2.0,
        )
    """

    wind_grid = np.full_like(tmrt, wind, dtype=np.float32)
    return utci_rust.utci_grid(ta, rh, tmrt, wind_grid)


def compute_pet_grid(
    tmrt: NDArray[np.floating],
    ta: float,
    rh: float,
    wind: float,
    human: HumanParams | None = None,
) -> NDArray[np.floating]:
    """
    Compute PET (Physiological Equivalent Temperature) for a single grid.

    This is a thin wrapper around the Rust PET implementation for in-memory processing.
    For batch processing of saved Tmrt files, use compute_pet() instead.

    Args:
        tmrt: Mean Radiant Temperature grid (°C).
        ta: Air temperature (°C).
        rh: Relative humidity (%).
        wind: Wind speed at 10m height (m/s).
        human: Human body parameters. Uses defaults if not provided.

    Returns:
        PET grid (°C).

    Example:
        # Compute PET for a single result
        pet = compute_pet_grid(
            tmrt=result.tmrt,
            ta=25.0,
            rh=60.0,
            wind=2.0,
            human=HumanParams(weight=75, height=1.75),
        )
    """

    if human is None:
        human = HumanParams()

    wind_grid = np.full_like(tmrt, wind, dtype=np.float32)
    return pet_rust.pet_grid(
        ta,
        rh,
        tmrt,
        wind_grid,
        human.weight,
        float(human.age),
        human.height,
        human.activity,
        human.clothing,
        human.sex,
    )


def compute_utci(
    tmrt_dir: str | Path,
    weather_series: list[Weather],
    output_dir: str | Path,
    location: Location | None = None,
) -> int:
    """
    Batch compute UTCI from saved Tmrt GeoTIFF files.

    Auto-discovers tmrt_*.tif files in tmrt_dir, matches them with weather_series
    by datetime, and saves utci_*.tif files to output_dir.

    Args:
        tmrt_dir: Directory containing tmrt_YYYYMMDD_HHMM.tif files.
        weather_series: List of Weather objects with datetime, ta, rh, ws.
        output_dir: Directory to save utci_YYYYMMDD_HHMM.tif files.
        location: Geographic location for weather.compute_derived().
            If None, assumes weather is already computed.

    Returns:
        Number of UTCI files processed.

    Example:
        # After running calculate_timeseries with output_dir
        n_processed = solweig.compute_utci(
            tmrt_dir="output/",
            weather_series=weather_list,
            output_dir="output_utci/",
        )
        print(f"Processed {n_processed} timesteps")
    """
    import re

    from . import io

    tmrt_dir = Path(tmrt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find tmrt_*.tif files (check tmrt/ subdirectory first, then flat layout)
    tmrt_files = sorted(tmrt_dir.glob("tmrt_*.tif"))
    if not tmrt_files and (tmrt_dir / "tmrt").exists():
        tmrt_files = sorted((tmrt_dir / "tmrt").glob("tmrt_*.tif"))
    if not tmrt_files:
        logger.warning(f"No tmrt_*.tif files found in {tmrt_dir}")
        return 0

    # Parse timestamps from filenames
    pattern = re.compile(r"tmrt_(\d{8})_(\d{4})\.tif")
    tmrt_map = {}
    for f in tmrt_files:
        match = pattern.match(f.name)
        if match:
            date_str, time_str = match.groups()
            timestamp = dt.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
            tmrt_map[timestamp] = f

    # Match weather with timestamps
    if location is not None:
        for w in weather_series:
            if not w._derived_computed:
                w.compute_derived(location)

    # Start timing
    start_time = time.time()
    processed = 0
    for weather in weather_series:
        if weather.datetime not in tmrt_map:
            logger.warning(f"No Tmrt file found for {weather.datetime}")
            continue

        # Load Tmrt
        tmrt_path = tmrt_map[weather.datetime]
        tmrt, transform, crs, _ = io.load_raster(str(tmrt_path))

        # Compute UTCI
        utci = compute_utci_grid(tmrt, weather.ta, weather.rh, weather.ws)

        # Save UTCI
        date_str = weather.datetime.strftime("%Y%m%d")
        time_str = weather.datetime.strftime("%H%M")
        utci_path = output_dir / f"utci_{date_str}_{time_str}.tif"

        io.save_raster(
            str(utci_path),
            utci,
            transform if isinstance(transform, list) else list(transform.to_gdal()),
            crs,
        )
        processed += 1

        if processed % 10 == 0:
            logger.info(f"Processed {processed}/{len(weather_series)} timesteps")

    total_time = time.time() - start_time
    rate = processed / total_time if total_time > 0 else 0
    logger.info(
        f"✓ UTCI computation complete: {processed} files saved to {output_dir} ({total_time:.1f}s, {rate:.2f} steps/s)"
    )
    return processed


def compute_pet(
    tmrt_dir: str | Path,
    weather_series: list[Weather],
    output_dir: str | Path,
    human: HumanParams | None = None,
    location: Location | None = None,
) -> int:
    """
    Batch compute PET from saved Tmrt GeoTIFF files.

    Auto-discovers tmrt_*.tif files in tmrt_dir, matches them with weather_series
    by datetime, and saves pet_*.tif files to output_dir.

    Args:
        tmrt_dir: Directory containing tmrt_YYYYMMDD_HHMM.tif files.
        weather_series: List of Weather objects with datetime, ta, rh, ws.
        output_dir: Directory to save pet_YYYYMMDD_HHMM.tif files.
        human: Human body parameters. Uses defaults if not provided.
        location: Geographic location for weather.compute_derived().
            If None, assumes weather is already computed.

    Returns:
        Number of PET files processed.

    Example:
        # After running calculate_timeseries with output_dir
        n_processed = solweig.compute_pet(
            tmrt_dir="output/",
            weather_series=weather_list,
            output_dir="output_pet/",
            human=HumanParams(weight=75, height=1.75),
        )
        print(f"Processed {n_processed} timesteps")
    """
    import re

    from . import io

    if human is None:
        human = HumanParams()

    tmrt_dir = Path(tmrt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find tmrt_*.tif files (check tmrt/ subdirectory first, then flat layout)
    tmrt_files = sorted(tmrt_dir.glob("tmrt_*.tif"))
    if not tmrt_files and (tmrt_dir / "tmrt").exists():
        tmrt_files = sorted((tmrt_dir / "tmrt").glob("tmrt_*.tif"))
    if not tmrt_files:
        logger.warning(f"No tmrt_*.tif files found in {tmrt_dir}")
        return 0

    # Parse timestamps from filenames
    pattern = re.compile(r"tmrt_(\d{8})_(\d{4})\.tif")
    tmrt_map = {}
    for f in tmrt_files:
        match = pattern.match(f.name)
        if match:
            date_str, time_str = match.groups()
            timestamp = dt.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
            tmrt_map[timestamp] = f

    # Match weather with timestamps
    if location is not None:
        for w in weather_series:
            if not w._derived_computed:
                w.compute_derived(location)

    # Start timing
    start_time = time.time()
    processed = 0
    for weather in weather_series:
        if weather.datetime not in tmrt_map:
            logger.warning(f"No Tmrt file found for {weather.datetime}")
            continue

        # Load Tmrt
        tmrt_path = tmrt_map[weather.datetime]
        tmrt, transform, crs, _ = io.load_raster(str(tmrt_path))

        # Compute PET
        pet = compute_pet_grid(tmrt, weather.ta, weather.rh, weather.ws, human)

        # Save PET
        date_str = weather.datetime.strftime("%Y%m%d")
        time_str = weather.datetime.strftime("%H%M")
        pet_path = output_dir / f"pet_{date_str}_{time_str}.tif"

        io.save_raster(
            str(pet_path),
            pet,
            transform if isinstance(transform, list) else list(transform.to_gdal()),
            crs,
        )
        processed += 1

        if processed % 10 == 0:
            logger.info(f"Processed {processed}/{len(weather_series)} timesteps")

    total_time = time.time() - start_time
    rate = processed / total_time if total_time > 0 else 0
    logger.info(
        f"✓ PET computation complete: {processed} files saved to {output_dir} ({total_time:.1f}s, {rate:.2f} steps/s)"
    )
    return processed
