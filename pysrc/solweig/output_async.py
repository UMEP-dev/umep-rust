"""Asynchronous GeoTIFF writing helpers for timeseries workflows."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .solweig_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models import SolweigResult, SurfaceData

logger = get_logger(__name__)


def async_output_enabled() -> bool:
    """Return whether asynchronous output writing is enabled."""
    raw = os.environ.get("SOLWEIG_ASYNC_OUTPUT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def collect_output_arrays(result: SolweigResult, outputs: list[str]) -> dict[str, NDArray[np.floating]]:
    """Collect requested output arrays from a result object."""
    available_outputs = {
        "tmrt": result.tmrt,
        "utci": result.utci,
        "pet": result.pet,
        "shadow": result.shadow,
        "kdown": result.kdown,
        "kup": result.kup,
        "ldown": result.ldown,
        "lup": result.lup,
    }

    selected: dict[str, NDArray[np.floating]] = {}
    for name in outputs:
        if name not in available_outputs:
            logger.warning(f"Unknown output '{name}', skipping. Valid: {list(available_outputs.keys())}")
            continue
        array = available_outputs[name]
        if array is None:
            logger.warning(f"Output '{name}' is None (not computed), skipping.")
            continue
        selected[name] = array
    return selected


class AsyncGeoTiffWriter:
    """
    Single-threaded async writer with bounded in-flight tasks.

    Writing runs on one background thread so compute can continue while I/O
    proceeds. ``max_pending`` provides backpressure and bounds memory use.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        surface: SurfaceData | None = None,
        max_pending: int = 2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pending = max(1, int(max_pending))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solweig-geotiff")
        self._pending: deque[Future[None]] = deque()

        self.transform: list[float] | None = None
        self.crs_wkt: str = ""
        if surface is not None:
            if surface._geotransform is not None:
                self.transform = surface._geotransform
            if surface._crs_wkt is not None:
                self.crs_wkt = surface._crs_wkt

    def submit(self, *, timestamp: dt, arrays: dict[str, NDArray[np.floating]]) -> None:
        """Queue one timestep worth of outputs for writing."""
        if not arrays:
            return

        self._drain_completed()
        while len(self._pending) >= self.max_pending:
            self._pending.popleft().result()

        ts_str = timestamp.strftime("%Y%m%d_%H%M")
        future = self._executor.submit(
            _write_outputs,
            output_dir=self.output_dir,
            ts_str=ts_str,
            arrays=arrays,
            transform=self.transform,
            crs_wkt=self.crs_wkt,
        )
        self._pending.append(future)

    def close(self) -> None:
        """Wait for all queued writes and stop background worker."""
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._executor.shutdown(wait=True)

    def _drain_completed(self) -> None:
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()


def _write_outputs(
    *,
    output_dir: Path,
    ts_str: str,
    arrays: dict[str, NDArray[np.floating]],
    transform: list[float] | None,
    crs_wkt: str,
) -> None:
    from . import io

    if not arrays:
        return

    write_transform = transform if transform is not None else [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

    for name, array in arrays.items():
        comp_dir = output_dir / name
        comp_dir.mkdir(parents=True, exist_ok=True)
        filepath = comp_dir / f"{name}_{ts_str}.tif"
        io.save_raster(
            out_path_str=str(filepath),
            data_arr=array,
            trf_arr=write_transform,
            crs_wkt=crs_wkt,
            no_data_val=np.nan,
        )


class TiledGeoTiffWriter:
    """Writes per-timestep GeoTIFFs tile-by-tile using windowed writes.

    Instead of assembling a full-raster array in memory, this writer
    creates empty GeoTIFF files at full dimensions and writes each tile's
    core region directly.  Only one tile's worth of data is in memory at
    a time.
    """

    def __init__(
        self,
        output_dir: str | Path,
        rows: int,
        cols: int,
        *,
        transform: list[float] | None = None,
        crs_wkt: str = "",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = rows
        self.cols = cols
        self.transform = transform if transform is not None else [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        self.crs_wkt = crs_wkt
        self._open_files: dict[str, Path] = {}
        self._all_files: dict[tuple[str, str], Path] = {}
        self._final_files: dict[tuple[str, str], Path] = {}

    def open_timestep(self, timestamp: dt, output_names: list[str]) -> None:
        """Create empty GeoTIFFs for the current timestep."""
        from . import io

        ts_str = timestamp.strftime("%Y%m%d_%H%M")
        self._open_files = {}
        for name in output_names:
            comp_dir = self.output_dir / name
            comp_dir.mkdir(parents=True, exist_ok=True)
            filepath = comp_dir / f"{name}_{ts_str}.tif"
            io.create_empty_raster(
                path_str=filepath,
                rows=self.rows,
                cols=self.cols,
                transform=self.transform,
                crs_wkt=self.crs_wkt,
                dtype=np.float32,
                nodata=np.nan,
            )
            self._open_files[name] = filepath

    def write_tile(self, write_slice: tuple[slice, slice], arrays: dict[str, NDArray[np.floating]]) -> None:
        """Write tile core data to the open GeoTIFFs at the correct window."""
        from . import io

        for name, data in arrays.items():
            if name in self._open_files:
                io.write_raster_window(
                    path_str=self._open_files[name],
                    data=data,
                    window=write_slice,
                )

    def close_timestep(self) -> None:
        """Mark the current timestep's files as complete."""
        self._open_files = {}

    def precreate_timesteps(self, timestamps: list[dt], output_names: list[str]) -> None:
        """Create empty temporary GeoTIFFs for all timesteps upfront."""
        from . import io

        self._all_files: dict[tuple[str, str], Path] = {}
        self._final_files = {}
        for ts in timestamps:
            ts_str = ts.strftime("%Y%m%d_%H%M")
            for name in output_names:
                comp_dir = self.output_dir / name
                comp_dir.mkdir(parents=True, exist_ok=True)
                finalpath = comp_dir / f"{name}_{ts_str}.tif"
                filepath = comp_dir / f"{name}_{ts_str}.partial.tif"
                io.create_empty_raster(
                    path_str=filepath,
                    rows=self.rows,
                    cols=self.cols,
                    transform=self.transform,
                    crs_wkt=self.crs_wkt,
                    dtype=np.float32,
                    nodata=np.nan,
                )
                self._all_files[(ts_str, name)] = filepath
                self._final_files[(ts_str, name)] = finalpath

    def write_tile_at(
        self, timestamp: dt, write_slice: tuple[slice, slice], arrays: dict[str, NDArray[np.floating]]
    ) -> None:
        """Write tile window to pre-created files for a specific timestep."""
        from . import io

        ts_str = timestamp.strftime("%Y%m%d_%H%M")
        for name, data in arrays.items():
            key = (ts_str, name)
            if key in self._all_files:
                io.write_raster_window(path_str=self._all_files[key], data=data, window=write_slice)

    def finalize_precreated(self) -> None:
        """Promote successfully written temporary rasters to final filenames."""
        for key, filepath in list(self._all_files.items()):
            finalpath = self._final_files.get(key)
            if finalpath is not None and filepath.exists():
                os.replace(filepath, finalpath)

    def cleanup_precreated(self) -> None:
        """Remove temporary rasters left behind by an aborted run."""
        for filepath in self._all_files.values():
            filepath.unlink(missing_ok=True)

    def close(self, *, success: bool | None = None) -> None:
        """Finalize or clean up pre-created files, then clear writer state."""
        if success is True:
            self.finalize_precreated()
        elif success is False:
            self.cleanup_precreated()
        self._open_files = {}
        self._all_files = {}
        self._final_files = {}
