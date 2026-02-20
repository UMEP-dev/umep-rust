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
