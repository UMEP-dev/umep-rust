"""
Profile SOLWEIG performance using the Athens demo dataset.

Measures wall-clock time for each phase:
  1. Surface preparation (walls, SVF)
  2. Weather loading
  3. Calculation (timeseries)

Run:
    SOLWEIG_TIMING=1 python demos/profile_athens.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import solweig

input_path = Path("demos/data/athens").absolute()
work_dir = Path("temp/athens_profile").absolute()

# Clean working dir so SVF is recomputed each run
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True, exist_ok=True)


def timed(label: str):
    """Context manager that prints elapsed time."""

    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  {label}: {self.elapsed:.3f}s")

    return Timer()


print("=== SOLWEIG Performance Profile (Athens 400x400, 35 days) ===\n")
print(f"GPU enabled: {solweig.GPU_ENABLED}")

# Phase 1: Surface preparation (walls + SVF)
with timed("Surface prepare (walls + SVF)") as t_prepare:
    surface = solweig.SurfaceData.prepare(
        dsm=str(input_path / "DSM.tif"),
        working_dir=str(work_dir / "working"),
        bbox=[476800, 4205850, 477200, 4206250],
        pixel_size=1.0,
    )

print(f"    Grid: {surface.dsm.shape[0]}x{surface.dsm.shape[1]}")

# Phase 2: Weather loading
with timed("Weather load (35 days)") as t_weather:
    weather_list = solweig.Weather.from_epw(
        str(input_path / "athens_2023.epw"),
        start="2023-07-01",
        end="2023-08-04",
    )
    location = solweig.Location.from_epw(str(input_path / "athens_2023.epw"))

print(f"    Timesteps: {len(weather_list)}")

# Phase 3: Calculation
with timed("Calculate (timeseries)") as t_calc:
    summary = solweig.calculate(
        surface=surface,
        weather=weather_list,
        location=location,
        use_anisotropic_sky=True,
        output_dir=str(work_dir / "output"),
        outputs=["tmrt"],
    )

print(f"    Daytime steps: {summary.n_daytime}, Night steps: {summary.n_nighttime}")

# Summary
total = t_prepare.elapsed + t_weather.elapsed + t_calc.elapsed
print(f"\n{'=' * 55}")
print(f"  Total:    {total:.3f}s")
print(f"  Prepare:  {t_prepare.elapsed:.3f}s ({t_prepare.elapsed / total * 100:.0f}%)")
print(f"  Weather:  {t_weather.elapsed:.3f}s ({t_weather.elapsed / total * 100:.0f}%)")
print(f"  Calculate:{t_calc.elapsed:.3f}s ({t_calc.elapsed / total * 100:.0f}%)")
print(f"  Per-step: {t_calc.elapsed / len(weather_list) * 1000:.1f}ms")
print(f"{'=' * 55}")
