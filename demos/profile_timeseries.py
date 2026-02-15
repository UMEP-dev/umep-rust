"""
Profile SOLWEIG timeseries to find per-timestep bottlenecks.

Instruments each component of calculate_core() and the I/O layer.
Patches at computation.py level to capture Python wrapper overhead too.
"""

import functools
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

# ── Monkey-patch component functions with timing ──────────────────────

_timings: dict[str, list[float]] = defaultdict(list)


def _timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _timings[name].append(time.perf_counter() - t0)
        return result

    return wrapper


# Patch at Rust FFI level (these are always called indirectly)
import solweig  # noqa: E402
from solweig.rustalgos import ground as ground_rust  # noqa: E402
from solweig.rustalgos import gvf as gvf_rust  # noqa: E402
from solweig.rustalgos import shadowing, sky, vegetation  # noqa: E402
from solweig.rustalgos import tmrt as tmrt_rust  # noqa: E402

shadowing.calculate_shadows_wall_ht_25 = _timed("rust:shadows", shadowing.calculate_shadows_wall_ht_25)
gvf_rust.gvf_calc = _timed("rust:gvf_calc", gvf_rust.gvf_calc)
vegetation.kside_veg = _timed("rust:kside_veg", vegetation.kside_veg)
vegetation.lside_veg = _timed("rust:lside_veg", vegetation.lside_veg)
sky.cylindric_wedge = _timed("rust:cylindric_wedge", sky.cylindric_wedge)
sky.anisotropic_sky = _timed("rust:aniso_sky", sky.anisotropic_sky)
sky.weighted_patch_sum = _timed("rust:patch_sum", sky.weighted_patch_sum)
ground_rust.compute_ground_temperature = _timed("rust:ground_temp", ground_rust.compute_ground_temperature)
ground_rust.ts_wave_delay_batch = _timed("rust:ts_wave_delay", ground_rust.ts_wave_delay_batch)
tmrt_rust.compute_tmrt = _timed("rust:tmrt", tmrt_rust.compute_tmrt)

# Patch at computation.py level (captures Python wrapper + Rust call)
from solweig import computation  # noqa: E402

# Explicit Any alias for monkey-patching dynamic module attributes used only in this profiler.
comp = cast(Any, computation)

# These are the functions imported by computation.py at module level
# We need to patch computation's references directly
comp.compute_shadows = _timed("py:shadows", comp.compute_shadows)
comp.resolve_svf = _timed("py:svf_resolve", comp.resolve_svf)
comp.compute_ground_temperature = _timed("py:ground_temp", comp.compute_ground_temperature)
comp.compute_gvf = _timed("py:gvf", comp.compute_gvf)
comp.compute_radiation = _timed("py:radiation", comp.compute_radiation)
comp.compute_tmrt = _timed("py:tmrt", comp.compute_tmrt)
comp._apply_thermal_delay = _timed("py:thermal_delay", comp._apply_thermal_delay)

# Patch I/O
from solweig.models import results as results_mod  # noqa: E402

if hasattr(results_mod, "SolweigResult"):
    orig_to_geotiff = results_mod.SolweigResult.to_geotiff
    results_mod.SolweigResult.to_geotiff = _timed("io:geotiff_write", orig_to_geotiff)

# Patch nighttime result
comp._nighttime_result = _timed("py:nighttime", comp._nighttime_result)

# Patch Python physics used in radiation (requires UMEP)
from solweig.components import radiation as rad_mod  # noqa: E402

if rad_mod.Kup_veg_2015a is not None:
    rad_mod.Kup_veg_2015a = _timed("py:Kup_veg_comp", rad_mod.Kup_veg_2015a)


def profile_period(weather_slice, label, surface, output_dir):
    """Profile a weather slice and return timing dict."""
    _timings.clear()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Profiling: {label}")
    print(f"  {len(weather_slice)} timesteps: {weather_slice[0].datetime} → {weather_slice[-1].datetime}")
    print(f"{'=' * 70}")

    t_total_start = time.perf_counter()
    solweig.calculate_timeseries(
        surface=surface,
        weather_series=weather_slice,
        output_dir=output_dir,
    )
    t_total = time.perf_counter() - t_total_start

    n_day = sum(1 for w in weather_slice if w.sun_altitude > 0)
    n_night = len(weather_slice) - n_day

    print(f"\n{'─' * 70}")
    print(f"RESULTS: {label}")
    print(f"{'─' * 70}")
    print(f"Total: {t_total:.2f}s | {len(weather_slice)} steps ({n_day} day, {n_night} night)")
    print(
        f"Per step: {t_total / len(weather_slice) * 1000:.1f}ms avg | "
        f"Per daytime step: {t_total / max(n_day, 1) * 1000:.1f}ms est."
    )

    # Collect all component-level (py:) timings
    py_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("py:")], key=lambda x: -x[1])
    rust_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("rust:")], key=lambda x: -x[1])
    io_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("io:")], key=lambda x: -x[1])

    # Compute real overhead
    py_total = sum(t for _, t in py_components)
    io_total = sum(t for _, t in io_components)
    nighttime_total = sum(t for n, t in py_components if n == "py:nighttime")
    overhead = t_total - py_total - io_total - nighttime_total

    print(f"\n{'Component':<25} {'Total':>8} {'Mean':>8} {'Med':>8} {'Max':>8} {'N':>5} {'%':>6}")
    print(f"{'─' * 70}")

    all_items = py_components + io_components
    all_items.sort(key=lambda x: -x[1])

    for name, total in all_items:
        times = _timings[name]
        ms = [t * 1000 for t in times]
        pct = total / t_total * 100
        print(
            f"  {name:<23} {total:>7.3f}s {statistics.mean(ms):>7.2f} "
            f"{statistics.median(ms):>7.2f} {max(ms):>7.2f} {len(ms):>5} {pct:>5.1f}%"
        )

    print(
        f"  {'overhead (precompute…)':<23} {overhead:>7.3f}s {'':>8} {'':>8} {'':>8} {'':>5} "
        f"{overhead / t_total * 100:>5.1f}%"
    )

    print(f"\n  {'Rust FFI detail:'}")
    for name, total in rust_components:
        times = _timings[name]
        ms = [t * 1000 for t in times]
        pct = total / t_total * 100
        print(
            f"    {name:<21} {total:>7.3f}s {statistics.mean(ms):>7.2f} "
            f"{statistics.median(ms):>7.2f} {max(ms):>7.2f} {len(ms):>5} {pct:>5.1f}%"
        )

    # Bar chart
    print("\n  Time budget:")
    bar_items = all_items + [("overhead", overhead)]
    for name, total in sorted(bar_items, key=lambda x: -x[1]):
        pct = total / t_total * 100
        bar = "█" * int(pct / 2) + "░" * (1 if pct % 2 > 0.5 else 0)
        if pct >= 1.0:
            print(f"    {name:<23} {pct:>5.1f}% {bar}")

    return dict(_timings)


# ── Setup ──────────────────────────────────────────────────────────────

working_path = Path("temp/goteborg").absolute()
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"

print("=" * 70)
print("SOLWEIG Timeseries Profiler")
print("=" * 70)
print(f"GPU: {'enabled' if solweig.GPU_ENABLED else 'disabled'}")

surface = solweig.SurfaceData.prepare(
    dsm=dsm_path,
    cdsm=cdsm_path,
    working_dir=str(working_path),
    trunk_ratio=0.25,
)
print(f"Grid: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} = {surface.dsm.size:,} pixels")

weather_all = solweig.Weather.from_umep_met("demos/data/Goteborg_SWEREF99_1200/GBG_TMY_1977.txt")

# ── Profile winter 48h (few daytime hours) ─────────────────────────────
winter_48h = weather_all[:48]
profile_period(winter_48h, "Winter 48h (Jan 1-2)", surface, str(working_path / "profile_winter"))

# ── Profile summer 48h (long days, most computation) ──────────────────
# June 21 = day 172, hour index = 172*24 = 4128
summer_start = 172 * 24
summer_48h = weather_all[summer_start : summer_start + 48]
profile_period(summer_48h, "Summer 48h (Jun 21-22)", surface, str(working_path / "profile_summer"))

print("\n" + "=" * 70)
print("Profiling complete.")
print("=" * 70)
