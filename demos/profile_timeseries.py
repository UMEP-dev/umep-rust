"""
Profile SOLWEIG timeseries to find per-timestep bottlenecks.

Instruments the fused Rust pipeline and Python-side precompute steps
called by calculate_core_fused().
"""

import functools
import statistics
import time
from collections import defaultdict
from pathlib import Path

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


import solweig  # noqa: E402

# Patch fused Rust pipeline (the main compute call per timestep)
from solweig.rustalgos import pipeline  # noqa: E402

pipeline.compute_timestep = _timed("rust:pipeline", pipeline.compute_timestep)
pipeline.precompute_gvf_cache = _timed("rust:gvf_precompute", pipeline.precompute_gvf_cache)

# Patch Python-side functions called by calculate_core_fused()
from solweig.components import gvf as gvf_mod  # noqa: E402
from solweig.components import shadows as shadows_mod  # noqa: E402
from solweig.components import svf_resolution as svf_mod  # noqa: E402
from solweig.physics import clearnessindex_2013b as ci_mod  # noqa: E402
from solweig.physics import daylen as daylen_mod  # noqa: E402
from solweig.physics import diffusefraction as df_mod  # noqa: E402

svf_mod.resolve_svf = _timed("py:svf_resolve", svf_mod.resolve_svf)
svf_mod.adjust_svfbuveg_with_psi = _timed("py:svf_psi_adjust", svf_mod.adjust_svfbuveg_with_psi)
shadows_mod.compute_transmissivity = _timed("py:transmissivity", shadows_mod.compute_transmissivity)
gvf_mod.detect_building_mask = _timed("py:building_mask", gvf_mod.detect_building_mask)
ci_mod.clearnessindex_2013b = _timed("py:clearness_idx", ci_mod.clearnessindex_2013b)
daylen_mod.daylen = _timed("py:daylen", daylen_mod.daylen)
df_mod.diffusefraction = _timed("py:diffuse_frac", df_mod.diffusefraction)

# Patch I/O
from solweig.models import results as results_mod  # noqa: E402

if hasattr(results_mod, "SolweigResult"):
    orig_to_geotiff = results_mod.SolweigResult.to_geotiff
    results_mod.SolweigResult.to_geotiff = _timed("io:geotiff_write", orig_to_geotiff)


def profile_period(weather_slice, label, surface, output_dir):
    """Profile a weather slice and return timing dict."""
    _timings.clear()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Profiling: {label}")
    print(f"  {len(weather_slice)} timesteps: {weather_slice[0].datetime} → {weather_slice[-1].datetime}")
    print(f"{'=' * 70}")

    t_total_start = time.perf_counter()
    solweig.calculate(
        surface=surface,
        weather=weather_slice,
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

    # Collect timings by category
    rust_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("rust:")], key=lambda x: -x[1])
    py_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("py:")], key=lambda x: -x[1])
    io_components = sorted([(n, sum(t)) for n, t in _timings.items() if n.startswith("io:")], key=lambda x: -x[1])

    all_items = rust_components + py_components + io_components
    all_items.sort(key=lambda x: -x[1])
    tracked_total = sum(t for _, t in all_items)
    overhead = t_total - tracked_total

    print(f"\n{'Component':<25} {'Total':>8} {'Mean':>8} {'Med':>8} {'Max':>8} {'N':>5} {'%':>6}")
    print(f"{'─' * 70}")

    for name, total in all_items:
        times = _timings[name]
        ms = [t * 1000 for t in times]
        pct = total / t_total * 100
        print(
            f"  {name:<23} {total:>7.3f}s {statistics.mean(ms):>7.2f} "
            f"{statistics.median(ms):>7.2f} {max(ms):>7.2f} {len(ms):>5} {pct:>5.1f}%"
        )

    print(
        f"  {'overhead (loop/alloc…)':<23} {overhead:>7.3f}s {'':>8} {'':>8} {'':>8} {'':>5} "
        f"{overhead / t_total * 100:>5.1f}%"
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
