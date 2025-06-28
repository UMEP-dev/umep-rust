import timeit

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from umep.functions.SOLWEIGpython import Solweig_run as sr
from umep.functions.svf_functions import svfForProcessing153
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23
from umepr import common
from umepr.hybrid.svf_hybrid import svfForProcessing153_rust_shdw
from umepr.rustalgos import shadowing, skyview
from umepr.solweig_config import solweig_run as srr


def test_shadowing():
    # Test shadowingfunction_wallheight_23 vs calculate_shadows_wall_ht_25 for speed and memory
    repeats = 3
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=1)

    # --- Timing only (no memory profiling) ---
    def run_py():
        shadowingfunction_wallheight_23(
            dsm,
            vegdsm,
            vegdsm2,
            azi,
            alt,
            scale,
            amaxvalue,
            bush,
            wall_hts,
            wall_asp * np.pi / 180.0,
        )

    def run_rust():
        shadowing.calculate_shadows_wall_ht_25(
            azi,
            alt,
            scale,
            amaxvalue,
            dsm,
            vegdsm,
            vegdsm2,
            bush,
            wall_hts,
            wall_asp * np.pi / 180.0,
            None,
            None,
        )

    py_timings = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("shadowingfunction_wallheight_23", py_timings)

    rust_timings = timeit.repeat(run_rust, number=1, repeat=repeats)
    print_timing_stats("shadowing.calculate_shadows_wall_ht_25", rust_timings)

    # Print relative speed as percentage
    relative_speed(py_timings, rust_timings)

    # --- Memory profiling only (no timing) ---
    py_memory = memory_usage(run_py, max_usage=True)
    print(f"shadowingfunction_wallheight_23: max memory usage: {py_memory:.2f} MiB")

    rust_memory = memory_usage(run_rust, max_usage=True)
    print(f"shadowing.calculate_shadows_wall_ht_25: max memory usage: {rust_memory:.2f} MiB")

    # Run Python version
    veg_sh, bldg_sh, veg_blocks_bldg_sh, wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun = (
        shadowingfunction_wallheight_23(
            dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp * np.pi / 180.0
        )
    )
    result_py = {
        "veg_sh": veg_sh,
        "bldg_sh": bldg_sh,
        "veg_blocks_bldg_sh": veg_blocks_bldg_sh,
        "wall_sh": wall_sh,
        "wall_sun": wall_sun,
        "wall_sh_veg": wall_sh_veg,
        "face_sh": face_sh,
        "face_sun": face_sun,
    }
    # Run Rust version
    result_rust = shadowing.calculate_shadows_wall_ht_25(
        azi,
        alt,
        scale,
        amaxvalue,
        dsm,
        vegdsm,
        vegdsm2,
        bush,
        wall_hts,
        wall_asp * np.pi / 180.0,
        None,
        None,
    )
    key_map = {
        "veg_sh": "veg_sh",
        "bldg_sh": "bldg_sh",
        "veg_blocks_bldg_sh": "veg_blocks_bldg_sh",
        "wall_sh": "wall_sh",
        "wall_sun": "wall_sun",
        "wall_sh_veg": "wall_sh_veg",
        "face_sh": "face_sh",
        "face_sun": "face_sun",
    }
    # Compare results
    compare_results(result_py, result_rust, key_map, atol=0.0001)
    # Plot visual residuals
    plot_visual_residuals(bldg_sh, result_rust.bldg_sh, title_prefix="Building Shadows")
    plot_visual_residuals(veg_sh, result_rust.veg_sh, title_prefix="Vegetation Shadows")
    plot_visual_residuals(veg_blocks_bldg_sh, result_rust.veg_blocks_bldg_sh, title_prefix="Veg Blocks Bldg Shadows")
    plot_visual_residuals(wall_sh, result_rust.wall_sh, title_prefix="Wall Shadows")
    plot_visual_residuals(wall_sun, result_rust.wall_sun, title_prefix="Wall Sun")
    plot_visual_residuals(wall_sh_veg, result_rust.wall_sh_veg, title_prefix="Wall Sh Veg")
    plot_visual_residuals(face_sh, result_rust.face_sh, title_prefix="Face Sh")
    plot_visual_residuals(face_sun, result_rust.face_sun, title_prefix="Face Sun")


def test_svf():
    # Test svfForProcessing153 vs skyview.calculate_svf_153 for speed
    repeats = 1
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays(resolution=2)

    # --- Timing only (no memory profiling) ---
    def run_py():
        svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)

    def run_hybrid():
        svfForProcessing153_rust_shdw(dsm, vegdsm, vegdsm2, scale, 1)

    def run_rust():
        skyview.calculate_svf(dsm, vegdsm, vegdsm2, scale, True, 2)

    times_py = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("svfForProcessing153 - (shadowingfunction_20)", times_py)

    times_hybrid = timeit.repeat(run_hybrid, number=1, repeat=repeats)
    print_timing_stats("svfForProcessing153 - hybrid w. rust shadows", times_hybrid)

    times_rust = timeit.repeat(run_rust, number=1, repeat=repeats)
    print_timing_stats("skyview.calculate_svf", times_rust)

    # Print relative speed as percentage
    print("\n--- Relative Speed shadowingfunction_20 - hybrid w. rust shadows vs. Python ---")
    relative_speed(times_py, times_hybrid)

    # Print relative speed as percentage
    print("\n--- Relative Speed shadowingfunction_20 - full rust SVF vs. Python ---")
    relative_speed(times_py, times_rust)

    # --- Memory profiling only (no timing) ---
    py_memory = memory_usage(run_py, max_usage=True)
    print(f"svfForProcessing153: max memory usage: {py_memory:.2f} MiB")

    hybrid_memory = memory_usage(run_hybrid, max_usage=True)
    print(f"svfForProcessing153 - hybrid w. rust shadows: max memory usage: {hybrid_memory:.2f} MiB")

    rust_memory = memory_usage(run_rust, max_usage=True)
    print(f"skyview.calculate_svf: max memory usage: {rust_memory:.2f} MiB")

    # For testing outputs use hybrid version - shadowing is tested separately in above test
    # (otherwise testing against outputs from underlying shadowingfunction_20 gives different results)
    # Run Python version
    result_py = svfForProcessing153_rust_shdw(dsm, vegdsm, vegdsm2, scale, 1)
    # Run Rust version
    result_rust = skyview.calculate_svf(dsm, vegdsm, vegdsm2, scale, True, 2)
    # Compare results
    key_map = {
        "svf": "svf",
        "svfE": "svf_east",
        "svfS": "svf_south",
        "svfW": "svf_west",
        "svfN": "svf_north",
        "svfveg": "svf_veg",
        "svfEveg": "svf_veg_east",
        "svfSveg": "svf_veg_south",
        "svfWveg": "svf_veg_west",
        "svfNveg": "svf_veg_north",
        "svfaveg": "svf_veg_blocks_bldg_sh",
        "svfEaveg": "svf_veg_blocks_bldg_sh_east",
        "svfSaveg": "svf_veg_blocks_bldg_sh_south",
        "svfWaveg": "svf_veg_blocks_bldg_sh_west",
        "svfNaveg": "svf_veg_blocks_bldg_sh_north",
    }
    compare_results(result_py, result_rust, key_map, atol=0.0001)

    # Plot visual residuals for all comparable SVF components explicitly
    print("\nGenerating residual plots...")
    plot_visual_residuals(result_py["svf"], result_rust.svf, title_prefix="Svf")
    plot_visual_residuals(result_py["svfE"], result_rust.svf_east, title_prefix="Svf East")
    plot_visual_residuals(result_py["svfS"], result_rust.svf_south, title_prefix="Svf South")
    plot_visual_residuals(result_py["svfW"], result_rust.svf_west, title_prefix="Svf West")
    plot_visual_residuals(result_py["svfN"], result_rust.svf_north, title_prefix="Svf North")
    plot_visual_residuals(result_py["svfveg"], result_rust.svf_veg, title_prefix="Svf Veg")
    plot_visual_residuals(result_py["svfEveg"], result_rust.svf_veg_east, title_prefix="Svf East Veg")
    plot_visual_residuals(result_py["svfSveg"], result_rust.svf_veg_south, title_prefix="Svf South Veg")
    plot_visual_residuals(result_py["svfWveg"], result_rust.svf_veg_west, title_prefix="Svf West Veg")
    plot_visual_residuals(result_py["svfNveg"], result_rust.svf_veg_north, title_prefix="Svf North Veg")
    plot_visual_residuals(result_py["svfaveg"], result_rust.svf_veg_blocks_bldg_sh, title_prefix="Svf vbssh Veg")
    plot_visual_residuals(
        result_py["svfEaveg"], result_rust.svf_veg_blocks_bldg_sh_east, title_prefix="Svf East vbssh Veg"
    )
    plot_visual_residuals(
        result_py["svfSaveg"], result_rust.svf_veg_blocks_bldg_sh_south, title_prefix="Svf South vbssh Veg"
    )
    plot_visual_residuals(
        result_py["svfWaveg"], result_rust.svf_veg_blocks_bldg_sh_west, title_prefix="Svf West vbssh Veg"
    )
    plot_visual_residuals(
        result_py["svfNaveg"], result_rust.svf_veg_blocks_bldg_sh_north, title_prefix="Svf North vbssh Veg"
    )


def test_solweig():
    repeats = 1

    # --- Timing only (no memory profiling) ---
    def run_py():
        sr.solweig_run("tests/rustalgos/test_config_solweig.ini", feedback=None)

    def run_hybrid():
        srr.solweig_run("tests/rustalgos/test_config_solweig.ini", feedback=None)

    py_timings = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("solweig_run", py_timings)

    hybrid_timings = timeit.repeat(run_hybrid, number=1, repeat=repeats)
    print_timing_stats("solweig_run w rust shadows", hybrid_timings)

    # Print relative speed as percentage
    relative_speed(py_timings, hybrid_timings)

    # --- Memory profiling only (no timing) ---
    py_memory = memory_usage(run_py, max_usage=True)
    print(f"solweig_run: max memory usage: {py_memory:.2f} MiB")

    rust_memory = memory_usage(run_hybrid, max_usage=True)
    print(f"solweig_run w rust shadows: max memory usage: {rust_memory:.2f} MiB")


def make_test_arrays(
    resolution,
    dsm_path="demos/data/athens/DSM_{res}m.tif",
    veg_dsm_path="demos/data/athens/CDSM_{res}m.tif",
    wall_hts_path="demos/data/athens/walls_{res}m/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls_{res}m/wall_aspects.tif",
):
    dsm, dsm_transf, _crs = common.load_raster(dsm_path.format(res=resolution), bbox=None)
    vegdsm, _transf, _crs = common.load_raster(veg_dsm_path.format(res=resolution), bbox=None)
    vegdsm2 = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf.a
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    wall_hts, _transf, _crs = common.load_raster(wall_hts_path.format(res=resolution), bbox=None)
    wall_asp, _transf, _crs = common.load_raster(wall_aspect_path.format(res=resolution), bbox=None)

    # Convert all loaded arrays to float32
    dsm = dsm.astype(np.float32)
    vegdsm = vegdsm.astype(np.float32)
    wall_hts = wall_hts.astype(np.float32)
    wall_asp = wall_asp.astype(np.float32)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


# Calculate and print per-array right percentage
def pct(a, b, atol=0.001):
    if a is None or b is None:
        return float("nan")
    # Ensure shapes match before comparison
    if a.shape != b.shape:
        return f"Shape mismatch: {a.shape} vs {b.shape}"
    return 100.0 * np.isclose(a, b, atol=atol, rtol=0, equal_nan=True).sum() / a.size


def compare_results(result_py, result_rust, key_map, atol=0.0001):
    print("\n--- Comparison ---")
    for py_key, rust_attr in key_map.items():
        py_val = result_py.get(py_key)
        rust_val = getattr(result_rust, rust_attr, None)
        match_pct = pct(py_val, rust_val, atol=atol)
        mean_diff = (
            np.nanmean(np.abs(py_val - rust_val)) if py_val is not None and rust_val is not None else float("nan")
        )
        range_diff = np.nanmax(py_val) - np.nanmin(py_val) if py_val is not None else float("nan")
        print(
            f"{py_key:<20} vs {rust_attr:<35} right: {match_pct:.2f} mean diff: {mean_diff:.3f} range: {range_diff:.2f}"
        )


def print_timing_stats(func_name, times):
    """Prints the min, max, and average timing statistics for a function."""
    if not times:
        print(f"\n{func_name}: No timing data available.")
        return
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    print(f"\n{func_name}: min={min_time:.3f}s, max={max_time:.3f}s, avg={avg_time:.3f}s")


def relative_speed(times_py, times_rust):
    """Calculates and prints how many times faster the Rust version is compared to Python."""
    rust_avg = sum(times_rust) / len(times_rust)
    py_avg = sum(times_py) / len(times_py)
    speedup_factor = py_avg / rust_avg
    print(f"\nRelative speed: {speedup_factor:.2f} times faster for given data.")


def plot_visual_residuals(
    py_array,
    rust_array,
    title_prefix="Visual",
    cmap="viridis",
    cmap_residuals="coolwarm",
    tick_fontsize="xx-small",
    colorbar_shrink=0.6,
):
    # check shape
    if py_array.shape != rust_array.shape:
        print(f"Error: Input arrays have different shapes: {py_array.shape} vs {rust_array.shape}")
        return

    residuals = rust_array - py_array

    # Determine the symmetric range for the residuals colormap
    max_abs_residual = np.abs(residuals).max()

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))  # 3 rows, 1 column

    # Plot Array 1 (Python)
    im1 = axes[0].imshow(py_array, cmap=cmap)
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=colorbar_shrink)
    cbar1.ax.tick_params(labelsize=tick_fontsize)
    axes[0].set_title(f"{title_prefix} - Array 1 (Python)")
    axes[0].axis("off")

    # Plot Array 2 (Rust)
    im2 = axes[1].imshow(rust_array, cmap=cmap)
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=colorbar_shrink)
    cbar2.ax.tick_params(labelsize=tick_fontsize)
    axes[1].set_title(f"{title_prefix} - Array 2 (Rust)")
    axes[1].axis("off")

    # Plot Residuals with centered colormap
    im3 = axes[2].imshow(residuals, cmap=cmap_residuals, vmin=-max_abs_residual, vmax=max_abs_residual)
    cbar3 = fig.colorbar(im3, ax=axes[2], shrink=colorbar_shrink)
    cbar3.ax.tick_params(labelsize=tick_fontsize)
    axes[2].set_title(f"{title_prefix} - Residuals (Rust - Python)")
    axes[2].axis("off")

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.savefig(f"temp/{title_prefix.lower().replace(' ', '_')}_residuals.png", dpi=150)
