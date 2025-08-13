import cProfile
import pstats
import timeit

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from umep import common
from umep.functions.SOLWEIGpython import Solweig_run
from umep.functions.SOLWEIGpython.daylen import daylen
from umep.functions.SOLWEIGpython.gvf_2018a import gvf_2018a
from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore
from umep.functions.svf_functions import svfForProcessing153
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23
from umepr.hybrid.svf import svfForProcessing153_rust_shdw
from umepr.rustalgos import gvf, shadowing, skyview
from umepr.solweig_runner_rust import SolweigRunRust


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

    # Origin
    def run_ori():
        Solweig_run.solweig_run("tests/rustalgos/test_config_solweig_old_fmt.ini", None)

    ori_timings = timeit.repeat(run_ori, number=1, repeat=repeats)
    print_timing_stats("solweig_run (old format)", ori_timings)

    # --- Timing only (no memory profiling) ---
    SWC = SolweigRunCore(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )
    SWC.config.output_dir = "temp/goteborg/test_py/"

    def run_py():
        SWC.run()

    py_timings = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("solweig_run", py_timings)

    SWR = SolweigRunRust(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )

    def run_hybrid():
        SWR.run()

    hybrid_timings = timeit.repeat(run_hybrid, number=1, repeat=repeats)
    print_timing_stats("solweig_run w rust shadows", hybrid_timings)

    # Print relative speed as percentage
    print("\n--- Relative Speed Original vs. Python ---")
    relative_speed(ori_timings, py_timings)
    print("\n--- Relative Speed Original vs. Rust ---")
    relative_speed(ori_timings, hybrid_timings)
    # NO ANISO - ~2.5
    # WITH ANISO - ~2.5
    print("\n--- Relative Speed Core vs. Rust ---")
    relative_speed(py_timings, hybrid_timings)

    # --- Memory profiling only (no timing) ---
    print("\n--- Memory Profiling ---")
    # Memory profiling for original Solweig run
    ori_memory = memory_usage(run_ori, max_usage=True)
    print(f"\nsolweig_run (old format): max memory usage: {ori_memory:.2f} MiB")

    py_memory = memory_usage(run_py, max_usage=True)
    print(f"\nsolweig_run: max memory usage: {py_memory:.2f} MiB")

    rust_memory = memory_usage(run_hybrid, max_usage=True)
    print(f"\nsolweig_run w rust shadows: max memory usage: {rust_memory:.2f} MiB")


def test_profile_solweig():
    SWR = SolweigRunRust(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )
    # ANI patch parallel        24   25.169    1.049   25.169    1.049 {built-in method sky.anisotropic_sky}
    # ANI pixel parallel        24    2.019    0.084    2.019    0.084 {built-in method sky.anisotropic_sky}
    # GVF                       18    1.407    0.078    1.407    0.078 {built-in method gvf.gvf_calc}
    # GVF pixel parallel        18    0.184    0.010    0.184    0.010 {built-in method gvf.gvf_calc}
    # shadowing                 18    0.918    0.051    0.918    0.051 {built-in method shadowing.calculate_shadows_wall_ht_25}

    """
    GVF time includes sun on surface!!
    NO ANISO
    18    4.748    0.264    4.748    0.264 {built-in method gvf.gvf_calc}
       96    1.682    0.018    1.682    0.018 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Lvikt_veg.py:1(Lvikt_veg)
      170    1.164    0.007    1.830    0.011 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/common.py:52(save_raster)
       18    0.761    0.042    0.761    0.042 {built-in method shadowing.calculate_shadows_wall_ht_25}
    WITH ANISO
    18   10.380    0.577   13.064    0.726 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Kside_veg_v2022a.py:6(Kside_veg_v2022a)
       18    5.385    0.299    5.385    0.299 {built-in method gvf.gvf_calc}
       24    4.778    0.199    4.778    0.199 {built-in method sky.anisotropic_sky}
     2754    2.101    0.001    2.101    0.001 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/sunlit_shaded_patches.py:6(shaded_or_sunlit)
       96    1.896    0.020    1.896    0.020 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Lvikt_veg.py:1(Lvikt_veg)
      170    1.325    0.008    2.124    0.012 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/common.py:52(save_raster)
       24    1.323    0.055   28.091    1.170 /Users/gareth/dev/umep-rust/pysrc/umepr/functions/solweig.py:35(Solweig_2025a_calc)
       18    1.139    0.063    1.139    0.063 {built-in method shadowing.calculate_shadows_wall_ht_25}
    """
    profiler = cProfile.Profile()
    profiler.enable()
    SWR.run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(30)  # Show top 30 lines

    SWC = SolweigRunCore(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )
    """
    NO ANISO
          324   16.477    0.051   17.924    0.055 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/sunonsurface_2018a.py:3(sunonsurface_2018a)
       18    3.574    0.199    3.813    0.212 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/util/SEBESOLWEIGCommonFiles/shadowingfunction_wallheight_23.py:42(shadowingfunction_wallheight_23)
       96    1.813    0.019    1.813    0.019 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Lvikt_veg.py:1(Lvikt_veg)
       18    1.275    0.071   18.695    1.039 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/gvf_2018a.py:6(gvf_2018a)
    WITH ANISO
    24   24.894    1.037   33.829    1.410 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/anisotropic_sky.py:11(anisotropic_sky)
      324   14.680    0.045   15.968    0.049 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/sunonsurface_2018a.py:3(sunonsurface_2018a)
       18    9.770    0.543   12.326    0.685 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Kside_veg_v2022a.py:6(Kside_veg_v2022a)
     6426    4.469    0.001    4.469    0.001 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/sunlit_shaded_patches.py:6(shaded_or_sunlit)
       18    3.817    0.212    4.071    0.226 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/util/SEBESOLWEIGCommonFiles/shadowingfunction_wallheight_23.py:42(shadowingfunction_wallheight_23)
    """
    profiler = cProfile.Profile()
    profiler.enable()
    SWC.run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(30)  # Show top 30 lines


def test_gvf():
    # prepare variables
    SWC = SolweigRunCore(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )
    idx = 12
    scale = 1 / SWC.dsm_trf_arr[1]
    SBC = 5.67051e-8
    if SWC.params.Tmrt_params.Value.posture == "Standing":
        posture = SWC.params.Posture.Standing.Value
    else:
        posture = SWC.params.Posture.Sitting.Value
    _, _, _, SNUP = daylen(SWC.environ_data.jday[idx], SWC.location["latitude"])
    first = np.round(posture.height)
    if first == 0.0:
        first = 1.0
    second = np.round(posture.height * 20.0)
    dectime = SWC.environ_data.dectime[idx]
    altmax = SWC.environ_data.altmax[idx]
    Ta = SWC.environ_data.Ta[idx]
    Tgamp = SWC.tg_maps.TgK * altmax + SWC.tg_maps.Tstart  # Fixed 2021
    # Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall) # Old
    Tgampwall = SWC.tg_maps.TgK_wall * altmax + SWC.tg_maps.Tstart_wall
    Tg = Tgamp * np.sin(
        (((dectime - np.floor(dectime)) - SNUP / 24) / (SWC.tg_maps.TmaxLST / 24 - SNUP / 24)) * np.pi / 2
    )  # 2015 a, based on max sun altitude
    Tgwall = Tgampwall * np.sin(
        (((dectime - np.floor(dectime)) - SNUP / 24) / (SWC.tg_maps.TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2
    )  # 2015a, based on max sun altitude
    if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
        # Tg = 0
        Tgwall = 0
    sh_results = shadowing.calculate_shadows_wall_ht_25(
        SWC.environ_data.azimuth[idx],
        SWC.environ_data.altitude[idx],
        scale,
        SWC.vegetation.amaxvalue,
        SWC.dsm_arr.astype(np.float32),
        SWC.vegetation.vegdsm.astype(np.float32),
        SWC.vegetation.vegdsm2.astype(np.float32),
        SWC.vegetation.bush.astype(np.float32),
        SWC.wallheight.astype(np.float32),
        SWC.wallaspect.astype(np.float32) * np.pi / 180.0,
        None,
        None,
    )
    shadow = sh_results.wall_sh - (1 - sh_results.veg_sh) * (1 - SWC.environ_data.psi[idx])

    repeats = 3

    def run_py():
        gvf_2018a(
            sh_results.wall_sun,
            SWC.wallheight,
            SWC.buildings,
            scale,
            shadow,
            first,
            second,
            SWC.wallaspect,
            Tg,
            Tgwall,
            Ta,
            SWC.tg_maps.emis_grid,
            SWC.params.Emissivity.Value.Walls,
            SWC.tg_maps.alb_grid,
            SBC,
            SWC.params.Albedo.Effective.Value.Walls,
            SWC.rows,
            SWC.cols,
            SWC.environ_data.Twater[idx],
            None,
            False,
        )

    def run_rust():
        gvf.gvf_calc(
            sh_results.wall_sun.astype(np.float32),
            SWC.wallheight.astype(np.float32),
            SWC.buildings.astype(np.float32),
            scale,
            shadow.astype(np.float32),
            first,
            second,
            SWC.wallaspect.astype(np.float32),
            Tg.astype(np.float32),
            Tgwall,
            Ta,
            SWC.tg_maps.emis_grid.astype(np.float32),
            SWC.params.Emissivity.Value.Walls,
            SWC.tg_maps.alb_grid.astype(np.float32),
            SBC,
            SWC.params.Albedo.Effective.Value.Walls,
            SWC.environ_data.Twater[idx],
            None,
            False,
        )

    py_timings = timeit.repeat(run_py, number=1, repeat=repeats)
    print_timing_stats("gvf_2018a", py_timings)

    rust_timings = timeit.repeat(run_rust, number=1, repeat=repeats)
    print_timing_stats("gvf.gvf_calc", rust_timings)

    # Print relative speed as percentage
    relative_speed(py_timings, rust_timings)

    (
        gvfLup,
        gvfalb,
        gvfalbnosh,
        gvfLupE,
        gvfalbE,
        gvfalbnoshE,
        gvfLupS,
        gvfalbS,
        gvfalbnoshS,
        gvfLupW,
        gvfalbW,
        gvfalbnoshW,
        gvfLupN,
        gvfalbN,
        gvfalbnoshN,
        gvfSum,
        gvfNorm,
    ) = gvf_2018a(
        sh_results.wall_sun,
        SWC.wallheight,
        SWC.buildings,
        scale,
        shadow,
        first,
        second,
        SWC.wallaspect,
        Tg,
        Tgwall,
        Ta,
        SWC.tg_maps.emis_grid,
        SWC.params.Emissivity.Value.Walls,
        SWC.tg_maps.alb_grid,
        SBC,
        SWC.params.Albedo.Effective.Value.Walls,
        SWC.rows,
        SWC.cols,
        SWC.environ_data.Twater[idx],
        None,
        False,
    )
    result_py = {
        "gvfLup": gvfLup,
        "gvfalb": gvfalb,
        "gvfalbnosh": gvfalbnosh,
        "gvfLupE": gvfLupE,
        "gvfalbE": gvfalbE,
        "gvfalbnoshE": gvfalbnoshE,
        "gvfLupS": gvfLupS,
        "gvfalbS": gvfalbS,
        "gvfalbnoshS": gvfalbnoshS,
        "gvfLupW": gvfLupW,
        "gvfalbW": gvfalbW,
        "gvfalbnoshW": gvfalbnoshW,
        "gvfLupN": gvfLupN,
        "gvfalbN": gvfalbN,
        "gvfalbnoshN": gvfalbnoshN,
        "gvfSum": gvfSum,
        "gvfNorm": gvfNorm,
    }

    result_rust = gvf.gvf_calc(
        sh_results.wall_sun.astype(np.float32),
        SWC.wallheight.astype(np.float32),
        SWC.buildings.astype(np.float32),
        scale,
        shadow.astype(np.float32),
        first,
        second,
        SWC.wallaspect.astype(np.float32),
        Tg.astype(np.float32),
        Tgwall,
        Ta,
        SWC.tg_maps.emis_grid.astype(np.float32),
        SWC.params.Emissivity.Value.Walls,
        SWC.tg_maps.alb_grid.astype(np.float32),
        SBC,
        SWC.params.Albedo.Effective.Value.Walls,
        SWC.environ_data.Twater[idx],
        None,
        False,
    )
    key_map = {
        "gvfSum": "gvf_sum",
        "gvfNorm": "gvf_norm",
        "gvfLup": "gvf_lup",
        "gvfLupN": "gvf_lup_n",
        "gvfLupS": "gvf_lup_s",
        "gvfLupE": "gvf_lup_e",
        "gvfLupW": "gvf_lup_w",
        "gvfalb": "gvfalb",
        "gvfalbN": "gvfalb_n",
        "gvfalbS": "gvfalb_s",
        "gvfalbE": "gvfalb_e",
        "gvfalbW": "gvfalb_w",
        "gvfalbnosh": "gvfalbnosh",
        "gvfalbnoshN": "gvfalbnosh_n",
        "gvfalbnoshS": "gvfalbnosh_s",
        "gvfalbnoshE": "gvfalbnosh_e",
        "gvfalbnoshW": "gvfalbnosh_w",
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


def make_test_arrays(
    resolution,
    dsm_path="demos/data/athens/DSM_{res}m.tif",
    veg_dsm_path="demos/data/athens/CDSM_{res}m.tif",
    wall_hts_path="demos/data/athens/walls_{res}m/wall_hts.tif",
    wall_aspect_path="demos/data/athens/walls_{res}m/wall_aspects.tif",
):
    dsm, dsm_transf, _crs, _nd_val = common.load_raster(dsm_path.format(res=resolution), bbox=None)
    vegdsm, _transf, _crs, _nd_val = common.load_raster(veg_dsm_path.format(res=resolution), bbox=None)
    vegdsm2 = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf[1]
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    wall_hts, _transf, _crs, _nd_val = common.load_raster(wall_hts_path.format(res=resolution), bbox=None)
    wall_asp, _transf, _crs, _nd_val = common.load_raster(wall_aspect_path.format(res=resolution), bbox=None)

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
