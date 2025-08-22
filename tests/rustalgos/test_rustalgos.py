import cProfile
import pstats
import timeit

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from umep import common
from umep.functions.SOLWEIGpython import Solweig_run
from umep.functions.SOLWEIGpython.anisotropic_sky import anisotropic_sky as ani_sky
from umep.functions.SOLWEIGpython.cylindric_wedge import cylindric_wedge
from umep.functions.SOLWEIGpython.daylen import daylen
from umep.functions.SOLWEIGpython.gvf_2018a import gvf_2018a
from umep.functions.SOLWEIGpython.Kside_veg_v2022a import Kside_veg_v2022a
from umep.functions.SOLWEIGpython.Kup_veg_2015a import Kup_veg_2015a
from umep.functions.SOLWEIGpython.Lside_veg_v2022a import Lside_veg_v2022a
from umep.functions.SOLWEIGpython.patch_radiation import patch_steradians
from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore
from umep.functions.SOLWEIGpython.TsWaveDelay_2015a import TsWaveDelay_2015a
from umep.functions.svf_functions import svfForProcessing153
from umep.util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b
from umep.util.SEBESOLWEIGCommonFiles.create_patches import create_patches
from umep.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23
from umepr.hybrid.svf import svfForProcessing153_rust_shdw
from umepr.rustalgos import gvf, shadowing, sky, skyview, vegetation
from umepr.solweig_runner_rust import SolweigRunRust


def test_shadowing():
    # Test shadowingfunction_wallheight_23 vs calculate_shadows_wall_ht_25 for speed and memory
    repeats = 3
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays()

    # --- Timing only (no memory profiling) ---
    def run_py():
        return shadowingfunction_wallheight_23(  # type: ignore
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
        return shadowing.calculate_shadows_wall_ht_25(
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
    veg_sh, bldg_sh, veg_blocks_bldg_sh, wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun = run_py()
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
    result_rust = run_rust()
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
    compare_results(result_py, result_rust, key_map)
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
    dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp = make_test_arrays()

    # --- Timing only (no memory profiling) ---
    def run_py():
        return svfForProcessing153(dsm, vegdsm, vegdsm2, scale, 1)

    def run_hybrid():
        return svfForProcessing153_rust_shdw(dsm, vegdsm, vegdsm2, scale, 1)

    def run_rust():
        return skyview.calculate_svf(dsm, vegdsm, vegdsm2, scale, True, 2, None)

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
    result_py = run_hybrid()
    # Run Rust version
    result_rust = run_rust()
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
    compare_results(result_py, result_rust, key_map)

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
    Running SOLWEIG: 100%|██████████| 24/24 [00:09<00:00,  2.61step/s]         100276 function calls (100107 primitive calls) in 9.306 seconds
       18    5.285    0.294    5.285    0.294 {built-in method gvf.gvf_calc}
      169    1.455    0.009    2.228    0.013 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/common.py:57(save_raster)
       18    1.095    0.061    1.095    0.061 {built-in method shadowing.calculate_shadows_wall_ht_25}
      169    0.558    0.003    0.596    0.004 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/rasterio/__init__.py:99(open)
       24    0.135    0.006    6.941    0.289 /Users/gareth/dev/umep-rust/pysrc/umepr/functions/solweig.py:30(Solweig_2025a_calc)
       18    0.126    0.007    0.129    0.007 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/cylindric_wedge.py:3(cylindric_wedge)
       24    0.090    0.004    0.090    0.004 {built-in method vegetation.lside_veg}
      2/1    0.070    0.035    9.290    9.290 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/solweig_runner.py:343(run)
       18    0.069    0.004    0.069    0.004 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Kup_veg_2015a.py:3(Kup_veg_2015a)
      990    0.053    0.000    0.053    0.000 {method 'astype' of 'numpy.ndarray' objects}
       18    0.053    0.003    0.053    0.003 {built-in method vegetation.kside_veg}
    WITH ANISO - CYLINDER
    Running SOLWEIG: 100%|██████████| 24/24 [00:15<00:00,  1.54step/s]         171116 function calls (170947 primitive calls) in 16.461 seconds
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       18    5.073    0.282    5.073    0.282 {built-in method gvf.gvf_calc}
       24    4.182    0.174    4.182    0.174 {built-in method sky.anisotropic_sky}
       24    1.394    0.058   13.789    0.575 /Users/gareth/dev/umep-rust/pysrc/umepr/functions/solweig.py:30(Solweig_2025a_calc)
      169    1.285    0.008    2.079    0.012 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/common.py:57(save_raster)
       18    1.031    0.057    1.031    0.057 {built-in method shadowing.calculate_shadows_wall_ht_25}
     1436    0.938    0.001    0.938    0.001 {method 'astype' of 'numpy.ndarray' objects}
       18    0.892    0.050    0.892    0.050 {built-in method vegetation.kside_veg}
      169    0.588    0.003    0.624    0.004 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/rasterio/__init__.py:99(open)
        1    0.391    0.391    0.391    0.391 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/solweig_runner.py:185(hemispheric_image)
       18    0.104    0.006    0.107    0.006 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/cylindric_wedge.py:3(cylindric_wedge)
      2/1    0.077    0.038   16.005   16.005 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/solweig_runner.py:343(run)
       18    0.052    0.003    0.052    0.003 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/Kup_veg_2015a.py:3(Kup_veg_2015a)
      169    0.050    0.000    0.053    0.000 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/pyproj/crs/crs.py:185(__init__)
       24    0.048    0.002   13.838    0.577 /Users/gareth/dev/umep-rust/pysrc/umepr/solweig_runner_rust.py:15(calc_solweig)
      108    0.026    0.000    0.026    0.000 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/TsWaveDelay_2015a.py:4(TsWaveDelay_2015a)
       24    0.024    0.001    0.024    0.001 {built-in method vegetation.lside_veg}
    WITH ANISO - BOX
    Running SOLWEIG: 100%|██████████| 24/24 [00:13<00:00,  1.71step/s]         171116 function calls (170947 primitive calls) in 14.697 seconds
           18    5.305    0.295    5.305    0.295 {built-in method gvf.gvf_calc}
       24    2.288    0.095    2.288    0.095 {built-in method sky.anisotropic_sky}
       24    1.603    0.067   12.705    0.529 /Users/gareth/dev/umep-rust/pysrc/umepr/functions/solweig.py:30(Solweig_2025a_calc)
       18    1.483    0.082    1.483    0.082 {built-in method shadowing.calculate_shadows_wall_ht_25}
      169    1.457    0.009    2.292    0.014 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/common.py:57(save_raster)
     1436    1.101    0.001    1.101    0.001 {method 'astype' of 'numpy.ndarray' objects}
       18    0.940    0.052    0.940    0.052 {built-in method vegetation.kside_veg}
      169    0.610    0.004    0.649    0.004 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/rasterio/__init__.py:99(open)
        1    0.384    0.384    0.384    0.384 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/solweig_runner.py:185(hemispheric_image)
       18    0.125    0.007    0.128    0.007 /Users/gareth/dev/umep-rust/.venv/lib/python3.12/site-packages/umep/functions/SOLWEIGpython/cylindric_wedge.py:3(cylindric_wedge)
     12/6    0.106    0.009    0.553    0.092 {method 'acquire' of '_thread.lock' objects}
       24    0.099    0.004    0.099    0.004 {built-in method vegetation.lside_veg}
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


def test_solweig_sub_funcs():
    # prepare variables
    SWC = SolweigRunCore(
        config_path_str="tests/rustalgos/test_config_solweig.ini",
        params_json_path="tests/rustalgos/test_params_solweig.json",
    )
    idx = 12
    scale = 1 / SWC.raster_data.trf_arr[1]
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
        SWC.raster_data.amaxvalue,
        SWC.raster_data.dsm.astype(np.float32),
        SWC.raster_data.cdsm.astype(np.float32),
        SWC.raster_data.tdsm.astype(np.float32),
        SWC.raster_data.bush.astype(np.float32),
        SWC.raster_data.wallheight.astype(np.float32),
        SWC.raster_data.wallaspect.astype(np.float32) * np.pi / 180.0,
        None,
        None,
    )
    shadow = sh_results.wall_sh - (1 - sh_results.veg_sh) * (1 - SWC.environ_data.psi[idx])

    repeats = 3

    def run_gvf_py():
        return gvf_2018a(  # type: ignore
            sh_results.wall_sun.astype(np.float32),
            SWC.raster_data.wallheight.astype(np.float32),
            SWC.raster_data.buildings.astype(np.float32),
            scale,
            shadow.astype(np.float32),
            first,
            second,
            SWC.raster_data.wallaspect.astype(np.float32),
            Tg.astype(np.float32),
            Tgwall,
            Ta,
            SWC.tg_maps.emis_grid.astype(np.float32),
            SWC.params.Emissivity.Value.Walls,
            SWC.tg_maps.alb_grid.astype(np.float32),
            SBC,
            SWC.params.Albedo.Effective.Value.Walls,
            SWC.raster_data.rows,
            SWC.raster_data.cols,
            SWC.environ_data.Twater[idx],
            None,
            False,
        )

    def run_gvf_rust():
        return gvf.gvf_calc(  # type: ignore
            sh_results.wall_sun.astype(np.float32),
            SWC.raster_data.wallheight.astype(np.float32),
            SWC.raster_data.buildings.astype(np.float32),
            scale,
            shadow.astype(np.float32),
            first,
            second,
            SWC.raster_data.wallaspect.astype(np.float32),
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

    py_gvf_timings = timeit.repeat(run_gvf_py, number=1, repeat=repeats)
    print_timing_stats("gvf_2018a", py_gvf_timings)

    rust_gvf_timings = timeit.repeat(run_gvf_rust, number=1, repeat=repeats)
    print_timing_stats("gvf.gvf_calc", rust_gvf_timings)

    # Print relative speed as percentage
    relative_speed(py_gvf_timings, rust_gvf_timings)

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
    ) = run_gvf_py()

    result_gvf_py = {
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

    result_gvf_rust = run_gvf_rust()

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
    compare_results(result_gvf_py, result_gvf_rust, key_map)
    # Plot visual residuals
    plot_visual_residuals(gvfSum, result_gvf_rust.gvf_sum, title_prefix="GVF Sum")
    plot_visual_residuals(gvfNorm, result_gvf_rust.gvf_norm, title_prefix="GVF Norm")
    plot_visual_residuals(gvfLup, result_gvf_rust.gvf_lup, title_prefix="GVF Lup")
    plot_visual_residuals(gvfLupN, result_gvf_rust.gvf_lup_n, title_prefix="GVF Lup N")
    plot_visual_residuals(gvfLupS, result_gvf_rust.gvf_lup_s, title_prefix="GVF Lup S")
    plot_visual_residuals(gvfLupW, result_gvf_rust.gvf_lup_w, title_prefix="GVF Lup W")
    plot_visual_residuals(gvfLupE, result_gvf_rust.gvf_lup_e, title_prefix="GVF Lup E")
    plot_visual_residuals(gvfalb, result_gvf_rust.gvfalb, title_prefix="GVF Albedo")
    plot_visual_residuals(gvfalbN, result_gvf_rust.gvfalb_n, title_prefix="GVF Albedo N")
    plot_visual_residuals(gvfalbS, result_gvf_rust.gvfalb_s, title_prefix="GVF Albedo S")
    plot_visual_residuals(gvfalbW, result_gvf_rust.gvfalb_w, title_prefix="GVF Albedo W")
    plot_visual_residuals(gvfalbE, result_gvf_rust.gvfalb_e, title_prefix="GVF Albedo E")
    plot_visual_residuals(gvfalbnosh, result_gvf_rust.gvfalbnosh, title_prefix="GVF Albedo No Shadow")
    plot_visual_residuals(gvfalbnoshN, result_gvf_rust.gvfalbnosh_n, title_prefix="GVF Albedo No Shadow N")
    plot_visual_residuals(gvfalbnoshS, result_gvf_rust.gvfalbnosh_s, title_prefix="GVF Albedo No Shadow S")
    plot_visual_residuals(gvfalbnoshW, result_gvf_rust.gvfalbnosh_w, title_prefix="GVF Albedo No Shadow W")
    plot_visual_residuals(gvfalbnoshE, result_gvf_rust.gvfalbnosh_e, title_prefix="GVF Albedo No Shadow E")

    ### KSIDE
    t = 0.0
    F_sh = cylindric_wedge(
        SWC.environ_data.zen[idx],
        SWC.svf_data.svfalfa,
        SWC.raster_data.rows,
        SWC.raster_data.cols,
    )
    Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(
        SWC.environ_data.radI[idx],
        SWC.environ_data.radD[idx],
        SWC.environ_data.radG[idx],
        SWC.environ_data.altitude[idx],
        SWC.raster_data.svfbuveg,
        SWC.params.Emissivity.Value.Walls,
        F_sh,
        result_gvf_rust.gvfalb,
        result_gvf_rust.gvfalb_e,
        result_gvf_rust.gvfalb_s,
        result_gvf_rust.gvfalb_w,
        result_gvf_rust.gvfalb_n,
        result_gvf_rust.gvfalbnosh,
        result_gvf_rust.gvfalbnosh_e,
        result_gvf_rust.gvfalbnosh_s,
        result_gvf_rust.gvfalbnosh_w,
        result_gvf_rust.gvfalbnosh_n,
    )
    zenDeg = SWC.environ_data.zen[idx] * (180 / np.pi)
    lv, pc_, pb_ = Perez_v3(
        zenDeg,
        SWC.environ_data.azimuth[idx],
        SWC.environ_data.radD[idx],
        SWC.environ_data.radI[idx],
        SWC.environ_data.jday[idx],
        1,
        2,
    )

    def run_kside_py():
        return Kside_veg_v2022a(  # type: ignore
            SWC.environ_data.radI[idx],
            SWC.environ_data.radD[idx],
            SWC.environ_data.radG[idx],
            shadow.astype(np.float32),
            SWC.svf_data.svf_south.astype(np.float32),
            SWC.svf_data.svf_west.astype(np.float32),
            SWC.svf_data.svf_north.astype(np.float32),
            SWC.svf_data.svf_east.astype(np.float32),
            SWC.svf_data.svf_veg_east.astype(np.float32),
            SWC.svf_data.svf_veg_south.astype(np.float32),
            SWC.svf_data.svf_veg_west.astype(np.float32),
            SWC.svf_data.svf_veg_north.astype(np.float32),
            SWC.environ_data.azimuth[idx],
            SWC.environ_data.altitude[idx],
            SWC.environ_data.psi[idx],
            t,
            SWC.params.Albedo.Effective.Value.Walls,
            F_sh.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            True,  # cylindrical
            lv.astype(np.float32) if lv is not None else None,
            True,  # anisotropic sky
            SWC.shadow_mats.diffsh.astype(np.float32) if SWC.shadow_mats.diffsh is not None else None,
            SWC.raster_data.rows,
            SWC.raster_data.cols,
            SWC.shadow_mats.asvf.astype(np.float32) if SWC.shadow_mats.asvf is not None else None,
            SWC.shadow_mats.shmat.astype(np.float32) if SWC.shadow_mats.shmat is not None else None,
            SWC.shadow_mats.vegshmat.astype(np.float32) if SWC.shadow_mats.vegshmat is not None else None,
            SWC.shadow_mats.vbshvegshmat.astype(np.float32) if SWC.shadow_mats.vbshvegshmat is not None else None,
        )

    def run_kside_rust():
        return vegetation.kside_veg(  # type: ignore
            SWC.environ_data.radI[idx],
            SWC.environ_data.radD[idx],
            SWC.environ_data.radG[idx],
            shadow.astype(np.float32),
            SWC.svf_data.svf_south.astype(np.float32),
            SWC.svf_data.svf_west.astype(np.float32),
            SWC.svf_data.svf_north.astype(np.float32),
            SWC.svf_data.svf_east.astype(np.float32),
            SWC.svf_data.svf_veg_east.astype(np.float32),
            SWC.svf_data.svf_veg_south.astype(np.float32),
            SWC.svf_data.svf_veg_west.astype(np.float32),
            SWC.svf_data.svf_veg_north.astype(np.float32),
            SWC.environ_data.azimuth[idx],
            SWC.environ_data.altitude[idx],
            SWC.environ_data.psi[idx],
            t,
            SWC.params.Albedo.Effective.Value.Walls,
            F_sh.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            True,  # cylindrical
            lv.astype(np.float32) if lv is not None else None,
            True,  # anisotropic sky
            SWC.shadow_mats.diffsh.astype(np.float32) if SWC.shadow_mats.diffsh is not None else None,
            SWC.shadow_mats.asvf.astype(np.float32) if SWC.shadow_mats.asvf is not None else None,
            SWC.shadow_mats.shmat.astype(np.float32) if SWC.shadow_mats.shmat is not None else None,
            SWC.shadow_mats.vegshmat.astype(np.float32) if SWC.shadow_mats.vegshmat is not None else None,
            SWC.shadow_mats.vbshvegshmat.astype(np.float32) if SWC.shadow_mats.vbshvegshmat is not None else None,
        )

    py_kside_timings = timeit.repeat(run_kside_py, number=1, repeat=repeats)
    print_timing_stats("kside_veg_v2022a", py_kside_timings)

    rust_kside_timings = timeit.repeat(run_kside_rust, number=1, repeat=repeats)
    print_timing_stats("vegetation.kside_veg", rust_kside_timings)

    # Print relative speed as percentage
    relative_speed(py_kside_timings, rust_kside_timings)

    (
        Keast,
        Ksouth,
        Kwest,
        Knorth,
        KsideI,
        KsideD,
        Kside,
    ) = run_kside_py()

    result_kside_py = {
        "Keast": Keast,
        "Ksouth": Ksouth,
        "Kwest": Kwest,
        "Knorth": Knorth,
        "KsideI": KsideI,
        "KsideD": KsideD,
        "Kside": Kside,
    }

    result_kside_rust = run_kside_rust()

    key_map = {
        "Keast": "keast",
        "Ksouth": "ksouth",
        "Kwest": "kwest",
        "Knorth": "knorth",
        "KsideI": "kside_i",
        "KsideD": "kside_d",
        "Kside": "kside",
    }
    # Compare results
    compare_results(result_kside_py, result_kside_rust, key_map)
    # Plot visual residuals
    plot_visual_residuals(Keast, result_kside_rust.keast, title_prefix="Keast_veg")
    plot_visual_residuals(Ksouth, result_kside_rust.ksouth, title_prefix="Ksouth_veg")
    plot_visual_residuals(Kwest, result_kside_rust.kwest, title_prefix="Kwest_veg")
    plot_visual_residuals(Knorth, result_kside_rust.knorth, title_prefix="Knorth_veg")
    plot_visual_residuals(KsideI, result_kside_rust.kside_i, title_prefix="KsideI_veg")
    plot_visual_residuals(KsideD, result_kside_rust.kside_d, title_prefix="KsideD_veg")
    plot_visual_residuals(Kside, result_kside_rust.kside, title_prefix="Kside_veg")

    ### LSIDE
    elvis = 0.0
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (SWC.environ_data.RH[idx] / 100.0)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis
    I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(
        SWC.environ_data.zen[idx],
        SWC.environ_data.jday[idx],
        Ta,
        SWC.environ_data.RH[idx] / 100.0,
        SWC.environ_data.radG[idx],
        SWC.location,
        SWC.environ_data.P[idx],
    )
    ewall = SWC.params.Albedo.Effective.Value.Walls
    Ldown = (
        (SWC.svf_data.svf + SWC.svf_data.svf_veg - 1) * esky * SBC * ((Ta + 273.15) ** 4)
        + (2 - SWC.svf_data.svf_veg - SWC.svf_data.svf_veg_blocks_bldg_sh) * ewall * SBC * ((Ta + 273.15) ** 4)
        + (SWC.svf_data.svf_veg_blocks_bldg_sh - SWC.svf_data.svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4)
        + (2 - SWC.svf_data.svf - SWC.svf_data.svf_veg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)
    )
    if CI < 0.95:
        c = 1 - CI
        Ldown = Ldown * (1 - c) + c * (
            (SWC.svf_data.svf + SWC.svf_data.svf_veg - 1) * SBC * ((Ta + 273.15) ** 4)
            + (2 - SWC.svf_data.svf_veg - SWC.svf_data.svf_veg_blocks_bldg_sh) * ewall * SBC * ((Ta + 273.15) ** 4)
            + (SWC.svf_data.svf_veg_blocks_bldg_sh - SWC.svf_data.svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4)
            + (2 - SWC.svf_data.svf - SWC.svf_data.svf_veg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)
        )
    timestepdec = 0
    timeadd = 0.0
    firstdaytime = 1.0
    Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, SWC.tg_maps.Tgmap1)
    LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, SWC.tg_maps.Tgmap1E)
    LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, SWC.tg_maps.Tgmap1S)
    LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, SWC.tg_maps.Tgmap1W)
    LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, SWC.tg_maps.Tgmap1N)

    def run_lside_py():
        return Lside_veg_v2022a(
            SWC.svf_data.svf_south.astype(np.float32),
            SWC.svf_data.svf_west.astype(np.float32),
            SWC.svf_data.svf_north.astype(np.float32),
            SWC.svf_data.svf_east.astype(np.float32),
            SWC.svf_data.svf_veg_east.astype(np.float32),
            SWC.svf_data.svf_veg_south.astype(np.float32),
            SWC.svf_data.svf_veg_west.astype(np.float32),
            SWC.svf_data.svf_veg_north.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_east.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_south.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_west.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_north.astype(np.float32),
            SWC.environ_data.azimuth[idx],
            SWC.environ_data.altitude[idx],
            Ta,
            Tgwall,
            SBC,
            SWC.params.Albedo.Effective.Value.Walls,
            Ldown.astype(np.float32),
            esky,
            t,
            F_sh.astype(np.float32),
            CI,
            LupE.astype(np.float32),
            LupS.astype(np.float32),
            LupW.astype(np.float32),
            LupN.astype(np.float32),
            0,
        )

    def run_lside_rust():
        return vegetation.lside_veg(
            SWC.svf_data.svf_south.astype(np.float32),
            SWC.svf_data.svf_west.astype(np.float32),
            SWC.svf_data.svf_north.astype(np.float32),
            SWC.svf_data.svf_east.astype(np.float32),
            SWC.svf_data.svf_veg_east.astype(np.float32),
            SWC.svf_data.svf_veg_south.astype(np.float32),
            SWC.svf_data.svf_veg_west.astype(np.float32),
            SWC.svf_data.svf_veg_north.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_east.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_south.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_west.astype(np.float32),
            SWC.svf_data.svf_veg_blocks_bldg_sh_north.astype(np.float32),
            SWC.environ_data.azimuth[idx],
            SWC.environ_data.altitude[idx],
            Ta,
            Tgwall,
            SBC,
            SWC.params.Albedo.Effective.Value.Walls,
            Ldown.astype(np.float32),
            esky,
            t,
            F_sh.astype(np.float32),
            CI,
            LupE.astype(np.float32),
            LupS.astype(np.float32),
            LupW.astype(np.float32),
            LupN.astype(np.float32),
            False,
        )

    py_lside_timings = timeit.repeat(run_lside_py, number=1, repeat=repeats)
    print_timing_stats("lside_veg_v2022a", py_lside_timings)

    rust_lside_timings = timeit.repeat(run_lside_rust, number=1, repeat=repeats)
    print_timing_stats("vegetation.lside_veg", rust_lside_timings)

    # Print relative speed as percentage
    relative_speed(py_lside_timings, rust_lside_timings)

    (
        Least,
        Lsouth,
        Lwest,
        Lnorth,
    ) = run_lside_py()

    result_lside_py = {
        "Least": Least,
        "Lsouth": Lsouth,
        "Lwest": Lwest,
        "Lnorth": Lnorth,
    }
    result_lside_rust = run_lside_rust()

    key_map = {
        "Least": "least",
        "Lsouth": "lsouth",
        "Lwest": "lwest",
        "Lnorth": "lnorth",
    }
    # Compare results
    compare_results(result_lside_py, result_lside_rust, key_map)
    # Plot visual residuals
    plot_visual_residuals(Least, result_lside_rust.least, title_prefix="Least_veg")
    plot_visual_residuals(Lsouth, result_lside_rust.lsouth, title_prefix="Lsouth_veg")
    plot_visual_residuals(Lwest, result_lside_rust.lwest, title_prefix="Lwest_veg")
    plot_visual_residuals(Lnorth, result_lside_rust.lnorth, title_prefix="Lnorth_veg")

    ### aniso
    skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(2)
    patch_emissivities = np.zeros(skyvaultalt.shape[0])
    x = np.transpose(np.atleast_2d(skyvaultalt))
    y = np.transpose(np.atleast_2d(skyvaultazi))
    z = np.transpose(np.atleast_2d(patch_emissivities))
    L_patches = np.append(np.append(x, y, axis=1), z, axis=1)
    steradians, skyalt, patch_altitude = patch_steradians(L_patches)
    Lup = SBC * SWC.tg_maps.emis_grid * ((SWC.tg_maps.Knight + Ta + Tg + 273.15) ** 4)

    def run_ani_py():
        return ani_sky(
            SWC.shadow_mats.shmat.astype(np.float32),
            SWC.shadow_mats.vegshmat.astype(np.float32),
            SWC.shadow_mats.vbshvegshmat.astype(np.float32),
            SWC.environ_data.altitude[idx],
            SWC.environ_data.azimuth[idx],
            SWC.shadow_mats.asvf.astype(np.float32),
            SWC.config.person_cylinder,
            esky,
            L_patches.astype(np.float32),
            0,  # wall scheme,
            SWC.walls_data.voxelTable.astype(np.float32) if SWC.walls_data.voxelTable is not None else None,
            SWC.walls_data.voxelMaps.astype(np.float32) if SWC.walls_data.voxelMaps is not None else None,
            steradians.astype(np.float32),
            Ta,
            Tgwall,
            SWC.params.Emissivity.Value.Walls,
            Lup.astype(np.float32),
            SWC.environ_data.radI[idx],
            SWC.environ_data.radD[idx],
            SWC.environ_data.radG[idx],
            lv.astype(np.float32),
            SWC.params.Albedo.Effective.Value.Walls,
            0,
            SWC.shadow_mats.diffsh.astype(np.float32),
            shadow.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            idx,
        )

    def run_ani_rust():
        return sky.anisotropic_sky(
            SWC.shadow_mats.shmat.astype(np.float32),
            SWC.shadow_mats.vegshmat.astype(np.float32),
            SWC.shadow_mats.vbshvegshmat.astype(np.float32),
            SWC.environ_data.altitude[idx],
            SWC.environ_data.azimuth[idx],
            SWC.shadow_mats.asvf.astype(np.float32),
            SWC.config.person_cylinder,
            esky,
            L_patches.astype(np.float32),
            False,  # wall scheme,
            SWC.walls_data.voxelTable.astype(np.float32) if SWC.walls_data.voxelTable is not None else None,
            SWC.walls_data.voxelMaps.astype(np.float32) if SWC.walls_data.voxelMaps is not None else None,
            steradians.astype(np.float32),
            Ta,
            Tgwall,
            SWC.params.Emissivity.Value.Walls,
            Lup.astype(np.float32),
            SWC.environ_data.radI[idx],
            SWC.environ_data.radD[idx],
            SWC.environ_data.radG[idx],
            lv.astype(np.float32),
            SWC.params.Albedo.Effective.Value.Walls,
            False,
            SWC.shadow_mats.diffsh.astype(np.float32),
            shadow.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            idx,
        )

    py_ani_timings = timeit.repeat(run_ani_py, number=1, repeat=repeats)
    print_timing_stats("anisotropic_sky", py_ani_timings)

    rust_ani_timings = timeit.repeat(run_ani_rust, number=1, repeat=repeats)
    print_timing_stats("sky.anisotropic_sky", rust_ani_timings)

    # Print relative speed as percentage
    relative_speed(py_ani_timings, rust_ani_timings)

    (
        Ldown,
        Lside,
        Lside_sky,
        Lside_veg,
        Lside_sh,
        Lside_sun,
        Lside_ref,
        Least,
        Lwest,
        Lnorth,
        Lsouth,
        Keast,
        Ksouth,
        Kwest,
        Knorth,
        KsideI,
        KsideD,
        Kside,
        steradians,
        skyalt,
    ) = run_ani_py()

    result_ani_py = {
        "Ldown": Ldown,
        "Lside": Lside,
        "Lside_sky": Lside_sky,
        "Lside_veg": Lside_veg,
        "Lside_sh": Lside_sh,
        "Lside_sun": Lside_sun,
        "Lside_ref": Lside_ref,
        "Least": Least,
        "Lwest": Lwest,
        "Lnorth": Lnorth,
        "Lsouth": Lsouth,
        "Keast": Keast,
        "Ksouth": Ksouth,
        "Kwest": Kwest,
        "Knorth": Knorth,
        "KsideI": KsideI,
        "KsideD": KsideD,
        "Kside": Kside,
        "steradians": steradians,
        "skyalt": skyalt,
    }

    result_ani_rust = run_ani_rust()

    key_map = {
        "Ldown": "ldown",
        "Lside": "lside",
        "Lside_sky": "lside_sky",
        "Lside_veg": "lside_veg",
        "Lside_sh": "lside_sh",
        "Lside_sun": "lside_sun",
        "Lside_ref": "lside_ref",
        "Least": "least",
        "Lwest": "lwest",
        "Lnorth": "lnorth",
        "Lsouth": "lsouth",
        "Keast": "keast",
        "Ksouth": "ksouth",
        "Kwest": "kwest",
        "Knorth": "knorth",
        "KsideI": "kside_i",
        "KsideD": "kside_d",
        "Kside": "kside",
        "steradians": "steradians",
        "skyalt": "skyalt",
    }

    # Compare results
    compare_results(result_ani_py, result_ani_rust, key_map)
    # Plot visual residuals
    plot_visual_residuals(Ldown, result_ani_rust.ldown, title_prefix="Ldown")
    plot_visual_residuals(Lside, result_ani_rust.lside, title_prefix="Lside")
    plot_visual_residuals(Lside_sky, result_ani_rust.lside_sky, title_prefix="Lside_sky")
    plot_visual_residuals(Lside_veg, result_ani_rust.lside_veg, title_prefix="Lside_veg")
    plot_visual_residuals(Lside_sh, result_ani_rust.lside_sh, title_prefix="Lside_sh")
    plot_visual_residuals(Lside_sun, result_ani_rust.lside_sun, title_prefix="Lside_sun")
    plot_visual_residuals(Lside_ref, result_ani_rust.lside_ref, title_prefix="Lside_ref")
    plot_visual_residuals(Least, result_ani_rust.least, title_prefix="Least")
    plot_visual_residuals(Lwest, result_ani_rust.lwest, title_prefix="Lwest")
    plot_visual_residuals(Lnorth, result_ani_rust.lnorth, title_prefix="Lnorth")
    plot_visual_residuals(Lsouth, result_ani_rust.lsouth, title_prefix="Lsouth")
    plot_visual_residuals(Keast, result_ani_rust.keast, title_prefix="Keast")
    plot_visual_residuals(Ksouth, result_ani_rust.ksouth, title_prefix="Ksouth")
    plot_visual_residuals(Kwest, result_ani_rust.kwest, title_prefix="Kwest")
    plot_visual_residuals(Knorth, result_ani_rust.knorth, title_prefix="Knorth")
    plot_visual_residuals(KsideI, result_ani_rust.kside_i, title_prefix="KsideI")
    plot_visual_residuals(KsideD, result_ani_rust.kside_d, title_prefix="KsideD")
    plot_visual_residuals(Kside, result_ani_rust.kside, title_prefix="Kside")


def make_test_arrays(
    dsm_path="demos/data/athens/DSM.tif",
    veg_dsm_path="temp/athens/CDSM.tif",
    wall_hts_path="temp/athens/walls/wall_hts.tif",
    wall_aspect_path="temp/athens/walls/wall_aspects.tif",
):
    dsm, dsm_transf, _crs, _nd_val = common.load_raster(dsm_path, bbox=None)
    vegdsm, _transf, _crs, _nd_val = common.load_raster(veg_dsm_path, bbox=None)
    vegdsm2 = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    azi = 45.0
    alt = 30.0
    scale = 1 / dsm_transf[1]
    vegmax = vegdsm.max()
    amaxvalue = dsm.max() - dsm.min()
    amaxvalue = np.maximum(amaxvalue, vegmax)
    bush = np.zeros(dsm.shape, dtype=np.float32)  # Ensure float32
    wall_hts, _transf, _crs, _nd_val = common.load_raster(wall_hts_path, bbox=None)
    wall_asp, _transf, _crs, _nd_val = common.load_raster(wall_aspect_path, bbox=None)

    # Convert all loaded arrays to float32
    dsm = dsm.astype(np.float32)
    vegdsm = vegdsm.astype(np.float32)
    wall_hts = wall_hts.astype(np.float32)
    wall_asp = wall_asp.astype(np.float32)

    return dsm, vegdsm, vegdsm2, azi, alt, scale, amaxvalue, bush, wall_hts, wall_asp


# Calculate and print per-array right percentage
def pct(a, b, atol, rtol):
    if a is None or b is None:
        return float("nan")
    # Ensure shapes match before comparison
    if a.shape != b.shape:
        return f"Shape mismatch: {a.shape} vs {b.shape}"
    return 100.0 * np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True).sum() / a.size


def compare_results(result_py, result_rust, key_map, atol=0.001, rtol=0.001):
    print("\n--- Comparison ---")
    for py_key, rust_attr in key_map.items():
        py_val = result_py.get(py_key)
        rust_val = getattr(result_rust, rust_attr, None)
        match_pct = pct(py_val, rust_val, atol=atol, rtol=rtol)
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

    # Determine the symmetric range for the residuals colormap
    min_extent = 0.001
    residuals = rust_array - py_array
    max_abs_residual = max(np.abs(residuals).max(), min_extent)

    im3 = axes[2].imshow(residuals, cmap=cmap_residuals, vmin=-max_abs_residual, vmax=max_abs_residual)
    cbar3 = fig.colorbar(im3, ax=axes[2], shrink=colorbar_shrink)
    cbar3.ax.tick_params(labelsize=tick_fontsize)
    axes[2].set_title(f"{title_prefix} - Residuals (Rust - Python)")
    axes[2].axis("off")

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.savefig(f"temp/{title_prefix.lower().replace(' ', '_')}_residuals.png", dpi=150)
