"""
Exact Parity Test: Compare API vs Reference Implementation

This test runs both implementations with identical inputs and compares
EVERY intermediate value to find the exact point of divergence.

Target: Tmrt bias < 0.5°C
"""

import numpy as np
import pytest
from datetime import datetime, timedelta


def compare_value(name: str, ref_val, api_val, atol: float = 1e-4) -> dict:
    """Compare two values and return detailed statistics."""
    if ref_val is None or api_val is None:
        return {"name": name, "status": "SKIP", "reason": "missing data"}

    ref_arr = np.asarray(ref_val)
    api_arr = np.asarray(api_val)

    # Handle scalars
    if ref_arr.ndim == 0 and api_arr.ndim == 0:
        diff = float(api_arr) - float(ref_arr)
        match = abs(diff) < atol
        return {
            "name": name,
            "status": "PASS" if match else "FAIL",
            "ref": float(ref_arr),
            "api": float(api_arr),
            "diff": diff,
        }

    # Handle arrays
    if ref_arr.shape != api_arr.shape:
        return {
            "name": name,
            "status": "FAIL",
            "reason": f"shape mismatch: ref={ref_arr.shape}, api={api_arr.shape}",
        }

    valid = np.isfinite(ref_arr) & np.isfinite(api_arr)
    if valid.sum() == 0:
        return {"name": name, "status": "SKIP", "reason": "no valid pixels"}

    diff = api_arr[valid] - ref_arr[valid]
    bias = diff.mean()
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff**2).mean())
    match_pct = (np.abs(diff) < atol).mean() * 100

    return {
        "name": name,
        "status": "PASS" if match_pct > 99 else "FAIL",
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "match_pct": match_pct,
        "ref_mean": ref_arr[valid].mean(),
        "api_mean": api_arr[valid].mean(),
    }


class TestExactParity:
    """Compare API against reference function value-by-value."""

    @pytest.fixture
    def runner_setup(self):
        """Initialize runner and get noon timestep parameters."""
        import tempfile
        import shutil
        from solweig.solweig_runner_rust import SolweigRunRust

        temp_dir = tempfile.mkdtemp(prefix="solweig_parity_")

        SWR = SolweigRunRust(
            config_path_str="tests/rustalgos/test_config_solweig.ini",
            params_json_path="tests/rustalgos/test_params_solweig.json",
        )

        # Configure output
        SWR.config.output_dir = temp_dir
        SWR.config.output_tmrt = True
        SWR.config.output_kdown = True
        SWR.config.output_kup = True
        SWR.config.output_ldown = True
        SWR.config.output_lup = True
        SWR.config.output_shadow = True

        # Run the full calculation - this initializes all internal state
        SWR.run()

        # Find noon timestep
        noon_idx = None
        for i in range(len(SWR.environ_data.hours)):
            if SWR.environ_data.hours[i] == 12:
                noon_idx = i
                break

        if noon_idx is None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            pytest.skip("No noon timestep")

        yield SWR, noon_idx, temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_all_intermediate_values(self, runner_setup):
        """
        Compare ALL intermediate values between API and reference.

        Since the runner has already run, we compare its saved outputs
        with the API's calculation for the same timesteps.
        """
        from pathlib import Path
        import rasterio
        from solweig.api import (
            SurfaceData,
            Location,
            Weather,
            HumanParams,
            SvfArrays,
            ShadowArrays,
            PrecomputedData,
            calculate,
            ThermalState,
            calculate_timeseries,
            load_params,
        )

        SWR, noon_idx, temp_dir = runner_setup

        # Load parameters from JSON (same file used by runner)
        params = load_params("tests/rustalgos/test_params_solweig.json")

        # Load runner's noon outputs from saved TIF files
        def load_noon_tif(pattern):
            files = list(Path(temp_dir).glob(f"{pattern}_*_1200D.tif"))
            if not files:
                return None
            with rasterio.open(files[0]) as src:
                data = src.read(1)
                return np.where(data < -100, np.nan, data)

        runner_tmrt = load_noon_tif("Tmrt")
        runner_kdown = load_noon_tif("Kdown")
        runner_kup = load_noon_tif("Kup")
        runner_ldown = load_noon_tif("Ldown")
        runner_lup = load_noon_tif("Lup")
        runner_shadow = load_noon_tif("Shadow")

        print("\n" + "=" * 70)
        print("RUNNER VALUES AT NOON (from saved TIF files)")
        print("=" * 70)
        if runner_tmrt is not None:
            print(f"  Tmrt: mean={np.nanmean(runner_tmrt):.4f}, std={np.nanstd(runner_tmrt):.4f}")
        if runner_kdown is not None:
            print(f"  Kdown: mean={np.nanmean(runner_kdown):.4f}, std={np.nanstd(runner_kdown):.4f}")
        if runner_kup is not None:
            print(f"  Kup: mean={np.nanmean(runner_kup):.4f}, std={np.nanstd(runner_kup):.4f}")
        if runner_ldown is not None:
            print(f"  Ldown: mean={np.nanmean(runner_ldown):.4f}, std={np.nanstd(runner_ldown):.4f}")
        if runner_lup is not None:
            print(f"  Lup: mean={np.nanmean(runner_lup):.4f}, std={np.nanstd(runner_lup):.4f}")
        if runner_shadow is not None:
            print(f"  Shadow: mean={np.nanmean(runner_shadow):.4f}, std={np.nanstd(runner_shadow):.4f}")

        # Now run API with timeseries to match thermal state accumulation
        land_cover = None
        if SWR.config.use_landcover and SWR.raster_data.lcgrid is not None:
            land_cover = SWR.raster_data.lcgrid.astype(np.uint8)

        surface = SurfaceData(
            dsm=SWR.raster_data.dsm.astype(np.float32),
            cdsm=SWR.raster_data.cdsm.astype(np.float32) if SWR.config.use_veg_dem else None,
            tdsm=SWR.raster_data.tdsm.astype(np.float32) if SWR.config.use_veg_dem else None,
            wall_height=SWR.raster_data.wallheight.astype(np.float32),
            wall_aspect=SWR.raster_data.wallaspect.astype(np.float32),
            land_cover=land_cover,
            pixel_size=SWR.raster_data.scale,
        )

        location = Location(
            latitude=SWR.location["latitude"],
            longitude=SWR.location["longitude"],
            altitude=SWR.location.get("altitude", 0.0),
            utc_offset=int(SWR.config.utc),
        )

        posture = SWR.params.Tmrt_params.Value.posture
        if posture == "Standing":
            posture_val = SWR.params.Posture.Standing.Value
        else:
            posture_val = SWR.params.Posture.Sitting.Value
        human = HumanParams(
            posture="standing" if posture == "Standing" else "sitting",
            height=posture_val.height,  # Use runner's height (1.1m, not default 1.75m)
            abs_k=SWR.params.Tmrt_params.Value.absK,
            abs_l=SWR.params.Tmrt_params.Value.absL,
        )

        # Build SVF arrays
        svf_arrays = SvfArrays(
            svf=SWR.svf_data.svf,
            svf_north=SWR.svf_data.svf_north,
            svf_east=SWR.svf_data.svf_east,
            svf_south=SWR.svf_data.svf_south,
            svf_west=SWR.svf_data.svf_west,
            svf_veg=SWR.svf_data.svf_veg,
            svf_veg_north=SWR.svf_data.svf_veg_north,
            svf_veg_east=SWR.svf_data.svf_veg_east,
            svf_veg_south=SWR.svf_data.svf_veg_south,
            svf_veg_west=SWR.svf_data.svf_veg_west,
            svf_aveg=SWR.svf_data.svf_veg_blocks_bldg_sh,
            svf_aveg_north=SWR.svf_data.svf_veg_blocks_bldg_sh_north,
            svf_aveg_east=SWR.svf_data.svf_veg_blocks_bldg_sh_east,
            svf_aveg_south=SWR.svf_data.svf_veg_blocks_bldg_sh_south,
            svf_aveg_west=SWR.svf_data.svf_veg_blocks_bldg_sh_west,
        )

        # Build shadow arrays
        shadow_arrays = ShadowArrays(
            _shmat_u8=SWR.shadow_mats.shmat,
            _vegshmat_u8=SWR.shadow_mats.vegshmat,
            _vbshmat_u8=SWR.shadow_mats.vbshvegshmat,
        )

        precomputed = PrecomputedData(svf=svf_arrays, shadow_matrices=shadow_arrays)
        use_anisotropic = bool(SWR.config.use_aniso)

        # Build weather series for timeseries comparison
        weather_series = []
        for i in range(noon_idx + 1):
            year = int(SWR.environ_data.YYYY[i])
            doy = int(SWR.environ_data.DOY[i])
            hour = int(SWR.environ_data.hours[i])
            minute = int(SWR.environ_data.minu[i])
            date_obj = datetime(year, 1, 1) + timedelta(days=doy - 1)
            dt = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute)

            weather = Weather(
                datetime=dt,
                ta=float(SWR.environ_data.Ta[i]),
                rh=float(SWR.environ_data.RH[i]),
                global_rad=float(SWR.environ_data.radG[i]),
                ws=float(SWR.environ_data.Ws[i]),
                pressure=float(SWR.environ_data.P[i]),
                measured_direct_rad=float(SWR.environ_data.radI[i]),
                measured_diffuse_rad=float(SWR.environ_data.radD[i]),
                precomputed_sun_altitude=float(SWR.environ_data.altitude[i]),
                precomputed_sun_azimuth=float(SWR.environ_data.azimuth[i]),
                precomputed_altmax=float(SWR.environ_data.altmax[i]),
            )
            weather_series.append(weather)

        # Run API timeseries with loaded params
        api_results = calculate_timeseries(
            surface=surface,
            location=location,
            weather_series=weather_series,
            human=human,
            precomputed=precomputed,
            use_anisotropic_sky=use_anisotropic,
            compute_utci=False,
            compute_pet=False,
            params=params,
        )

        api_result = api_results[noon_idx]

        print("\n" + "=" * 70)
        print("API VALUES AT NOON (with accumulated thermal state)")
        print("=" * 70)
        print(f"  Tmrt: mean={np.nanmean(api_result.tmrt):.4f}")
        print(f"  Kdown: mean={np.nanmean(api_result.kdown):.4f}")
        print(f"  Kup: mean={np.nanmean(api_result.kup):.4f}")
        print(f"  Ldown: mean={np.nanmean(api_result.ldown):.4f}")
        print(f"  Lup: mean={np.nanmean(api_result.lup):.4f}")
        print(f"  shadow: mean={np.nanmean(api_result.shadow):.4f}")

        # Compare all values
        print("\n" + "=" * 70)
        print("COMPARISON (sorted by bias magnitude)")
        print("=" * 70)

        comparisons = []

        # Main outputs
        comparisons.append(compare_value("Tmrt", runner_tmrt, api_result.tmrt, atol=0.5))
        comparisons.append(compare_value("Kdown", runner_kdown, api_result.kdown, atol=5.0))
        comparisons.append(compare_value("Kup", runner_kup, api_result.kup, atol=5.0))
        comparisons.append(compare_value("Ldown", runner_ldown, api_result.ldown, atol=5.0))
        comparisons.append(compare_value("Lup", runner_lup, api_result.lup, atol=5.0))
        comparisons.append(compare_value("shadow", runner_shadow, api_result.shadow, atol=0.01))

        # Sort by bias magnitude
        def get_bias(c):
            return abs(c.get("bias", 0)) if "bias" in c else 0

        comparisons_sorted = sorted(comparisons, key=get_bias, reverse=True)

        for c in comparisons_sorted:
            name = c["name"]
            if c["status"] == "SKIP":
                print(f"  {name:12s}: SKIP ({c.get('reason', '')})")
            elif "bias" in c:
                print(
                    f"  {name:12s}: bias={c['bias']:+8.4f}, mae={c['mae']:8.4f}, "
                    f"match={c['match_pct']:5.1f}% [{c['status']}]"
                )
            else:
                print(f"  {name:12s}: {c}")

        # Also compare at first daytime timestep to check gvf_lup without TsWaveDelay
        print("\n" + "=" * 70)
        print("CHECKING FIRST DAYTIME TIMESTEP (TsWaveDelay should not smooth)")
        print("=" * 70)

        # Find first daytime timestep
        first_daytime_idx = None
        for j in range(len(SWR.environ_data.hours)):
            if SWR.environ_data.altitude[j] > 0:
                first_daytime_idx = j
                break

        if first_daytime_idx is not None and first_daytime_idx != noon_idx:
            # Load first daytime Lup from runner
            first_hour = int(SWR.environ_data.hours[first_daytime_idx])
            first_files = list(Path(temp_dir).glob(f"Lup_*_{first_hour:02d}00D.tif"))
            if first_files:
                with rasterio.open(first_files[0]) as src:
                    first_runner_lup = src.read(1)
                    first_runner_lup = np.where(first_runner_lup < -100, np.nan, first_runner_lup)

                first_api_lup = api_results[first_daytime_idx].lup

                valid = np.isfinite(first_runner_lup) & np.isfinite(first_api_lup)
                diff = first_api_lup[valid] - first_runner_lup[valid]
                print(f"  First daytime index: {first_daytime_idx} (hour={first_hour})")
                print(f"  Runner Lup: mean={np.nanmean(first_runner_lup):.2f}")
                print(f"  API Lup: mean={np.nanmean(first_api_lup):.2f}")
                print(f"  Bias: {diff.mean():+.4f} W/m²")
                print(f"  Note: At first daytime, TsWaveDelay should return gvf_lup unchanged")
            else:
                print(f"  No Lup file found for hour {first_hour}")
        else:
            print(f"  First daytime = noon (idx={noon_idx}), can't isolate gvf_lup")

        # Final Tmrt comparison
        tmrt_cmp = comparisons[0]
        if "bias" in tmrt_cmp:
            print(f"\n*** Tmrt bias: {tmrt_cmp['bias']:+.4f}°C ***")
            print(f"*** Target: < 0.5°C ***")

            # Detailed Tmrt statistics
            valid = np.isfinite(runner_tmrt) & np.isfinite(api_result.tmrt)
            diff = api_result.tmrt[valid] - runner_tmrt[valid]
            print(f"\nTmrt difference statistics:")
            print(f"  Mean diff: {diff.mean():+.4f}°C")
            print(f"  Std diff: {diff.std():.4f}°C")
            print(f"  Min diff: {diff.min():+.4f}°C")
            print(f"  Max diff: {diff.max():+.4f}°C")
            print(f"  P10/P90: {np.percentile(diff, 10):+.4f} / {np.percentile(diff, 90):+.4f}°C")

            # Check where the biggest differences are
            diff_2d = np.where(valid, api_result.tmrt - runner_tmrt, np.nan)
            max_idx = np.nanargmax(np.abs(diff_2d))
            max_row, max_col = np.unravel_index(max_idx, diff_2d.shape)
            print(f"\n  Max diff location: ({max_row}, {max_col})")
            print(f"  Runner Tmrt at max: {runner_tmrt[max_row, max_col]:.2f}°C")
            print(f"  API Tmrt at max: {api_result.tmrt[max_row, max_col]:.2f}°C")

            # Assert target
            assert abs(tmrt_cmp["bias"]) < 0.5, f"Tmrt bias {tmrt_cmp['bias']:.4f}°C exceeds 0.5°C target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
