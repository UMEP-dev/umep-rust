"""
Visual Comparison Test: API vs Runner

Generates side-by-side comparison plots for visual inspection.
Run with: pytest tests/test_api_visual_comparison.py -v -s

Plots saved to: temp/visual_comparison/
"""

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest


def plot_comparison(runner_arr, api_arr, name, output_dir):
    """Generate side-by-side comparison with residual plot."""
    if runner_arr is None or api_arr is None:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Runner
    im1 = axes[0].imshow(runner_arr, cmap="viridis")
    fig.colorbar(im1, ax=axes[0], shrink=0.6)
    axes[0].set_title(f"{name} - Runner")
    axes[0].axis("off")

    # API
    im2 = axes[1].imshow(api_arr, cmap="viridis")
    fig.colorbar(im2, ax=axes[1], shrink=0.6)
    axes[1].set_title(f"{name} - API")
    axes[1].axis("off")

    # Residuals
    residuals = api_arr - runner_arr
    max_abs = max(np.nanmax(np.abs(residuals)), 0.001)
    im3 = axes[2].imshow(residuals, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    fig.colorbar(im3, ax=axes[2], shrink=0.6)
    axes[2].set_title(f"{name} - Residuals (API - Runner)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name.lower()}_comparison.png", dpi=150)
    plt.close()


def compare_arrays(name, runner_arr, api_arr, atol=0.5):
    """Compare arrays and return statistics."""
    valid = np.isfinite(runner_arr) & np.isfinite(api_arr)
    if valid.sum() == 0:
        return {"name": name, "status": "SKIP"}

    diff = api_arr[valid] - runner_arr[valid]
    mae = np.abs(diff).mean()
    bias = diff.mean()
    match_pct = 100.0 * np.isclose(runner_arr[valid], api_arr[valid], atol=atol).sum() / valid.sum()

    print(f"{name}:")
    print(f"  Runner: mean={np.nanmean(runner_arr):.2f}, std={np.nanstd(runner_arr):.2f}")
    print(f"  API:    mean={np.nanmean(api_arr):.2f}, std={np.nanstd(api_arr):.2f}")
    print(f"  MAE={mae:.4f}, Bias={bias:+.4f}, Match={match_pct:.1f}%")

    return {"name": name, "mae": mae, "bias": bias, "match_pct": match_pct}


class TestVisualComparison:
    """Generate visual comparison plots for API vs Runner."""

    def test_visual_parity(self):
        """
        Compare all outputs between API and runner, generating visual plots.

        This test is primarily for visual inspection of residuals.
        """
        from solweig.solweig_runner_rust import SolweigRunRust
        from solweig.api import (
            SurfaceData, Location, Weather, HumanParams,
            SvfArrays, ShadowArrays, PrecomputedData, calculate_timeseries,
            load_params,
        )
        import rasterio

        output_dir = "temp/visual_comparison"
        temp_dir = tempfile.mkdtemp(prefix="solweig_visual_")

        try:
            # Load params
            params = load_params("tests/rustalgos/test_params_solweig.json")

            # Initialize and run runner
            SWR = SolweigRunRust(
                config_path_str="tests/rustalgos/test_config_solweig.ini",
                params_json_path="tests/rustalgos/test_params_solweig.json",
            )
            SWR.config.output_dir = temp_dir
            SWR.config.output_tmrt = True
            SWR.config.output_kdown = True
            SWR.config.output_kup = True
            SWR.config.output_ldown = True
            SWR.config.output_lup = True
            SWR.config.output_shadow = True
            SWR.run()

            # Find noon timestep
            noon_idx = None
            for i in range(len(SWR.environ_data.hours)):
                if SWR.environ_data.hours[i] == 12:
                    noon_idx = i
                    break

            if noon_idx is None:
                pytest.skip("No noon timestep")

            # Load runner outputs
            def load_tif(pattern):
                files = list(Path(temp_dir).glob(f"{pattern}_*_1200D.tif"))
                if not files:
                    return None
                with rasterio.open(files[0]) as src:
                    data = src.read(1)
                    return np.where(data < -100, np.nan, data)

            runner_tmrt = load_tif("Tmrt")
            runner_kdown = load_tif("Kdown")
            runner_kup = load_tif("Kup")
            runner_ldown = load_tif("Ldown")
            runner_lup = load_tif("Lup")
            runner_shadow = load_tif("Shadow")

            # Build API inputs
            land_cover = SWR.raster_data.lcgrid.astype(np.uint8) if SWR.config.use_landcover else None

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
            posture_val = SWR.params.Posture.Standing.Value if posture == "Standing" else SWR.params.Posture.Sitting.Value
            human = HumanParams(
                posture="standing" if posture == "Standing" else "sitting",
                height=posture_val.height,
                abs_k=SWR.params.Tmrt_params.Value.absK,
                abs_l=SWR.params.Tmrt_params.Value.absL,
            )

            # Build precomputed data
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

            shadow_arrays = None
            use_anisotropic = bool(SWR.config.use_aniso)
            if hasattr(SWR, "shadow_mats") and SWR.shadow_mats is not None:
                shadow_arrays = ShadowArrays(
                    _shmat_u8=SWR.shadow_mats.shmat,
                    _vegshmat_u8=SWR.shadow_mats.vegshmat,
                    _vbshmat_u8=SWR.shadow_mats.vbshvegshmat,
                )

            precomputed = PrecomputedData(svf=svf_arrays, shadow_matrices=shadow_arrays)

            # Build weather series
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

            # Run API
            results = calculate_timeseries(
                surface=surface,
                location=location,
                weather_series=weather_series,
                human=human,
                precomputed=precomputed,
                use_anisotropic_sky=use_anisotropic,
                params=params,
            )

            api_result = results[noon_idx]

            # Compare and plot
            print("\n" + "=" * 60)
            print("VISUAL COMPARISON (API vs Runner at Noon)")
            print("=" * 60)

            comparisons = []
            for name, runner_arr, api_arr, atol in [
                ("Tmrt", runner_tmrt, api_result.tmrt, 0.5),
                ("Kdown", runner_kdown, api_result.kdown, 5.0),
                ("Kup", runner_kup, api_result.kup, 5.0),
                ("Ldown", runner_ldown, api_result.ldown, 5.0),
                ("Lup", runner_lup, api_result.lup, 5.0),
                ("Shadow", runner_shadow, api_result.shadow, 0.01),
            ]:
                if runner_arr is not None and api_arr is not None:
                    comparisons.append(compare_arrays(name, runner_arr, api_arr, atol=atol))
                    plot_comparison(runner_arr, api_arr, name, output_dir)

            print(f"\nPlots saved to: {output_dir}/")

            # Summary
            print("\n--- Summary ---")
            for c in sorted(comparisons, key=lambda x: abs(x.get("bias", 0)), reverse=True):
                print(f"  {c['name']:8s}: Bias={c['bias']:+.4f}, MAE={c['mae']:.4f}, Match={c['match_pct']:.1f}%")

            # Assert Tmrt is close
            tmrt_cmp = next(c for c in comparisons if c["name"] == "Tmrt")
            assert abs(tmrt_cmp["bias"]) < 0.5, f"Tmrt bias {tmrt_cmp['bias']:.4f}°C exceeds 0.5°C"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
