# %%
"""
Demo: Athens SOLWEIG - Simplified API

This demo shows how to use the solweig package with the new simplified API.
The simplified API automatically handles:
- Wall height and aspect computation from DSM
- Sky View Factor (SVF) calculation on-the-fly
- Extent intersection and resampling
- CRS validation and extraction

For config file-driven workflows, see the legacy API at the bottom.
"""

from pathlib import Path

import geopandas as gpd
import solweig
from pyproj import CRS

# Working folders
input_folder = "demos/data/athens"
input_path = Path(input_folder).absolute()
output_folder = "temp/athens"
output_folder_path = Path(output_folder).absolute()
output_folder_path.mkdir(parents=True, exist_ok=True)
output_dir = output_folder_path / "output_simplified"

# Extents for Athens demo area
EXTENTS_BBOX = [476800, 4205850, 477200, 4206250]
TARGET_CRS = 2100

# %%
# =============================================================================
# SIMPLIFIED API (Recommended)
# =============================================================================

# Generate CDSM from tree vector data
trees_gdf = gpd.read_file(input_folder + "/trees.gpkg")
trees_gdf = trees_gdf.to_crs(TARGET_CRS)
cdsm_rast, cdsm_transf = solweig.io.rasterise_gdf(
    trees_gdf,
    "geometry",
    "height",
    bbox=EXTENTS_BBOX,
    pixel_size=1.0,
)
solweig.io.save_raster(
    str(output_folder_path / "CDSM.tif"),
    cdsm_rast,
    cdsm_transf.to_gdal(),
    CRS.from_epsg(TARGET_CRS).to_wkt(),
)

# %%
# Step 1: Prepare surface data
# - CRS automatically extracted from DSM
# - Walls and SVF computed and cached to working_dir if not provided
# - Extent and resolution handled automatically
# - Resampled data saved to working_dir for inspection
surface = solweig.SurfaceData.prepare(
    dsm=str(input_path / "DSM.tif"),
    working_dir=str(output_folder_path / "working"),  # Cache preprocessing here
    cdsm=str(output_folder_path / "CDSM.tif"),
    bbox=EXTENTS_BBOX,  # Optional: specify extent
    pixel_size=1.0,  # Optional: specify resolution (default: from DSM)
)

# Step 2: Load weather data from EPW file
weather_list = solweig.Weather.from_epw(
    str(input_path / "athens_2023.epw"),
    start="2023-07-01",
    end="2023-07-03",  # 3 days: July 1-3
)

# %%
# Step 3: Calculate and auto-save results
# Location is auto-extracted from surface CRS metadata
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    use_anisotropic_sky=True,  # Uses SVF (computed automatically if needed)
    output_dir=str(output_dir),
    outputs=["tmrt", "shadow"],
)

print(f"\n✓ Simplified API complete! Processed {len(results)} timesteps.")
print(f"  Mean Tmrt: {results[0].tmrt.mean():.1f}°C")
print(f"\nNote: Preprocessing cached in {output_folder_path / 'working'}")
print("      Use force_recompute=True to regenerate walls/SVF.")

# %%
# Step 4: Post-process thermal comfort indices (UTCI/PET)
# UTCI and PET are computed separately for better performance
# This allows you to:
# - Skip thermal comfort if you only need Tmrt
# - Compute for subset of timesteps
# - Compute for different human parameters without re-running main calculation

# Compute UTCI (fast polynomial, ~1 second for full timeseries)
utci_dir = output_folder_path / "output_utci"
n_utci = solweig.compute_utci(
    tmrt_dir=str(output_dir),
    weather_series=weather_list,
    output_dir=str(utci_dir),
)
print(f"\n✓ UTCI post-processing complete! Processed {n_utci} timesteps.")

# Compute PET (slower iterative solver, optional)
# pet_dir = output_folder_path / "output_pet"
# n_pet = solweig.compute_pet(
#     tmrt_dir=str(output_dir),
#     weather_series=weather_list,
#     output_dir=str(pet_dir),
#     human=solweig.HumanParams(weight=75, height=1.75),
# )
# print(f"\n✓ PET post-processing complete! Processed {n_pet} timesteps.")

# %%
# =============================================================================
# LEGACY API (for backwards compatibility)
# =============================================================================
# This is the original API using config files. Maintained for:
# - Backwards compatibility with existing workflows
# - Comparison with simplified API

# Pre-generate walls and SVF (required for legacy API)
solweig.walls.generate_wall_hts(
    dsm_path=str(input_path / "DSM.tif"),
    bbox=EXTENTS_BBOX,
    out_dir=str(output_folder_path / "walls"),
)

solweig.svf.generate_svf(
    dsm_path=str(input_path / "DSM.tif"),
    bbox=EXTENTS_BBOX,
    out_dir=str(output_folder_path / "svf"),
    cdsm_path=str(output_folder_path / "CDSM.tif"),
    trans_veg_perc=3,
)

# Uncomment to run with legacy API:

# Rust-optimized runner (fast)
SRR = solweig.SolweigRunRust(
    "demos/data/athens/configsolweig.ini",
    "demos/data/athens/parametersforsolweig.json",
    use_tiled_loading=False,
    tile_size=200,
)
SRR.run()
# # Performance: ~1.63 steps/s on Athens demo (400x400 grid, 72 timesteps)

# Pure Python runner (for debugging or environments without Rust)
# SRC = solweig.SolweigRunCore(
#     "demos/data/athens/configsolweig.ini",
#     "demos/data/athens/parametersforsolweig.json",
#     use_tiled_loading=False,
# )
# SRC.run()
# # Performance: ~4.02 s/step on Athens demo (slower than Rust)

# %%
