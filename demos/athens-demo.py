# %%
"""
Demo: Athens SOLWEIG - Simplified API

This demo shows how to use the solweig package with the new simplified API.
The simplified API automatically handles:
- Wall height and aspect computation from DSM
- Sky View Factor (SVF) calculation on-the-fly
- Extent intersection and resampling
- CRS validation and extraction
- NaN filling in DSM/CDSM/TDSM with ground reference (DEM or DSM)

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
# - NaN in DSM/CDSM/TDSM filled with ground reference (DEM or DSM)
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

# Step 2: Load weather data and location from EPW file
epw_path = str(input_path / "athens_2023.epw")
weather_list = solweig.Weather.from_epw(
    epw_path,
    start="2023-07-01",
    end="2023-07-04",  # 4 days: July 1-4
)
location = solweig.Location.from_epw(epw_path)  # lat, lon, UTC offset, elevation

# %%
# Step 3: Calculate Tmrt with all defaults
# All parameters use bundled defaults:
#  - Human: abs_k=0.7, abs_l=0.95, standing, 75kg, 180cm, 35yo, 80W activity
#  - Physics: Tree transmissivity=0.03, seasonal dates, posture geometry
#  - No materials needed (no landcover grid)
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    use_anisotropic_sky=True,  # Uses SVF (computed automatically if needed)
    conifer=False,  # Use seasonal leaf on/off (set True for evergreen trees)
    output_dir=str(output_dir),
    outputs=["tmrt", "shadow"],
)

print(f"\n✓ Simplified API complete! Processed {len(results)} timesteps.")
print(f"  Mean Tmrt: {results[0].tmrt.mean():.1f}°C")
print(f"\nNote: Preprocessing cached in {output_folder_path / 'working'}")
print("      Use force_recompute=True to regenerate walls/SVF.")
print(f"      Run metadata saved to {output_dir / 'run_metadata.json'}")

# %%
# Optional: Load and inspect run metadata
# This metadata captures all parameters used in the calculation for reproducibility
metadata = solweig.load_run_metadata(output_dir / "run_metadata.json")
print("\nRun metadata loaded:")
print(f"  Timestamp: {metadata['run_timestamp']}")
print(f"  SOLWEIG version: {metadata['solweig_version']}")
print(f"  Location: {metadata['location']['latitude']:.2f}°N, {metadata['location']['longitude']:.2f}°E")
print(f"  Human posture: {metadata.get('human', {}).get('posture', 'default (standing)')}")
print(f"  Anisotropic sky: {metadata['parameters']['use_anisotropic_sky']}")
print(f"  Weather timesteps: {metadata['timeseries']['timesteps']}")
print(f"  Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")

# %%
# Optional parameter customization examples:

# Example 1: Custom human parameters (common use case)
# results = solweig.calculate_timeseries(
#     surface=surface,
#     weather_series=weather_list,
#     human=solweig.HumanParams(
#         abs_k=0.65,       # Lower shortwave absorption
#         abs_l=0.97,       # Higher longwave absorption
#         weight=70,        # 70 kg
#         height=1.65,      # 165 cm
#         posture="sitting",
#     ),
#     output_dir=str(output_dir),
# )

# Example 2: Custom physics (e.g., different tree transmissivity)
# Create custom_trees.json with:
# {
#   "Tree_settings": {"Value": {"Transmissivity": 0.05, ...}},
#   "Posture": {"Standing": {...}, "Sitting": {...}}
# }
# physics = solweig.load_physics("custom_trees.json")
# results = solweig.calculate_timeseries(
#     surface=surface,
#     weather_series=weather_list,
#     physics=physics,
#     output_dir=str(output_dir),
# )

# Example 3: Custom materials (requires landcover grid)
# surface_with_lc = solweig.SurfaceData.prepare(
#     dsm="dsm.tif",
#     land_cover="landcover.tif",  # Grid with class IDs (0-7, 99-102)
#     working_dir="cache/",
# )
# materials = solweig.load_materials("site_materials.json")  # Albedo, emissivity per class
# results = solweig.calculate_timeseries(
#     surface=surface_with_lc,
#     weather_series=weather_list,
#     materials=materials,
#     output_dir=str(output_dir),
# )

# Legacy: Old unified params file (still supported for backwards compatibility)
# params = solweig.load_params("parametersforsolweig.json")
# Contains human + physics + materials in one file
# Note: Prefer the new three-parameter model for clarity!

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
# NOTE: Legacy API (SolweigRunRust, SolweigRunCore, configs.py) removed in Phase 5.6
# =============================================================================
# The legacy config-file-driven API has been removed. Use the modern simplified API above.
# For tiled processing of large rasters, use:
#
# results = solweig.calculate_tiled(
#     surface=surface,
#     location=location,
#     weather=weather,
#     tile_size=256,  # Tile size in pixels
#     overlap=50,     # Overlap in pixels for shadow continuity
#     output_dir=str(output_dir),
# )
#
# Performance: The modern API with Rust algorithms is comparable to the old runner.

# %%
