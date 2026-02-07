# %%
"""
Demo: Gothenburg SOLWEIG preprocessing

This demo shows how to use the solweig package for:
1. Wall height and aspect generation
2. Sky View Factor (SVF) calculation

Uses SurfaceData.prepare() which automatically computes and caches
walls and SVF in the working directory.
"""

from pathlib import Path

import solweig

# %%
# Working folder and input files
working_folder = "temp/goteborg"
working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)

# Input files
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"

# Setup parameters
trunk_ratio = 0.25  # Trunk height as fraction of canopy height

# %%
# Prepare surface data with automatic wall and SVF computation
# SurfaceData.prepare() will:
#   - Compute wall heights/aspects and cache in working_dir/walls/
#   - Compute SVF and cache in working_dir/svf/
#   - Reuse cached data on subsequent runs (use force_recompute=True to regenerate)
print("Preparing surface data (walls and SVF will be computed if not cached)...")
print(f"  Working dir: {working_path}")
print(f"GPU acceleration: {'enabled' if solweig.GPU_ENABLED else 'disabled'}")

surface = solweig.SurfaceData.prepare(
    dsm=dsm_path,
    cdsm=cdsm_path,
    working_dir=str(working_path),
    trunk_ratio=trunk_ratio,
    # bbox=None,  # Full extent (default)
    # force_recompute=False,  # Use cached data if available (default)
)

print("\nPreprocessing complete!")
print(f"  DSM shape: {surface.dsm.shape}")
print(f"  Walls cached: {working_path}/walls/")
print(f"  SVF cached: {working_path}/svf/")

# %%
# The surface object is now ready for SOLWEIG calculations:
#
weather_list = solweig.Weather.from_umep_met(
    "demos/data/Goteborg_SWEREF99_1200/GBG_TMY_1977.txt",
    start="1975-07-01",
    end="1975-07-02",  # 2 days: July 1-2
)
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    output_dir=str(working_path / "output"),
)

# %%
