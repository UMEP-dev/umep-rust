# Quick Start Guide

Get up and running with SOLWEIG in 5 minutes.

## Installation

```bash
pip install solweig
```

For full features including POI analysis and fast raster I/O:

```bash
pip install solweig[full]
```

## Basic Workflow

SOLWEIG calculates Mean Radiant Temperature (Tmrt) and thermal comfort indices (UTCI, PET) from urban surface data and weather conditions.

### Step 1: Prepare Your Data

You'll need:

- **DSM** (Digital Surface Model) - GeoTIFF showing building and ground heights
- **Weather data** - EPW file with hourly temperature, humidity, radiation, wind
- **Pre-computed preprocessing** - Wall heights/aspects and Sky View Factor (see below)

### Step 2: Preprocess (One-time Setup)

Preprocessing generates wall geometries and sky view factors. This is done once per DSM:

```python
import solweig

# Generate wall heights and aspects (~30 seconds for 500x500 grid)
solweig.walls.generate_wall_hts(
    dsm_path="path/to/dsm.tif",
    out_dir="preprocessed/walls/"
)

# Generate Sky View Factor (~3 minutes for 500x500 grid)
solweig.svf.generate_svf(
    dsm_path="path/to/dsm.tif",
    cdsm_path="path/to/cdsm.tif",  # Optional: vegetation heights
    out_dir="preprocessed/svf/",
    trans_veg_perc=3,  # Vegetation transmissivity %
)
```

**Performance estimates:**

| Grid Size | Wall Generation | SVF Generation | Total Time |
| --------- | --------------- | -------------- | ---------- |
| 250×250   | ~5 sec          | ~30 sec        | ~1 min     |
| 500×500   | ~15 sec         | ~3 min         | ~3-4 min   |
| 1000×1000 | ~45 sec         | ~15 min        | ~15-20 min |

_With GPU acceleration. CPU-only may be 2-3x slower._

### Step 3: Calculate Tmrt

Once preprocessing is complete, calculating Tmrt is fast:

```python
import solweig
from datetime import datetime

# Load surface data (returns tuple: surface, precomputed)
surface, precomputed = solweig.SurfaceData.from_geotiff(
    dsm="path/to/dsm.tif",
    cdsm="path/to/cdsm.tif",  # Optional: vegetation
    walls_dir="preprocessed/walls/",
    svf_dir="preprocessed/svf/",
)

# Auto-extract location from DSM
location = solweig.Location.from_dsm_crs("path/to/dsm.tif", utc_offset=2)

# Load weather data
weather_list = solweig.Weather.from_epw(
    "path/to/weather.epw",
    start="2023-07-01",
    end="2023-07-01",
    hours=[12, 13, 14, 15],  # Optional: specific hours only
)

# Calculate timeseries with auto-save
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    precomputed=precomputed,
    output_dir="output/",  # Auto-saves results incrementally
    outputs=["tmrt", "utci"],  # Which outputs to save
)

# Access results (NumPy arrays) - files already saved!
print(f"Mean Tmrt: {results[0].tmrt.mean():.1f}°C")
print(f"Mean UTCI: {results[0].utci.mean():.1f}°C")
```

**For single timestep (no auto-save):**

```python
# Single timestep - use calculate() instead
result = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list[0],
    precomputed=precomputed,
)

# Manually save if needed
result.to_geotiff(output_dir="output/", outputs=["tmrt", "utci"], surface=surface)
```

## Complete Example (4 Lines)

Once preprocessing is done, the core workflow is just 4 lines:

```python
import solweig

surface = solweig.SurfaceData.from_geotiff(dsm="dsm.tif", walls_dir="walls/", svf_dir="svf/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01", end="2023-07-01")
result = solweig.calculate(surface, weather)
result.to_geotiff("output/")
```

## Project API (Even Simpler)

For complete workflows, the Project API manages everything automatically:

```python
import solweig

# Create project - auto-discovers preprocessing in cache_dir
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",  # Preprocessing goes here
)

# Check status
project.print_status()

# Calculate (auto-prepares if needed)
results = project.calculate(
    start="2023-07-01",
    end="2023-07-01",
    auto_prepare=True,  # Generates preprocessing if missing
)
```

**Benefits:**

- Automatic path management and discovery
- Auto-preparation of missing preprocessing
- Status checking: `project.print_status()`
- Save/load project configuration

**With explicit paths:**

```python
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    walls_dir="preprocessed/walls/",  # Explicit override
    svf_dir="preprocessed/svf/",      # Explicit override
)
```

## Working with Sample Data

Download pre-computed demo data to get started instantly:

```python
import solweig

# Download Athens demo with pre-computed preprocessing
solweig.download_sample_data(output_dir="demo/")

# Use it immediately (no preprocessing needed!)
surface = solweig.SurfaceData.from_geotiff(
    dsm="demo/dsm.tif",
    walls_dir="demo/preprocessed/walls/",
    svf_dir="demo/preprocessed/svf/",
)
weather = solweig.Weather.from_epw("demo/weather.epw", start="2023-07-15", end="2023-07-15")
result = solweig.calculate(surface, weather)
```

## Next Steps

- **[User Guide](../user-guide/preprocessing.md)** - Detailed preprocessing instructions
- **[Model Options](../user-guide/model-options.md)** - Configure anisotropic sky, vegetation schemes
- **[API Reference](../api/reference.md)** - Complete API documentation
- **[QGIS Plugin](../qgis/installation.md)** - Use SOLWEIG in QGIS with GUI

## Common Issues

### "SVF data not found"

You need to run preprocessing first. SVF generation takes 3-20 minutes depending on grid size.

### "No module named 'rasterio'"

Install the full package: `pip install solweig[full]`

Or use GDAL backend: `export UMEP_USE_GDAL=1`

### GPU not available

The package automatically falls back to CPU. GPU acceleration is optional.

Check GPU status:

```python
import solweig
print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Using backend: {solweig.get_compute_backend()}")
```

## Getting Help

- **GitHub Issues**: [github.com/UMEP-dev/solweig/issues](https://github.com/UMEP-dev/solweig/issues)
- **Documentation**: [umep-docs.readthedocs.io](https://umep-docs.readthedocs.io)
- **Mailing List**: <umep-dev@googlegroups.com>
