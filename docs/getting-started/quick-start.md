# Quick Start

This guide walks you through your first SOLWEIG calculation — from raw inputs to a Tmrt map.

## What you'll do

1. Create a surface with buildings
2. Define where and when
3. Calculate Mean Radiant Temperature
4. Interpret the results

## Option A: From numpy arrays (no files needed)

Use this when you want to experiment quickly or don't have GeoTIFF data yet.

```python
import numpy as np
import solweig
from datetime import datetime

# --- 1. Create a surface ---
# A 200×200 m flat area at 2 m elevation, with a 15 m tall building in the centre
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0  # 40×40 m building

surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)  # 1 pixel = 1 metre
# SVF is required before calculate(); compute once and reuse on this surface
surface.compute_svf()

# --- 2. Define location and weather ---
location = solweig.Location(
    latitude=48.8,      # Paris
    longitude=2.3,
    utc_offset=1,       # Central European Time (UTC+1)
)

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),  # 2pm, July 15
    ta=32.0,            # Air temperature (°C)
    rh=40.0,            # Relative humidity (%)
    global_rad=850.0,   # Global horizontal irradiance (W/m²)
)

# --- 3. Calculate ---
result = solweig.calculate(surface, location, weather)

# --- 4. Inspect results ---
print(f"Mean Tmrt:   {result.tmrt.mean():.1f}°C")
print(f"Sunlit Tmrt: {result.tmrt[result.shadow > 0.5].mean():.1f}°C")
print(f"Shaded Tmrt: {result.tmrt[result.shadow < 0.5].mean():.1f}°C")
```

!!! note "SVF is explicit"
    `calculate()` requires SVF to already be available. For array-based workflows, call `surface.compute_svf()` once before the first calculation. For GeoTIFF workflows, `SurfaceData.prepare()` computes/caches SVF for you.
    If you explicitly set `use_anisotropic_sky=True`, shadow matrices must also already be available (prepared via the same preprocessing step).

## Option B: From GeoTIFF files (real-world data)

This is the typical workflow for real projects. You need:

- A **DSM** GeoTIFF (Digital Surface Model — building/terrain heights)
- An **EPW** weather file (standard format, downloadable from climate databases)

```python
import solweig

# --- 1. Load and prepare surface ---
# Walls, sky view factors, and NaN handling are all automatic
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",        # Preprocessing cached here for reuse
    cdsm="data/trees.tif",       # Optional: vegetation canopy heights
)

# --- 2. Load weather and location from EPW ---
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",          # Date range to simulate
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

print(f"Location: {location.latitude:.1f}°N, {location.longitude:.1f}°E")
print(f"Loaded {len(weather_list)} hourly timesteps")

# --- 3. Run timeseries ---
# Results saved as GeoTIFFs; thermal state carried between timesteps
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

print(f"Done — {len(results)} timesteps saved to output/")
```

If disk space is limited, omit `output_dir` and aggregate in memory, or keep
`output_dir` and set `return_results=False` for low-memory streaming. See
[Timeseries](../guide/timeseries.md#choose-an-output-strategy).

### What `prepare()` does behind the scenes

When you call `SurfaceData.prepare()`, it automatically:

1. Loads the DSM (and optional CDSM, DEM, land cover)
2. Fills NaN/nodata values using the ground reference
3. Computes **wall heights and aspects** from the DSM edges
4. Computes **Sky View Factors** (15 directional grids)
5. Caches everything to `working_dir/` so the next run is instant

## Adding thermal comfort

Tmrt tells you how much radiation a person absorbs, but thermal comfort also depends on air temperature, humidity, and wind. UTCI and PET combine all of these.

```python
# From a single-timestep result:
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")

# From saved timeseries files (batch):
solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)
```

| UTCI range | Meaning |
| ---------- | ------- |
| > 46°C | Extreme heat stress |
| 38–46°C | Very strong heat stress |
| 32–38°C | Strong heat stress |
| 26–32°C | Moderate heat stress |
| 9–26°C | No thermal stress |
| < 9°C | Cold stress categories |

## Common setup patterns

Use these patterns when your input data and workflow differ from the basic examples above.

### Surface setup patterns

#### Pattern 1: GeoTIFF workflow (recommended)

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",      # Optional
    dem="data/dem.tif",         # Optional
    working_dir="cache/",
)
```

`prepare()` computes/caches walls and SVF automatically.

#### Pattern 2: In-memory arrays with absolute heights

```python
import numpy as np

dsm_abs = np.array(...)         # Absolute elevation (e.g., m above sea level)
cdsm_abs = np.array(...)        # Optional canopy elevation (absolute)

surface = solweig.SurfaceData(
    dsm=dsm_abs,
    cdsm=cdsm_abs,              # Optional
    dsm_relative=False,
    cdsm_relative=False,
    pixel_size=1.0,
)
surface.compute_svf()           # Required before calculate()
```

#### Pattern 3: In-memory arrays with relative heights

```python
import numpy as np

dsm_rel = np.array(...)         # Height above ground
cdsm_rel = np.array(...)        # Optional canopy height above ground
dem = np.array(...)             # Ground elevation

surface = solweig.SurfaceData(
    dsm=dsm_rel,
    dem=dem,
    cdsm=cdsm_rel,              # Optional
    dsm_relative=True,
    cdsm_relative=True,
    pixel_size=1.0,
)
surface.preprocess()            # Converts relative -> absolute
surface.compute_svf()           # Required before calculate()
```

### Weather setup patterns

#### Pattern 1: Existing EPW file

```python
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
```

#### Pattern 2: Download EPW, then load

```python
epw_path = solweig.download_epw(
    latitude=37.98,
    longitude=23.73,
    output_path="athens.epw",
)
weather_list = solweig.Weather.from_epw(epw_path)
```

#### Pattern 3: Manually create one timestep

```python
from datetime import datetime

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,
    rh=40.0,
    global_rad=850.0,
    wind_speed=2.0,             # Optional but useful for UTCI/PET
)
```

### Location setup patterns

#### Pattern 1: From EPW metadata (recommended with EPW weather)

```python
location = solweig.Location.from_epw("data/weather.epw")
```

#### Pattern 2: From surface CRS

```python
location = solweig.Location.from_surface(surface, utc_offset=1)
```

#### Pattern 3: Manual coordinates

```python
location = solweig.Location(
    latitude=48.8,
    longitude=2.3,
    utc_offset=1,
)
```

!!! warning "Always set `utc_offset` correctly"
    UTC offset directly affects sun position timing and therefore shadows and Tmrt.

## Where to get input data

### DSM (Digital Surface Model)

A raster grid where each pixel contains the height in metres (including buildings and terrain). Common sources:

- **LiDAR point clouds** processed to raster (national mapping agencies often provide these)
- **Photogrammetry** from drone surveys
- **OpenStreetMap building footprints** extruded to heights

### EPW (EnergyPlus Weather)

Hourly weather data in a standard format. Free sources:

- [Climate.OneBuilding.Org](https://climate.onebuilding.org/) — global coverage
- [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) — European Commission tool

You can also download an EPW directly from PVGIS (no API key needed):

```python
# Download weather data for any location
epw_path = solweig.download_epw(
    latitude=37.98,
    longitude=23.73,
    output_path="athens.epw",
)

# Then load it
weather_list = solweig.Weather.from_epw(epw_path)
location = solweig.Location.from_epw(epw_path)
```

!!! note "Data attribution"
    PVGIS weather data is derived from ERA5 reanalysis and contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

### CDSM (Canopy Digital Surface Model)

Vegetation canopy heights, either from LiDAR or by rasterising tree survey data:

```python
import geopandas as gpd

trees = gpd.read_file("trees.gpkg")
cdsm, transform = solweig.io.rasterise_gdf(
    trees, "geometry", "height",
    bbox=[minx, miny, maxx, maxy],
    pixel_size=1.0,
)
```

## Key classes at a glance

| Class | What it holds |
| ----- | ------------- |
| `SurfaceData` | DSM, optional vegetation/DEM/land cover, preprocessed walls and SVF |
| `Location` | Latitude, longitude, altitude, UTC offset |
| `Weather` | Air temperature, humidity, radiation for one timestep |
| `HumanParams` | Body parameters for Tmrt/PET (optional — sensible defaults provided) |
| `SolweigResult` | Output grids: Tmrt, shadow, radiation components |

## Complete working demos

The repository includes full end-to-end demos you can run directly:

- **[demos/athens-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/athens-demo.py)** — Full workflow: rasterise tree vectors, load GeoTIFFs, run a multi-day timeseries, post-process UTCI. The best starting point for real projects.
- **[demos/solweig_gbg_test.py](https://github.com/UMEP-dev/solweig/blob/main/demos/solweig_gbg_test.py)** — Gothenburg test data: surface preparation, SVF caching, and timeseries calculation.

## Next steps

- [Basic Usage](../guide/basic-usage.md) — Vegetation, height conventions, custom parameters, validation
- [Working with GeoTIFFs](../guide/geotiffs.md) — File loading, caching, saving results
- [Timeseries](../guide/timeseries.md) — Multi-day simulations with thermal state
- [Thermal Comfort](../guide/thermal-comfort.md) — UTCI and PET in depth
- [API Reference](../api/index.md) — All classes and functions
