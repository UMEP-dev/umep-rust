# Basic Usage

This guide covers the core concepts and common patterns for day-to-day use of SOLWEIG.

## The three required inputs

Every SOLWEIG calculation needs exactly three things:

### 1. Surface — the physical environment

A `SurfaceData` object holds the building/terrain heights and optional vegetation. The only required field is a DSM (Digital Surface Model).

**From numpy arrays** (in-memory workflows):

```python
import numpy as np
import solweig

# Minimum: just a height grid
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0  # A building

surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)
# SVF is required before calculate(); compute once for this surface
surface.compute_svf()
```

**From GeoTIFF files** (file-based workflows — walls and SVF are computed and cached to `working_dir/`):

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
)
```

### 2. Location — where on Earth

A `Location` tells SOLWEIG where the site is, so it can compute sun position correctly.

```python
# Manual — when you know the coordinates
location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)

# From a GeoTIFF's coordinate system — extracts lat/lon from the DSM's CRS
location = solweig.Location.from_surface(surface, utc_offset=1)

# From an EPW weather file header — lat, lon, and UTC offset all extracted
location = solweig.Location.from_epw("weather.epw")
```

!!! warning "Always set `utc_offset`"
    The UTC offset determines how clock time maps to sun position. Getting it wrong shifts shadows by hours. When creating a `Location` manually, always provide `utc_offset` explicitly.

### 3. Weather — atmospheric conditions

A `Weather` object holds the meteorological data for one point in time.

```python
from datetime import datetime

weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,          # Air temperature (°C)
    rh=40.0,          # Relative humidity (%)
    global_rad=850.0, # Global horizontal irradiance (W/m²)
    ws=2.0,           # Wind speed (m/s) — used for UTCI/PET
)
```

`ws` is optional for Tmrt calculations (defaults to 1.0 m/s) but directly affects UTCI and PET results.

For timeseries, load from an EPW file:

```python
weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
```

### Downloading weather data

Don't have an EPW file? Download one directly from PVGIS (no API key needed):

```python
epw_path = solweig.download_epw(
    latitude=37.98,      # Athens
    longitude=23.73,
    output_path="athens.epw",
)
weather_list = solweig.Weather.from_epw(epw_path)
location = solweig.Location.from_epw(epw_path)
```

## Running a calculation

### Single timestep

```python
result = solweig.calculate(surface, location, weather)
```

### Multiple timesteps (timeseries)

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    output_dir="output/",          # Save GeoTIFFs as they're computed
    outputs=["tmrt", "shadow"],    # Which outputs to save
)
```

`calculate_timeseries` automatically carries **thermal state** between timesteps — ground and wall temperatures from one hour affect the next. This matters for accuracy; avoid looping over `calculate()` manually.

Two common output patterns:

1. Stream outputs to disk as they are computed (`output_dir=...`) for long runs and low RAM.
2. Keep outputs in memory (no `output_dir`), aggregate manually, and save only final products to minimize disk usage.

See [Timeseries](timeseries.md#choose-an-output-strategy) for full examples of both patterns.

## Understanding the output

`SolweigResult` contains 2D grids with the same shape as your DSM:

| Field | Unit | What it means |
| ----- | ---- | ------------- |
| `tmrt` | °C | **Mean Radiant Temperature** — how much radiation a person absorbs. The main output. |
| `shadow` | 0–1 | Shadow fraction. 1 = fully sunlit, 0 = fully shaded. |
| `kdown` | W/m² | Incoming shortwave radiation (sun + diffuse sky). |
| `kup` | W/m² | Reflected shortwave radiation from the ground. |
| `ldown` | W/m² | Incoming longwave radiation (thermal, from sky + walls). |
| `lup` | W/m² | Emitted longwave radiation from the ground. |

```python
result = solweig.calculate(surface, location, weather)

# Tmrt difference between sun and shade
sunlit = result.tmrt[result.shadow > 0.5].mean()
shaded = result.tmrt[result.shadow < 0.5].mean()
print(f"Sun-shade Tmrt difference: {sunlit - shaded:.0f}°C")
```

## Adding vegetation

Trees reduce Tmrt significantly through shading. Provide a Canopy DSM (CDSM) — a grid of vegetation canopy heights above ground.

```python
# From arrays
cdsm = np.zeros_like(dsm)
cdsm[10:40, 50:80] = 8.0  # 8 m tall trees

surface = solweig.SurfaceData(dsm=dsm, cdsm=cdsm, pixel_size=1.0)

# From GeoTIFF
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",
    working_dir="cache/",
)
```

### Relative vs. absolute heights

CDSM (and TDSM) heights can be either **relative** (height above ground) or **absolute** (elevation above sea level). By default, SOLWEIG assumes CDSM values are relative.

```python
# Relative CDSM (default): values are height above ground
# e.g. 8.0 = an 8 m tall tree
surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm,
    cdsm_relative=True,   # Default — SOLWEIG adds ground elevation automatically
    pixel_size=1.0,
)
surface.preprocess()  # Converts relative → absolute using DEM or DSM as base

# Absolute CDSM: values are elevation above sea level
# e.g. 135.0 = tree canopy is at 135 m elevation
surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm_absolute,
    cdsm_relative=False,  # Already absolute — no conversion needed
    pixel_size=1.0,
)
```

When using `SurfaceData.prepare()` with GeoTIFFs, this conversion happens automatically.

The same flags exist for DSM and TDSM:

```python
# DSM with relative heights (height above ground) — requires a DEM
surface = solweig.SurfaceData(
    dsm=dsm_relative,
    dem=dem,
    dsm_relative=True,
    pixel_size=1.0,
)
surface.preprocess()  # dsm_absolute = dem + dsm_relative
```

### Deciduous vs. evergreen trees

By default, SOLWEIG uses seasonal leaf-on/leaf-off based on the date. For evergreen trees (conifers), set:

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    conifer=True,  # Trees always have full canopy
)
```

## Customising human parameters

The default person is standing, 75 kg, 1.75 m tall, 35 years old. You can change this:

```python
result = solweig.calculate(
    surface, location, weather,
    human=solweig.HumanParams(
        posture="sitting",   # or "standing" (default)
        abs_k=0.7,           # Shortwave absorption (0–1)
        abs_l=0.97,          # Longwave absorption (0–1)
        weight=65,           # kg (affects PET only)
        height=1.65,         # m (affects PET only)
    ),
)
```

## Anisotropic sky model

SOLWEIG supports both isotropic and anisotropic sky models. For reproducible
behavior, set `use_anisotropic_sky` explicitly in your call:

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    use_anisotropic_sky=True,  # More accurate, slightly slower
)
```

If you explicitly set `use_anisotropic_sky=True`, shadow matrices must already
be available. They are prepared alongside SVF via `SurfaceData.prepare(...)`
or `surface.compute_svf()`. Otherwise, `calculate*()` raises
`MissingPrecomputedData`.

## Input validation

Catch problems before the expensive SVF computation:

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")
    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid size mismatch: {e.field} expected {e.expected}, got {e.got}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

## Common issues

### NaN pixels in output

NaN values in DSM/CDSM are automatically filled with the ground reference (DEM, or the DSM itself if no DEM is provided). If output still contains NaN, provide a DEM to maximise valid coverage.

### Slow first calculation

SVF must be prepared before `calculate()`. For in-memory array workflows, call `surface.compute_svf()` once and reuse the surface. For production file workflows, use `SurfaceData.prepare()` with a persistent `working_dir` so SVF is cached and reused across runs.

### GPU not detected

```python
print(f"GPU: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")
```

The package falls back to CPU automatically. GPU gives ~5–10x speedup for shadow and SVF computation.

## Complete working demos

For end-to-end examples you can run directly:

- **[demos/athens-demo.py](https://github.com/UMEP-dev/solweig/blob/main/demos/athens-demo.py)** — Full GeoTIFF workflow with tree vectors, multi-day timeseries, and UTCI post-processing.
- **[demos/solweig_gbg_test.py](https://github.com/UMEP-dev/solweig/blob/main/demos/solweig_gbg_test.py)** — Gothenburg test data: surface preparation, SVF caching, and timeseries.
