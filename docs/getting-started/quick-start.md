# Quick Start Guide

Get up and running with SOLWEIG in minutes.

## Installation

```bash
# From source (development)
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
pip install -e .
maturin develop  # Build Rust extension
```

## Basic Workflow

SOLWEIG calculates Mean Radiant Temperature (Tmrt) and thermal comfort indices (UTCI, PET) from urban surface data and weather conditions.

### Step 1: Prepare Surface Data

Use `SurfaceData.prepare()` to load GeoTIFFs and auto-compute walls/SVF:

```python
import solweig

# Load DSM and optional vegetation, compute walls/SVF automatically
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",       # Walls/SVF cached here for reuse
    cdsm="data/cdsm.tif",       # Optional: vegetation heights
    bbox=[476800, 4205850, 477200, 4206250],  # Optional: crop extent
    pixel_size=1.0,             # Optional: resolution in meters
)
```

**What `prepare()` does:**

- Loads and validates DSM (required) and optional CDSM/DEM/land cover
- Computes wall heights and aspects (cached to `working_dir/walls/`)
- Computes Sky View Factor (cached to `working_dir/svf/`)
- Aligns all rasters to common extent and resolution

**Performance:** First run computes walls/SVF (slow). Subsequent runs reuse cached data (fast).

### Step 2: Load Weather Data

Load weather from an EPW file:

```python
# Load 3 days of weather data
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2023-07-01",
    end="2023-07-03",
)

print(f"Loaded {len(weather_list)} timesteps")
```

### Step 3: Calculate Tmrt

```python
# Calculate timeseries with auto-save
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    output_dir="output/",           # Auto-saves Tmrt rasters
    outputs=["tmrt", "shadow"],     # Which outputs to save
)

print(f"Processed {len(results)} timesteps")
print(f"Mean Tmrt: {results[0].tmrt.mean():.1f}°C")
```

### Step 4: Post-process Thermal Comfort (Optional)

UTCI and PET are computed separately for better performance:

```python
# Compute UTCI (fast polynomial, ~1 second)
n_utci = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)

# Compute PET (slower iterative solver, optional)
# n_pet = solweig.compute_pet(
#     tmrt_dir="output/",
#     weather_series=weather_list,
#     output_dir="output_pet/",
# )
```

## Complete Example

See [demos/athens-demo.py](../../demos/athens-demo.py) for a complete working example.

```python
import solweig
from pathlib import Path

# Setup paths
input_path = Path("demos/data/athens")
output_path = Path("temp/athens")
output_path.mkdir(parents=True, exist_ok=True)

# Step 1: Prepare surface (walls/SVF auto-computed and cached)
surface = solweig.SurfaceData.prepare(
    dsm=str(input_path / "DSM.tif"),
    working_dir=str(output_path / "working"),
    cdsm=str(output_path / "CDSM.tif"),  # Optional vegetation
)

# Step 2: Load weather
weather_list = solweig.Weather.from_epw(
    str(input_path / "athens_2023.epw"),
    start="2023-07-01",
    end="2023-07-01",
)

# Step 3: Calculate
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    output_dir=str(output_path / "output"),
)

print(f"Done! Processed {len(results)} timesteps")
```

## Single Timestep Calculation

For quick single-timestep calculations without file I/O:

```python
import numpy as np
from datetime import datetime
import solweig

# Create surface from arrays (no files needed)
dsm = np.ones((200, 200), dtype=np.float32) * 10.0
dsm[50:100, 50:100] = 25.0  # Add a building

surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)

# IMPORTANT: Always specify UTC offset for correct sun position calculations
location = solweig.Location(
    latitude=37.98,    # Athens, Greece
    longitude=23.73,
    utc_offset=2,      # Eastern European Time (UTC+2)
)

weather = solweig.Weather(
    datetime=datetime(2024, 7, 15, 12, 0),
    ta=30.0,      # Air temperature (°C)
    rh=50.0,      # Relative humidity (%)
    global_rad=800.0,  # Global radiation (W/m²)
)

# Calculate (SVF computed on first call, cached for subsequent calls)
result = solweig.calculate(surface, location, weather)
print(f"Tmrt: {result.tmrt.mean():.1f}°C")

# Compute thermal comfort indices from the result
utci = result.compute_utci(weather)  # Fast polynomial
pet = result.compute_pet(weather)    # Slower iterative solver
print(f"UTCI: {utci.mean():.1f}°C, PET: {pet.mean():.1f}°C")
```

### Location from GeoTIFF

When loading data from GeoTIFFs, you can extract the location automatically:

```python
surface = solweig.SurfaceData.prepare(dsm="data/dsm.tif", working_dir="cache/")

# Auto-extract location from CRS (requires explicit UTC offset!)
location = solweig.Location.from_surface(surface, utc_offset=2)

# Warning: If you omit utc_offset, it defaults to 0 with a warning
# This can cause incorrect sun position calculations!
```

## Key Classes

| Class | Purpose |
|---|---|
| `SurfaceData` | Terrain data (DSM, CDSM, walls, SVF) |
| `Location` | Geographic coordinates (lat, lon, UTC offset) |
| `Weather` | Weather conditions (temp, humidity, radiation) |
| `HumanParams` | Human body parameters (optional customization) |
| `SolweigResult` | Calculation output (Tmrt, shadow, radiation) |

## Common Options

### Custom Human Parameters

```python
result = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather,
    human=solweig.HumanParams(
        abs_k=0.7,        # Shortwave absorption (0-1)
        abs_l=0.97,       # Longwave absorption (0-1)
        posture="standing",  # or "sitting"
        weight=75,        # kg
        height=1.80,      # m
    ),
)
```

### Anisotropic Sky Model

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    use_anisotropic_sky=True,  # More accurate but slower
)
```

## Input Validation

Validate inputs before expensive calculations:

```python
try:
    # Preflight check - catches errors before SVF computation
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")

    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field} expected {e.expected}, got {e.got}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

## Common Issues

### "SVF data not found"

SVF is computed automatically on first run. If using `SurfaceData.prepare()`, ensure `working_dir` is writable.

### Slow first calculation

First `calculate()` call computes SVF (expensive). Subsequent calls reuse cached SVF and are ~200× faster.

### GPU not available

The package automatically falls back to CPU. Check status:

```python
print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")
```

## Next Steps

- See [demos/athens-demo.py](../../demos/athens-demo.py) for a complete example
- Check [specs/](../../specs/) for physics documentation
- Run `pytest tests/` to verify installation
