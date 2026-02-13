# Basic Usage

This guide covers the core workflow for SOLWEIG calculations.

## Minimal Example

```python
import numpy as np
import solweig
from datetime import datetime

# 1. Create surface from DSM array
dsm = np.ones((100, 100), dtype=np.float32) * 10.0
dsm[30:70, 30:70] = 25.0  # Add a 15m tall building

surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)

# 2. Define location (always include UTC offset!)
location = solweig.Location(
    latitude=57.7,
    longitude=12.0,
    utc_offset=1,  # CET
)

# 3. Define weather conditions
weather = solweig.Weather(
    datetime=datetime(2024, 7, 15, 12, 0),
    ta=25.0,        # Air temperature (°C)
    rh=50.0,        # Relative humidity (%)
    global_rad=800.0  # Global radiation (W/m²)
)

# 4. Calculate Tmrt
result = solweig.calculate(surface, location, weather)

print(f"Mean Tmrt: {result.tmrt.mean():.1f}°C")
print(f"Max Tmrt: {result.tmrt.max():.1f}°C")
```

## Understanding the Output

`SolweigResult` contains:

| Field | Description |
|-------|-------------|
| `tmrt` | Mean radiant temperature grid (°C) |
| `shadow` | Shadow fraction (1=sunlit, 0=shaded) |
| `kdown` | Downwelling shortwave radiation (W/m²) |
| `kup` | Upwelling shortwave radiation (W/m²) |
| `ldown` | Downwelling longwave radiation (W/m²) |
| `lup` | Upwelling longwave radiation (W/m²) |

## Thermal Comfort Indices

Compute UTCI or PET directly from the result:

```python
# UTCI - fast polynomial (~1ms)
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")

# PET - iterative solver (~50× slower)
pet = result.compute_pet(weather)
print(f"Mean PET: {pet.mean():.1f}°C")
```

## Adding Vegetation

Include canopy data for tree shading effects:

```python
# CDSM = Canopy Digital Surface Model (vegetation heights)
cdsm = np.zeros_like(dsm)
cdsm[10:30, 40:60] = 8.0  # 8m tall trees

surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm,
    pixel_size=1.0,
    cdsm_relative=True,  # CDSM is height above ground (default)
)

# Preprocess converts relative → absolute heights
surface.preprocess()
```

### NaN Handling

NaN values in DSM, CDSM, and TDSM are automatically filled with the ground
reference before calculation:

| Layer | NaN meaning     | Filled with           |
| ----- | --------------- | --------------------- |
| DSM   | No surface data | DEM (if provided)     |
| CDSM  | No canopy       | DEM, or DSM if no DEM |
| TDSM  | No trunk        | DEM, or DSM if no DEM |
| DEM   | No ground data  | Not filled (baseline) |

After filling, pixels within 0.1 m of the ground reference are clamped to
exactly the ground value to prevent shadow/SVF artefacts from resampling noise.

This happens automatically — `fill_nan()` is called inside both `preprocess()`
and `calculate()`. You can also call it explicitly:

```python
surface.fill_nan()           # idempotent — safe to call multiple times
surface.fill_nan(tolerance=0.2)  # custom noise tolerance (default 0.1 m)
```

## Performance Tips

### First vs Repeat Calculations

The first calculation computes Sky View Factor (SVF), which is expensive:

```
First call:  ~67 seconds (200×200 grid)
Repeat call: ~0.3 seconds (210× faster!)
```

SVF is automatically cached on the `surface` object.

### Pre-computing for Large Areas

For production workflows, pre-compute walls and SVF:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",  # Walls/SVF saved here
)

# Subsequent runs reuse cached preprocessing
```

## Input Validation

Validate inputs before expensive calculations:

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")
    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field}")
```
