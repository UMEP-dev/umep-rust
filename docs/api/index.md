# API Reference

SOLWEIG provides a clean, minimal API for urban microclimate calculations.

## Quick Overview

### Core Functions

| Function | Description |
|----------|-------------|
| [`calculate()`](functions.md#calculate) | Single timestep Tmrt calculation |
| [`calculate_timeseries()`](functions.md#calculate_timeseries) | Multi-timestep with thermal state |
| [`calculate_tiled()`](functions.md#calculate_tiled) | Large raster processing |
| [`validate_inputs()`](functions.md#validate_inputs) | Pre-flight input validation |

### Data Classes

| Class | Description |
|-------|-------------|
| [`SurfaceData`](dataclasses.md#surfacedata) | Terrain data (DSM, CDSM, walls, SVF) |
| [`Location`](dataclasses.md#location) | Geographic coordinates |
| [`Weather`](dataclasses.md#weather) | Meteorological conditions |
| [`HumanParams`](dataclasses.md#humanparams) | Human body parameters |
| [`SolweigResult`](dataclasses.md#solweigresult) | Calculation output |
| [`ModelConfig`](dataclasses.md#modelconfig) | Model configuration |

### Post-Processing

| Function | Description |
|----------|-------------|
| [`compute_utci()`](functions.md#compute_utci) | Batch UTCI from Tmrt files |
| [`compute_pet()`](functions.md#compute_pet) | Batch PET from Tmrt files |

### GPU Utilities

| Function | Description |
|----------|-------------|
| `is_gpu_available()` | Check if GPU acceleration is available |
| `get_compute_backend()` | Returns `"gpu"` or `"cpu"` |
| `disable_gpu()` | Disable GPU, fall back to CPU |

## Import Pattern

```python
import solweig

# All public API is available at the top level
surface = solweig.SurfaceData(dsm=my_dsm, pixel_size=1.0)
result = solweig.calculate(surface, location, weather)
```

## Type Annotations

SOLWEIG is fully typed. Enable type checking in your IDE for the best experience:

```python
from solweig import SurfaceData, Location, Weather, SolweigResult

def process_area(dsm: np.ndarray) -> SolweigResult:
    surface: SurfaceData = SurfaceData(dsm=dsm, pixel_size=1.0)
    location: Location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather: Weather = Weather(...)
    return solweig.calculate(surface, location, weather)
```
