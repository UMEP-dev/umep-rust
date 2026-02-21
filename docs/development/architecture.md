# Architecture

SOLWEIG follows a layered architecture with a fused Rust compute pipeline.

## Layer Overview

```
┌─────────────────────────────────────────────┐
│  Layer 1: User API (api.py)                 │
│  calculate(), SurfaceData, Weather, etc.    │
├─────────────────────────────────────────────┤
│  Layer 2: Orchestration                     │
│  computation.py, timeseries.py, tiling.py   │
├─────────────────────────────────────────────┤
│  Layer 3: Fused Rust Pipeline               │
│  pipeline.compute_timestep() via PyO3       │
│  + Python helpers (SVF, transmissivity,     │
│    building mask, ground temperature)       │
├─────────────────────────────────────────────┤
│  Layer 4: Rust Algorithms                   │
│  shadowing, skyview, gvf, vegetation,       │
│  tmrt, utci, pet, sky (via maturin/PyO3)    │
└─────────────────────────────────────────────┘
```

## Layer 1: User API

**File**: `api.py`

Public interface that users import:

```python
import solweig

summary = solweig.calculate(
    surface=solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="output"),
    weather=solweig.Weather.from_umep_met("weather.txt"),
    location=solweig.Location.from_surface(surface, utc_offset=1),
)
summary.report()
```

Key types:

- `SurfaceData` — DSM, vegetation, walls, land cover, SVF (via `.prepare()`)
- `Weather` — per-timestep meteorological data
- `Location` — geographic coordinates with UTC offset
- `TimeseriesSummary` — returned by `calculate()`, with summary statistics and GeoTIFF export

## Layer 2: Orchestration

**Files**: `computation.py`, `timeseries.py`, `tiling.py`, `summary.py`

Coordinates the pipeline and manages state:

```python
# timeseries.py — iterates over weather list
for weather in weather_list:
    result = calculate_core_fused(surface, location, weather, state, ...)
    accumulator.update(result)       # GridAccumulator tracks min/max/mean
    state = result.state             # carry thermal state forward

# computation.py — single-timestep entry point
def calculate_core_fused(surface, location, weather, state, ...):
    svf = resolve_svf(precomputed, ...)           # Python (cached)
    psi = compute_transmissivity(doy, ...)        # Python
    buildings = detect_building_mask(dsm, ...)     # Python
    result = pipeline.compute_timestep(...)        # Fused Rust FFI call
    lup = _apply_thermal_delay(...)                # Rust (TsWaveDelay)
    return SolweigResult(tmrt, shadow, ...)
```

Responsibilities:

- Pre-compute Python-side inputs (SVF resolution, transmissivity, building mask)
- Hand off to fused Rust pipeline for per-pixel computation
- Manage thermal state across timesteps
- Accumulate summary statistics (GridAccumulator)
- Route large rasters to tiled processing

## Layer 3: Fused Rust Pipeline

**Rust entry point**: `pipeline.compute_timestep()`

A single FFI call performs the full per-pixel computation:

```text
Shadows → Ground temperature → GVF → Radiation → Tmrt
```

This eliminates intermediate numpy allocations and FFI round-trips between
Python and Rust. The pipeline accepts all inputs at once and returns the
complete result.

**Python helpers** still called by the orchestration layer (Layer 2):

| Module | Function | Purpose |
|--------|----------|---------|
| `components/svf_resolution.py` | `resolve_svf()` | SVF lookup and adjustment (cached) |
| `components/svf_resolution.py` | `adjust_svfbuveg_with_psi()` | Vegetation transmissivity correction |
| `components/shadows.py` | `compute_transmissivity()` | Seasonal leaf-on/off transmissivity |
| `components/gvf.py` | `detect_building_mask()` | Building footprint detection for GVF |
| `components/ground.py` | `compute_ground_temperature()` | Sinusoidal ground/wall temperature model |

## Layer 4: Rust Algorithms

**Directory**: `rust/src/`

Performance-critical algorithms in Rust, exposed via maturin/PyO3:

| Module | Purpose |
|--------|---------|
| `pipeline` | Fused per-timestep compute (shadows → Tmrt) |
| `shadowing` | Ray-traced shadow computation (CPU + GPU) |
| `skyview` | Sky View Factor calculation |
| `gvf` | Ground View Factor with wall radiation |
| `vegetation` | Kside/Lside vegetation radiation |
| `sky` | Anisotropic (Perez) sky model |
| `tmrt` | Mean Radiant Temperature from radiation budget |
| `ground` | Ground/wall temperature and TsWaveDelay |
| `utci` | Universal Thermal Climate Index polynomial |
| `pet` | Physiological Equivalent Temperature solver |
| `morphology` | Binary dilation (building mask) |

## Data Flow

```
SurfaceData ──┐
              │
Location ─────┼──► calculate() ──► TimeseriesSummary
              │         │               │
Weather[] ────┘         │               ├── tmrt_mean / tmrt_max
                        │               ├── shadow_fraction
                        ▼               ├── sun_hours
                  timeseries loop       ├── utci_mean
                        │               └── to_geotiff() / report()
                        ▼
              calculate_core_fused()
                        │
              ┌─────────┼──────────┐
              │ Python   │  Rust    │
              │ helpers  │ pipeline │
              └─────────┴──────────┘
```

## Bundle Classes

Components communicate via typed dataclass bundles:

```python
@dataclass
class GroundBundle:
    tg: np.ndarray          # Ground temperature deviation (K)
    tg_wall: float          # Wall temperature deviation
    ci_tg: float            # Clearness index correction
    alb_grid: np.ndarray    # Albedo per pixel
    emis_grid: np.ndarray   # Emissivity per pixel

@dataclass
class LupBundle:
    lup: np.ndarray         # Upwelling longwave (center)
    lup_e: np.ndarray       # Upwelling longwave (east)
    lup_s: np.ndarray       # ... south, west, north
    state: ThermalState     # Updated state for next timestep
```

Active bundles: `DirectionalArrays`, `SvfBundle`, `GroundBundle`,
`GvfBundle`, `LupBundle`, `WallBundle`, `VegetationBundle`.

## Caching Strategy

Expensive computations are cached:

| Data | Cached Where | Invalidation |
|------|-------------|--------------|
| Wall heights/aspects | `working_dir/walls/` | DSM change |
| SVF arrays | `working_dir/svf/` | DSM change |
| GVF geometry cache | `PrecomputedData` | Per-run |
| Land cover properties | `SurfaceData._land_cover_props_cache` | Identity change |
| Valid-pixel bounding box | `SurfaceData._valid_bbox_cache` | Identity change |

## Dual Environment Support

SOLWEIG runs in both standalone Python and QGIS:

| Component | Python | QGIS/OSGeo4W |
|-----------|--------|--------------|
| Raster I/O | rasterio | GDAL |
| Progress | tqdm | QgsProcessingFeedback |
| Logging | logging | QgsProcessingFeedback |

Backend detection is automatic in `_compat.py`.
