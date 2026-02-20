# Architecture

SOLWEIG follows a 4-layer architecture separating concerns cleanly.

## Layer Overview

```
┌─────────────────────────────────────────────┐
│  Layer 1: User API (api.py)                 │
│  calculate(), SurfaceData, Weather, etc.    │
├─────────────────────────────────────────────┤
│  Layer 2: Orchestration                     │
│  computation.py, timeseries.py, tiling.py   │
├─────────────────────────────────────────────┤
│  Layer 3: Component Functions               │
│  shadows.py, svf.py, radiation.py, etc.     │
├─────────────────────────────────────────────┤
│  Layer 4: Rust Computation                  │
│  rustalgos (via maturin/PyO3)               │
└─────────────────────────────────────────────┘
```

## Layer 1: User API

**File**: `api.py` (~244 lines)

Public interface that users import:

```python
import solweig

result = solweig.calculate(surface, location, weather)
```

Responsibilities:

- Re-export public classes and functions
- Input validation
- Documentation (docstrings)

## Layer 2: Orchestration

**Files**: `computation.py`, `timeseries.py`, `tiling.py`

Coordinates component functions and manages state:

```python
# computation.py
def _compute_single_timestep(surface, location, weather, state):
    shadows = compute_shadows(...)
    svf = resolve_svf(...)
    ground = compute_ground_temperature(...)
    gvf = compute_gvf(...)
    radiation = compute_radiation(...)
    tmrt = compute_tmrt(...)
    return SolweigResult(...)
```

Responsibilities:

- Call components in correct order
- Manage thermal state across timesteps
- Handle caching and buffer pools
- Coordinate parallel processing

## Layer 3: Component Functions

**Directory**: `components/`

Pure functions that implement physical models:

| Module | Function | Output |
|--------|----------|--------|
| `shadows.py` | `compute_shadows()` | ShadowBundle |
| `svf_resolution.py` | `resolve_svf()` | SvfBundle |
| `ground.py` | `compute_ground_temperature()` | GroundBundle |
| `gvf.py` | `compute_gvf()` | GvfBundle |
| `radiation.py` | `compute_radiation()` | RadiationBundle |
| `tmrt.py` | `compute_tmrt()` | TmrtResult |

Design principles:

- Pure functions (no side effects)
- Explicit inputs and outputs
- Bundle classes for multiple return values
- Testable in isolation

## Layer 4: Rust Computation

**Directory**: `rust/`

Performance-critical algorithms in Rust:

- `shadowing` - Ray-traced shadow computation
- `skyview` - Sky View Factor calculation
- `gvf` - Ground View Factor
- `vegetation` - Vegetation transmissivity
- `utci` - UTCI polynomial
- `pet` - PET iterative solver

Exposed to Python via maturin/PyO3:

```python
from solweig import rustalgos
shadows = rustalgos.compute_shadows(dsm, sun_altitude, sun_azimuth)
```

## Data Flow

```
SurfaceData ──┐
              │
Location ─────┼──► calculate() ──► SolweigResult
              │         │               │
Weather ──────┘         │               ├── tmrt
                        │               ├── shadow
                        ▼               ├── kdown
                  Component             ├── kup
                  Functions             ├── ldown
                        │               └── lup
                        ▼
                  Rust Algorithms
```

## Bundle Classes

Components communicate via typed bundles:

```python
@dataclass
class ShadowBundle:
    shadow: np.ndarray      # Combined shadow fraction
    shadow_building: np.ndarray
    shadow_vegetation: np.ndarray

@dataclass
class RadiationBundle:
    kdown: np.ndarray       # Downwelling shortwave
    kup: np.ndarray         # Upwelling shortwave
    ldown: np.ndarray       # Downwelling longwave
    lup: np.ndarray         # Upwelling longwave
    kside: DirectionalData  # Lateral shortwave
    lside: DirectionalData  # Lateral longwave
```

## Caching Strategy

Expensive computations are cached:

| Data | Cached Where | Invalidation |
|------|-------------|--------------|
| SVF | `PrecomputedData` | DSM hash change |
| Wall heights | `working_dir/walls/` | DSM change |
| Shadow matrices | `PrecomputedData` | DSM change |

## Dual Environment Support

SOLWEIG runs in both standalone Python and QGIS:

| Component | Python | QGIS/OSGeo4W |
|-----------|--------|--------------|
| Raster I/O | rasterio | GDAL |
| Progress | tqdm | QgsProcessingFeedback |
| Logging | logging | QgsProcessingFeedback |

Backend detection is automatic in `io.py`.
