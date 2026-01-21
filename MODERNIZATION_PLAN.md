# SOLWEIG Modernization Plan

**Updated: January 2026** - Simplified based on expert analysis

## Core Principles

1. **Simplicity first** - Minimize parameters, auto-compute what we can
2. **Spec-driven testing** - Simple markdown specs, handwritten tests
3. **POI-focused** - Support computing only at points of interest
4. **Progressive complexity** - Simple defaults, advanced options available

---

## 1. Current State → Target State

### Parameter Explosion Problem

Current API has 13+ parameters scattered across functions:

```python
# CURRENT: Too many parameters, hard to use
calculate_tmrt(
    dsm, cdsm, dem, svf_grid, shadow_grid,
    kdown, kup, ldown, lup,
    sun_altitude, sun_azimuth,
    ta, rh, ws,
    abs_k=0.7, abs_l=0.97,
    posture="standing",
    pixel_size=1.0,
    max_height=None,  # User must calculate
    ...
)
```

### Target: Grouped Configuration

```python
# TARGET: Clean, grouped parameters
@dataclass
class SurfaceData:
    dsm: np.ndarray           # Required
    cdsm: np.ndarray | None   # Optional vegetation
    dem: np.ndarray | None    # Optional terrain
    pixel_size: float = 1.0

@dataclass
class Location:
    latitude: float
    longitude: float
    utc_offset: int = 0

@dataclass
class Weather:
    datetime: datetime        # Sun position computed internally
    ta: float                 # Air temperature (°C)
    rh: float                 # Relative humidity (%)
    global_rad: float         # Global radiation (W/m²)
    ws: float = 1.0           # Wind speed (m/s)
    # Direct/diffuse split computed internally via Erbs model

@dataclass
class HumanParams:
    """Optional - defaults to standard reference person"""
    posture: str = "standing"
    abs_k: float = 0.7
    abs_l: float = 0.97
    # PET-specific params only if using PET
    age: int = 35
    weight: float = 75
    height: float = 1.75

# CLEAN API
result = solweig.calculate(
    surface=SurfaceData(dsm=dsm),
    location=Location(lat=57.7, lon=12.0),
    weather=Weather(datetime=dt, ta=25, rh=50, global_rad=800),
)
```

---

## 2. Auto-Computed Values

Values that should be computed internally, not required from user:

| Value            | Computation                          | Source              |
| ---------------- | ------------------------------------ | ------------------- |
| `sun_altitude`   | From lat/lon/datetime                | pysolar or internal |
| `sun_azimuth`    | From lat/lon/datetime                | pysolar or internal |
| `max_dsm_height` | `np.nanmax(dsm) - np.nanmin(dsm)`    | Automatic           |
| `direct_rad`     | Erbs model from global_rad + sun_alt | Literature          |
| `diffuse_rad`    | `global_rad - direct_rad`            | Derived             |
| `svf_grid`       | Computed once, cached                | Internal            |
| `gvf_grid`       | `1 - svf` for flat ground            | Simplified          |

### Erbs Diffuse Fraction Model

```python
def erbs_diffuse_fraction(kt: float, altitude_deg: float) -> float:
    """
    Erbs et al. (1982) correlation for diffuse fraction.

    kt = clearness index = global_rad / extraterrestrial_rad
    """
    if kt <= 0.22:
        return 1.0 - 0.09 * kt
    elif kt <= 0.80:
        return 0.9511 - 0.1604*kt + 4.388*kt**2 - 16.638*kt**3 + 12.336*kt**4
    else:
        return 0.165
```

---

## 3. POI-Only Mode

For most use cases, users want Tmrt at specific points, not full grids.

### Current: Full Grid (slow)

```python
# Computes Tmrt for ALL 1 million pixels
tmrt_grid = calculate_tmrt(dsm_1000x1000, ...)  # ~30 seconds
```

### Target: POI Mode (fast)

```python
# Computes Tmrt only at 10 points of interest
poi_coords = [(100, 200), (150, 300), ...]  # (row, col) pairs
tmrt_values = calculate_tmrt_poi(dsm, poi_coords, ...)  # ~0.1 seconds
```

### Implementation Strategy

1. **Shadow at POI**: Only trace rays from POI locations
2. **SVF at POI**: Compute view factors only at POI
3. **Radiation at POI**: Full calculation but only at points
4. **Result**: 99%+ reduction for typical monitoring use case

---

## 4. Memory Optimizations

| Current                 | Optimized       | Savings |
| ----------------------- | --------------- | ------- |
| `shadow: float32`       | `shadow: uint8` | 75%     |
| `svf: float64`          | `svf: float32`  | 50%     |
| Full intermediate grids | POI arrays      | 99%+    |

### Shadow Storage

```rust
// Current: wasteful
shadow_grid: Array2<f32>  // 4 bytes per pixel, values only 0.0 or 1.0

// Optimized
shadow_grid: Array2<u8>   // 1 byte per pixel, values 0 or 1
```

---

## 5. Simplified Algorithm Options

### UTCI vs PET

| Index    | Use When                 | Computation                  |
| -------- | ------------------------ | ---------------------------- |
| **UTCI** | Standard outdoor comfort | Fast polynomial (~200 terms) |
| **PET**  | Person-specific analysis | Slow iterative solver        |

**Recommendation**: Make PET optional, UTCI is sufficient for most urban climate studies.

```python
# Default: UTCI only
result = solweig.calculate(surface, location, weather)
print(result.utci)  # Available

# Optional: Include PET
result = solweig.calculate(surface, location, weather, compute_pet=True)
print(result.pet)   # Now available
```

### Simplified GVF

For areas without tall walls (parks, open spaces), GVF ≈ 1 - SVF:

```python
def calculate_gvf(svf: np.ndarray, has_walls: bool = False) -> np.ndarray:
    """
    Ground View Factor.

    For open areas: GVF = 1 - SVF (horizontal surface assumption)
    For urban canyons: Full wall view factor calculation
    """
    if not has_walls:
        return 1.0 - svf  # Fast path
    else:
        return _calculate_gvf_with_walls(svf)  # Full calculation
```

---

## 6. Specification Format (Simplified)

We use **simple markdown files** that scientists can read and edit.
Tests are handwritten based on these specs (no YAML parsing).

### Current Specs

```
specs/
├── OVERVIEW.md      # Pipeline diagram, module relationships
├── shadows.md       # Shadow calculation (8 properties)
├── svf.md           # Sky View Factor (8 properties)
├── gvf.md           # Ground View Factor (6 properties)
├── radiation.md     # Shortwave + Longwave components
├── tmrt.md          # Mean Radiant Temperature (10 properties)
├── utci.md          # Universal Thermal Climate Index (8 properties)
├── pet.md           # Physiological Equivalent Temperature (8 properties)
└── technical.md     # Float32, tiling, GPU, NaN handling
```

### Spec → Test Mapping

Each property in specs becomes one or more test functions:

```markdown
# In specs/shadows.md

## Properties

1. No shadows when sun ≤ 0° (below horizon)
2. Flat terrain → no shadows
3. Lower sun → longer shadows
```

```python
# In tests/spec/test_shadows.py
def test_no_shadows_below_horizon():
    """Property 1: No shadows when sun ≤ 0°"""
    ...

def test_flat_terrain_no_shadows():
    """Property 2: Flat terrain → no shadows"""
    ...

def test_lower_sun_longer_shadows():
    """Property 3: Lower sun → longer shadows"""
    ...
```

---

## 7. Rust Integration Strategy

### Move Time Loop to Rust

Current: Python loops over timesteps, calling Rust for each

```python
for hour in range(24):
    shadow = rust.calculate_shadow(dsm, sun_pos[hour])
    svf = rust.calculate_svf(dsm)  # Redundant!
    tmrt[hour] = rust.calculate_tmrt(shadow, svf, ...)
```

Target: Single Rust call for full day

```python
# One call, Rust handles time loop internally
results = rust.calculate_day(
    dsm=dsm,
    timestamps=timestamps,
    weather=weather_array,
    location=location,
)
# Returns: Dict with tmrt[24], utci[24], shadows[24]
```

### Shared Weight Polynomial

Both `calculate_tmrt()` and other functions use similar weighting:

```rust
// Extract common helper
fn directional_weights(altitude: f64, azimuth: f64) -> [f64; 6] {
    // Returns weights for [up, down, N, E, S, W]
    ...
}
```

---

## 8. Implementation Phases

### Phase 1: API Simplification ✓ (Specs Complete)

- [x] Create simple markdown specs
- [x] Define property-based tests
- [x] Document physical invariants

### Phase 2: Config Consolidation

- [ ] Create `SurfaceData`, `Location`, `Weather`, `HumanParams` dataclasses
- [ ] Add internal sun position calculation
- [ ] Add Erbs diffuse fraction model
- [ ] Auto-compute `max_dsm_height`

### Phase 3: POI Mode

- [ ] Add `poi_coords` parameter to key functions
- [ ] Implement shadow-at-point calculation
- [ ] Implement SVF-at-point calculation
- [ ] Benchmark: target 100x speedup for 10 POIs vs 1M grid

### Phase 4: Memory Optimization

- [ ] Change shadow storage to uint8
- [ ] Ensure float32 throughout
- [ ] Add POI-only intermediate storage

### Phase 5: Rust Time Loop

- [ ] Move hourly loop into Rust
- [ ] Single `calculate_day()` entry point
- [ ] Cache SVF/GVF (computed once per DSM)

### Phase 6: Optional Complexity

- [ ] Make PET computation optional
- [ ] Add simplified GVF for open areas
- [ ] UTCI lookup table option for extreme speed

---

## 9. Migration Path

### Breaking Changes

| Old API                                    | New API                                 |
| ------------------------------------------ | --------------------------------------- |
| `calculate_tmrt(dsm, cdsm, dem, svf, ...)` | `calculate(surface, location, weather)` |
| User computes sun position                 | Auto from datetime + location           |
| User splits direct/diffuse                 | Auto via Erbs model                     |
| Full grid always                           | POI mode available                      |

### Compatibility Layer

```python
# Deprecated wrapper for old code
def calculate_tmrt_legacy(dsm, cdsm, dem, svf, shadow, ...):
    """DEPRECATED: Use solweig.calculate() instead"""
    warnings.warn("Use solweig.calculate()", DeprecationWarning)
    surface = SurfaceData(dsm=dsm, cdsm=cdsm, dem=dem)
    ...
    return calculate(surface, location, weather).tmrt
```

---

## 10. Success Metrics

| Metric               | Current | Target              |
| -------------------- | ------- | ------------------- |
| API parameters       | 13+     | 3-4 config objects  |
| POI calculation time | N/A     | <0.1s for 10 points |
| Memory per megapixel | ~200MB  | <50MB               |
| Lines to basic usage | ~50     | ~10                 |

### Example: Before vs After

**Before (50+ lines)**:

```python
import numpy as np
from pysolar.solar import get_altitude, get_azimuth
from solweig import calculate_svf, calculate_shadow, calculate_tmrt

# Load data
dsm = load_raster("dsm.tif")
cdsm = load_raster("cdsm.tif")

# Calculate sun position
lat, lon = 57.7, 12.0
dt = datetime(2024, 7, 15, 12, 0)
altitude = get_altitude(lat, lon, dt)
azimuth = get_azimuth(lat, lon, dt)

# Pre-compute SVF
svf = calculate_svf(dsm, cdsm)

# Calculate shadows
max_height = np.nanmax(dsm) - np.nanmin(dsm)
shadow = calculate_shadow(dsm, cdsm, altitude, azimuth, 1.0, max_height)

# Split radiation
global_rad = 800
# ... complex diffuse fraction calculation ...

# Finally calculate Tmrt
tmrt = calculate_tmrt(dsm, cdsm, None, svf, shadow, kdown, kup, ...)
```

**After (10 lines)**:

```python
import solweig
from datetime import datetime

result = solweig.calculate(
    surface=solweig.SurfaceData(dsm="dsm.tif", cdsm="cdsm.tif"),
    location=solweig.Location(lat=57.7, lon=12.0),
    weather=solweig.Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=25, rh=50, global_rad=800
    ),
)
print(f"Tmrt: {result.tmrt.mean():.1f}°C")
print(f"UTCI: {result.utci.mean():.1f}°C")
```
