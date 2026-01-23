# SOLWEIG Modernization Plan

**Updated: January 2026** - Post-Processing Architecture Complete

> **Major Update:** UTCI/PET moved to post-processing architecture (January 2026).
> Main calculation computes Tmrt only. UTCI/PET are separate post-processing steps.
> Working directory caching complete. SVF caching bug fixed (~72× speedup).
> See Section 1.8 for architecture overview.

## Core Principles

1. **Simplicity first** - Minimize parameters, auto-compute what we can
2. **Spec-driven testing** - Simple markdown specs, handwritten tests
3. **POI-focused** - Support computing only at points of interest
4. **Progressive complexity** - Simple defaults, advanced options available
5. **QGIS compatible** - GDAL support, minimal external dependencies

---

## Stakeholder Roles

| Role          | Responsibility                             | Key Focus Areas                                 |
| ------------- | ------------------------------------------ | ----------------------------------------------- |
| **Engineer**  | API elegance, performance, maintainability | Memory efficiency, tiling, Rust/Python boundary |
| **Scientist** | Scientific accuracy, proper references     | Formula validation, spec completeness           |
| **User**      | Usability, intuitive workflows             | GeoTIFF loading, simple configs, QGIS           |

---

## 1. Current State Summary (January 2026)

### Phase 2 & 3 Achievements ✅

**Phase 2 (100% Complete):**
- 100% parity with reference implementation
- Simplified API: 3-4 config objects instead of 13+ parameters
- Auto-computation of sun position, radiation split, max_height
- Timeseries support with thermal state accumulation

**Phase 3 (Partially Complete - Major Progress):**
- ✅ `SurfaceData.prepare()` - Auto-preprocessing with working directory caching
- ✅ `Weather.from_epw()` - EPW file loading with date range filtering
- ✅ `SolweigResult.to_geotiff()` - Save outputs to GeoTIFF
- ✅ Auto-location extraction from DSM CRS
- ✅ **Post-processing architecture** - UTCI/PET computed separately after main calculation
- ✅ Progress reporting with timing metrics
- ✅ SVF caching bug fix (~72× speedup potential)

### Current Architecture (NEW - January 2026)

**Main Calculation Loop (Inline):**
```python
# Computes Tmrt for all timesteps
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    use_anisotropic_sky=True,
    output_dir="output/",
    outputs=["tmrt", "shadow"],  # Inline: Tmrt computed in loop
)
```

**Post-Processing (Separate):**
```python
# UTCI computed separately from saved Tmrt files
n_utci = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)

# PET computed separately (optional, slower)
n_pet = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
    human=solweig.HumanParams(weight=75, height=1.75),
)
```

**Why This Architecture:**
1. **Tmrt is always needed** - Computed inline for performance
2. **UTCI/PET are optional** - Many users only need Tmrt for radiation analysis
3. **PET is expensive** - 50× slower than UTCI due to iterative solver
4. **Flexibility** - Compute UTCI/PET for subset of timesteps or different human parameters
5. **No thermal state dependency** - UTCI/PET are pure functions, safe to post-process

### Remaining Issues

**Engineering Debt:**

- `_calculate_core()` is 900+ lines (monolithic) - needs decomposition
- `calc_solweig()` passes 70+ positional parameters (legacy runner)
- ShadowArrays properties allocate new arrays on every access
- ~~SVF loads 16 separate arrays instead of lazy/batched~~ ✅ **FIXED** - SVF caching working
- **NEW:** Working directory cache needs validation/invalidation strategy
- **NEW:** Post-processing functions need standardized progress reporting

**Scientific Gaps:**

- 8 missing secondary references in specs
- Ground temperature model (TsWaveDelay) undocumented
- 4 formula discrepancies between specs and implementation
- 18 edge cases not documented
- **NEW:** Post-processing validation tests missing (UTCI accuracy, PET convergence)
- **NEW:** Error propagation analysis (Tmrt uncertainty → UTCI/PET uncertainty)

**Usability Issues:**

- ~~No `SurfaceData.from_geotiff()` convenience method~~ ✅ **COMPLETE** - `SurfaceData.prepare()`
- ~~No `result.to_geotiff()` for easy export~~ ✅ **COMPLETE**
- No POI/point-based calculation API - **Next priority**
- QGIS plugin integration not documented
- **NEW:** Workflow documentation needs clarification (inline vs. post-processing)
- **NEW:** Migration guide from legacy API needed

---

## 1.5 Critical Issues from Stakeholder Review (January 2026)

Three-stakeholder review identified the following critical issues that MUST be addressed:

### Engineering Critical Issues

| Issue | Status | Problem | Resolution |
|-------|--------|---------|------------|
| **E1: SVF caching** | ✅ **FIXED** | SVF was computed every timestep. | Fixed: Check `surface.svf` before `precomputed.svf`. ~72× speedup. |
| **E2: params.json not addressed** | TODO | Simplified API doesn't show how to specify physical parameters (albedo, emissivity, vegetation properties). | Add `params` parameter to `calculate()`. Bundle default params.json with package. |
| **E3: Land cover omitted** | TODO | `SurfaceData` table omits `land_cover` input, but model needs it for surface property variation. | Add `land_cover` to inputs. Document default behavior (uniform properties). |
| **E4: Old API deprecation** | OBSOLETE | Legacy config-based API still referenced. | **NEW DECISION:** Remove all references to old API. Building new API from scratch. |
| **E5: CRS mismatch handling** | TODO | No handling for DSM/CDSM/DEM with different CRS, resolution, or extent. | Validate on load. Require matching CRS/extent or raise descriptive error. |
| **E6: Cache invalidation** | NEW | Working directory cache has no validation if DSM changes. | Add cache metadata (shape, CRS, pixel_size). Validate on load. |
| **E7: Post-processing weather mismatch** | NEW | `compute_utci()` silently skips non-matching timestamps. | Warn if <50% matched. Add `require_all=True` option. |

### Scientific Critical Issues

| Issue | Problem | Resolution |
|-------|---------|------------|
| **S1: Kelvin offset wrong** | Implementation uses -273.2, spec uses -273.15. Systematic 0.05K bias. | Fix implementation to use -273.15 (correct value). |
| **S2: Hidden model defaults** | Simplified API hides `use_aniso`, `use_veg`, `conifer`, `use_landcover` with undocumented defaults. | Add `ModelOptions` dataclass. Document defaults and their scientific implications. |
| **S3: Perez coefficients undocumented** | Code uses coefficients that differ from comments. No clear citation for variant in use. | Document which Perez variant is used. Add proper citation. |
| **S4: POI mode accuracy undefined** | "SVF-at-point (sample rays)" - no error bounds, no validation against full-grid extraction. | Define acceptable tolerance. Add validation test comparing POI to grid extraction. |
| **S5: TsWaveDelay undocumented** | Ground temperature model uses magic constant (33.27) with no reference. | Add citation (Lindberg et al. 2015?). Document in specs. |

### User Critical Issues

| Issue | Status | Problem | Resolution |
|-------|--------|---------|------------|
| **U1: API confusion** | ✅ **RESOLVED** | Legacy config-based API confusing. | **NEW DECISION:** Remove all references to old API. Single simplified API only. |
| **U2: Preprocessing time unknown** | TODO | "Slower first run" is vague. Users need time estimates. | Add performance table: SVF for 500x500 = 2min, 1000x1000 = 10min, etc. |
| **U3: Model options inaccessible** | TODO | Simplified API doesn't expose `use_aniso`, `use_veg`, etc. | Add `ModelConfig` parameter to `calculate()`. |
| **U4: to_geotiff() unspecified** | ✅ **COMPLETE** | Output format unclear. | Implemented: `outputs=["tmrt", "shadow"]`, filename template `{output}_YYYYMMDD_HHMM.tif`. |
| **U5: EPW date mismatch handling** | TODO | What if EPW doesn't cover requested dates? | Error with available date range and suggestions. |
| **U6: Workflow documentation** | NEW | Inline vs. post-processing unclear. | Document architecture: Tmrt inline, UTCI/PET post-processing. |
| **U7: Working directory confusion** | NEW | Cache location and cleanup strategy unclear. | Document working_dir, add cache metadata, cleanup guide. |

---

## 1.6 Final Review Findings (January 2026)

Three-stakeholder final review identified these additional items:

### Engineering Findings

| Finding | Resolution |
|---------|------------|
| **API return type ambiguity**: `calculate()` with/without `output_dir` unclear | Add explicit mode parameter OR require `output_dir` always |
| **Progress callback signature inconsistent**: Line 251 vs line 748 differ | Standardize to `on_timestep(result, timestamp, step, total)` → bool |
| **Tile overlap formula missing**: Shadow extent depends on building height | Add formula: `overlap = ceil(max_height / tan(radians(10)))` |
| **POI 100x speedup claim unrealistic**: Shadow ray-casting is the bottleneck | Revise to "10-50x speedup depending on POI density" |
| **NumPy ABI risk underestimated**: 1.x vs 2.x incompatibility crashes | Add Sprint 0: NumPy compatibility testing FIRST |
| **rasterio/GDAL layer complexity**: Full wrapper is 500+ lines | Acknowledge scope; consider `rioxarray` as intermediate |

### Scientific Findings

| Finding | Resolution |
|---------|------------|
| **Missing spec files critical**: diffuse_fraction, surface_temperature, sky_emissivity, anisotropic_sky | Elevate to Sprint 1 alongside UX work |
| **Validation tolerances undefined**: What is acceptable Tmrt error? | Define: Tmrt ±0.1K vs UMEP, UTCI ±0.5K, SVF ±0.01 |
| **18 edge cases not enumerated**: Plan mentions them, doesn't list | Add edge_cases.md spec with test coverage |
| **Kelvin offset fix breaks reproducibility**: Old results differ by 0.05K | Document in release notes; version specs |
| **Perez model variant unspecified**: 1987 vs 1990 vs 1993? | Document: using Perez et al. 1990 coefficients |

### User Findings

| Finding | Resolution |
|---------|------------|
| **Preprocessing disconnected**: Two separate functions awkward | Add `solweig.preprocess()` wrapper function |
| **No input validation helper**: Users discover errors during run | Add `SurfaceData.validate()` method |
| **TMY date format inconsistent**: `"07-01"` vs `"2023-07-01"` | Standardize ISO 8601; use `year=None` parameter instead |
| **Missing aggregation outputs**: Users want daily max/mean | Add `outputs=["tmrt", "tmrt_max", "tmrt_mean"]` option |
| **No sample data bundle**: Users can't test without preprocessing | Provide pre-computed demo data on GitHub releases |
| **File structure undocumented**: What goes in walls_dir? | Add explicit file structure reference |

### Sprint 0: Prerequisites (NEW)

Before Phase 3 work begins:

- [ ] **NumPy ABI compatibility**: Test Rust extension with NumPy 1.26 AND NumPy 2.0+
- [ ] **Build wheel variants**: Verify CI builds for Python 3.9-3.12 on all platforms
- [ ] **Sample data bundle**: Create pre-computed walls/SVF for Athens demo (~10MB)

---

## 1.7 Code-Informed Engineering Review (January 2026)

Deep code review by engineering stakeholder revealed implementation-specific issues:

### Critical Bugs to Fix

| Bug | Location | Description | Fix |
|-----|----------|-------------|-----|
| **B1: `SurfaceData.from_files()` broken** | api.py:647-661 | Doesn't unpack tuple return from `common.load_raster()` | Change to `dsm, _, _, _ = common.load_raster(str(dsm_path))` |
| **B2: EPW parser needs pvlib** | runner.py | `pvlib.iotools.read_epw()` used for EPW parsing, not just solar position | Implement internal EPW parser OR document pvlib requirement |
| **B3: GPU requires runtime detection** | rust/Cargo.toml, api.py | No graceful fallback when GPU unavailable; crashes or requires separate builds | Implement runtime GPU detection with automatic CPU fallback |

### API Inconsistencies

| Issue | Location | Resolution |
|-------|----------|------------|
| **`SurfaceData.from_geotiff()` vs `from_files()`** | api.py | Plan references `from_geotiff()` but implementation has `from_files()` - rename to `from_geotiff()` |
| **`Weather.from_epw()` doesn't exist** | N/A | Currently uses runner.py's config loading - implement new method |
| **Kelvin offset hardcoded** | api.py | Uses -273.2, should be configurable with `use_legacy_kelvin_offset=True` default for backwards compat |

### Build Configuration Issues

**GPU Support Strategy (B3):**

**Goal:** GPU acceleration available to all users (including QGIS) with graceful fallback.

**Problem with current approach:**
- GPU as compile-time feature means users must choose at install time
- CI builds fail on runners without CUDA
- QGIS users on GPU-capable machines can't easily get acceleration

**Solution: Runtime GPU detection with graceful fallback**

```toml
# rust/Cargo.toml - Build WITH GPU support, detect at runtime
[tool.maturin]
features = ["pyo3/extension-module"]  # GPU detection happens at runtime

[features]
default = []
gpu = ["wgpu", "metal"]  # Optional compile-time GPU backends
```

```python
# pysrc/solweig/gpu.py - Runtime detection
def is_gpu_available() -> bool:
    """Check if GPU acceleration is available at runtime."""
    try:
        from solweig.rustalgos import check_gpu_available
        return check_gpu_available()
    except (ImportError, RuntimeError):
        return False

def get_compute_backend() -> str:
    """Return 'gpu' or 'cpu' based on availability."""
    return "gpu" if is_gpu_available() else "cpu"
```

```python
# User API - automatic GPU usage with override option
solweig.calculate(
    surface, weather,
    output_dir="output/",
    use_gpu=True,   # Default: use GPU if available, else CPU
    # use_gpu=False,  # Force CPU (useful for debugging/comparison)
)

# Check GPU status
print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Using backend: {solweig.get_compute_backend()}")
```

**Build Strategy:**
- **Development:** `maturin develop` (CPU-only, fast builds)
- **CI testing:** CPU-only wheels (no CUDA on runners)
- **PyPI release:** Wheels with runtime GPU detection
- **Local GPU builds:** `maturin develop --features gpu` for explicit GPU backend

**QGIS Integration:**
- Plugin checks `solweig.is_gpu_available()` on load
- Shows GPU status in plugin info/about dialog
- Processing algorithms use GPU automatically when available

**NumPy ABI Compatibility (Critical Risk):**

The `numpy` Rust crate v0.24 claims support for NumPy 1.16+, but:
- NumPy 2.0 broke ABI compatibility with 1.x
- Wheels built against NumPy 1.x may crash on NumPy 2.x
- QGIS 3.34 LTR ships NumPy 1.26.x, but future versions may use 2.x

**Mitigation:**
1. Build separate wheels for NumPy 1.x and 2.x
2. Test in CI matrix with both NumPy versions
3. Use `abi3` wheel tag if possible for better compatibility

### Units Documentation Table

All inputs and outputs should have documented units for scientific reproducibility:

| Variable | Unit | Notes |
|----------|------|-------|
| DSM/DEM/CDSM | metres | Height above datum |
| Wall heights | metres | Building height |
| Wall aspects | degrees | 0=North, clockwise |
| SVF | 0-1 | Dimensionless fraction |
| Temperature (input) | °C | Air temperature |
| Temperature (output) | °C | Tmrt, UTCI, PET |
| Radiation | W/m² | Kdown, Kup, Ldown, Lup |
| Humidity | % | Relative humidity |
| Wind speed | m/s | At reference height |
| Latitude/Longitude | degrees | WGS84 |

---

## 2. Revised Phase Plan

### Phase 3: User Experience & API Simplification

**Goal:** Make the package usable for researchers with minimal code

**Owner:** User-focused, with Engineer support

**Status:** Partially complete - major progress on core workflows

| Task | Description                                                           | Priority | Status      |
| ---- | --------------------------------------------------------------------- | -------- | ----------- |
| 3.1  | `SurfaceData.prepare()` with auto-preprocessing & caching             | HIGH     | ✅ COMPLETE |
| 3.2  | `Weather.from_epw(path, start, end)` - EPW support with date filter  | HIGH     | ✅ COMPLETE |
| 3.3  | `SolweigResult.to_geotiff()` with output control                      | HIGH     | ✅ COMPLETE |
| 3.4  | Auto-extract location from DSM CRS or EPW header                      | HIGH     | ✅ COMPLETE |
| 3.5  | `ModelConfig` dataclass for scientific settings                       | HIGH     | TODO        |
| 3.6  | `params` parameter with bundled defaults                              | HIGH     | TODO        |
| 3.7  | ~~`from_config()` migration helper~~ REMOVED                          | ~~MEDIUM~~   | ~~OBSOLETE~~ |
| 3.8  | Improved error messages with context and suggestions                  | MEDIUM   | TODO        |
| 3.9  | Input validation (CRS match, extent match, resolution match)          | MEDIUM   | TODO        |
| 3.10 | ~~`solweig.preprocess()` unified wrapper~~ REMOVED                    | ~~HIGH~~     | ~~DEPRECATED~~ |
| 3.11 | `SurfaceData.validate()` pre-flight validation method                 | MEDIUM   | TODO        |
| 3.12 | ~~Aggregation outputs~~ DEPRIORITIZED (use pandas/numpy)              | LOW      | TODO        |
| 3.13 | `solweig.validate_inputs()` standalone validation helper              | MEDIUM   | TODO        |
| 3.14 | `solweig.download_sample_data()` for quick start                      | MEDIUM   | TODO        |
| 3.15 | QGIS-compatible logging infrastructure (`logging.py`)                 | HIGH     | ✅ COMPLETE |
| 3.16 | Integrate automatic logging in key operations                         | HIGH     | ✅ COMPLETE |
| 3.17 | **Post-processing architecture** (UTCI/PET separate from main loop)   | **HIGH** | ✅ **COMPLETE** |
| 3.18 | Progress reporting with timing metrics                                | HIGH     | ✅ COMPLETE |
| 3.19 | Working directory cache metadata & validation                         | MEDIUM   | TODO        |
| 3.20 | Document post-processing workflow & architecture                      | HIGH     | TODO        |
| 3.21 | Standardize progress reporting across all functions                   | MEDIUM   | TODO        |

**SurfaceData Inputs (all GeoTIFF):**

| Input       | Description                              | Required | Notes                                      |
| ----------- | ---------------------------------------- | -------- | ------------------------------------------ |
| DSM         | Digital Surface Model (buildings+ground) | Yes      | Must have valid CRS for auto-location      |
| DEM         | Digital Elevation Model (ground only)    | No       | Used for building detection if provided    |
| CDSM        | Canopy DSM (vegetation heights)          | No       | Must match DSM CRS and extent              |
| TDSM        | Trunk zone DSM                           | No       | Must match DSM CRS and extent              |
| land_cover  | Land cover classification grid           | No       | UMEP IDs (0=paved, 5=grass, 7=water, etc.) |
| walls_dir   | Pre-computed wall heights/aspects        | No*      | *Required unless auto-compute enabled      |
| svf_dir     | Pre-computed SVF files                   | No*      | *Required unless auto-compute enabled      |

**ModelOptions Dataclass (addresses S2, U3):**

```python
@dataclass
class ModelOptions:
    """Scientific model settings with documented defaults."""
    use_veg: bool = True           # Vegetation scheme (Lindberg & Grimmond 2011)
    use_aniso: bool = False        # Anisotropic sky (Perez model) - 50% slower, +2-5°C accuracy
    use_landcover: bool = False    # Land cover variation (requires land_cover input)
    use_wall_scheme: bool = False  # Wall temperature scheme (Wallenberg et al. 2025)
    conifer: bool = False          # Evergreen trees (True) vs deciduous (False)
    person_cylinder: bool = True   # Cylindric (True) vs box (False) human geometry
```

**Location Auto-Detection:**

Location (lat/lon) is auto-extracted from:

1. **DSM raster CRS** - converts center point to WGS84 (already implemented in runner.py:178-182)
2. **EPW file header** - line 1 contains lat/lon (e.g., `LOCATION,...,38.00,23.75,...`)

**Error if CRS missing:**
```
ValueError: DSM has no CRS metadata. Cannot auto-detect location.
Either:
  1. Add CRS to GeoTIFF: gdal_edit.py -a_srs EPSG:32633 dsm.tif
  2. Provide location manually: solweig.Location(latitude=38.0, longitude=23.75)
```

**Current API (January 2026):**

```python
import solweig

# Step 1: Prepare surface data (auto-computes walls/SVF, caches to working_dir)
surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    working_dir="cache/",            # Cache preprocessing here
    cdsm="cdsm.tif",                 # Optional: vegetation
    bbox=[minx, miny, maxx, maxy],   # Optional: crop extent
    pixel_size=1.0,                  # Optional: resample resolution
)

# Step 2: Load weather from EPW file
weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2023-07-01",
    end="2023-07-03",  # 3 days
)

# Step 3: Calculate Tmrt (main calculation)
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    use_anisotropic_sky=True,
    output_dir="output/",
    outputs=["tmrt", "shadow"],  # Tmrt computed inline
)

# Step 4: Post-process UTCI (separate, fast)
n_utci = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)

# Step 5: Post-process PET (separate, slower, optional)
n_pet = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
    human=solweig.HumanParams(weight=75, height=1.75),
)
```

**Minimal Example (4 lines):**

```python
import solweig
surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01", end="2023-07-01")
solweig.calculate_timeseries(surface, weather, output_dir="output/")
```

**Current API Architecture:**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    SOLWEIG API Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SurfaceData.prepare(dsm, working_dir, cdsm=..., bbox=...)       │
│      → Auto-computes walls and SVF                                  │
│      → Caches to working_dir for reuse                              │
│      → Second run is instant (uses cache)                           │
│                                                                     │
│  2. Weather.from_epw(path, start, end)                              │
│      → Loads EPW file with date filtering                           │
│      → Returns list of Weather objects                              │
│                                                                     │
│  3. calculate_timeseries(surface, weather_series, output_dir=...)   │
│      → Main calculation loop (Tmrt computed inline)                 │
│      → Writes tmrt_*.tif, shadow_*.tif to output_dir                │
│      → Memory efficient: one timestep at a time                     │
│      → Progress reporting with timing                               │
│                                                                     │
│  4. compute_utci(tmrt_dir, weather_series, output_dir)              │
│      → Post-processing: UTCI from saved Tmrt files                  │
│      → Fast polynomial (~60 steps/s for 1M pixels)                  │
│                                                                     │
│  5. compute_pet(tmrt_dir, weather_series, output_dir, human=...)    │
│      → Post-processing: PET from saved Tmrt files                   │
│      → Slower iterative solver (~1 step/s for 1M pixels)            │
│                                                                     │
│  FUTURE: POI mode (Phase 4)                                         │
│      → calculate_at_points(surface, weather, points)                │
│      → 10-50x faster than full grid                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Post-Processing Architecture:**

1. **Tmrt is always needed** - Computed inline, saved to disk
2. **UTCI/PET are optional** - Many users only need Tmrt for radiation analysis
3. **PET is expensive** - 50× slower than UTCI, make it truly optional
4. **Flexibility** - Compute UTCI/PET for subset of timesteps or different human parameters without re-running main calculation
5. **No thermal state dependency** - UTCI/PET are pure functions, safe to separate

**ModelConfig Dataclass (TODO - addresses S2, U3):**

```python
@dataclass
class ModelConfig:
    """Scientific model settings with documented defaults."""
    use_veg: bool = True           # Vegetation scheme (Lindberg & Grimmond 2011)
    use_aniso: bool = False        # Anisotropic sky (Perez model) - 50% slower, +2-5°C accuracy
    use_landcover: bool = False    # Land cover variation (requires land_cover input)
    use_wall_scheme: bool = False  # Wall temperature scheme (Wallenberg et al. 2025)
    conifer: bool = False          # Evergreen trees (True) vs deciduous (False)
    person_cylinder: bool = True   # Cylindric (True) vs box (False) human geometry

# Usage (planned):
results = solweig.calculate_timeseries(
    surface, weather,
    config=solweig.ModelConfig(use_aniso=True, use_veg=True),
    output_dir="output/",
)
```

**Output Specification (addresses U4):**

| Output Name | Description                        | Units   | File Created                          |
|-------------|------------------------------------|---------|---------------------------------------|
| `tmrt`      | Mean Radiant Temperature           | °C      | `tmrt_YYYYMMDD_HHMM.tif`              |
| `utci`      | Universal Thermal Climate Index    | °C      | `utci_YYYYMMDD_HHMM.tif`              |
| `pet`       | Physiologically Equivalent Temp    | °C      | `pet_YYYYMMDD_HHMM.tif`               |
| `shadow`    | Shadow fraction (0=sun, 1=shadow)  | 0-1     | `shadow_YYYYMMDD_HHMM.tif`            |
| `kdown`     | Downward shortwave radiation       | W/m²    | `kdown_YYYYMMDD_HHMM.tif`             |
| `kup`       | Upward shortwave radiation         | W/m²    | `kup_YYYYMMDD_HHMM.tif`               |
| `ldown`     | Downward longwave radiation        | W/m²    | `ldown_YYYYMMDD_HHMM.tif`             |
| `lup`       | Upward longwave radiation          | W/m²    | `lup_YYYYMMDD_HHMM.tif`               |

**Default outputs:** If `outputs` parameter omitted, only `tmrt` is saved (most common use case).

**Post-Processing Functions:**

| Function | Purpose | Speed | Notes |
|----------|---------|-------|-------|
| `compute_utci(tmrt_dir, weather, output_dir)` | UTCI from saved Tmrt files | Fast (~60 steps/s) | Polynomial approximation |
| `compute_pet(tmrt_dir, weather, output_dir, human)` | PET from saved Tmrt files | Slow (~1 step/s) | Iterative solver |
| `compute_utci_grid(tmrt, ta, rh, wind)` | In-memory UTCI for single grid | Instant | For custom workflows |
| `compute_pet_grid(tmrt, ta, rh, wind, human)` | In-memory PET for single grid | Fast | For custom workflows |

**Sample Data Download (for quick start):**

```python
# Download pre-computed sample data (Athens demo ~15MB)
solweig.download_sample_data(output_dir="demo/")
# Downloads: demo/dsm.tif, demo/weather.epw, demo/preprocessed/walls/, demo/preprocessed/svf/

# Also available as CLI command:
# $ solweig download-sample-data --output ./demo
```

**Validation Helper (catch errors early):**

```python
# Validate inputs before expensive computation
issues = solweig.validate_inputs(
    dsm="dsm.tif",
    cdsm="cdsm.tif",
    walls_dir="walls/",
    svf_dir="svf/",
)
if issues:
    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
    # Example output:
    #   ERROR: DSM and CDSM have different CRS (EPSG:32633 vs EPSG:4326)
    #   WARNING: SVF directory missing shadowmats.npz (required for use_aniso=True)
```

**Debug Mode (for troubleshooting):**

```python
# Enable verbose logging for debugging
solweig.calculate(
    surface, weather,
    output_dir="output/",
    verbose=True,  # Logs intermediate values
)
# Logs: "Loading DSM: 1000x1000, EPSG:32633"
# Logs: "Sun position: altitude=45.2°, azimuth=180.5°"
# Logs: "Shadow calculation: 15% building shadow"
# Logs: "Timestep 1/24: mean Tmrt=42.3°C, max=58.1°C"
```

**Working Directory Structure:**

After `SurfaceData.prepare()` creates preprocessing cache:

```
working/
├── walls/
│   ├── wall_hts.tif           # Wall heights (metres)
│   └── wall_aspects.tif       # Wall orientations (degrees, 0=N)
├── svf/
│   ├── svfs.zip               # Sky View Factor arrays (16 directions)
│   └── shadowmats.npz         # Shadow matrices (for use_anisotropic_sky=True)
└── resampled/                 # Resampled inputs (for inspection)
    ├── dsm_resampled.tif
    └── cdsm_resampled.tif
```

**Cache Behavior:**

- Second run with same `working_dir`: Instant (loads from cache)
- Use `force_recompute=True` to regenerate walls/SVF
- Delete `working/` directory to clear cache

**Pre-flight Validation (catch errors before expensive computation):**

```python
surface = solweig.SurfaceData.from_geotiff(dsm="dsm.tif", ...)
issues = surface.validate()
# Returns: ValidationResult with warnings/errors
#   - CRS mismatch between DSM and CDSM
#   - Resolution mismatch (DSM 1m, CDSM 2m)
#   - Extent mismatch
#   - Missing SVF files for anisotropic mode
#   - NoData pixels in DSM
if issues.errors:
    print(issues.summary())  # Human-readable error list
```

**Preprocessing Workflow (addresses E1, U2):**

Preprocessing is **automatic** via `SurfaceData.prepare()`:
- Computes walls and SVF on first run
- Caches to `working_dir` for reuse
- Second run is instant (loads from cache)

**Performance Estimates:**

| Grid Size     | Wall Generation | SVF (no veg) | SVF (with CDSM) | Total (First Run) |
|---------------|-----------------|--------------|-----------------|-------------------|
| 250×250       | ~5 sec          | ~30 sec      | ~45 sec         | ~1 min            |
| 500×500       | ~15 sec         | ~2 min       | ~3 min          | ~3-4 min          |
| 1000×1000     | ~45 sec         | ~10 min      | ~15 min         | ~15-20 min        |
| 2000×2000     | ~3 min          | ~45 min      | ~60 min         | ~1 hour           |

*Times measured on Apple M1 with GPU acceleration. CPU-only may be 2-3x slower.*

**Example File Structure:**

```text
project/
├── data/
│   ├── dsm.tif                    # Input: Digital Surface Model
│   ├── cdsm.tif                   # Input: Vegetation (optional)
│   ├── weather.epw                # Input: Weather data
│   └── working/                   # Cache: Auto-created by prepare()
│       ├── walls/
│       │   ├── wall_hts.tif
│       │   └── wall_aspects.tif
│       ├── svf/
│       │   ├── svfs.zip
│       │   └── shadowmats.npz
│       └── resampled/
│           ├── dsm_resampled.tif
│           └── cdsm_resampled.tif
├── output/                        # Results: Tmrt files
│   ├── tmrt_20230701_1200.tif
│   └── ...
└── output_utci/                   # Post-processing: UTCI files
    ├── utci_20230701_1200.tif
    └── ...
```

**Advanced: Manual Preprocessing (for power users)**

If you need fine control, you can still call preprocessing functions directly:

```python
# Manual preprocessing (advanced)
solweig.walls.generate_wall_hts(dsm_path="dsm.tif", out_dir="walls/")
solweig.svf.generate_svf(dsm_path="dsm.tif", cdsm_path="cdsm.tif", out_dir="svf/")

# Then load manually
surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    working_dir="manual_preprocessing/",
    force_recompute=False,  # Use existing walls/svf from working_dir
)
```

---

### Phase 4: POI Mode (Performance)

**Goal:** 10-50x speedup for point-based calculations (revised from 100x based on engineering review)

**Owner:** Engineer

**Raster vs POI Mode:**

| Mode | Use Case | Outputs | Speed |
|------|----------|---------|-------|
| **Raster** | Heat maps, spatial analysis, planning | Full GeoTIFFs (Tmrt, UTCI, PET, etc.) | Baseline |
| **POI** | Sensor validation, specific locations | Values at points only | 10-50x faster |

Both modes compute UTCI/PET - the difference is whether you need the full spatial grid or just specific points.

**When to use POI mode:**
- Validating against field measurements at weather stations
- Quick thermal comfort assessment at specific locations
- Batch processing many scenarios where only summary stats are needed

**When to use Raster mode:**
- Creating heat maps for urban planning
- Identifying hot spots across an area
- Any spatial analysis requiring the full grid

| Task | Description                                                 | Priority |
| ---- | ----------------------------------------------------------- | -------- |
| 4.1  | Add `poi_coords` parameter to `calculate()`                 | HIGH     |
| 4.2  | Implement shadow-at-point (localized ray-casting)           | HIGH     |
| 4.3  | Implement SVF-at-point (sample rays, ±5% tolerance)         | HIGH     |
| 4.4  | Lat/lon to pixel coordinate conversion                      | HIGH     |
| 4.5  | POI-to-tile ownership mapping (avoid duplicate computation) | MEDIUM   |
| 4.6  | Benchmark: target 10-50x speedup depending on POI density   | HIGH     |
| 4.7  | Validation: compare POI results to grid extraction (S4)     | HIGH     |

**Target API:**

```python
points = [(57.705, 11.985), (57.708, 11.990)]  # lat/lon
results = solweig.calculate_at_points(surface, weather, points)
# Returns: [{"lat": 57.705, "lon": 11.985, "tmrt": 42.3, "utci": 35.1}, ...]

# For timeseries at a point (sensor validation workflow)
timeseries = solweig.calculate_timeseries_at_point(surface, weather, point=(57.705, 11.985))
# Returns: pandas DataFrame with timestamp, tmrt, utci, etc.
```

**POI Mode Accuracy (addresses S4):**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| SVF at point | ±5% of grid value | Compare 100 random points to full grid extraction |
| Shadow at point | Exact match | Binary comparison at known shadow boundaries |
| Tmrt at point | ±0.5K of grid | Statistical comparison across test cases |

---

### Phase 5: Code Simplification (Engineering)

**Goal:** Reduce complexity, improve maintainability

**Owner:** Engineer

| Task | Description                                                   | File           | Priority |
| ---- | ------------------------------------------------------------- | -------------- | -------- |
| 5.1  | Decompose `_calculate_core()` into composable functions       | api.py         | HIGH     |
| 5.2  | Create structured bundles (SvfBundle, RadiationBundle)        | api.py         | HIGH     |
| 5.3  | Fix ShadowArrays property caching (allocates on every access) | api.py:858-871 | HIGH     |
| 5.4  | Replace 70-parameter function with dataclass bundles          | runner.py      | MEDIUM   |
| 5.5  | Lazy SVF loading (load directions on-demand)                  | configs.py     | MEDIUM   |
| 5.6  | Add tile overlap validation (tile_size > 2\*overlap)          | tiles.py       | LOW      |

**Decomposition target for `_calculate_core()`:**

```python
def _calculate_core(surface, location, weather, params, precomputed, ...):
    svf_bundle = _compute_svf(surface, precomputed, ...)
    shadow = _compute_shadows(surface, weather, ...)
    tg_bundle = _compute_ground_temperature(surface, weather, location, params)
    gvf_bundle = _compute_gvf(svf_bundle, shadow, surface, ...)
    radiation = _compute_radiation(svf_bundle, shadow, gvf_bundle, weather, ...)
    tmrt = _compute_tmrt(radiation, human_params)
    return SolweigResult(tmrt=tmrt, ...)
```

---

### Phase 6: Memory Optimization

**Goal:** Handle large rasters (10k x 10k) without memory exhaustion

**Owner:** Engineer

| Task | Description                                           | Est. Memory Savings    |
| ---- | ----------------------------------------------------- | ---------------------- |
| 6.1  | Ensure float32 throughout (check before converting)   | 50% for float64 inputs |
| 6.2  | Shadow storage as uint8 (0-255 scale)                 | 75% for shadow arrays  |
| 6.3  | Compute `diffsh` lazily instead of pre-allocating     | 25% for aniso mode     |
| 6.4  | Streaming shadow patches (don't load all 153 at once) | 2+ GB for large grids  |
| 6.5  | Zero-copy Rust returns via PyO3 `as_pyarray()`        | Variable               |
| 6.6  | Memory-mapped SVF for tiled processing                | Variable               |

**Memory budget for 1000x1000 grid:**

| Component       | Current | Target             |
| --------------- | ------- | ------------------ |
| DSM             | 4 MB    | 4 MB               |
| SVF (16 arrays) | 64 MB   | 4 MB (lazy)        |
| Shadow (aniso)  | 2.3 GB  | 150 MB (streaming) |
| Total           | ~2.5 GB | ~200 MB            |

---

### Phase 7: Rust Optimization

**Goal:** Move hot loops to Rust, minimize Python/Rust boundary crossings

**Owner:** Engineer

| Task | Description                                                       | Priority |
| ---- | ----------------------------------------------------------------- | -------- |
| 7.1  | Batch processing entry point `calculate_timesteps()` in Rust      | HIGH     |
| 7.2  | Move hourly loop into Rust (avoid per-timestep boundary crossing) | HIGH     |
| 7.3  | Zero-copy array returns from Rust                                 | MEDIUM   |
| 7.4  | GPU fallback handling (fail gracefully to CPU)                    | MEDIUM   |
| 7.5  | Document GPU build requirements                                   | LOW      |

---

### Phase 8: Scientific Documentation

**Goal:** Complete specification coverage with proper references

**Owner:** Scientist

| Task | Description                                                                            | Priority |
| ---- | -------------------------------------------------------------------------------------- | -------- |
| 8.1  | Add missing references (see list below)                                                | HIGH     |
| 8.2  | Document ground temperature model (TsWaveDelay)                                        | HIGH     |
| 8.3  | Fix formula discrepancies (spec vs implementation)                                     | MEDIUM   |
| 8.4  | Document 18 identified edge cases                                                      | MEDIUM   |
| 8.5  | Create new spec files (diffuse_fraction.md, surface_temperature.md, sky_emissivity.md) | LOW      |

**Missing References to Add:**

| Formula                   | Required Citation                |
| ------------------------- | -------------------------------- |
| Diffuse fraction          | Reindl et al. (1990)             |
| Sky emissivity            | Brutsaert (1975) or Prata (1996) |
| Anisotropic sky           | Perez et al. (1993)              |
| Clearness index           | Crawford & Duchon (1999)         |
| Cylindric projection      | Fanger (1970) or ISO 7726        |
| Surface temperature       | Lindberg et al. (2015)           |
| Vegetation transmissivity | Lindberg & Grimmond (2011)       |
| Clothing area factor      | ISO 9920                         |

**Formula Discrepancies to Fix:**

1. **Tmrt offset:** Spec uses -273.15, implementation uses -273.2
2. **Ldown formula:** Spec shows simplified, implementation has full wall/veg terms
3. **SVF+vegetation:** Transmissivity-weighted combination not in spec
4. **Ground temperature:** Not documented in any spec

---

### Phase 9: QGIS Integration

**Goal:** First-class QGIS plugin support with Rust acceleration and minimal dependencies

**Owner:** User + Engineer

| Task | Description                                          | Priority |
| ---- | ---------------------------------------------------- | -------- |
| 9.1  | Progress callback API for QGIS QProgressDialog       | HIGH     |
| 9.2  | Make pvlib optional (implement solar position internally) | HIGH  |
| 9.3  | Replace rasterio with osgeo.gdal for QGIS compatibility | HIGH  |
| 9.4  | Make geopandas optional (use osgeo.ogr for vectors)  | HIGH     |
| 9.5  | Example QGIS Processing script                       | MEDIUM   |
| 9.6  | Thread safety audit (no global state mutation)       | MEDIUM   |
| 9.7  | Document QGIS Python console usage                   | LOW      |
| 9.8  | **Rust extension QGIS compatibility testing**        | HIGH     |
| 9.9  | Build wheels for QGIS Python versions (3.9-3.12)     | HIGH     |
| 9.10 | Helpful error messages when Rust import fails        | MEDIUM   |

**Dependency Minimization Strategy (for QGIS portability):**

Goal: Use only QGIS-bundled packages where possible to avoid pip install headaches.

| Current Dep    | QGIS-bundled? | Strategy                                           | Status  |
|----------------|---------------|----------------------------------------------------|---------|
| `numpy`        | ✅ Yes        | Keep (core dependency)                             | ✅ Done |
| `scipy`        | ✅ Yes        | Keep (interpolation)                               | ✅ Done |
| `pandas`       | ✅ Yes        | Keep (weather data)                                | ✅ Done |
| `pyproj`       | ✅ Yes        | Keep (CRS transforms)                              | ✅ Done |
| `shapely`      | ✅ Yes        | Keep (geometry ops)                                | ✅ Done |
| `osgeo.gdal`   | ✅ Yes        | Use instead of rasterio in QGIS                    | ✅ Done |
| `rasterio`     | ❌ No         | Optional; use osgeo.gdal via `UMEP_USE_GDAL=1`     | ✅ Done |
| `pvlib`        | ❌ No         | Not used; standalone EPW parser implemented        | ✅ Done |
| `geopandas`    | ❌ No         | Optional; only for POI/WOI features                | ✅ Done |
| `tqdm`         | ❌ No         | Optional; uses QGIS feedback or silent fallback    | ✅ Done |

**Rasterio/GDAL Compatibility Layer:**

```python
# pysrc/solweig/io.py
def _get_raster_backend():
    """Use osgeo.gdal in QGIS, rasterio otherwise."""
    try:
        from qgis.core import QgsApplication
        # We're in QGIS - use osgeo.gdal
        from osgeo import gdal
        return "gdal"
    except ImportError:
        # Standalone Python - prefer rasterio
        try:
            import rasterio
            return "rasterio"
        except ImportError:
            from osgeo import gdal
            return "gdal"
```

**Solar Position Without pvlib:**

pvlib is only used for sun position (altitude/azimuth). This is standard astronomy that can be implemented internally:

```python
# pysrc/solweig/sun.py
def solar_position(timestamp, latitude, longitude):
    """
    Calculate solar position using NOAA Solar Calculator algorithm.
    Eliminates pvlib dependency for QGIS compatibility.

    Reference: https://gml.noaa.gov/grad/solcalc/calcdetails.html
    """
    # ~50 lines of trig, already exists in legacy UMEP code
    ...
    return altitude, azimuth
```

**QGIS Plugin Distribution Options:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: pip install** | Plugin requires `pip install solweig` first | Simple to maintain | Users must use terminal |
| **B: Bundled wheels** | Include pre-built Rust wheels in plugin zip | No pip needed | Large plugin (50-100MB), complex CI |
| **C: Auto-install** | Plugin downloads/installs solweig on first run | Best UX | Requires network, may need admin |
| **D: Hybrid** | Bundle for common platforms, auto-install fallback | Good coverage | Complex build process |

**Recommended: Option D (Hybrid)**

```
qgis_plugin/
├── __init__.py           # Plugin entry point with install logic
├── metadata.txt          # QGIS plugin metadata
├── wheels/               # Pre-built wheels for common platforms
│   ├── solweig-0.1.0-cp311-cp311-win_amd64.whl
│   ├── solweig-0.1.0-cp311-cp311-macosx_arm64.whl
│   └── solweig-0.1.0-cp311-cp311-manylinux_x86_64.whl
└── provider.py           # QgsProcessingProvider
```

```python
# qgis_plugin/__init__.py
import sys
import subprocess
from pathlib import Path

def classFactory(iface):
    try:
        import solweig
        from .provider import SolweigProvider
        return SolweigProvider(iface)
    except ImportError:
        # Try bundled wheel first
        plugin_dir = Path(__file__).parent
        wheel = _find_matching_wheel(plugin_dir / "wheels")

        if wheel:
            # Install from bundled wheel (no network needed)
            subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel)])
        else:
            # Fall back to PyPI (needs network)
            from qgis.PyQt.QtWidgets import QMessageBox
            reply = QMessageBox.question(None, "Install SOLWEIG",
                "SOLWEIG package not found. Install from PyPI?")
            if reply == QMessageBox.Yes:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "solweig"])
            else:
                _show_manual_instructions()
                return None

        import solweig
        from .provider import SolweigProvider
        return SolweigProvider(iface)

def _find_matching_wheel(wheels_dir):
    """Find wheel matching current platform and Python version."""
    import platform
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = {"Darwin": "macosx", "Windows": "win", "Linux": "linux"}[platform.system()]

    for wheel in wheels_dir.glob("*.whl"):
        if py_ver in wheel.name and plat in wheel.name:
            return wheel
    return None
```

**QGIS Plugin Architecture (same repo, layered design):**

The plugin lives in the same repository as the core package, enabling parallel use from Python directly or via QGIS:

```text
solweig/
├── pysrc/solweig/              # Core Python package
│   ├── __init__.py             # Main API
│   ├── api.py                  # calculate(), calculate_iter()
│   ├── io.py                   # Rasterio/GDAL compatibility layer
│   ├── sun.py                  # Solar position (no pvlib needed)
│   └── qgis/                   # QGIS-specific code (optional import)
│       ├── __init__.py
│       ├── provider.py         # QgsProcessingProvider
│       └── algorithms.py       # Processing algorithms wrapping core API
├── rust/                       # Rust implementation
├── qgis_plugin/                # Thin wrapper for QGIS plugin manager
│   ├── __init__.py             # classFactory + install logic
│   ├── metadata.txt            # QGIS plugin metadata
│   └── wheels/                 # Pre-built wheels (optional)
└── pyproject.toml
```

**Usage modes:**

```python
# 1. Direct Python usage (researchers, scripts)
import solweig
solweig.calculate(surface, weather, output_dir="output/")

# 2. QGIS Python console
import solweig
# Same API, but uses osgeo.gdal instead of rasterio automatically

# 3. QGIS Processing toolbox
# User interacts via GUI, plugin calls solweig internally
```

**Rust Extension QGIS Compatibility (Critical for Performance):**

This package is **Rust-only** - no pure-Python fallback is maintained. Focus effort on making Rust work everywhere.

| Compatibility Issue   | Problem                                                   | Resolution                                      |
| --------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| **Python version**    | QGIS bundles specific Python (varies by platform/version) | Build wheels for Python 3.9, 3.10, 3.11, 3.12   |
| **NumPy ABI**         | Rust compiled against NumPy 1.x may crash with 2.x        | Use `numpy>=1.20,<3` in build, test both        |
| **GDAL conflicts**    | rasterio links different GDAL than QGIS                   | Use `osgeo.gdal` when in QGIS, rasterio otherwise |
| **OSGeo4W (Windows)** | Special DLL paths, environment setup                      | Test in OSGeo4W shell, document PATH requirements |
| **GPU/CUDA**          | CUDA may not be available in QGIS env                     | Graceful fallback to CPU (Phase 7.4)            |

**Error Handling (no fallback - Rust required):**

```python
# In solweig/__init__.py
try:
    from solweig._rust import calculate_rust
except ImportError as e:
    raise ImportError(
        f"Failed to load solweig Rust extension: {e}\n\n"
        "This package requires the compiled Rust extension.\n"
        "Possible fixes:\n"
        "  1. Install from PyPI: pip install solweig (includes pre-built wheels)\n"
        "  2. Check Python version matches wheel (need 3.9-3.12)\n"
        "  3. On Windows QGIS: run from OSGeo4W shell\n"
        "  4. For pure-Python version: use solweig-legacy package\n"
    ) from e
```

**Legacy Package (separate repo):**
- Users who cannot run Rust extension should use `solweig-legacy` (the older pure-Python UMEP implementation)
- This repo (`solweig`) is Rust-accelerated only
- No effort spent maintaining Python parity

**QGIS Testing Matrix:**

| Platform | QGIS Version | Python | Test Status |
|----------|--------------|--------|-------------|
| Windows (OSGeo4W) | 3.34 LTR | 3.12.8 | Required |
| Windows (OSGeo4W) | 3.36+ | 3.12.x | Required |
| macOS (Homebrew) | 3.34+ | 3.11+ | Required |
| Linux (Ubuntu) | 3.34+ | 3.10+ | Required |

**OSGeo4W v2 Package Versions (January 2025):**

Source: [OSGeo4W v2 repository](http://download.osgeo.org/osgeo4w/v2/x86_64/release/)

| Package | Version         | Notes                              |
| ------- | --------------- | ---------------------------------- |
| Python  | 3.12.8          | Updated from 3.9.18 in April 2024  |
| NumPy   | 1.26.4          | No NumPy 2.x yet                   |
| SciPy   | 1.13.0          |                                    |
| Pandas  | 2.2.2 / 2.3.1   |                                    |
| GDAL    | 3.8.x - 3.12.x  | Use osgeo.gdal, not rasterio       |
| PyProj  | 3.6.x - 3.7.x   |                                    |
| Shapely | 2.0.x           |                                    |

Note: The April 2024 OSGeo4W update rebuilt everything with Visual C++ 2022,
moving from Python 3.9.18 to 3.12.3. This removed legacy GRASS 7 support.

**Progress callback API:**

```python
def calculate(surface, location, weather, progress_callback=None):
    """
    progress_callback: Callable[[int, int, str], bool]
        - Called with (current_step, total_steps, description)
        - Return False to cancel operation
    """
```

---

### Phase 10: Documentation

**Goal:** Comprehensive documentation for users, developers, and scientists

**Owner:** User + Scientist

| Task  | Description                                              | Priority |
| ----- | -------------------------------------------------------- | -------- |
| 10.1  | **Quick Start Guide** - 5-minute intro with sample data  | HIGH     |
| 10.2  | **User Guide** - Complete workflow documentation         | HIGH     |
| 10.3  | **API Reference** - Auto-generated from docstrings       | HIGH     |
| 10.4  | **Scientific Manual** - Model theory and equations       | MEDIUM   |
| 10.5  | **QGIS Tutorial** - Step-by-step plugin usage            | MEDIUM   |
| 10.6  | **Developer Guide** - Contributing, architecture         | LOW      |
| 10.7  | **Migration Guide** - From UMEP/legacy to solweig        | MEDIUM   |
| 10.8  | **Changelog** - Version history with breaking changes    | HIGH     |

**Documentation Structure:**

```text
docs/
├── index.md                    # Landing page with quick links
├── getting-started/
│   ├── installation.md         # pip, conda, QGIS plugin
│   ├── quick-start.md          # 5-minute tutorial
│   └── sample-data.md          # Download and use demo data
├── user-guide/
│   ├── preprocessing.md        # Wall heights, SVF generation
│   ├── running-solweig.md      # Main calculation workflow
│   ├── outputs.md              # Understanding Tmrt, UTCI, PET
│   ├── model-options.md        # Configuring anisotropic, vegetation
│   └── troubleshooting.md      # Common errors and fixes
├── qgis/
│   ├── installation.md         # Plugin installation guide
│   ├── tutorial.md             # Step-by-step QGIS workflow
│   ├── processing-tools.md     # Each algorithm explained
│   └── tips.md                 # QGIS-specific tips
├── api/
│   ├── reference.md            # Auto-generated API docs
│   ├── examples.md             # Code examples by use case
│   └── advanced.md             # Iterator API, callbacks
├── science/
│   ├── overview.md             # Model theory overview
│   ├── equations.md            # All equations with citations
│   ├── validation.md           # Comparison with measurements
│   └── references.md           # Full bibliography
├── development/
│   ├── contributing.md         # How to contribute
│   ├── architecture.md         # Code organization
│   ├── rust-python.md          # Rust/Python boundary
│   └── testing.md              # Test infrastructure
└── changelog.md                # Version history
```

**Documentation Tools:**

| Tool | Purpose |
|------|---------|
| **MkDocs + Material** | Documentation site generator |
| **mkdocstrings** | Auto-generate API docs from docstrings |
| **mike** | Documentation versioning |
| **GitHub Pages** | Hosting at `umep-dev.github.io/solweig` |

**GitHub Pages Deployment:**

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths: ['docs/**', 'pysrc/**/*.py']  # Rebuild on doc or code changes
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for mike versioning

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install mkdocs-material mkdocstrings[python] mike

      - name: Configure git
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Deploy docs
        run: |
          # For releases: deploy versioned docs
          if [[ "${{ github.event_name }}" == "release" ]]; then
            VERSION="${{ github.ref_name }}"
            mike deploy --push --update-aliases $VERSION latest
          else
            # For main branch: deploy dev docs
            mike deploy --push dev
          fi
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

permissions:
  contents: write
  pages: write
```

**Documentation URLs:**

| Version | URL |
|---------|-----|
| Latest stable | `https://umep-dev.github.io/solweig/latest/` |
| Development | `https://umep-dev.github.io/solweig/dev/` |
| Specific version | `https://umep-dev.github.io/solweig/0.2.0/` |

**Quick Start Guide Outline:**

```markdown
# Quick Start (5 minutes)

## 1. Install
pip install solweig

## 2. Download Sample Data
solweig download-sample-data --output ./demo

## 3. Run Preprocessing (one-time)
import solweig
solweig.preprocess(dsm="demo/dsm.tif", output_dir="demo/preprocessed/")

## 4. Calculate Tmrt
surface = solweig.SurfaceData.from_geotiff(
    dsm="demo/dsm.tif",
    walls_dir="demo/preprocessed/walls/",
    svf_dir="demo/preprocessed/svf/",
)
weather = solweig.Weather.from_epw("demo/weather.epw", start="2023-07-01", end="2023-07-01")
solweig.calculate(surface, weather, output_dir="output/")

## 5. View Results
# Open output/tmrt_20230701_1200.tif in QGIS or Python
```

---

### Phase 11: QGIS Plugin Development

**Goal:** Full-featured QGIS Processing plugin with GUI

**Owner:** User + Engineer

| Task  | Description                                              | Priority |
| ----- | -------------------------------------------------------- | -------- |
| 11.1  | **Plugin skeleton** - metadata.txt, __init__.py, icons   | HIGH     |
| 11.2  | **Processing Provider** - Register with QGIS toolbox     | HIGH     |
| 11.3  | **Preprocessing Algorithm** - Walls + SVF generation     | HIGH     |
| 11.4  | **SOLWEIG Algorithm** - Main calculation with GUI        | HIGH     |
| 11.5  | **POI Algorithm** - Point-based calculation              | MEDIUM   |
| 11.6  | **Batch Processing** - Multiple scenarios                | MEDIUM   |
| 11.7  | **Result Styling** - Auto-apply color ramps to outputs   | MEDIUM   |
| 11.8  | **Temporal Support** - Load outputs as temporal layer    | HIGH     |
| 11.9  | **Help Integration** - Context-sensitive help links      | MEDIUM   |
| 11.10 | **Plugin Installer** - Auto-install solweig package      | HIGH     |

**Temporal Layer Support (elevated priority per user feedback):**

QGIS temporal layers allow animation and time-slider navigation of multi-timestep outputs:

```python
# In SOLWEIG algorithm output handling:
def _add_temporal_outputs(self, output_dir, context, feedback):
    """Add outputs as temporal layer group with time awareness."""
    from qgis.core import QgsRasterLayer, QgsProject
    from datetime import datetime

    # Load all timestep outputs
    for tif in sorted(Path(output_dir).glob("tmrt_*.tif")):
        # Parse timestamp from filename: tmrt_20230701_1200.tif
        ts_str = tif.stem.split("_", 1)[1]  # "20230701_1200"
        timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M")

        layer = QgsRasterLayer(str(tif), f"Tmrt {timestamp}")
        # Set temporal properties
        layer.temporalProperties().setIsActive(True)
        layer.temporalProperties().setFixedTemporalRange(
            QgsDateTimeRange(timestamp, timestamp)
        )
        QgsProject.instance().addMapLayer(layer)
```

This enables:
- Time slider in QGIS to animate through hours
- Temporal controller for playback
- Comparison of different time periods

**QGIS Processing Algorithms:**

| Algorithm | Inputs | Outputs | Description |
|-----------|--------|---------|-------------|
| **Preprocess DSM** | DSM, CDSM (opt) | walls/, svf/ | One-time preprocessing |
| **Run SOLWEIG** | Surface layers, EPW, dates | Tmrt, UTCI, etc. | Main thermal comfort calculation |
| **SOLWEIG at Points** | Surface, EPW, points layer | Point layer with attributes | POI mode |
| **Batch SOLWEIG** | Folder of inputs | Folder of outputs | Multi-scenario |

**Algorithm Implementation Pattern:**

```python
# pysrc/solweig/qgis/algorithms.py
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
)
import solweig

class SolweigPreprocessAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing algorithm for preprocessing."""

    DSM = 'DSM'
    CDSM = 'CDSM'
    OUTPUT_DIR = 'OUTPUT_DIR'
    COMPUTE_ANISO = 'COMPUTE_ANISO'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.DSM, 'Digital Surface Model (DSM)', optional=False))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.CDSM, 'Canopy DSM (vegetation)', optional=True))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIR, 'Output directory'))
        self.addParameter(QgsProcessingParameterBoolean(
            self.COMPUTE_ANISO, 'Generate anisotropic sky data', defaultValue=True))

    def processAlgorithm(self, parameters, context, feedback):
        dsm_layer = self.parameterAsRasterLayer(parameters, self.DSM, context)
        cdsm_layer = self.parameterAsRasterLayer(parameters, self.CDSM, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        compute_aniso = self.parameterAsBool(parameters, self.COMPUTE_ANISO, context)

        # Progress callback for QGIS feedback
        def on_progress(step, total, message):
            feedback.setProgress(int(100 * step / total))
            feedback.pushInfo(message)
            return not feedback.isCanceled()

        solweig.preprocess(
            dsm=dsm_layer.source(),
            cdsm=cdsm_layer.source() if cdsm_layer else None,
            output_dir=output_dir,
            compute_aniso=compute_aniso,
            progress_callback=on_progress,
        )

        return {self.OUTPUT_DIR: output_dir}

    def name(self):
        return 'preprocess_dsm'

    def displayName(self):
        return 'Preprocess DSM for SOLWEIG'

    def group(self):
        return 'SOLWEIG'

    def groupId(self):
        return 'solweig'

    def shortHelpString(self):
        return """Generate wall heights/aspects and Sky View Factor from DSM.
        This is a one-time preprocessing step required before running SOLWEIG.
        For a 1000x1000 grid, expect ~15-20 minutes processing time."""
```

**Main SOLWEIG Algorithm GUI:**

```python
class SolweigRunAlgorithm(QgsProcessingAlgorithm):
    """Main SOLWEIG thermal comfort calculation."""

    DSM = 'DSM'
    CDSM = 'CDSM'
    WALLS_DIR = 'WALLS_DIR'
    SVF_DIR = 'SVF_DIR'
    EPW_FILE = 'EPW_FILE'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    HOURS = 'HOURS'
    USE_ANISO = 'USE_ANISO'
    OUTPUTS = 'OUTPUTS'
    OUTPUT_DIR = 'OUTPUT_DIR'

    def initAlgorithm(self, config=None):
        # Input layers
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.DSM, 'Digital Surface Model'))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.CDSM, 'Canopy DSM', optional=True))

        # Preprocessing paths
        self.addParameter(QgsProcessingParameterFile(
            self.WALLS_DIR, 'Walls directory', behavior=1))  # 1 = folder
        self.addParameter(QgsProcessingParameterFile(
            self.SVF_DIR, 'SVF directory', behavior=1))

        # Weather
        self.addParameter(QgsProcessingParameterFile(
            self.EPW_FILE, 'EPW weather file', extension='epw'))
        self.addParameter(QgsProcessingParameterString(
            self.START_DATE, 'Start date (YYYY-MM-DD)', defaultValue='2023-07-01'))
        self.addParameter(QgsProcessingParameterString(
            self.END_DATE, 'End date (YYYY-MM-DD)', defaultValue='2023-07-01'))
        self.addParameter(QgsProcessingParameterString(
            self.HOURS, 'Hours (comma-separated, empty=all)',
            defaultValue='6,7,8,9,10,11,12,13,14,15,16,17,18', optional=True))

        # Options
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_ANISO, 'Use anisotropic sky model', defaultValue=False))
        self.addParameter(QgsProcessingParameterEnum(
            self.OUTPUTS, 'Outputs to generate',
            options=['tmrt', 'utci', 'pet', 'shadow', 'kdown'],
            allowMultiple=True, defaultValue=[0]))

        # Output
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIR, 'Output directory'))

    def processAlgorithm(self, parameters, context, feedback):
        # ... implementation with progress callbacks
        pass
```

**Plugin metadata.txt:**

```ini
[general]
name=SOLWEIG
qgisMinimumVersion=3.28
description=High-performance urban thermal comfort modelling (Rust-accelerated)
version=0.1.0
author=UMEP Developers
email=umep-dev@googlegroups.com
about=Calculate Mean Radiant Temperature (Tmrt), UTCI, and PET for urban environments.
      Rust-accelerated implementation of the SOLWEIG model.
tracker=https://github.com/UMEP-dev/solweig/issues
repository=https://github.com/UMEP-dev/solweig
tags=urban climate, thermal comfort, SOLWEIG, Tmrt, UTCI, PET, microclimate
homepage=https://umep-docs.readthedocs.io/
category=Analysis
icon=icons/solweig.png
experimental=True
deprecated=False
hasProcessingProvider=yes
```

**Plugin Release Workflow:**

```yaml
# .github/workflows/qgis-plugin.yml
name: Build QGIS Plugin

on:
  release:
    types: [published]

jobs:
  build-plugin:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Build wheels for all platforms
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist

      # Package plugin with wheels
      - name: Package plugin
        run: |
          mkdir -p qgis_plugin/wheels
          cp dist/*.whl qgis_plugin/wheels/
          cd qgis_plugin && zip -r ../solweig-qgis-${{ github.ref_name }}.zip .

      # Upload to GitHub release
      - name: Upload plugin
        uses: softprops/action-gh-release@v1
        with:
          files: solweig-qgis-*.zip
```

---

## 3. Priority Matrix

| Phase                   | Priority      | Dependencies    | Est. Effort |
| ----------------------- | ------------- | --------------- | ----------- |
| **Phase 3: UX**         | P0 (Critical) | None            | Medium      |
| **Phase 4: POI**        | P0 (Critical) | Phase 3.5       | High        |
| **Phase 5: Simplify**   | P1 (High)     | None            | High        |
| **Phase 6: Memory**     | P1 (High)     | Phase 5         | Medium      |
| **Phase 7: Rust**       | P2 (Medium)   | Phase 5         | High        |
| **Phase 8: Sci Docs**   | P2 (Medium)   | None            | Low         |
| **Phase 9: QGIS Compat**| P1 (High)     | Phase 3         | Medium      |
| **Phase 10: User Docs** | P1 (High)     | Phase 3         | Medium      |
| **Phase 11: QGIS Plugin**| P2 (Medium)  | Phase 9, 10     | High        |

---

## 4. Implementation Schedule

### Sprint 0: Prerequisites (MUST complete before Phase 3)

**NumPy ABI Compatibility (Critical):**

- [ ] Test Rust extension with NumPy 1.26.x (current QGIS)
- [ ] Test Rust extension with NumPy 2.0+ (future-proofing)
- [ ] Configure maturin build for ABI compatibility
- [ ] Add CI matrix for NumPy version testing

**Sample Data Bundle:**

- [ ] Create pre-computed walls/SVF for Athens demo
- [ ] Package as GitHub release asset (~10MB)
- [ ] Update demos to use pre-computed data option

**Wheel Build Verification:**

- [ ] Verify CI builds wheels for Python 3.9, 3.10, 3.11, 3.12
- [ ] Test wheel installation on Windows, macOS, Linux
- [ ] Document any platform-specific issues

---

### Sprint 1: User Experience Foundation (Priority: Critical)

**Critical Bug Fixes (from Section 1.7):**

- [ ] **B1** Fix `SurfaceData.from_files()` tuple unpacking (api.py:647-661)
- [ ] **B2** Implement internal EPW parser (replace `pvlib.iotools.read_epw()`)
- [ ] **B3** Implement runtime GPU detection with graceful CPU fallback
- [ ] Rename `from_files()` to `from_geotiff()` for API consistency

**SurfaceData & Weather Loading:**

- [ ] 3.1 `SurfaceData.from_geotiff()` with all inputs
- [ ] 3.2 `Weather.from_epw()` with date range validation, `year=None` for TMY (U5)
- [ ] 3.9 Input validation: CRS match, extent match, resolution match (E5)
- [ ] 3.10 `solweig.preprocess()` unified wrapper
- [ ] 3.11 `SurfaceData.validate()` pre-flight validation
- [ ] 3.13 `solweig.validate_inputs()` standalone validation helper (user request)

**Model Options & Parameters:**

- [ ] 3.5 `ModelOptions` dataclass (S2, U3)
- [ ] 3.6 `params` parameter with bundled defaults (E2)

**Scientific Fixes (elevated from Phase 8):**

- [ ] S1 Fix Kelvin offset: -273.2 → -273.15 with `use_legacy_kelvin_offset` config option
- [ ] S5 Document TsWaveDelay constant (33.27) with citation
- [ ] Create `surface_temperature.md` spec
- [ ] Create `diffuse_fraction.md` spec (Reindl model)

### Sprint 2: Output & Migration

**Output Control:**

- [ ] 3.3 `SolweigResult.to_geotiff()` with output specification (U4)
- [ ] 3.4 Auto-extract location from DSM CRS (verify existing)

**Migration Support:**

- [ ] 3.7 `from_config()` migration helper (E4, U1)
- [ ] 3.8 Improved error messages with context

**Documentation:**

- [ ] 8.1 Add missing references to specs (S3, S5)
- [ ] Document Perez coefficients variant in use (S3)
- [ ] Document TsWaveDelay constant (S5)

### Sprint 3: POI Mode Core

- [ ] 4.1 Add `poi_coords` parameter
- [ ] 4.2 Shadow-at-point (skip full grid)
- [ ] 4.3 SVF-at-point (sample rays) with defined tolerance (S4)
- [ ] 4.4 Lat/lon to pixel conversion
- [ ] 4.6 Validation: POI vs grid extraction comparison (S4)

### Sprint 4: Code Health

- [ ] 5.1 Decompose `_calculate_core()` (900+ lines → <200 per function)
- [ ] 5.3 Fix ShadowArrays caching (allocates on every access)
- [ ] 6.1 Ensure float32 throughout Rust/Python boundary

### Sprint 5: Performance & QGIS

- [ ] 6.2 Shadow storage as uint8
- [ ] 7.1 Rust batch processing
- [ ] 9.1 Progress callback API for QGIS

### Sprint 6: QGIS Compatibility (Critical)

**Dependency Minimization:**

- [x] 9.2 pvlib not used; standalone EPW parser already implemented ✅ COMPLETE
- [x] 9.3 Create rasterio/osgeo.gdal compatibility layer (auto-detect environment) ✅ COMPLETE
- [x] 9.4 Make geopandas optional; use osgeo.ogr for vector operations ✅ COMPLETE
- [x] Make tqdm optional; use QGIS QgsProcessingFeedback or no-op fallback ✅ COMPLETE

**Completed QGIS Compatibility Work (January 2026):**

1. **Progress Abstraction** (`progress.py`): Auto-detects environment:
   - QGIS: Uses `QgsProcessingFeedback` for native progress bar
   - Terminal: Uses `tqdm` when available
   - Fallback: Silent iteration when neither available

2. **Optional Dependencies** (`pyproject.toml`):
   - Core dependencies: numpy, pandas, scipy, pyproj, shapely (all QGIS-bundled)
   - Optional `[full]` extras: geopandas, rasterio, tqdm
   - Optional `[qgis]` extras: none (uses bundled packages)

3. **Graceful Fallbacks**:
   - `runner.py`: geopandas optional with helpful error for POI/WOI features
   - `io.py`: GDAL backend via `UMEP_USE_GDAL=1` environment variable
   - `io.py`: Standalone EPW parser (no pvlib dependency)
   - All progress loops: Use `progress.py` abstraction

**Rust Extension Testing:**

- [ ] 9.8 Test Rust extension in QGIS environments (OSGeo4W, macOS, Linux)
- [ ] 9.9 Build wheels for QGIS Python versions (3.9, 3.10, 3.11, 3.12)
- [ ] 9.10 Helpful error messages when Rust import fails (point to solweig-legacy)

**Plugin Distribution:**

- [ ] Create QGIS plugin structure (qgis_plugin/ directory)
- [ ] Implement hybrid install: bundled wheels + PyPI fallback
- [ ] Build CI pipeline to create wheels for common platforms
- [ ] Test plugin install workflow on Windows, macOS, Linux

### Sprint 7: Documentation Foundation

**Setup & Quick Start:**

- [ ] 10.1 Set up MkDocs + Material theme
- [ ] 10.1 Configure GitHub Pages deployment workflow
- [ ] 10.1 Write Quick Start guide with sample data
- [ ] 10.2 Write installation guide (pip, conda, QGIS)
- [ ] 3.14 Implement `solweig.download_sample_data()` CLI and Python API
- [ ] 3.15 Implement debug/verbose mode for troubleshooting

**User Guide:**

- [ ] 10.2 Document preprocessing workflow
- [ ] 10.2 Document main calculation workflow
- [ ] 10.2 Document output interpretation (Tmrt, UTCI, PET)
- [ ] 10.2 Write troubleshooting guide

**API Reference:**

- [ ] 10.3 Configure mkdocstrings for auto-generated docs
- [ ] 10.3 Add docstrings to all public API functions
- [ ] 10.3 Write code examples for common use cases

### Sprint 8: QGIS Plugin Development

**Core Plugin:**

- [ ] 11.1 Create plugin skeleton (metadata.txt, __init__.py, icons)
- [ ] 11.2 Implement QgsProcessingProvider
- [ ] 11.10 Implement auto-install logic for solweig package

**Processing Algorithms:**

- [ ] 11.3 Implement Preprocess DSM algorithm
- [ ] 11.4 Implement Run SOLWEIG algorithm
- [ ] 11.5 Implement SOLWEIG at Points algorithm (POI mode)

**Polish:**

- [ ] 11.9 Add help strings to all algorithms
- [ ] 11.7 Implement result layer styling (color ramps)
- [ ] 11.8 Implement temporal layer support (HIGH priority per user feedback)
- [ ] Test full workflow in QGIS 3.34 LTR

### Sprint 9: Documentation Completion & Release

**QGIS Documentation:**

- [ ] 10.5 Write QGIS plugin installation tutorial
- [ ] 10.5 Write step-by-step QGIS workflow tutorial
- [ ] 10.5 Document each Processing algorithm

**Scientific Documentation:**

- [ ] 10.4 Write model theory overview
- [ ] 10.4 Document all equations with citations
- [ ] 10.4 Add validation section with measurement comparisons

**Migration & Developer Docs:**

- [ ] 10.7 Write migration guide from UMEP/legacy
- [ ] 10.6 Write contributing guide
- [ ] 10.8 Create changelog with version history

**Release:**

- [ ] Publish v0.2.0 to PyPI
- [ ] Publish plugin to QGIS Plugin Repository
- [ ] Announce on UMEP mailing list

---

## 5. Success Metrics

| Metric                          | Current | Phase 3 Target | Phase 7 Target |
| ------------------------------- | ------- | -------------- | -------------- |
| Lines for basic use             | 10      | 4              | 4              |
| POI speedup (10 pts vs 1M grid) | 1x      | 10-50x         | 10-50x         |
| Memory (1k x 1k, aniso)         | ~2.5 GB | ~2.5 GB        | ~200 MB        |
| Timestep boundary crossings     | N       | N              | 1              |
| Spec reference coverage         | 70%     | 90%            | 100%           |
| QGIS usability score            | 7/10    | 8/10           | 9/10           |

**Validation Tolerances (addresses scientific review):**

| Output | Tolerance | Reference Standard |
|--------|-----------|-------------------|
| Tmrt | ±0.1 K | vs UMEP Python reference |
| UTCI | ±0.5 K | vs Fiala full model |
| PET | ±1.0 K | vs Mayer-Höppe model |
| SVF | ±0.01 | vs analytical hemisphere |
| POI SVF | ±5% of grid | statistical validation |

---

## 6. Reporting Structure

### Weekly Check-ins

Each stakeholder reports on their assigned phases:

**Engineer Report:**

- Tasks completed this week
- Blockers encountered
- Memory/performance measurements
- Code review requests

**Scientist Report:**

- Specs updated
- References verified
- Formula discrepancies found/fixed
- Edge cases documented

**User Report:**

- API tested from user perspective
- Pain points discovered
- Documentation gaps
- QGIS testing results

### Definition of Done (per task)

- [ ] Implementation complete
- [ ] Tests added/updated
- [ ] Specs updated (if applicable)
- [ ] Parity maintained (run `test_api_exact_parity.py`)
- [ ] Memory usage verified (for Phase 6 tasks)
- [ ] QGIS tested (for Phase 9 tasks)

---

## 7. Key Files Reference

| File                                                 | Purpose                     | Owner     |
| ---------------------------------------------------- | --------------------------- | --------- |
| [pysrc/solweig/api.py](pysrc/solweig/api.py)         | Simplified API (Phase 3, 5) | Engineer  |
| [pysrc/solweig/runner.py](pysrc/solweig/runner.py)   | Reference runner (Phase 5)  | Engineer  |
| [pysrc/solweig/io.py](pysrc/solweig/io.py)           | I/O operations (Phase 3, 9) | Engineer  |
| [pysrc/solweig/tiles.py](pysrc/solweig/tiles.py)     | Tiling system (Phase 6)     | Engineer  |
| [pysrc/solweig/configs.py](pysrc/solweig/configs.py) | Configuration (Phase 5, 6)  | Engineer  |
| [rust/src/lib.rs](rust/src/lib.rs)                   | Rust bindings (Phase 7)     | Engineer  |
| [specs/\*.md](specs/)                                | Specifications (Phase 8)    | Scientist |
| [tests/test_api.py](tests/test_api.py)               | Unit tests                  | All       |

---

## 8. Risk Register

| Risk                           | Impact | Mitigation                              |
| ------------------------------ | ------ | --------------------------------------- |
| Breaking parity with reference | HIGH   | Run parity test after every change      |
| **Rust ext fails in QGIS**     | HIGH   | Build wheels for all platforms, test matrix, helpful errors |
| **NumPy ABI mismatch**         | HIGH   | Pin numpy version, test both 1.x and 2.x |
| QGIS DLL conflicts             | MEDIUM | Use osgeo.gdal in QGIS, test in OSGeo4W |
| Memory regression              | MEDIUM | Add memory benchmarks to CI             |
| Python version mismatch        | MEDIUM | Build wheels for 3.9, 3.10, 3.11, 3.12  |
| POI mode numerical differences | LOW    | Document acceptable tolerance           |
| Rust compilation issues        | LOW    | Publish pre-built wheels to PyPI        |

---

## 9. Appendix: Complexity Hotspots

### Top 5 Functions to Refactor

1. **`_calculate_core()`** - api.py:1356-2172 (816 lines)
2. **`calc_solweig()`** - runner.py:430-539 (70+ params)
3. **`SolweigRunCore.__init__()`** - runner.py:86-163 (config explosion)
4. **`SvfData.__init__()`** - configs.py:389-483 (16 separate loads)
5. **`ShadowArrays.shmat`** - api.py:858-871 (allocation on access)

### Memory Hotspots

1. Anisotropic shadow matrices: 2.3 GB for 1k x 1k x 153 patches
2. SVF 16-direction arrays: 64 MB for 1k x 1k
3. GVF result extraction: 15 separate array copies
4. Redundant float32 conversions throughout

---

## 10. Previous Phase History

### Phase 1: API Simplification ✅ COMPLETE

- Created simple markdown specs
- Defined property-based tests
- Documented physical invariants

### Phase 2: Config Consolidation ✅ COMPLETE

- Created `SurfaceData`, `Location`, `Weather`, `HumanParams` dataclasses
- Added internal sun position calculation
- Added Reindl diffuse fraction model
- **Achieved 100% parity with reference implementation**

### Bug Fixes Applied (Phase 2)

1. Zenith angle units - converted degrees to radians for cylindric_wedge
2. Anisotropic diffuse radiation - use correct drad variable
3. Land cover properties - load from params JSON instead of hardcoded
