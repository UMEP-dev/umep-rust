# SOLWEIG Roadmap

**Updated: February 2026**

This document outlines the development priorities for SOLWEIG.

## Priorities (in order)

1. **Scientific Rigor & Validation** - Complete specifications, add missing references, validate implementations
2. **Memory & Computational Improvements** - Optimize for large rasters and efficiency
3. **Performance (POI Mode)** - Deferred until core science is solid

---

## Next Tasks (Prioritized)

| #   | Task                                     | Section | Impact                                 | Status      |
| --- | ---------------------------------------- | ------- | -------------------------------------- | ----------- |
| 1   | ~~Result methods (compute_utci/pet)~~    | E.1     | HIGH - API discoverability             | ✅ Complete |
| 2   | ~~Location UTC offset warning~~          | E.2     | HIGH - prevents silent bugs            | ✅ Complete |
| 3   | ~~Structured errors + validate_inputs()~~ | E.4    | MEDIUM - better error messages         | ✅ Complete |
| 4   | ~~Config precedence (explicit wins)~~    | E.3     | MEDIUM - API consistency               | ✅ Complete |
| 5   | ~~API cleanup (factories, docs)~~        | E.5     | LOW - polish                           | ✅ Complete |
| 6   | Documentation update                     | D       | MEDIUM - user adoption                 | In progress |
| 7   | POI Mode                                 | C       | HIGH - 10-100x speedup                 | Deferred    |
| 8   | Cache validation (hashes)                | B.3     | LOW - safety feature                   | Deferred    |

**Current status:** Phase E (API Improvements) complete. Focus on documentation (Phase D).

### Recently Completed

| Task                                 | Section | Status      |
| ------------------------------------ | ------- | ----------- |
| np.memmap for SVF caching            | B.1     | ✅ Complete |
| Pre-allocated buffer pools           | B.1     | ✅ Complete |
| Batch thermal delay (Rust)           | B.5     | ✅ Complete |
| Constants consolidation              | B.4     | ✅ Complete |
| GVF golden tests                     | A.4     | ✅ Complete |
| Radiation golden tests (Kside/Lside) | A.4     | ✅ Complete |
| SurfaceData.prepare() refactor       | B.4     | ✅ Complete |
| Rust parameter structs               | B.5     | ✅ Complete |
| SVF auto-caching in calculate()      | B.2     | ✅ Complete |

---

## Session Log (Feb 2026)

**Latest session (Feb 3):**

- ✅ **SVF auto-caching** - Fresh-computed SVF now cached on `surface.svf` for reuse
  - First call: computes SVF (~67s for 200×200)
  - Subsequent calls: **0.3s** (210× speedup)
- ✅ Fixed `_compute_and_cache_svf()` - was referencing non-existent `svf` module
- ✅ Added `SvfArrays.from_bundle()` - converts computation result to cacheable format
- ✅ Confirmed `SurfaceData.prepare()` refactor already complete (methods split into focused functions)

**Previous session:**

- ✅ Batch thermal delay (Rust) - 6 FFI calls → 1
- ✅ Constants consolidation - `SBC`, `KELVIN_OFFSET` centralized in `constants.py`
- ✅ `as_float32()` helper - avoids unnecessary dtype copies
- ✅ Rust parameter structs - `GvfScalarParams` (20→11 params), `TmrtParams` (18→15 params)

**API cleanup:**
- `gvf.gvf_calc(arrays..., GvfScalarParams)` - clean struct-based API
- `tmrt.compute_tmrt(arrays..., TmrtParams)` - clean struct-based API
- Old 20+ param functions removed (no backward compat needed for new API)

---

## Completed Work

| Phase   | Description                        | Status                 |
| ------- | ---------------------------------- | ---------------------- |
| Phase 1 | Spec-driven testing infrastructure | ✅ Complete            |
| Phase 2 | API simplification (100% parity)   | ✅ Complete            |
| Phase 3 | User experience improvements       | ✅ Complete            |
| Phase 5 | Middle layer refactoring           | ✅ Complete (Jan 2026) |

**Key metrics achieved:**

- api.py reduced from 3,976 → 244 lines (-93.9%)
- 6,100 lines of legacy code deleted
- models.py split into models/ package (6 modules, ~2,850 lines)
- 197 tests passing (including spec validation tests)
- 100% parity with reference UMEP implementation

---

## Phase A: Scientific Rigor & Validation

**Goal:** Ensure all physics models are properly documented, referenced, and validated.

### A.1 Specification Gaps

| Gap                     | Spec File      | Status      | Notes                                           |
| ----------------------- | -------------- | ----------- | ----------------------------------------------- |
| Sky emissivity formula  | radiation.md   | ✅ Complete | Jonsson et al. 2006 formula documented          |
| Diffuse fraction model  | radiation.md   | ✅ Complete | Reindl et al. 1990 piecewise correlations       |
| Anisotropic radiation   | radiation.md   | ✅ Complete | Perez et al. 1993 sky luminance model           |
| Absorption coefficients | tmrt.md        | ✅ Complete | ISO 7726:1998 reference added                   |
| absL discrepancy        | tmrt.md + JSON | ✅ Fixed    | Updated JSON files from 0.95 → 0.97             |
| Posture view factors    | tmrt.md        | ✅ Complete | Mayer & Höppe 1987 reference, derivations added |
| SVF calculation method  | svf.md         | ✅ Complete | Patch-based method, Robinson & Stone 1990       |
| GVF calculation method  | gvf.md         | ✅ Complete | Wall integration, Lindberg et al. 2008          |

### A.2 Ground Temperature Model ✅ Complete

**Files:**

- [specs/ground_temperature.md](specs/ground_temperature.md) - NEW
- [TsWaveDelay_2015a.py](pysrc/solweig/algorithms/TsWaveDelay_2015a.py)
- [components/ground.py](pysrc/solweig/components/ground.py)

**Completed:**

- [x] Created specs/ground_temperature.md specification
- [x] Documented thermal mass parameters (decay constant 33.27 day⁻¹, τ ≈ 43 min)
- [x] Added reference: Lindberg et al. (2016)
- [x] Documented exponential decay formula: `w = exp(-33.27 × Δt)`

### A.3 Missing References ✅ Complete

All key citations have been added to specifications:

| Parameter                | Location   | Citation Added             | Status      |
| ------------------------ | ---------- | -------------------------- | ----------- |
| DuBois body surface area | pet.md     | DuBois & DuBois 1916       | ✅ Complete |
| MEMI energy balance      | pet.md     | Höppe 1984, 1999           | ✅ Complete |
| Metabolic rates          | pet.md     | ISO 8996:2021              | ✅ Complete |
| Clothing insulation      | pet.md     | ISO 9920:2007, Fanger 1970 | ✅ Complete |
| Tree transmissivity      | shadows.md | Konarska et al. 2014       | ✅ Complete |
| Trunk ratio (0.25)       | shadows.md | Lindberg & Grimmond 2011   | ✅ Complete |

### A.4 Validation Tests ✅ Complete

**Spec compliance tests:** ✅ Complete (16 tests)

- [x] Sky emissivity formula validation (Jonsson et al. 2006)
- [x] Diffuse fraction model tests (Reindl et al. 1990)
- [x] Absorption coefficient tests (ISO 7726:1998)
- [x] Posture view factor tests (Mayer & Höppe 1987)
- [x] TsWaveDelay thermal delay tests (decay constant, morning reset)

See: [tests/spec/test_radiation_formulas.py](tests/spec/test_radiation_formulas.py)

**Thermal comfort validation:** ✅ Complete (19 tests)

- [x] UTCI polynomial accuracy and stress categories
- [x] UTCI wind/humidity/radiation effects
- [x] PET solver with DuBois body surface area
- [x] PET stress categories and radiation effects
- [x] Default parameter validation

See: [tests/spec/test_thermal_comfort.py](tests/spec/test_thermal_comfort.py)

**Component validation:** ✅ Complete

- [x] GVF golden tests (physical property validation, regression detection)
- [x] Radiation component golden tests (Kside, Lside directional components)

### A.5 Formula Reconciliation ✅ Complete

**Resolved:**

- [x] absL coefficient: Updated from 0.95 → 0.97 in JSON files to match ISO 7726:1998
- [x] All specs reviewed and aligned with implementation
- [x] Validation tests confirm spec-implementation consistency

---

## Phase B: Memory & Computational Improvements

**Goal:** Handle large rasters efficiently without compromising accuracy.

### B.1 Memory Optimization (HIGH priority)

| Issue               | Current                | Target              | Approach                           | Status             |
| ------------------- | ---------------------- | ------------------- | ---------------------------------- | ------------------ |
| Array precision     | Mixed float32/float64  | float32 throughout  | Audit and convert                  | ✅ Complete        |
| Shadow storage      | float32                | float32             | Continuous values (transmissivity) | ⚠️ Cannot compress |
| SVF caching         | Full arrays in memory  | Memory-mapped files | Use np.memmap for tiled processing | ✅ Complete (Feb 2026) |
| Intermediate arrays | Allocated per timestep | Pre-allocated pools | Reuse buffers                      | ⏳ Pending         |

**Tasks:**

- [x] Audit all array allocations for dtype consistency (53 allocations fixed)
- [x] Investigate shadow compression - **Finding:** Shadow masks are NOT binary due to vegetation transmissivity formula `shadow = bldg_sh - (1 - veg_sh) * (1 - psi)` where psi is continuous (0.03-0.5)
- [x] Memory profiling script created: `scripts/profile_memory.py`
- [x] Benchmark memory usage - **Results (Feb 2026):**
  - ~370 bytes/pixel peak memory at scale
  - 800×800 grid: 225 MB peak
  - Estimated 10k×10k: **34 GB** (requires optimization before large-scale use)
  - Memory overhead decreases with grid size (fixed module overhead amortized)
- [x] Add memory profiling to CI (tests/benchmarks/test_memory_benchmark.py)

**Bug fixes (Feb 2026):**

- [x] Fixed missing imports in tiling.py (`SurfaceData`, `PrecomputedData`, `SolweigResult`, `SimpleNamespace`)
- [x] Fixed Rust function call in ground.py (positional vs keyword arguments)
- [x] Fixed `max_height` to include vegetation heights for buffer calculation

### B.2 Computational Efficiency (MEDIUM priority)

| Optimization                     | Benefit                    | Approach                                       | Status                                     |
| -------------------------------- | -------------------------- | ---------------------------------------------- | ------------------------------------------ |
| Reduce Python/Rust crossings     | Less FFI overhead          | Batch operations in Rust                       | ⏳ Pending                                 |
| Lazy SVF loading                 | Faster startup             | Load on first access                           | ✅ Already implemented                     |
| ~~Parallel timestep processing~~ | ~~Better CPU utilization~~ | ~~Process independent timesteps concurrently~~ | ❌ Not feasible (thermal state dependency) |
| **Altmax caching**               | 17x faster timeseries      | Cache max sun altitude per day                 | ✅ Complete (Feb 2026)                     |
| **SVF auto-caching**             | **210× faster repeats**    | Cache fresh-computed SVF on surface object     | ✅ Complete (Feb 2026)                     |

**Completed optimizations:**

- [x] **SVF auto-caching** (Feb 3, 2026) - Fresh-computed SVF is now cached back to `surface.svf` after first `calculate()` call. Subsequent calls reuse cached SVF. Result: **210× speedup** on repeat timesteps (67s → 0.3s for 200×200 grid).

- [x] **Altmax caching** - Weather.compute_derived() iterated 96 times to find max sun altitude. For timeseries, this is now computed once per unique day and cached. Result: **17.6x speedup** for weather pre-computation (4.04s → 0.23s for 72 timesteps).

- [x] **SVF lazy loading** - SVF resolution checks cached/precomputed sources before computing fresh (see [components/svf_resolution.py](pysrc/solweig/components/svf_resolution.py)).

### B.3 Cache Validation (LOW priority)

Working directory cache needs validation/invalidation strategy:

- [ ] Store input hashes with cached data
- [ ] Validate cache on load
- [ ] Clear stale cache automatically

### B.4 Code Quality (LOW priority)

Optional refactoring for maintainability. No behavioral changes.

| Task                               | Goal                                                     | Status      |
| ---------------------------------- | -------------------------------------------------------- | ----------- |
| Q.1 Constants consolidation        | Eliminate duplicate `SBC = 5.67e-8` etc. across 5+ files | ✅ Complete |
| Q.2 Models package split           | Split 2,238-line models.py into modules                  | ✅ Complete |
| Q.3 Tiling consolidation           | Merge duplicate tiling implementations                   | ✅ Complete |
| Q.4 SurfaceData.prepare() refactor | Break 400+ line method into focused functions            | ✅ Complete |

**Q.4 Details:** `prepare()` is now ~50 lines of orchestration calling focused helpers:
`_load_and_validate_dsm()`, `_load_terrain_rasters()`, `_load_preprocessing_data()`,
`_align_rasters()`, `_create_surface_instance()`, `_compute_and_cache_walls()`, `_compute_and_cache_svf()`

### B.5 Rust FFI Optimization (LOW priority)

Optional Rust improvements to reduce Python/Rust crossing overhead.

| Task                       | Goal                                                           | Status                          |
| -------------------------- | -------------------------------------------------------------- | ------------------------------- |
| Batch thermal delay        | Combine 6 `ts_wave_delay` calls into 1                         | ✅ Complete                     |
| Rust parameter structs     | Replace 29-param functions with structs                        | ✅ Complete                     |
| Fused radiation+tmrt       | Combine radiation calc + Tmrt in single Rust call              | Deferred (marginal gain)        |
| Parallel SVF patches       | Rayon parallelization of patch calculations                    | Deferred (SVF is one-time cost) |
| Mega-kernel                | Combine SVF→shadows→ground→GVF→radiation→Tmrt into single call | Deferred (0.3s/step is fast)    |

**Note:** With SVF auto-caching, per-timestep cost is ~0.3s for 200×200. Further FFI optimization offers diminishing returns.

---

## Phase C: Performance (Deferred)

**POI-only mode** - Deferred until scientific validation is complete.

When prioritized, this phase would enable 10-100× speedup for point-based calculations through localized ray-casting and SVF sampling. See archived MODERNIZATION_PLAN.md for detailed design.

---

## Phase D: Documentation & Integration (Future)

- Quick Start Guide
- API Reference (auto-generated)
- QGIS plugin integration
- Build wheels for multiple platforms

---

## Phase E: API Improvements (Complete)

**Goal:** Improve API ergonomics, consistency, and error handling.

**Status:** ✅ Complete (Feb 2026)

### E.1 Result Methods Pattern (P0)

Add `compute_utci()` and `compute_pet()` methods directly on `SolweigResult` for discoverability.

| Task | File | Status |
|------|------|--------|
| Add `SolweigResult.compute_utci(weather)` method | models/results.py | ✅ Complete |
| Add `SolweigResult.compute_pet(weather, human)` method | models/results.py | ✅ Complete |
| Support both `result.compute_utci(weather)` and `result.compute_utci(ta, rh, wind)` | models/results.py | ✅ Complete |
| Update README with new pattern | README.md | ✅ Complete |
| Add tests for result methods | tests/test_api.py | ✅ Complete |

**Usage after implementation:**

```python
result = solweig.calculate(surface, location, weather)

# Pattern A: Pass weather object (convenient)
utci = result.compute_utci(weather)

# Pattern B: Pass individual values (explicit)
utci = result.compute_utci(ta=25.0, rh=50.0, wind=2.0)
```

### E.2 Location Auto-Extraction Warning (P0)

Fix silent UTC offset defaulting when location is auto-extracted from CRS.

| Task | File | Status |
|------|------|--------|
| Change `Location.from_surface()` to require explicit `utc_offset` or warn | models/weather.py | ✅ Complete |
| Add warning in `calculate_timeseries()` when location=None | timeseries.py | ✅ Complete |
| Update quick-start guide with explicit location examples | docs/getting-started/quick-start.md | ✅ Complete |

**Behavior after implementation:**

```python
# This will emit a warning about UTC offset defaulting to 0
results = calculate_timeseries(surface, weather_list)  # location=None

# Recommended: explicit location
location = solweig.Location(latitude=37.98, longitude=23.73, utc_offset=2)
results = calculate_timeseries(surface, weather_list, location=location)
```

### E.3 Config Harmonization - Explicit Wins (P1)

Change precedence so explicit parameters override `config` values (Python's "explicit is better than implicit").

| Task | File | Status |
|------|------|--------|
| Change `calculate()` to let explicit params override config | api.py | ✅ Complete |
| Change `calculate_timeseries()` to let explicit params override config | timeseries.py | ✅ Complete |
| Change `use_anisotropic_sky` default to `None` (means "use config or False") | api.py | ✅ Complete |
| Add debug logging when explicit params override config | api.py | ✅ Complete |
| Document new precedence in docstrings | api.py, timeseries.py | ✅ Complete |
| Add tests for precedence behavior | tests/test_api.py | ✅ Complete |

**Current behavior (config wins):**

```python
# config.use_anisotropic_sky=True overrides explicit False - CONFUSING
calculate(..., config=config, use_anisotropic_sky=False)  # Uses True!
```

**New behavior (explicit wins):**

```python
# Explicit parameter takes precedence - INTUITIVE
calculate(..., config=config, use_anisotropic_sky=False)  # Uses False
```

### E.4 Validation & Structured Errors (P1)

Add typed exceptions and preflight validation for better error messages.

| Task | File | Status |
|------|------|--------|
| Create `errors.py` with `SolweigError`, `InvalidSurfaceData`, `GridShapeMismatch`, `MissingPrecomputedData`, `WeatherDataError` | errors.py (new) | ✅ Complete |
| Add `validate_inputs()` preflight function | api.py | ✅ Complete |
| Update `calculate()` to raise structured errors | api.py | ✅ Complete |
| Export errors in `__all__` | api.py | ✅ Complete |
| Add tests for error cases | tests/test_errors.py (new) | ✅ Complete |

**Usage after implementation:**

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field} expected {e.expected}, got {e.got}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

### E.5 API Cleanup (P2)

Minor cleanup tasks.

| Task | File | Status |
|------|------|--------|
| Remove `poi_coords` from public signature (keep internal) | api.py | ⏳ Deferred |
| Add `Weather.from_values()` factory for quick testing | models/weather.py | ✅ Complete |
| Document result methods and validation in README | README.md | ✅ Complete |

### E.6 Implementation Order

| Step | Task | Effort | Risk | Dependencies |
|------|------|--------|------|--------------|
| 1 | E.1: Result methods | 1 hour | None | - |
| 2 | E.2: Location warning | 30 min | None | - |
| 3 | E.4: errors.py + validate_inputs() | 2 hours | None | - |
| 4 | E.3: Config precedence | 2 hours | **Low** - behavioral change | - |
| 5 | E.5: API cleanup | 30 min | None | - |
| 6 | Update README and docs | 1 hour | None | E.1, E.2 |

**Total estimated effort:** ~7 hours

---

## Wish List (Future Features)

Ideas for future development, not yet prioritized.

### High Value

| Feature                    | Description                                                                                                                                              | Complexity |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **Shade Duration Mapping** | Compute hours of shade per pixel over a day. Useful for urban planning and tree placement. Builds on existing shadow infrastructure.                     | Low        |
| **WBGT Index**             | Wet Bulb Globe Temperature - occupational heat stress index (OSHA, military, sports). Formula: `WBGT = 0.7×Tw + 0.2×Tg + 0.1×Ta`. Simpler than UTCI/PET. | Low        |
| **Tree Canopy Scenarios**  | "What-if" analysis: add hypothetical trees to CDSM and quantify cooling effect. Useful for urban forestry and climate adaptation planning.               | Medium     |
| **Heat Exposure Duration** | Cumulative hours above thermal stress thresholds (e.g., UTCI > 32°C). Time-weighted exposure mapping.                                                    | Low        |

### Medium Value

| Feature                     | Description                                                   | Complexity |
| --------------------------- | ------------------------------------------------------------- | ---------- |
| **SET\* Index**             | Standard Effective Temperature - ASHRAE thermal comfort index | Medium     |
| **GeoTIFF Export**          | Export results with proper CRS metadata for GIS integration   | Low        |
| **Animation Export**        | Time-lapse visualization of Tmrt/UTCI over a day              | Medium     |
| **Weather API Integration** | Fetch real-time weather from OpenWeather, etc.                | Medium     |

### Exploratory

| Feature                      | Description                                        | Notes                                       |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------- |
| Wind comfort                 | Simple wind amplification from building geometry   | Requires wind field data or simplifications |
| Cool corridor identification | Automated detection of thermally comfortable paths | Builds on shade duration                    |
| Optimal tree placement       | Algorithmic tree positioning for maximum cooling   | Requires optimization framework             |

---

## Testing Requirements

All changes must maintain:

- **Tmrt bias < 0.1°C** vs reference implementation
- **310 tests passing** (current baseline, including spec validation tests)
- No memory regression on standard benchmarks

Gate command: `pytest tests/`

---

## File Reference

| File                                           | Purpose                 | Lines   |
| ---------------------------------------------- | ----------------------- | ------- |
| [api.py](pysrc/solweig/api.py)                 | Public API entry point  | 244     |
| [models/](pysrc/solweig/models/)               | Dataclasses (6 modules) | ~2,850  |
| [computation.py](pysrc/solweig/computation.py) | Core orchestration      | 385     |
| [components/](pysrc/solweig/components/)       | Physics modules         | ~1,365  |
| [specs/](specs/)                               | Physics specifications  | 10 files|

---

## Risk Register

| Risk                              | Impact | Mitigation                          |
| --------------------------------- | ------ | ----------------------------------- |
| Breaking parity during spec fixes | HIGH   | Run parity tests after every change |
| Memory regression                 | MEDIUM | Add memory benchmarks to CI         |
| NumPy ABI mismatch                | HIGH   | Pin version, test 1.x and 2.x       |
