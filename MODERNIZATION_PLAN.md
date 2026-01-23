# SOLWEIG Modernization Plan

**Updated: January 2026**

## Core Principles

1. **Simplicity first** - Minimize parameters, auto-compute what we can
2. **Spec-driven testing** - Simple markdown specs, handwritten tests
3. **POI-focused** - Support computing only at points of interest
4. **Progressive complexity** - Simple defaults, advanced options available
5. **QGIS compatible** - GDAL support, minimal external dependencies

---

## Current Architecture (Post-Processing Model)

**Main Calculation (Inline):**

```python
# Computes Tmrt for all timesteps
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    use_anisotropic_sky=True,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
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

- Tmrt is always needed → computed inline for performance
- UTCI/PET are optional → many users only need Tmrt
- PET is expensive → 50× slower than UTCI due to iterative solver
- Flexibility → compute UTCI/PET for subset of timesteps or different parameters
- No thermal state dependency → UTCI/PET are pure functions, safe to post-process

---

## Phase 5: Middle Layer Refactoring - ✅ **COMPLETE**

**Goal:** Streamline the middle layer between user API and Rust internals to enable easier Rust migration

**Status:** All phases complete (5.1-5.6). Legacy code deleted (6,100 lines). api.py reduced from 3,976 → 256 lines (93.6%). Test suite: 146/146 passing. Completed January 23, 2026.

**Why This Matters:**

- `_calculate_core()` is 841 lines doing 14 different things - unmaintainable
- `calc_solweig()` takes 70+ positional parameters - error-prone
- ShadowArrays allocates 98MB on every property access - memory waste
- Unclear boundaries between Python orchestration and Rust computation

### Target Architecture (4 Layers)

```
Layer 1: User API (UNCHANGED)
    calculate(), calculate_timeseries()
    SurfaceData, Location, Weather, HumanParams

Layer 2: Orchestration (NEW - computation.py)
    _compute_timeseries_batch() - manages state across timesteps
    _compute_single_timestep() - coordinates all components

Layer 3: Component Functions (NEW - components/)
    svf_resolution.py:  resolve_svf() -> SvfBundle
    shadows.py:         compute_shadows() -> ShadowBundle
    ground.py:          compute_ground_temperature() -> GroundBundle
    gvf.py:             compute_gvf() -> GvfBundle
    radiation.py:       compute_radiation() -> RadiationBundle
    tmrt.py:            compute_tmrt() -> TmrtResult

Layer 4: Rust Computation (UNCHANGED)
    shadowing, skyview, gvf, sky, vegetation, utci, pet
```

### Implementation Tasks

| Phase                              | Task                                                    | Risk   | Priority | Status      | Files                                 |
| ---------------------------------- | ------------------------------------------------------- | ------ | -------- | ----------- | ------------------------------------- |
| **Phase 1: Data Structures**       |                                                         |        |          |             |                                       |
| 5.1.1                              | Create `bundles.py` with new dataclasses                | LOW    | HIGH     | ✅ COMPLETE | pysrc/solweig/bundles.py              |
| 5.1.2                              | Fix ShadowArrays property caching (98MB waste)          | LOW    | HIGH     | ✅ COMPLETE | api.py:1539-1607                      |
| **Phase 2: Component Extraction**  |                                                         |        |          |             |                                       |
| 5.2.1                              | Extract ground temperature → `components/ground.py`     | MEDIUM | HIGH     | ✅ COMPLETE | components/ground.py                  |
| 5.2.2                              | Extract SVF resolution → `components/svf_resolution.py` | MEDIUM | HIGH     | ✅ COMPLETE | components/svf_resolution.py          |
| 5.2.3                              | Extract shadow computation → `components/shadows.py`    | MEDIUM | HIGH     | ✅ COMPLETE | components/shadows.py                 |
| 5.2.4                              | Extract GVF calculation → `components/gvf.py`           | MEDIUM | HIGH     | ✅ COMPLETE | components/gvf.py                     |
| 5.2.5                              | Extract radiation → `components/radiation.py`           | MEDIUM | HIGH     | ✅ COMPLETE | components/radiation.py               |
| 5.2.6                              | Extract Tmrt calculation → `components/tmrt.py`         | MEDIUM | HIGH     | ✅ COMPLETE | components/tmrt.py                    |
| **Phase 3: Orchestration**         |                                                         |        |          |             |                                       |
| 5.3.1                              | Create new orchestration in `computation.py`            | LOW    | HIGH     | ✅ COMPLETE | pysrc/solweig/computation.py          |
| 5.3.2                              | Add feature flag to switch old/new paths                | LOW    | HIGH     | ⏭️ SKIPPED  | Not needed - clean break from legacy  |
| 5.3.3                              | Remove legacy `_calculate_core`                         | LOW    | MEDIUM   | ✅ COMPLETE | api.py (deleted 843 lines)            |
| **Phase 4: Runner Simplification** |                                                         |        |          |             |                                       |
| 5.4                                | Simplify legacy runner                                  | MEDIUM | LOW      | ⏭️ SKIPPED  | Legacy API deleted instead            |
| **Phase 5: API Organization**      |                                                         |        |          |             |                                       |
| 5.5.1                              | Extract dataclasses to `models.py`                      | LOW    | HIGH     | ✅ COMPLETE | pysrc/solweig/models.py (2,238 lines) |
| 5.5.2                              | Extract orchestration to modular files                  | LOW    | HIGH     | ✅ COMPLETE | metadata.py, timeseries.py, etc.      |
| 5.5.3                              | Extract utilities to `utils.py`, `config.py`            | LOW    | HIGH     | ✅ COMPLETE | utils.py, config.py                   |
| 5.5.4                              | Reduce api.py to re-exports only                        | LOW    | HIGH     | ✅ COMPLETE | api.py (3,976 → 256 lines)            |
| **Phase 6: Legacy Deletion**       |                                                         |        |          |             |                                       |
| 5.6.1                              | Delete legacy runner, configs, wrappers                 | LOW    | HIGH     | ✅ COMPLETE | 6,100 lines removed                   |
| 5.6.2                              | Update tests, remove legacy comparisons                 | LOW    | HIGH     | ✅ COMPLETE | 146/146 tests passing                 |
| 5.6.3                              | Verify Athens demo with modern API                      | LOW    | HIGH     | ✅ COMPLETE | 2.03 steps/s, 72 timesteps            |

### Refactored Core (841 lines → 50 lines)

```python
def _calculate_core(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    use_anisotropic_sky: bool,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
) -> SolweigResult:
    """Core calculation - orchestrates component functions."""

    # Early exit for nighttime
    if weather.sun_altitude <= 0:
        return _nighttime_result(surface, weather, state)

    # Prepare inputs
    veg_bundle = prepare_vegetation(surface)
    wall_bundle = resolve_walls(surface, precomputed)
    psi = compute_transmissivity(weather.datetime, physics)

    # SVF resolution
    svf_bundle = resolve_svf(surface, precomputed, veg_bundle, psi)

    # Shadow computation
    shadow_bundle = compute_shadows(weather, surface, veg_bundle, wall_bundle, psi)

    # Ground temperature model
    ground_bundle = compute_ground_temperature(
        weather, location, surface, materials, shadow_bundle
    )

    # GVF calculation
    gvf_bundle = compute_gvf(surface, wall_bundle, shadow_bundle, ground_bundle, weather, human)

    # Apply thermal delay
    lup_bundle = apply_thermal_delay(gvf_bundle, state)

    # Radiation calculation
    radiation_bundle = compute_radiation(
        weather, svf_bundle, shadow_bundle, gvf_bundle, lup_bundle,
        human, use_anisotropic_sky, precomputed
    )

    # Final Tmrt
    tmrt = compute_tmrt(radiation_bundle, human)

    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow_bundle.shadow,
        kdown=radiation_bundle.kdown,
        kup=radiation_bundle.kup,
        ldown=radiation_bundle.ldown,
        lup=lup_bundle.lup,
        state=lup_bundle.state,
    )
```

### Testing Strategy

Every refactoring step includes:

1. Run `test_api_exact_parity.py` before and after (100% parity required)
2. Add component-level parity tests (compare extracted function to inline version)
3. Keep legacy code path with feature flag until validated
4. Memory profiling to verify caching improvements

### Success Metrics

| Metric                             | Before                            | After                | Status      |
| ---------------------------------- | --------------------------------- | -------------------- | ----------- |
| `_calculate_core` lines            | 841                               | 50                   | ✅ ACHIEVED |
| Max function length                | 841 (api.py)                      | 532 (computation.py) | ✅ ACHIEVED |
| api.py lines                       | 3,976                             | 256 (-93.6%)         | ✅ ACHIEVED |
| `calc_solweig` parameters          | 70+                               | N/A (deleted)        | ✅ ACHIEVED |
| Files >1000 lines                  | 3 (api.py, configs.py, runner.py) | 1 (models.py)        | ✅ ACHIEVED |
| Memory per ShadowArrays access     | 98 MB                             | 0 (cached)           | ✅ ACHIEVED |
| SVF loading code paths             | 3 (duplicated)                    | 1 (unified)          | ✅ ACHIEVED |
| Python/Rust crossings per timestep | ~15                               | ~8                   | ✅ ACHIEVED |
| Legacy code lines                  | ~8,500                            | ~2,400 (-6,100)      | ✅ ACHIEVED |

### Key Achievements (Completed January 23, 2026)

#### Phase 5.5: API Organization

- Created [models.py](pysrc/solweig/models.py) (2,238 lines) - All 11 dataclasses extracted
- Created [computation.py](pysrc/solweig/computation.py) (532 lines) - Core orchestration logic
- Created [timeseries.py](pysrc/solweig/timeseries.py) (237 lines) - Batch time series processing
- Created [tiling.py](pysrc/solweig/tiling.py) (382 lines) - Large raster tiling support
- Created [postprocess.py](pysrc/solweig/postprocess.py) (314 lines) - UTCI/PET thermal comfort indices
- Created [metadata.py](pysrc/solweig/metadata.py) (143 lines) - Run provenance tracking
- Created [config.py](pysrc/solweig/config.py) (206 lines) - Parameter loading
- Created [utils.py](pysrc/solweig/utils.py) (182 lines) - Utility functions
- Reduced [api.py](pysrc/solweig/api.py) from 3,976 → 256 lines (93.6% reduction)

#### Phase 5.6: Legacy Deletion

- Deleted `runner.py` (1,847 lines) - Legacy config-driven runner
- Deleted `configs.py` (1,234 lines) - Legacy config loading
- Deleted `functions.py` (987 lines) - Legacy wrapper functions
- Deleted `wrappers.py` (834 lines) - Legacy compatibility layer
- Deleted obsolete tests and benchmarks (~1,200 lines)
- Total deletion: 6,100 lines of legacy code

#### Runtime Verification

- All 146 tests passing (100% pass rate)
- Athens demo verified: 72 timesteps in 35.5s (2.03 steps/s)
- EPW parser implemented (no external dependencies)
- Cloud-Optimized GeoTIFF output working
- GPU acceleration active (Metal backend)

#### Architecture Quality

- Zero circular imports (clean dependency graph)
- Largest file now 2,238 lines (models.py with 11 dataclasses)
- Longest function now 532 lines (computation.py orchestration)
- All component functions <200 lines
- Modern API only - clean break from legacy

---

## Outstanding Engineering Issues

### Engineering Debt

- ✅ ~~`_calculate_core()` is 841 lines (monolithic)~~ - RESOLVED in Phase 5.3
- ✅ ~~`calc_solweig()` passes 70+ positional parameters (legacy runner)~~ - RESOLVED in Phase 5.6 (deleted)
- ✅ ~~EPW weather file parser dependency~~ - RESOLVED in Phase 5.6 (implemented in io.py)
- Working directory cache needs validation/invalidation strategy
- Post-processing functions need standardized progress reporting

### Scientific Gaps

- 8 missing secondary references in specs
- Ground temperature model (TsWaveDelay) undocumented
- 4 formula discrepancies between specs and implementation
- Post-processing validation tests missing (UTCI accuracy, PET convergence)

### Usability Issues

- No POI/point-based calculation API - TARGET for Phase 6
- QGIS plugin integration not documented
- Workflow documentation needs clarification (inline vs. post-processing)
- ✅ ~~Migration guide from legacy API needed~~ - NOT NEEDED (legacy API deleted)

---

## Future Phases

### Phase 6: POI Mode Implementation - **NEXT PRIORITY**

**Goal:** Enable point-of-interest calculations for 10-100× speedup over full-grid mode

**Status:** Design complete. Ready to implement (Est. 8 days).

**Key Features:**

- Shadow-at-point (localized ray-casting, no full shadow grid)
- SVF-at-point (sample-based, no full SVF grid)
- POI batch API for multiple points and timesteps
- Automatic crossover detection (use full-grid when POI count > threshold)

**Expected Performance:**

- 10 points: ~32,500× speedup vs full grid
- 100 points: ~3,250× speedup
- 1,000 points: ~325× speedup
- Crossover point: ~500 points (where full-grid becomes competitive)

**Implementation Tasks:**

| Phase | Task                                          | Est. Days | Status      |
| ----- | --------------------------------------------- | --------- | ----------- |
| 6.1   | POI data structures and API design            | 1         | Not started |
| 6.2   | Shadow-at-point (Rust ray-casting)            | 2         | Not started |
| 6.3   | SVF-at-point (Rust sample-based)              | 2         | Not started |
| 6.4   | POI orchestration and batch processing        | 1         | Not started |
| 6.5   | Validation against full-grid (tolerance test) | 2         | Not started |

### Phase 7: Memory Optimization

**Goal:** Handle large rasters (10k x 10k) without memory exhaustion

- Ensure float32 throughout
- Shadow storage as uint8
- Streaming shadow patches
- Memory-mapped SVF for tiled processing

### Phase 8: Rust Optimization

**Goal:** Move hot loops to Rust, minimize Python/Rust boundary crossings

- Batch processing entry point in Rust
- Move hourly loop into Rust
- Zero-copy array returns
- GPU fallback handling

### Phase 9: Scientific Documentation

**Goal:** Complete specification coverage with proper references

- Add missing references
- Document ground temperature model
- Fix formula discrepancies
- Document edge cases

### Phase 10: QGIS Integration

**Goal:** First-class QGIS plugin support

- Progress callback API for QGIS
- Dependency minimization (use QGIS-bundled packages)
- Runtime GPU detection with graceful fallback
- Build wheels for QGIS Python versions

### Phase 11: Documentation

**Goal:** Comprehensive user and developer documentation

- Quick Start Guide
- User Guide with complete workflow
- API Reference (auto-generated)
- QGIS Tutorial

### Phase 12: QGIS Plugin Development

**Goal:** Full-featured QGIS Processing plugin

- Plugin skeleton with Processing Provider
- Preprocessing and main calculation algorithms
- POI mode algorithm
- Temporal layer support

---

## Key Files Reference

| File                           | Purpose                                      | Lines | Owner     |
| ------------------------------ | -------------------------------------------- | ----- | --------- |
| pysrc/solweig/api.py           | Public API re-exports (entry point)          | 256   | Engineer  |
| pysrc/solweig/models.py        | All dataclasses (11 models)                  | 2,238 | Engineer  |
| pysrc/solweig/computation.py   | Core orchestration logic                     | 532   | Engineer  |
| pysrc/solweig/timeseries.py    | Batch time series processing                 | 237   | Engineer  |
| pysrc/solweig/tiling.py        | Large raster tiling support                  | 382   | Engineer  |
| pysrc/solweig/postprocess.py   | UTCI/PET thermal comfort indices             | 314   | Engineer  |
| pysrc/solweig/metadata.py      | Run provenance tracking                      | 143   | Engineer  |
| pysrc/solweig/config.py        | Parameter loading (human/physics/materials)  | 206   | Engineer  |
| pysrc/solweig/utils.py         | Utility functions                            | 182   | Engineer  |
| pysrc/solweig/bundles.py       | Data transfer objects (DTOs)                 | 449   | Engineer  |
| pysrc/solweig/components/      | Component functions (shadows, SVF, GVF, etc.)| ~1,100| Engineer  |
| pysrc/solweig/io.py            | I/O operations, EPW parser                   | 863   | Engineer  |
| rust/src/lib.rs                | Rust bindings                                | -     | Engineer  |
| specs/\*.md                    | Specifications                               | -     | Scientist |
| tests/test_api.py              | Integration tests                            | -     | All       |

---

## Risk Register

| Risk                           | Impact | Mitigation                                  |
| ------------------------------ | ------ | ------------------------------------------- |
| Breaking parity with reference | HIGH   | Run parity test after every change          |
| Rust extension fails in QGIS   | HIGH   | Build wheels for all platforms, test matrix |
| NumPy ABI mismatch             | HIGH   | Pin numpy version, test both 1.x and 2.x    |
| Memory regression              | MEDIUM | Add memory benchmarks to CI                 |
| POI mode numerical differences | LOW    | Document acceptable tolerance               |

---

## Previous Phases (Complete)

### Phase 1: API Simplification ✅

- Created simple markdown specs
- Defined property-based tests
- Documented physical invariants

### Phase 2: Config Consolidation ✅

- Created `SurfaceData`, `Location`, `Weather`, `HumanParams` dataclasses
- Added internal sun position calculation
- Added Reindl diffuse fraction model
- **Achieved 100% parity with reference implementation**

### Phase 3: User Experience & API Simplification ✅ (Partially Complete)

**Major achievements:**

- ✅ `SurfaceData.prepare()` with auto-preprocessing & caching
- ✅ `Weather.from_epw()` with date range filtering
- ✅ `SolweigResult.to_geotiff()` with output control
- ✅ Auto-location extraction from DSM CRS
- ✅ Three-parameter model (human/physics/materials)
- ✅ Post-processing architecture (UTCI/PET separate from main loop)
- ✅ Progress reporting with timing metrics
- ✅ Working directory caching (~72× speedup potential)

**Remaining:**

- Input validation (CRS match, extent match)
- Improved error messages
- Cache metadata & validation
- Workflow documentation

### Phase 4: POI Mode ⏳ (Not Started)

**Goal:** 10-50x speedup for point-based calculations

- Shadow-at-point (localized ray-casting)
- SVF-at-point (sample rays)
- POI-to-tile ownership mapping
