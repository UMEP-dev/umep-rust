# SOLWEIG Roadmap

**Updated: February 2026**

This document outlines the development priorities for SOLWEIG.

## Priorities (in order)

1. **Scientific Rigor & Validation** - Complete specifications, add missing references, validate implementations
2. **Memory & Computational Improvements** - Optimize for large rasters and efficiency
3. **Performance (POI Mode)** - Deferred until core science is solid

---

## Next Tasks (Prioritized)

| #   | Task                                      | Section | Impact                             | Status      |
| --- | ----------------------------------------- | ------- | ---------------------------------- | ----------- |
| 1   | ~~Result methods (compute_utci/pet)~~     | E.1     | HIGH - API discoverability         | ✅ Complete |
| 2   | ~~Location UTC offset warning~~           | E.2     | HIGH - prevents silent bugs        | ✅ Complete |
| 3   | ~~Structured errors + validate_inputs()~~ | E.4     | MEDIUM - better error messages     | ✅ Complete |
| 4   | ~~Config precedence (explicit wins)~~     | E.3     | MEDIUM - API consistency           | ✅ Complete |
| 5   | ~~API cleanup (factories, docs)~~         | E.5     | LOW - polish                       | ✅ Complete |
| 6   | ~~Cache validation (hashes)~~             | B.3     | LOW - safety feature               | ✅ Complete |
| 7   | ~~Tests for `calculate_timeseries()`~~    | F.1     | HIGH - primary workflow untested   | ✅ Complete |
| 8   | ~~Fix GPU function docs~~                 | D       | HIGH - breaks new user experience  | ✅ Complete |
| 9   | ~~Fix EPW parser tests~~                  | F       | HIGH - 8 tests failing silently    | ✅ Complete |
| 10  | ~~Rename `algorithms/` → `physics/`~~     | B.4     | MEDIUM - misleading "Legacy" label | ✅ Complete |
| 11  | ~~Slim down `__all__` exports~~           | E.5     | MEDIUM - internal bundles exposed  | ✅ Complete |
| 12  | ~~Rename `config.py` → `loaders.py`~~     | B.4     | LOW - two-config ambiguity         | ✅ Complete |
| 13  | ~~Move `cylindric_wedge` to Rust~~        | G.2     | HIGH - per-timestep hotspot        | ✅ Complete |
| 14  | ~~GPU buffer reuse / persistence~~        | G.3     | HIGH - eliminates per-call alloc   | ✅ Complete |
| 15  | ~~Move aniso patch loop to Rust~~         | G.2     | MEDIUM - anisotropic mode speedup  | ✅ Complete |
| 16  | ~~QGIS plugin testing (Phase 11)~~        | D       | HIGH - blocks plugin adoption      | ✅ Complete |
| 17  | ~~Orchestration layer unit tests~~        | F.1     | MEDIUM - regression safety         | ✅ Complete |
| 18  | ~~API reference with mkdocstrings~~       | D       | MEDIUM - user adoption             | ✅ Complete |
| 19  | Field-data validation                     | H       | HIGH - scientific credibility      | In Progress |
| 20  | POI Mode                                  | C       | HIGH - 10-100x speedup             | Deferred    |

**Current status:** Phases A, B, D, E, F.1, G.2, G.3.1–G.3.2, H.1, H.2 complete. 612+ tests total. Perez sky luminance fully ported to Rust (`crate::perez::perez_v3()`). GPU acceleration covers shadows, SVF (pipelined dispatch via `svf_accumulation.wgsl`), and anisotropic sky (`anisotropic_sky.wgsl`). Fused Rust pipeline (`pipeline.compute_timestep`) handles all per-timestep computation in a single FFI call. Next: POI mode.

### Recently Completed

| Task                                 | Section | Status      |
| ------------------------------------ | ------- | ----------- |
| Cache validation (hash-based)        | B.3     | ✅ Complete |
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

**Session (Feb 6, JSON parameter integration):**

- ✅ **Full JSON parameter integration** - `parametersforsolweig.json` as single source of truth
  - Created `pysrc/solweig/data/default_materials.json` (bundled UMEP JSON with wall values filled in)
  - Auto-load materials in `calculate()` / `calculate_timeseries()` when `materials=None`
  - Wall params flow from JSON → Python → Rust: tgk_wall, tstart_wall, tmaxlst_wall
  - Rust `ground.rs`: 3 `Option<f32>` wall params with cobblestone defaults
  - Fixed phase clamping bug (unclamped to allow afternoon cooling per UMEP)
  - Fixed wall denominator division-by-zero guard
  - 5 new sinusoidal golden tests + 12 parametrized UMEP agreement tests
  - Golden report generator: HTML → Markdown rewrite, added sinusoidal section
  - Wall material defaults: Brick (TgK=0.40), Wood (TgK=0.50), Concrete (TgK=0.35)

**Session (Feb 6, validation):**

- ✅ **Field-data validation (Phase H)** - Kolumbus dataset from Zenodo (Wallenberg et al. 2025)
  - Downloaded: kolumbus.csv (wall temps), geodata (DSM/DEM/CDSM/groundcover), met forcing
  - Added `Weather.from_umep_met()` classmethod for SUEWS-format meteorological files
  - 12 validation tests: data loading (7), wall temperature (3), full pipeline (2)
  - Wall temp RMSE: 6.67°C (PB) / 8.96°C (wood) with generic params vs ~2°C in paper (tuned)
  - Full pipeline: Tmrt 31.7°C at noon (Ta=20.5°C), peak 41.8°C at 15:00
  - Confirmed land cover support exists throughout pipeline and QGIS plugin
  - Added `tests/conftest.py` to fix QGIS test imports (pre-existing sys.path issue)
  - Investigated Montpellier dataset: reduced-scale canyon, globe thermometer, needs synthetic DSM

**Session (Feb 6, continued):**

- ✅ **QGIS plugin tests (Phase 11)** - 40 tests for converters and base algorithm
  - `tests/qgis_mocks.py`: shared mock infrastructure (install/uninstall osgeo separately)
  - `tests/test_qgis_converters.py`: 25 tests (HumanParams, Weather, Location, EPW)
  - `tests/test_qgis_base.py`: 15 tests (grid validation, output paths, georeferenced save)
  - Fixed osgeo mock pollution (split install/install_osgeo to prevent cross-test contamination)
- ✅ **Orchestration unit tests (F.1)** - 57 tests for computation internals
  - `_nighttime_result()`: 13 tests (Tmrt=Ta, longwave physics, state reset)
  - `_apply_thermal_delay()`: 7 tests (state transitions, Rust FFI mock, day/night flags)
  - `_precompute_weather()`: 5 tests (altmax caching, multi-day, derived computation)
  - ThermalState/TileSpec: 11 tests, tiling helpers: 21 tests
- ✅ **API reference with mkdocstrings** - docs build with `--strict`
  - Added mkdocs/mkdocstrings[python] to dev dependencies
  - `poe docs` / `poe docs_build` tasks for local serving and strict build
  - All 6 functions + 9 dataclasses + 5 error classes auto-documented from docstrings

**Session (Feb 6, first half):**

- ✅ **GPU buffer reuse (G.3.1)** - `CachedBuffers` struct persists 17 wgpu buffers across shadow calls
  - Buffers reallocated only when grid dimensions change
  - Uses `queue.write_buffer()` instead of `create_buffer_init()` per call
- ✅ **Test infrastructure** - `poe test_quick` (221 tests, ~4 min) / `poe test_full` (357 tests)
  - `@pytest.mark.slow` on 7 modules (api, timeseries, tiling, memory, svf, gvf, wall_geometry)
  - CI expanded from 55 → 221 tests per Python version
  - `ty check` scope fixed in CI to match pre-commit hook
- ✅ **Phase G.2 complete** - Moved Python hotspots to Rust with rayon parallelism
  - `cylindric_wedge()`: per-pixel wall shadow fraction → `sky.rs`
  - `weighted_patch_sum()`: anisotropic patch summation → `sky.rs`
  - Both include low-sun guards matching Python reference
- ✅ **Type checking expanded** - `ty check` now covers all directories (pysrc/, tests/, demos/, scripts/, qgis_plugin/)
  - Fixed 8 type errors across codebase
  - Pre-commit hook and poe tasks updated to match
- ✅ Fixed real bug: QGIS converters.py `sex` field mapped to string instead of int

**Session (Feb 5):**

- ✅ **Low sun angle handling** - Fixed numerical issues at low solar altitudes
  - `Perez_v3.py`: robust handling of edge-case zenith angles
  - `cylindric_wedge.py`: clamp/guard for near-horizon sun positions
  - `io.py`: related fixes for sun position edge cases
  - Added `tests/spec/test_low_sun_angles.py` validation tests
- ✅ QGIS plugin scaffolded and documented (Phases 1-10 complete)
- ✅ MkDocs documentation site scaffolded (25 pages under `docs/`)

**Session (Feb 3):**

- ✅ **SVF auto-caching** - Fresh-computed SVF now cached on `surface.svf` for reuse
  - First call: computes SVF (~67s for 200×200)
  - Subsequent calls: **0.3s** (210× speedup)
- ✅ Fixed `_compute_and_cache_svf()` - was referencing non-existent `svf` module
- ✅ Added `SvfArrays.from_bundle()` - converts computation result to cacheable format
- ✅ Confirmed `SurfaceData.prepare()` refactor already complete (methods split into focused functions)

**Earlier sessions:**

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

- api.py reduced from 3,976 → 403 lines (-89.9%)
- 6,100 lines of legacy code deleted
- models.py split into models/ package (6 modules, ~3,080 lines)
- 612+ tests passing (including spec, golden, benchmark, and validation tests)
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

- [specs/ground_temperature.md](specs/ground_temperature.md)
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

See: [tests/spec/test_utci.py](tests/spec/test_utci.py), [tests/spec/test_pet.py](tests/spec/test_pet.py)

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

| Issue               | Current                | Target              | Approach                           | Status                 |
| ------------------- | ---------------------- | ------------------- | ---------------------------------- | ---------------------- |
| Array precision     | Mixed float32/float64  | float32 throughout  | Audit and convert                  | ✅ Complete            |
| Shadow storage      | float32                | float32             | Continuous values (transmissivity) | ⚠️ Cannot compress     |
| SVF caching         | Full arrays in memory  | Memory-mapped files | Use np.memmap for tiled processing | ✅ Complete (Feb 2026) |
| Intermediate arrays | Allocated per timestep | Pre-allocated pools | Reuse buffers                      | ⏳ Pending             |

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
| ~~Reduce Python/Rust crossings~~ | ~~Less FFI overhead~~      | ~~Batch operations in Rust~~                   | Deferred (diminishing returns)             |
| Lazy SVF loading                 | Faster startup             | Load on first access                           | ✅ Already implemented                     |
| ~~Parallel timestep processing~~ | ~~Better CPU utilization~~ | ~~Process independent timesteps concurrently~~ | ❌ Not feasible (thermal state dependency) |
| **Altmax caching**               | 17x faster timeseries      | Cache max sun altitude per day                 | ✅ Complete (Feb 2026)                     |
| **SVF auto-caching**             | **210× faster repeats**    | Cache fresh-computed SVF on surface object     | ✅ Complete (Feb 2026)                     |
| **Algorithm optimizations**      | 1.6-2x faster functions    | Vectorized numpy, pre-compute common terms     | ✅ Complete (Feb 2026)                     |

**Completed optimizations:**

- [x] **SVF auto-caching** (Feb 3, 2026) - Fresh-computed SVF is now cached back to `surface.svf` after first `calculate()` call. Subsequent calls reuse cached SVF. Result: **210× speedup** on repeat timesteps (67s → 0.3s for 200×200 grid).

- [x] **Altmax caching** - Weather.compute_derived() iterated 96 times to find max sun altitude. For timeseries, this is now computed once per unique day and cached. Result: **17.6x speedup** for weather pre-computation (4.04s → 0.23s for 72 timesteps).

- [x] **SVF lazy loading** - SVF resolution checks cached/precomputed sources before computing fresh (see [components/svf_resolution.py](pysrc/solweig/components/svf_resolution.py)).

- [x] **Algorithm optimizations** (Feb 2026) - Optimized Python algorithms:
  - `cylindric_wedge.py`: 1.6× faster via vectorized np.where and pre-computed trig values
  - `Kup_veg_2015a.py`: 2× faster via pre-computing common terms (5 sin/multiply → 1)

### B.3 Cache Validation ✅ Complete

Working directory cache now validates against input data:

- [x] Store input hashes with cached data (via `cache.py` module)
- [x] Validate cache on load (hash comparison of DSM/CDSM/pixel_size)
- [x] Clear stale cache automatically (auto-clears and recomputes if inputs changed)

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

| Task                   | Goal                                                           | Status                          |
| ---------------------- | -------------------------------------------------------------- | ------------------------------- |
| Batch thermal delay    | Combine 6 `ts_wave_delay` calls into 1                         | ✅ Complete                     |
| Rust parameter structs | Replace 29-param functions with structs                        | ✅ Complete                     |
| Fused radiation+tmrt   | Combine radiation calc + Tmrt in single Rust call              | Deferred (marginal gain)        |
| Parallel SVF patches   | Rayon parallelization of patch calculations                    | Deferred (SVF is one-time cost) |
| Mega-kernel            | Combine SVF→shadows→ground→GVF→radiation→Tmrt into single call | Deferred (0.3s/step is fast)    |

**Note:** With SVF auto-caching, per-timestep cost is ~0.3s for 200×200. Further FFI optimization offers diminishing returns.

---

## Phase C: Performance (Deferred)

**POI-only mode** - Deferred until scientific validation is complete.

When prioritized, this phase would enable 10-100× speedup for point-based calculations through localized ray-casting and SVF sampling. See archived MODERNIZATION_PLAN.md for detailed design.

---

## Phase G: GPU & Rust-Python Interface Design

**Goal:** Extend GPU acceleration beyond shadowing, adopt a principled Rust/Python boundary, and move remaining Python hotspots to Rust where the gain justifies the complexity.

### Current State (Feb 2026)

**Rust modules** (5,341 lines, 15 files):

- `shadowing.rs` (812 lines) - GPU-accelerated ray-marching via wgpu compute shader
- `skyview.rs` (550 lines) - Hemispherical SVF (calls shadowing 32-248×)
- `gvf.rs` (390 lines) - Ground view factor with wall radiation
- `sky.rs` (550 lines) - Anisotropic sky longwave
- `vegetation.rs` (800 lines) - Directional radiation from vegetation/buildings
- `ground.rs` (350 lines) - TgMaps ground temperature model
- `utci.rs` (350 lines) - Fast polynomial (125 terms)
- `pet.rs` (370 lines) - Iterative thermal comfort solver
- `tmrt.rs` (240 lines) - Mean radiant temperature integration
- Internal helpers: `sun.rs`, `patch_radiation.rs`, `emissivity_models.rs`, `sunlit_shaded_patches.rs`

**GPU status:** Three GPU-accelerated paths via wgpu compute shaders. All fall back to CPU automatically.

- `shadowing.rs` — ray-marching shadows (`shadow_propagation.wgsl`)
- `skyview.rs` — SVF accumulation with pipelined dispatch (`svf_accumulation.wgsl`)
- `aniso_gpu.rs` — anisotropic sky longwave (`anisotropic_sky.wgsl`)

**Python physics** (`physics/`) — reference implementations retained for readability and validation:

- `sun_position.py` (1,061 lines) - ASTM solar position algorithm (kept in Python: once per timestep, scalar)
- `Perez_v3.py` (313 lines) - Reference only; production uses `crate::perez::perez_v3()` in Rust
- `cylindric_wedge.py` (109 lines) - Reference only; production uses `crate::sky::cylindric_wedge()` in Rust
- `morphology.py` (188 lines) - Reference only; production uses `crate::morphology` in Rust
- `wallalgorithms.py` (158 lines) - Wall height/aspect detection (setup, not per-timestep)
- Scalars: `clearnessindex_2013b.py`, `diffusefraction.py`, `daylen.py`, etc. (kept in Python: once per timestep)

### G.1 Principled Rust/Python Boundary

**Decision framework** - move to Rust when ALL of:

1. Per-pixel computation (not scalar/once-per-timestep)
2. Called in the per-timestep hot path
3. Measurable bottleneck (>5% of timestep time)

**Keep in Python** when:

- Scalar computation (clearnessindex, diffusefraction, daylen)
- Called once per scenario, not per timestep
- Complex control flow better expressed in Python
- Debugging/readability priority outweighs performance

### G.2 Python → Rust Migration Candidates

| Priority | Function                         | Current                          | Per-Timestep?          | Expected Speedup | Effort |
| -------- | -------------------------------- | -------------------------------- | ---------------------- | ---------------- | ------ |
| **P0**   | `cylindric_wedge()`              | Python (109 lines)               | Yes, always            | 3-5×             | Low    |
| **P0**   | Anisotropic patch summation loop | Python (5 lines in radiation.py) | Yes (aniso mode)       | 5-10×            | Low    |
| **P1**   | ~~`binary_dilation()`~~          | Rust (morphology.rs)             | No (setup)             | 2.5× (measured)  | ✅ Done |
| **P2**   | ~~`Perez_v3()`~~                 | Rust (`crate::perez::perez_v3`)  | Yes (aniso mode)       | 2-3×             | ✅ Done  |
| **P3**   | `wallalgorithms.py`              | Python (158 lines)               | No (setup)             | 3-5×             | Medium |
| Keep     | `sun_position.py`                | Python (1,061 lines)             | Once/timestep (scalar) | Negligible       | —      |
| Keep     | `clearnessindex_2013b.py`        | Python (88 lines)                | Once/timestep (scalar) | Negligible       | —      |
| Keep     | `diffusefraction.py`             | Python (47 lines)                | Once/timestep (scalar) | Negligible       | —      |
| Keep     | `daylen.py`                      | Python (22 lines)                | Once/scenario (scalar) | Negligible       | —      |

**P0: cylindric_wedge → Rust**

- Currently: vectorized numpy with trig ops (tan, arctan, sqrt, cos, sin) over full 2D grid
- Why: Called every timestep, pure math, no complex control flow
- How: Add `cylindric_wedge()` to existing `sky.rs` module
- Test: Validate against golden regression tests

**P0: Patch summation loop → Rust**

- Currently: `for idx in range(lv.shape[0]): ani_lum += diffsh[:,:,idx] * lv[idx, 2]`
- Why: ~150 iterations × full grid per timestep (anisotropic mode)
- How: Add batch dot-product function to `sky.rs`
- Test: Bit-exact comparison with Python loop

**P1: binary_dilation → Rust**

- Currently: Python nested loop replacing scipy.ndimage.binary_dilation
- Why: O(rows × cols × iterations × 9) where iterations ≈ 25/pixel_size
- How: Add to a new `morphology.rs` or to `gvf.rs`
- Alternative: Could use ndarray + rayon in Rust for immediate 10×

### G.3 GPU Acceleration Roadmap

**Current GPU architecture (wgpu):**

- Framework: wgpu 27.0 (WebGPU standard, cross-platform)
- Shader: WGSL compute shader (shadow_propagation.wgsl, 346 lines)
- Context: `ShadowGpuContext` with 17 storage buffers
- Dispatch: 16×16×1 workgroups
- Lifecycle: Context persisted via `OnceLock`, but buffers recreated per call

**Phase G.3.1: GPU Buffer Reuse** (HIGH priority)

- Problem: Per-call buffer allocation overhead - every `calculate_shadows_wall_ht_25()` creates ~10 new GPU buffers, bind groups, staging buffers, and command encoders
- Context itself already persisted via `OnceLock<Option<ShadowGpuContext>>` in `shadowing.rs`
- Fix: Add `GpuResourcePool` with buffer caching by size, persistent staging buffer
- Alternative: Python-side `ShadowGpuRunner` class (matches existing `SkyviewRunner` pattern)
- Expected benefit: Eliminate per-call allocation overhead
- Risk: Low (architectural change, no algorithm changes)

**Phase G.3.2: GPU-Accelerated SVF** ✅ Complete

- Implemented pipelined GPU dispatch via `svf_accumulation.wgsl`
- SVF accumulation runs entirely on GPU with batch patch processing
- Falls back to CPU automatically when GPU unavailable
- SVF is cached after first computation (210× speedup on repeat)

**Phase G.3.3: GPU cylindric_wedge** (MEDIUM priority)

- Candidate for GPU: Pure per-pixel trig operations
- Could run as a simple compute shader alongside shadow GPU
- Expected benefit: Marginal (already fast with numpy vectorization)
- Recommendation: Move to Rust first (G.2 P0), consider GPU later if needed

**Phase G.3.4: GPU UTCI/PET** (LOW priority)

- Both are embarrassingly parallel (per-pixel, no data dependencies)
- UTCI: 125-term polynomial (fast, already rayon-parallel in Rust)
- PET: 50-iteration solver (slower, could benefit from GPU for very large grids)
- Expected benefit: Only significant for grids >5000×5000
- Recommendation: Defer unless handling very large rasters

### G.4 FFI Boundary Optimization

**Current pattern** (good):

```
Python orchestration → Rust computation → Python result handling
```

**Identified improvements:**

| Issue                   | Current                            | Target                      | Priority |
| ----------------------- | ---------------------------------- | --------------------------- | -------- |
| Radiation orchestration | Python loops + multiple Rust calls | Single fused Rust call      | P2       |
| GPU context lifecycle   | Per-call init                      | Persistent across timesteps | P0       |
| Array transfer overhead | Copy per call                      | Zero-copy via PyArray views | P3       |
| Parameter passing       | Mix of structs and positional args | Consistent struct-based API | P2       |

**Fused radiation kernel** (deferred):

- Currently: Python calls `cylindric_wedge` → `Perez_v3` → `vegetation.kside_veg` → `vegetation.lside_veg` → `sky.anisotropic_sky` → `tmrt.compute_tmrt`
- Could be: Single `compute_full_radiation(inputs) → RadiationResult` in Rust
- Benefit: Eliminate ~6 Python/Rust crossings per timestep
- Risk: Reduces modularity, harder to debug intermediate values
- Recommendation: Defer until per-timestep time exceeds 1s for target grid sizes

### G.5 Implementation Order

| Step | Task                                           | Est. Effort | Dependencies  | Status      |
| ---- | ---------------------------------------------- | ----------- | ------------- | ----------- |
| 1    | Move `cylindric_wedge()` to Rust (`sky.rs`)    | 2-3 hours   | None          | ✅ Complete |
| 2    | Move anisotropic patch loop to Rust (`sky.rs`) | 1-2 hours   | None          | ✅ Complete |
| 3    | GPU buffer reuse (persistent resource pool)    | 3-4 hours   | None          | ✅ Complete |
| 4    | Move `binary_dilation()` to Rust               | 2-3 hours   | None          | ✅ Complete |
| 5    | ~~Move `Perez_v3()` to Rust~~                  | 4-6 hours   | Step 1        | ✅ Complete |
| 6    | ~~GPU-accelerated SVF~~                        | 2-3 days    | Step 3        | ✅ Complete |
| 7    | ~~Fused radiation kernel~~                     | 1-2 days    | Steps 1, 2, 5 | ✅ Complete |

**Milestone targets:**

- After steps 1-2: ~2× faster per timestep (anisotropic mode)
- After step 3: Eliminated GPU init overhead
- After step 4: 10-100× faster wall setup
- After step 6: 5-50× faster fresh SVF computation

---

## Phase H: Field Data Validation (In Progress)

**Goal:** Validate SOLWEIG outputs against measured field data from real-world observation campaigns.

**Why this matters:** Currently all validation is against the reference UMEP Python implementation (computational parity). This confirms the code is _equivalent_, but not that it's _correct_. Field-data validation would confirm that:

1. Tmrt predictions match actual measurements within published error bounds
2. UTCI/PET thermal comfort categories are realistic
3. Shadow patterns match observed conditions
4. The Perez anisotropic sky model improves accuracy vs isotropic

### H.1 Kolumbus Wall Temperature Validation (Complete)

**Dataset:** Wallenberg et al. (2025) - Zenodo record 15309445
- Site: Gothenburg, Sweden (57.697°N, 11.930°E), EPSG:3007
- Period: 2023-05-15 to 2023-08-31 (summer months)
- Grid: 80×81 pixels at 0.5m resolution
- Geodata: DSM, DEM, CDSM, groundcover GeoTIFFs + WOI shapefile
- Met forcing: UMEP/SUEWS format (10-min resolution, 4 monthly files)
- Observations: IR radiometer wall surface temperatures (plastered brick + wood)

**Results (generic cobblestone parameters):**

| Metric | Plastered Brick | Wood |
|--------|----------------|------|
| Monthly RMSE (July) | 6.67°C | 8.96°C |
| Monthly Bias | -2.53°C | -3.17°C |
| Single-day RMSE | 8.53°C | 11.57°C |
| Published RMSE (tuned params) | ~2°C | ~2°C |

**Key finding:** Our generic model (hardcoded tgk=0.37, tstart=-3.41, tmaxlst=15.0) is 3-4× worse than the paper's per-material tuned parameters. This validates the importance of land cover support (which the full pipeline already has).

**Full pipeline results (noon, July 15):**
- WOI Tmrt: 31.7°C at Ta=20.5°C (+11.2°C excess radiation)
- Peak Tmrt: 41.8°C at 15:00, Ta=23.9°C (+17.9°C excess)

**Tasks:**

- [x] Download Zenodo validation dataset (kolumbus.csv + geodata + met forcing)
- [x] Add `Weather.from_umep_met()` for SUEWS-format met files
- [x] Write 12 validation tests (data loading, wall temp, full pipeline)
- [x] Register `validation` pytest marker
- [x] Add material-specific wall temperature parameters (Rust optional params + JSON defaults)
- [x] ~~Land cover-aware wall temperature model~~ — Not needed: scalar `wall_material` param (brick/concrete/wood/cobblestone) is the correct abstraction for SOLWEIG's wall radiation model

### H.2 Montpellier Tmrt Validation (Complete)

**Dataset:** INRAE PRESTI experimental canyon, Montpellier, France (43.64°N, 3.87°E)
- Reduced-scale urban canyon (2.3m concrete walls, 12m long, 5m apart, E-W orientation)
- 15 grey globe thermometers (40mm, RAL 7001, PT100) at 1.3m height
- Period: 2023-07-21 to 2024-07-31 (10-min intervals)
- Clear-sky GHI model (Ineichen, Linke turbidity 3.5)

**Results (isotropic sky, Aug 4 2023):**

| Metric | Value |
|--------|-------|
| Single-day RMSE | 7.59°C |
| Single-day Bias | +4.45°C |
| Multi-day RMSE (3 days) | 9.06°C |
| Multi-day Bias | +5.18°C |
| Noon Tmrt | 52.2°C (Ta=25.6°C) |
| Peak Tmrt | 52.8°C |

**Tasks:**

- [x] Construct synthetic DSM from known canyon dimensions (30×40 at 0.5m)
- [x] Download and parse globe thermometer measurements (presti_subset.csv)
- [x] Write Tmrt validation tests (20 tests: data, globe-to-Tmrt, DSM, model vs obs)
- [x] Compare isotropic vs anisotropic sky model accuracy (aniso requires shadow matrices, deferred)

### H.3 Additional Validation Opportunities

**Potential data sources:**

- UMEP validation datasets (Gothenburg, London)
- Published SOLWEIG validation studies (Lindberg et al. 2008, 2016)
- COSMO/CLM urban datasets
- Local university weather stations with globe thermometer data

---

## Phase D: Documentation & Integration (In Progress)

- [x] Quick Start Guide ([docs/getting-started/quick-start.md](docs/getting-started/quick-start.md))
- [x] MkDocs site scaffolded (25 pages under `docs/`)
- [x] API Reference with mkdocstrings (auto-generated)
- [x] QGIS plugin scaffolded (Phases 1-10, see [qgis_plugin/README.md](qgis_plugin/README.md))
- [x] QGIS plugin testing & polish (Phase 11)
- [x] CI/CD for cross-platform plugin builds
- [ ] Build and publish wheels for multiple platforms

### D.1 Documentation Fixes (Pending)

| Task                                    | Impact | Notes                                                                                                              |
| --------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| Fix GPU function docs                   | HIGH   | `disable_gpu()` referenced in quick-start & installation but doesn't exist. `is_gpu_available()` not in `__all__`. |
| Document undocumented `__all__` exports | MEDIUM | `compute_utci_grid`, `compute_pet_grid`, data bundles, tiling utils undocumented                                   |
| Fix `ThermalState` import path in docs  | LOW    | `docs/api/dataclasses.md` uses deep path instead of `solweig.ThermalState`                                         |

---

## Phase F: Test Coverage (Pending)

**Goal:** Close critical gaps in test coverage. The physics (golden tests) and API surface are well tested, but the orchestration layer and primary workflow have blind spots.

### F.1 Critical Test Gaps

| Gap                          | Risk   | What's missing                                                              |
| ---------------------------- | ------ | --------------------------------------------------------------------------- |
| ~~`calculate_timeseries()`~~ | HIGH   | ✅ 13 tests added in `tests/test_timeseries.py`                             |
| ~~`validate_inputs()`~~      | MEDIUM | ✅ 8 tests added in `tests/test_timeseries.py`                              |
| `compute_utci_grid/pet_grid` | MEDIUM | Grid-level postprocessing exported in `__all__` but untested                |
| Orchestration unit tests     | MEDIUM | `computation.py` and `timeseries.py` only tested indirectly via integration |
| Multi-timestep thermal state | MEDIUM | No test verifies state persistence/accumulation across timesteps            |

**Current coverage by layer:**

| Layer                          | Coverage                      | Notes                                                       |
| ------------------------------ | ----------------------------- | ----------------------------------------------------------- |
| Layer 1: Public API (`api.py`) | Good (70 tests)               | Missing timeseries, validate_inputs                         |
| Layer 2: Orchestration         | Poor (indirect only)          | No unit tests for compute_single_timestep, state management |
| Layer 3: Components            | Moderate (indirect)           | Tested through golden tests, not directly                   |
| Layer 4: Rust                  | Excellent (100+ golden tests) | No gaps identified                                          |

---

## Phase E: API Improvements (Complete)

**Goal:** Improve API ergonomics, consistency, and error handling.

**Status:** ✅ Complete (Feb 2026)

### E.1 Result Methods Pattern (P0)

Add `compute_utci()` and `compute_pet()` methods directly on `SolweigResult` for discoverability.

| Task                                                                                | File              | Status      |
| ----------------------------------------------------------------------------------- | ----------------- | ----------- |
| Add `SolweigResult.compute_utci(weather)` method                                    | models/results.py | ✅ Complete |
| Add `SolweigResult.compute_pet(weather, human)` method                              | models/results.py | ✅ Complete |
| Support both `result.compute_utci(weather)` and `result.compute_utci(ta, rh, wind)` | models/results.py | ✅ Complete |
| Update README with new pattern                                                      | README.md         | ✅ Complete |
| Add tests for result methods                                                        | tests/test_api.py | ✅ Complete |

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

| Task                                                                      | File                                | Status      |
| ------------------------------------------------------------------------- | ----------------------------------- | ----------- |
| Change `Location.from_surface()` to require explicit `utc_offset` or warn | models/weather.py                   | ✅ Complete |
| Add warning in `calculate_timeseries()` when location=None                | timeseries.py                       | ✅ Complete |
| Update quick-start guide with explicit location examples                  | docs/getting-started/quick-start.md | ✅ Complete |

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

| Task                                                                         | File                  | Status      |
| ---------------------------------------------------------------------------- | --------------------- | ----------- |
| Change `calculate()` to let explicit params override config                  | api.py                | ✅ Complete |
| Change `calculate_timeseries()` to let explicit params override config       | timeseries.py         | ✅ Complete |
| Change `use_anisotropic_sky` default to `None` (means "use config or False") | api.py                | ✅ Complete |
| Add debug logging when explicit params override config                       | api.py                | ✅ Complete |
| Document new precedence in docstrings                                        | api.py, timeseries.py | ✅ Complete |
| Add tests for precedence behavior                                            | tests/test_api.py     | ✅ Complete |

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

| Task                                                                                                                            | File                       | Status      |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------- |
| Create `errors.py` with `SolweigError`, `InvalidSurfaceData`, `GridShapeMismatch`, `MissingPrecomputedData`, `WeatherDataError` | errors.py (new)            | ✅ Complete |
| Add `validate_inputs()` preflight function                                                                                      | api.py                     | ✅ Complete |
| Update `calculate()` to raise structured errors                                                                                 | api.py                     | ✅ Complete |
| Export errors in `__all__`                                                                                                      | api.py                     | ✅ Complete |
| Add tests for error cases                                                                                                       | tests/test_errors.py (new) | ✅ Complete |

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

| Task                                                      | File              | Status      |
| --------------------------------------------------------- | ----------------- | ----------- |
| Remove `poi_coords` from public signature (keep internal) | api.py            | ⏳ Deferred |
| Add `Weather.from_values()` factory for quick testing     | models/weather.py | ✅ Complete |
| Document result methods and validation in README          | README.md         | ✅ Complete |

### E.6 Implementation Order

| Step | Task                               | Effort  | Risk                        | Dependencies |
| ---- | ---------------------------------- | ------- | --------------------------- | ------------ |
| 1    | E.1: Result methods                | 1 hour  | None                        | -            |
| 2    | E.2: Location warning              | 30 min  | None                        | -            |
| 3    | E.4: errors.py + validate_inputs() | 2 hours | None                        | -            |
| 4    | E.3: Config precedence             | 2 hours | **Low** - behavioral change | -            |
| 5    | E.5: API cleanup                   | 30 min  | None                        | -            |
| 6    | Update README and docs             | 1 hour  | None                        | E.1, E.2     |

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
- **612+ tests passing** (current baseline, including spec, golden, benchmark, and validation tests)
- No memory regression on standard benchmarks

Gate command: `pytest tests/`

---

## File Reference

| File                                           | Purpose                 | Lines    |
| ---------------------------------------------- | ----------------------- | -------- |
| [api.py](pysrc/solweig/api.py)                 | Public API entry point  | 403      |
| [models/](pysrc/solweig/models/)               | Dataclasses (6 modules) | ~3,080   |
| [computation.py](pysrc/solweig/computation.py) | Core orchestration      | 389      |
| [components/](pysrc/solweig/components/)       | Physics modules         | ~1,365   |
| [specs/](specs/)                               | Physics specifications  | 10 files |

---

## Risk Register

| Risk                              | Impact | Mitigation                          |
| --------------------------------- | ------ | ----------------------------------- |
| Breaking parity during spec fixes | HIGH   | Run parity tests after every change |
| Memory regression                 | MEDIUM | Add memory benchmarks to CI         |
| NumPy ABI mismatch                | HIGH   | Pin version, test 1.x and 2.x       |
