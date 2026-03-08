# SOLWEIG Code Review TODO

## Bugs & Correctness Issues

### High Priority

- [x] **Southern Hemisphere clearnessindex bug** — `physics/clearnessindex_2013b.py`: negative latitudes fall through to 80+ degree coefficient band; use `abs(latitude)` for lookup
- [x] **UTC offset truncation** — `models/weather.py:201`: `int(metadata["tz_offset"])` truncates half-hour offsets (India +5:30, Nepal +5:45); change `Location.utc_offset` from `int` to `float`
- [x] **Hardcoded version in metadata** — `metadata.py:48`: version string `"0.0.1a1"` is stale; read from `importlib.metadata.version("solweig")`
- [x] **Hardcoded altitude=0.0** — `components/ground.py:95` and `computation.py:336`: clearness index ignores site elevation; pass actual altitude from `Location`
- [x] **No chronological order enforcement for Weather list** — `api.py:436` forwards `list[Weather]` unchanged; `timeseries.py:410` and `tiling.py:1233` derive `timestep_dec` from `weather_series[1] - weather_series[0]`. Out-of-order inputs (e.g., 13:00 before 12:00) produce negative `timestep_dec`, silently inverting thermal-state stepping and hour accumulation instead of failing fast. Validate chronological order and positive timestep at the API boundary.
- [x] **Stale wall cache reuse after DSM changes** — `surface.py:829` loads cached wall rasters unconditionally; `surface.py:1237` writes cache metadata but never reads it back for validation. File-mode `SurfaceData.prepare()` can reuse stale wall rasters after the DSM changes. Validate wall cache against current DSM hash before reuse.

### Medium Priority

- [x] **SVF cache invalidates on every run with relative inputs** — `surface.py:461` hashes raw aligned rasters before relative-height preprocessing (`surface.py:510`), so `dsm_relative=True` with nonzero DEM produces a different hash each run even when inputs are identical. Move cache validation after preprocessing, or hash the post-processed arrays.
- [x] **NaN to integer array** — `io.py:594`: `rast_arr[rast_arr == no_data_val] = np.nan` silently fails when `ensure_float32=False` and dtype is integer
- [x] **Assert for runtime validation** — `loaders.py:106`: `assert isinstance(result, SimpleNamespace)` is stripped by `python -O`; use `if not isinstance(...): raise TypeError(...)`
- [x] **Unclamped shadow fraction** — `physics/cylindric_wedge.py:73`: `F_sh` not clamped to [0, 1]; can produce out-of-range values
- [x] **Input array mutation** — `physics/wallalgorithms.py:144`: `walls[walls > 0.5] = 1` mutates caller's array in-place; copy first
- [x] **Non-deterministic default datetime** — `models/weather.py:481`: `from_values()` defaults to `dt.now()`; consider requiring explicit datetime or raising
- [x] **Unguarded log(RH)** — `physics/clearnessindex_2013b.py:67`: `np.log(RH)` with no guard for RH <= 0; add validation or clamp
- [ ] **Silent PET non-convergence** — Rust `pet.rs`: iterative solver has 200-iteration limit with no warning on non-convergence

## Dead Code & Redundancy

- [x] **Dead branch in api.py** — `api.py:318-319`: `if human is None` can never be reached; remove
- [x] **Unused `_apply_thermal_delay`** — `computation.py`: function appears never called (fused Rust path handles it); verify and remove
- [x] **`needs_psi_adjustment` always False** — `components/svf_resolution.py`: flag is always `False`; 7 unused parameters in `resolve_svf`; clean up
- [x] **Dead `requested_outputs is None` checks** — `tiling.py:1126-1130`: `requested_outputs` is always a set; remove dead branches
- [x] **Unused variable `zen`** — `components/ground.py:91`: computed but never referenced; remove — *Actually used by clearnessindex call; not dead code*
- [x] **Legacy `material_params` field** — `models/config.py`: duplicates `materials` field; remove or unify
- [ ] **SVF array construction duplicated 3x** — `models/surface.py`: extract into a factory method like `SvfArrays.from_rust_result()`
- [ ] **Orchestration duplication** — `tiling.py` and `timeseries.py`: large blocks of config resolution, weather precomputation, and output logic duplicated; extract shared helpers

## Performance

- [x] **Vectorize land cover properties** — `loaders.py` `get_lc_properties_from_params()`: replace 8 full-array scans with `np.take` lookup table
- [ ] **Avoid redundant `astype(np.float32)`** — `models/surface.py`: use `np.asarray(arr, dtype=np.float32)` to avoid copying when already float32
- [ ] **Pre-allocate scratch buffers** — `summary.py` `update()`: allocates temporary arrays every call; pre-allocate in `__init__`
- [ ] **Cache `altmax` per day** — `models/weather.py:354-383`: 96-iteration sun position loop runs per timestep even for same-day calls
- [ ] **Fragile `id()`-based caching** — `computation.py`: reused memory addresses could return stale data; consider content-based keys

## Architecture & Design

- [ ] **Inconsistent error handling** — `svf_resolution.py` raises custom errors, `gvf.py` returns defaults silently, `ground.py` crashes with `AttributeError`; unify strategy
- [x] **Private attribute access** — `models/results.py`: accesses `surface._geotransform` and `surface._crs_wkt` instead of public properties `.geotransform` and `.crs`
- [ ] **Overloaded `SurfaceData`** — 17 cache/internal fields on the dataclass; consider separating into a dedicated cache container
- [ ] **Eager import-time I/O** — `_compat.py:123`: backend detection runs at import, blocking unrelated imports; consider lazy initialization
- [ ] **Mutex poisoning in GPU contexts** — `aniso_gpu.rs`, `gvf_gpu.rs`: poisoned mutex leaves context permanently unusable; add recovery path
- [x] **Stale docstring reference** — `loaders.py:172`: references `configs.py` which was renamed to `loaders.py`
- [x] **Fragile `getattr` without default** — `loaders.py:264`: `resolve_wall_params` crashes with `AttributeError` on custom materials JSON missing expected keys
- [x] **`check_path` directory heuristic** — `io.py:161`: `not path.suffix` misidentifies extensionless files as directories
- [x] **Redundant top-level imports** — `utils.py:20-23`: rasterio imports at module scope are re-imported inside functions; remove top-level copies

## Constants Accuracy

- [x] **Stefan-Boltzmann constant** — `constants.py:16`: comment says "f32-rounded CODATA 2018" but value `5.67051e-8` is the legacy UMEP constant (relative error ~1.3e-4 vs actual CODATA `5.670374419e-8`); fix comment or value
- [x] **Sitting view factors don't sum to 1** — `constants.py:49`: `F_UP_SITTING = 0.166666` truncated; six factors sum to 0.999996; use `0.166667` or compute from `1/6`

## Test Coverage Gaps

- [ ] **GPU vs CPU parity tests** — add tests comparing GPU and CPU outputs for shadows, SVF, and GVF (only anisotropic has one)
- [ ] **UTCI/PET edge-case tests** — test extreme inputs, clamping boundaries, zero wind speed, non-convergence
- [ ] **GVF property-based tests** — no physics property tests (e.g., "closer buildings produce higher GVF")
- [ ] **Ground temperature property tests** — only covered by golden regression tests
- [ ] **Concurrency tests for GPU dispatch** — exercise `readback_inflight` guards with concurrent access
- [ ] **Missing SVF property test 5** — numbering jumps from 4 to 6; add or document why skipped
- [ ] **Shadow length tolerance** — 15% + 3 pixels (~30% effective) is generous; tighten if possible

## Minor / Code Quality

- [ ] **`save_raster` COG memory leak** — `io.py:313-335`: `memfile.open()` dataset passed to `copy()` but never explicitly closed
- [x] **`as_float32` duplicates `ensure_float32_inplace`** — `buffers.py`: two identical functions; consolidate
- [ ] **`progress()` swallows unknown kwargs** — `progress.py:234-250`: typos in keyword arguments silently ignored
- [ ] **`patch_radiation.py` East condition** — `_cardinal_components` line 37: `patch_azimuth > 360` is dead code (azimuths never exceed 360)
- [ ] **`Lside_sh` always zero** — `physics/patch_radiation.py:130-144`: shaded longwave component initialized to zero and never modified but returned
- [x] **`svfalfa` via log/exp** — `components/svf_resolution.py:112-115`: `arcsin(exp(log(x)/2))` is slower and less stable than equivalent `arcsin(sqrt(x))`
- [x] **GVF dilation magic number** — `components/gvf.py:76`: `iterations = int(25 / pixel_size) + 1` assumes max ~50m buildings; undocumented
- [x] **String path concatenation** — `walls.py:56`: uses `+` instead of `Path /` operator
- [x] **`bbox` typed as `list[int]`** — `io.py:501`, `walls.py:30`: should be `list[float]` since coordinates are floats
