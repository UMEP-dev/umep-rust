# Runtime Contract

This document defines the normative runtime behavior for the unified
`calculate()` Python API. If implementation and documentation conflict, this
contract is the source of truth.

## Preconditions

1. **SVF availability is required for all `calculate()` calls**
   - `calculate()` requires SVF data to already be present on
     `surface.svf` or `precomputed.svf`.
   - SVF may be prepared by:
     - `SurfaceData.prepare(...)` (computes/caches SVF when missing), or
     - `surface.compute_svf()` for in-memory/manual workflows.
   - Runtime calculation paths must not silently compute SVF as a fallback.

2. **Explicit anisotropic requests have strict input requirements**
   - If anisotropic sky is explicitly requested via
     `use_anisotropic_sky=True`, shadow matrices must already be available on
     `surface.shadow_matrices` or `precomputed.shadow_matrices`.
   - Missing shadow matrices must raise `MissingPrecomputedData`.
   - Runtime must not silently downgrade to isotropic sky when anisotropic is
     explicitly requested.

## Output Conventions

1. **Shadow convention**
   - `shadow` uses `1.0 = sunlit`, `0.0 = shaded`.
   - This convention applies to daytime and nighttime outputs.

2. **Timeseries return semantics**
   - `timestep_outputs=["tmrt", "shadow", ...]`: returned `SolweigResult` objects
     in `summary.results` keep the requested arrays in memory.
   - `timestep_outputs=None` (default): implementation frees arrays after
     aggregation to minimize memory use; `summary.results` is empty.

## Default Behavior

1. Default anisotropic behavior is consistent across all `calculate()` calls:
   - The unified `calculate()` function uses the same anisotropic default regardless
     of whether tiling or timeseries mode is active.
   - `ModelConfig.load()` must use `True` as the default for `use_anisotropic_sky`
     when deserializing JSON that omits the key, matching the dataclass field default.
2. Thermal state chaining for timeseries remains automatic and is unaffected by
   output streaming mode.
3. `max_shadow_distance_m` defaults to `MAX_BUFFER_M = 1000.0` meters across all
   entry points (`calculate()`, `calculate_buffer_distance()`).

## Documentation Requirements

1. User docs and examples must state SVF is explicit at `calculate()` runtime.
2. Docs must not claim anisotropic shadow matrices are auto-generated during
   calculation; preparation must be explicit via preprocessing helpers.
