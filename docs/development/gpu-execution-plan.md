# GPU Execution Plan

This document defines a methodical process for GPU work in SOLWEIG across:

- SVF preprocessing (`skyview`, tiled and non-tiled)
- Timestep and timeseries runtime (`shadowing`, anisotropic sky, tiled orchestration)

The goal is to make GPU changes **predictable, measurable, and reversible**.

## Objectives

1. Improve runtime and scalability without silent correctness regressions.
2. Keep fallback behavior explicit and diagnosable.
3. Guarantee large-grid behavior on all supported backends.

## Core Policy

### 1) No silent fallback

If a GPU stage fails and CPU fallback is used, we must log a clear reason.

Required log pattern:

- `"[GPU] <stage> failed: <reason>. CPU fallback."`

### 2) Kernel geometry must be size-safe

For grid kernels, prefer 2D dispatch (`x`, `y`) over 1D dispatch over `total_pixels`.

Reason: dispatch dimensions are limited by backend (e.g., 65535 per dimension in wgpu validation).

### 3) GPU changes must pass parity + performance gates

No GPU change is complete without both functional parity and benchmark evidence.

## Acceptance Criteria

A GPU change is accepted only if all items pass.

### A. Correctness gates

1. SVF kernel/property tests:

```bash
uv run pytest tests/spec/test_svf.py -q
```

2. SVF core-path regression (Rust core output must match full-tile slicing):

```bash
uv run pytest tests/spec/test_svf_core_api.py -q
```

3. Anisotropic GPU/CPU parity:

```bash
uv run pytest tests/spec/test_aniso_gpu_parity.py -q
```

4. Tiled parity checks:

```bash
uv run pytest tests/test_tiling_integration.py -k "multitile_vs_nontiled_comparison or anisotropic_tiled_vs_nontiled" -q
```

### B. Runtime gates

Run the CI-stable tiled performance benchmark:

```bash
uv run pytest tests/benchmarks/test_tiling_benchmark.py -q
```

Expected:

- Worker-scaling sanity passes
- Bounded in-flight scheduling passes
- Anisotropic tiled runtime smoke passes

For deeper diagnostics (including API vs plugin matrix ratios), run:

```bash
uv run pytest tests/benchmarks/test_performance_matrix_benchmark.py -q
```

### C. Build gate

Rebuild release extension with GPU features and retest:

```bash
uv run maturin develop --release --manifest-path rust/Cargo.toml --features "pyo3/extension-module,pyo3/abi3-py39,gpu"
```

## Metrics To Track

Track both absolute and relative metrics.

### SVF metrics

- `svf_tile_wall_time_s`
- `svf_patch_progress_rate` (patches/s)
- `svf_stitch_copy_time_ms`
- `svf_gpu_fallback_count`

### Timeseries metrics

- `timestep_wall_time_s`
- `tile_turnaround_ms` (mean, p95)
- `gpu_stage_time_ms` (shadow + anisotropic when active)
- `queue_depth_peak`
- `gpu_fallback_count` per stage

### Quality metrics

- `mean_abs_diff` and `max_abs_diff` vs CPU reference for parity fixtures
- `% finite pixels` equality checks where applicable

## Standard Profiling Modes

### Lightweight timing (developer)

```bash
SOLWEIG_TIMING=1 uv run pytest tests/spec/test_aniso_gpu_parity.py -q
```

### Anisotropic overlap mode experimentation

```bash
SOLWEIG_ANISO_GPU_OVERLAP=1 uv run pytest tests/spec/test_aniso_gpu_parity.py -q
```

### Full benchmark matrix

```bash
uv run pytest tests/benchmarks/test_performance_matrix_benchmark.py -q
```

## Workstreams

## 1) Kernel Safety

Scope:

- Keep all grid kernels dispatch-safe for large tiles.
- Ensure host/shader uniform structs stay layout-aligned.

Checklist:

1. Dispatch dimensions are bounded by `rows/cols`, not `total_pixels` on one axis.
2. Shader bounds checks match host-provided dimensions.
3. Test a case where `rows * cols / workgroup_size > 65535` would have failed in 1D mode.

## 2) Data Movement Reduction

Scope:

- Minimize GPU↔CPU transfers and Python-side copies.

Checklist:

1. No per-patch readback when accumulation can remain on GPU.
2. Prefer core-window outputs for tiled stitching paths.
3. Measure and report copy/stitch time explicitly.

## 3) Tiled Orchestration Efficiency

Scope:

- Keep GPU fed while avoiding CPU-side queue/mem thrash.

Checklist:

1. Validate `tile_workers` and `inflight_limit` under realistic memory pressure.
2. Keep telemetry for `mean_turnaround`, `max_queue`.
3. Verify parity for tiled vs non-tiled after scheduling changes.

## 4) Backend-Aware Resource Policy

Scope:

- Keep tile sizing stable across Metal/DX12/Vulkan/GL behavior differences.

Checklist:

1. Use backend metadata in sizing decisions.
2. Keep total-memory vs single-buffer heuristics documented and tested.
3. Preserve CPU fallback when limits are exceeded unexpectedly.

## 5) Observability and Operations

Scope:

- Make performance and fallback behavior visible in user logs.

Checklist:

1. Every fallback path logs stage + reason.
2. Progress bars use bounded ranges when embedded in multi-phase workflows.
3. Provide concise per-run telemetry summaries.

## Change Workflow (Required)

For each GPU PR:

1. Describe expected bottleneck shift (e.g., readback -> compute).
2. Attach before/after metrics from benchmark matrix and one realistic dataset.
3. Run all correctness gates.
4. Run release build + spot parity check.
5. Document fallback behavior and any new env flags.

## Immediate Priorities

1. Keep dispatch geometry audits active for all remaining GPU kernels.
2. Add explicit fallback counters (not only logs) for shadow/SVF/anisotropic stages.
3. Add one large-grid stress test for anisotropic GPU dispatch limits.
4. Add a CI lane for GPU parity + tiled parity (nightly if runtime is high).

## Quick Command Bundle

```bash
# Correctness gates
uv run pytest tests/spec/test_svf.py tests/spec/test_svf_core_api.py tests/spec/test_aniso_gpu_parity.py -q
uv run pytest tests/test_tiling_integration.py -k "multitile_vs_nontiled_comparison or anisotropic_tiled_vs_nontiled" -q

# Performance gates
uv run pytest tests/benchmarks/test_tiling_benchmark.py -q

# Release build
uv run maturin develop --release --manifest-path rust/Cargo.toml --features "pyo3/extension-module,pyo3/abi3-py39,gpu"
```

## Poe Shortcuts

```bash
uv run poe test_gpu_gates
uv run poe test_gpu_perf_gate
```

## Definition of Done

A GPU optimization is done when:

1. Correctness gates pass.
2. Performance gates pass or regressions are explicitly accepted with rationale.
3. No silent fallback remains in changed paths.
4. Release build is validated.
5. Documentation is updated here if behavior/policy changed.
