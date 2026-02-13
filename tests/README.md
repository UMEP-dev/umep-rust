# SOLWEIG Test Suite

This document explains the testing strategy used during the SOLWEIG modernization effort.

## Three-Layer Testing Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: SPEC PROPERTY TESTS (mock data)                       │
│  - Verify physical invariants from specs                        │
│  - Fast, deterministic, easy to debug                           │
│  - "Does the algorithm behave correctly?"                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: GOLDEN REGRESSION TESTS (demo data)                   │
│  - Pre-computed outputs from known-good runs                    │
│  - Catch any numerical drift during refactoring                 │
│  - "Does output still match what we expect?"                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: RUST vs UMEP COMPARISON (demo data)                   │
│  - Existing test_rustalgos.py                                   │
│  - Verify Rust matches original Python implementation           │
│  - "Does Rust produce same results as reference?"               │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1: Spec Property Tests

**Location:** `tests/spec/`

**Purpose:** Verify that algorithms satisfy physical invariants defined in `specs/*.md`.

**Data:** Synthetic/mock DSMs created in test fixtures (small, fast, deterministic).

**Examples:**

- "No shadows when sun altitude ≤ 0"
- "SVF always in range [0, 1]"
- "Flat terrain has SVF = 1"
- "Taller buildings cast longer shadows"

**Rationale:** These tests verify the algorithm behaves correctly according to physics. They use simple synthetic data so failures are easy to diagnose. If a spec test fails, you know exactly which physical property was violated.

**Files:**

- `test_shadows.py` - 8 shadow properties
- `test_svf.py` - 8 SVF properties
- `test_tmrt.py` - 10 Tmrt properties (planned)
- `test_radiation.py` - 7 radiation properties (planned)
- `test_utci.py` - 8 UTCI properties (planned)
- `test_pet.py` - 8 PET properties (planned)

## Layer 2: Golden Regression Tests

**Location:** `tests/golden/`

**Purpose:** Ensure outputs don't change unexpectedly during refactoring.

**Data:** Real demo data (Athens, Gothenburg) with pre-computed reference outputs.

**How it works:**

1. Run the algorithm on demo data with current (known-good) code
2. Save outputs as `.npy` files (golden fixtures)
3. Future test runs compare new output against golden fixtures
4. Any numerical difference fails the test

**Rationale:** During modernization, we'll change APIs, consolidate parameters, and refactor code. Golden tests catch any accidental changes to numerical output, even subtle floating-point differences. If a golden test fails, the output changed - investigate whether it's intentional.

**Files:**

- `fixtures/` - Pre-computed reference outputs
- `test_golden_shadows.py` - Shadow regression tests
- `test_golden_svf.py` - SVF regression tests
- `test_golden_tmrt.py` - Tmrt regression tests

## Layer 3: Rust vs UMEP Comparison

**Location:** `tests/rustalgos/`

**Purpose:** Verify Rust implementations match the original Python UMEP code exactly.

**Data:** Real demo data (Athens, Gothenburg).

**How it works:**

1. Run the original Python UMEP implementation
2. Run the Rust implementation with identical inputs
3. Compare outputs element-by-element
4. Report match percentage and mean difference

**Rationale:** The Rust code is a performance-optimized rewrite of the original Python. This layer ensures the Rust code produces identical results to the reference Python implementation. These tests also measure speedup (typically 10-30x faster).

**Files:**

- `test_rustalgos.py` - Comprehensive Rust vs Python comparison

## When to Run Each Layer

| Situation                      | Layer 1 | Layer 2 | Layer 3 |
| ------------------------------ | ------- | ------- | ------- |
| Quick check during development | ✅      |         |         |
| Before committing changes      | ✅      | ✅      |         |
| Before merging PR              | ✅      | ✅      | ✅      |
| After changing algorithm logic | ✅      | ✅      | ✅      |
| After Rust code changes        | ✅      |         | ✅      |

## Running Tests

```bash
# Run all spec tests (fast, ~10 seconds)
uv run pytest tests/spec/ -v

# Run golden regression tests
uv run pytest tests/golden/ -v

# Run Rust vs UMEP comparison (requires demo data)
uv run python -c "from tests.rustalgos.test_rustalgos import test_shadowing; test_shadowing()"

# Run everything
uv run pytest tests/ -v

# Run performance regression benchmarks
uv run pytest tests/benchmarks/ -v

# If CI is slower, scale runtime budgets (example: +50% headroom)
SOLWEIG_PERF_BUDGET_SCALE=1.5 uv run pytest tests/benchmarks/ -v
```

Performance benchmarks are intended for local/reproducible environments and are not run in CI.
Each run appends logs to:
- `tests/benchmarks/logs/performance_matrix_history.csv` (long-form records)
- `tests/benchmarks/logs/performance_matrix_history.md` (matrix snapshot per run)
Logged metadata includes hardware context (CPU counts, RAM total/available, GPU availability/backend/max buffer size).

## Adding New Tests

### Adding a Spec Property Test

1. Check the relevant spec file in `specs/*.md`
2. Identify the property to test
3. Create a test function with synthetic data that verifies the property
4. Name it `test_property_N_description` where N matches the spec

### Adding a Golden Test

1. Run the algorithm on demo data with current code
2. Save output: `np.save("tests/golden/fixtures/name.npy", output)`
3. Create test that loads fixture and compares with `np.testing.assert_allclose()`

### Adding a Rust vs UMEP Test

1. Follow the pattern in `test_rustalgos.py`
2. Run both Python and Rust implementations
3. Use `compare_results()` helper to check match percentage
