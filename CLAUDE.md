# CLAUDE.md - Project Notes for Claude Code

## Project Overview

SOLWEIG is a high-performance urban microclimate model (Rust + Python via maturin).
It calculates Mean Radiant Temperature (Tmrt), UTCI, and PET for urban thermal comfort analysis.

## Directory Structure

```
pysrc/solweig/          # Python source
  api.py                # Simplified API (Phase 2 complete - 100% parity)
  functions/solweig.py  # Reference implementation (legacy runner)
  runner.py             # Full runner class
rust/                   # Rust extensions via maturin
specs/                  # Markdown specs for each module
tests/
  test_api.py           # Unit tests (31 tests)
  test_api_exact_parity.py  # 100% parity verification (REQUIRED)
  golden/fixtures/      # Pre-computed reference data from UMEP
```

## Key Files

- `MODERNIZATION_PLAN.md` - API simplification roadmap (Phase 2 complete, 100% parity achieved)
- `specs/` - Markdown specifications for each module (shadows, svf, radiation, tmrt, utci, pet)
- [pysrc/solweig/api.py](pysrc/solweig/api.py) - Current simplified API implementation
- [tests/test_api_exact_parity.py](tests/test_api_exact_parity.py) - Gate for all API changes

## Agent Model Selection Preferences

When spawning agents via the Task tool, use these guidelines:

| Task Type                                       | Model  |
| ----------------------------------------------- | ------ |
| Quick file searches, grep patterns              | haiku  |
| Codebase exploration                            | haiku  |
| Implementing features, writing tests            | sonnet |
| Complex debugging, architectural analysis       | opus   |
| Multi-step reasoning with significant ambiguity | opus   |

Default to **haiku** for exploration/search to minimize cost and latency.
Escalate to **sonnet** for implementation work.
Reserve **opus** for higher-level or complex reasoning tasks.

## Build & Test Commands

```bash
# Build Rust extension
maturin develop

# Run tests
pytest ./tests

# Full verification (format, lint, typecheck, test)
poe verify_project
```

## Tooling Preferences

Use these tools instead of alternatives:

| Tool     | Use For                | Instead Of                   |
| -------- | ---------------------- | ---------------------------- |
| **uv**   | Package management     | pip, poetry, pipenv          |
| **ruff** | Linting and formatting | black, isort, flake8, pylint |
| **ty**   | Type checking          | mypy, pyright                |

When writing commands or suggesting workflows, prefer these tools.

## Architecture Notes

- Shadow calculations use ray tracing from sun position
- SVF (Sky View Factor) computed once per DSM, should be cached
- UTCI uses fast polynomial (~200 terms), PET uses slow iterative solver
- Target: POI-only mode for 100x speedup when sampling specific points

## Key Design Decisions

### 100% Parity Requirement

All API changes MUST maintain exact parity with the reference runner implementation.

- Tmrt bias must be < 0.1°C
- Run `pytest tests/test_api_exact_parity.py` before any PR
- This is non-negotiable - parity gates all changes

### Auto-Computation Strategy

The API auto-computes values to reduce parameter burden:

- Sun position (altitude/azimuth) from lat/lon/datetime
- Direct/diffuse radiation split using Reindl model
- Max DSM height from grid analysis
- See [MODERNIZATION_PLAN.md](MODERNIZATION_PLAN.md) for complete list

### Performance Constraints

- **SVF/GVF**: Expensive to compute, must be cached between timesteps
- **PET**: Slow iterative solver, may become optional in Phase 6
- **Thermal state**: Ground temperature requires full timeseries (TsWaveDelay model)

## Common Workflows

### Adding New Features

1. Read relevant spec in `specs/` to understand physical model
2. Implement in [pysrc/solweig/api.py](pysrc/solweig/api.py)
3. **CRITICAL**: Run parity test: `pytest tests/test_api_exact_parity.py`
4. Add property-based test in `tests/spec/` if adding new physics
5. Full verification: `poe verify_project`

### Debugging Parity Issues

If parity test fails (Tmrt bias > 0.1°C):

1. Run visual comparison: `pytest tests/test_api_visual_comparison.py -s`
2. Check intermediate outputs (Kdown, Ldown, Lup, Kup, shadows)
3. Common culprits:
   - Units (degrees vs radians) - especially zenith angle
   - Land cover properties not loaded from params JSON
   - Anisotropic vs isotropic radiation toggle
   - Missing unit conversions

## Important Gotchas

### 1. Zenith Angle Units

**Problem**: `cylindric_wedge()` expects radians, but sun_zenith is in degrees
**Solution**: Always convert before use

```python
zen_rad = weather.sun_zenith * (np.pi / 180.0)
```

See [api.py:1581](pysrc/solweig/api.py#L1581) for reference

### 2. Land Cover Properties

**Problem**: Albedo/emissivity must match params JSON, not be hardcoded
**Solution**: Use `_get_lc_properties_from_params()` helper

```python
lc_props = _get_lc_properties_from_params(params, 'LANDCOVER_ID')
```

See [api.py:309-380](pysrc/solweig/api.py#L309)

### 3. Thermal State Accumulation

**Problem**: Ground temperature model requires previous timesteps (thermal mass)
**Solution**: Use `calculate_timeseries()` for accurate multi-timestep runs

```python
# WRONG: Single timestep misses thermal accumulation
result = calculate(surface, location, weather_noon)

# CORRECT: Full timeseries accumulates thermal state
results = calculate_timeseries(surface, location, weather_all_hours)
noon_result = results[12]
```

### 4. Anisotropic Radiation Toggle

**Problem**: Must use correct diffuse radiation variable based on `use_aniso` flag
**Solution**: Use `drad` (which handles both cases), not hardcoded `drad_iso`
See [api.py:1863](pysrc/solweig/api.py#L1863)

## Current Status (Updated: January 2026)

- Phase 1: ✅ Spec-driven testing infrastructure
- Phase 2: ✅ API simplification (100% parity achieved)
- Phase 3: ⏳ POI-only mode (next priority - 100x speedup target)
- Phase 4-6: ⏳ Memory optimization, Rust time loop, optional complexity

See [MODERNIZATION_PLAN.md](MODERNIZATION_PLAN.md) for detailed roadmap.
