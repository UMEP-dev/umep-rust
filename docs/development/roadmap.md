# Roadmap

The canonical roadmap lives at **[ROADMAP.md](https://github.com/gushogg-blake/solweig/blob/main/ROADMAP.md)** in the project root.

This page summarizes current status. For full details including session logs, phase breakdowns, and the wish list, see the root file.

## Current Status (February 2026)

**Phases A, B, E complete.** Code quality sweep done. GPU/Rust plan written.

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 1-2 | API simplification | Complete |
| 5 | Middle layer refactoring | Complete |
| A | Scientific rigor & validation | Complete |
| B | Memory & computational improvements | Complete |
| E | API improvements | Complete |
| D | Documentation & integration | **In progress** |
| F | Test coverage | **In progress** |
| G | GPU & Rust-Python interface | **Planned** |
| H | Field-data validation | **Planned** |
| C | Performance (POI mode) | Deferred |

## Next Tasks

| # | Task | Impact | Status |
| - | ---- | ------ | ------ |
| 1 | Move `cylindric_wedge` to Rust | HIGH - per-timestep hotspot | Pending |
| 2 | GPU context persistence | HIGH - eliminates init overhead | Pending |
| 3 | QGIS plugin testing (Phase 11) | HIGH - blocks plugin adoption | Pending |
| 4 | Field-data validation | HIGH - scientific credibility | Pending |
| 5 | Orchestration layer unit tests | MEDIUM - regression safety | Pending |
| 6 | API reference with mkdocstrings | MEDIUM - user adoption | Pending |
| 7 | POI mode | HIGH - 10-100x speedup | Deferred |

## Test Suite

353 tests across 4 categories:

- **Spec property tests** (`tests/spec/`) - Physical invariants
- **Golden regression tests** (`tests/golden/`) - Reference output comparison
- **Integration tests** (`tests/test_*.py`) - API and feature tests
- **Benchmarks** (`tests/benchmarks/`) - Memory profiling

## Contributing

See [Contributing](contributing.md) for how to help with these priorities.
