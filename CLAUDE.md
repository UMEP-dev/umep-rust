# CLAUDE.md - Project Notes for Claude Code

## Project Overview

SOLWEIG is a high-performance urban microclimate model (Rust + Python via maturin).
It calculates Mean Radiant Temperature (Tmrt), UTCI, and PET for urban thermal comfort analysis.

## Key Files

- `MODERNIZATION_PLAN.md` - API simplification roadmap (Phase 1 complete, Phases 2-6 pending)
- `specs/` - Markdown specifications for each module (shadows, svf, radiation, tmrt, utci, pet)
- `rust/` - Rust implementation via maturin
- `pysrc/` - Python source code

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
