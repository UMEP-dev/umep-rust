# Project API Design

The Project API is the highest-level interface for SOLWEIG workflows, designed to manage paths, preprocessing, and calculations with minimal boilerplate.

## Design Principles

1. **Explicit when needed, implicit when safe**
   - Users can specify exact paths for full control
   - Smart defaults and auto-discovery reduce boilerplate
   - Convention-based structure when paths not specified

2. **Fail-safe with clear errors**
   - Missing preprocessing detected early
   - `auto_prepare` flag controls automatic generation
   - Status checking shows exactly what's ready

3. **Persistent and resumable**
   - Save project configuration to JSON
   - Load and resume later
   - Relative paths for portability

## Path Resolution Priority

The Project API resolves paths with this priority:

### Walls Directory
1. **Explicit `walls_dir` parameter** - Use if provided
2. **Auto-discover in `cache_dir/walls/`** - Use if exists
3. **Generate on `prepare()`** - Create in `cache_dir/walls/`

### SVF Directory
1. **Explicit `svf_dir` parameter** - Use if provided
2. **Auto-discover in `cache_dir/svf/`** - Use if exists
3. **Generate on `prepare()`** - Create in `cache_dir/svf/`

### Cache Directory Default
- If not specified: `{dsm_directory}/solweig_cache/`
- Convention: All preprocessing goes in `cache_dir/`

## Usage Patterns

### Pattern 1: Explicit Paths (Full Control)

```python
import solweig

project = solweig.Project(
    dsm="data/dsm.tif",
    cdsm="data/cdsm.tif",
    weather="data/weather.epw",
    walls_dir="custom/walls/",  # Explicit
    svf_dir="custom/svf/",      # Explicit
)

# No auto-discovery, uses exactly what you specify
results = project.calculate(start="2023-07-01", end="2023-07-01")
```

### Pattern 2: Convention-based (Auto-discovery)

```python
import solweig

project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/solweig_cache/",
)

# Auto-discovers:
# - data/solweig_cache/walls/ (if exists)
# - data/solweig_cache/svf/ (if exists)

project.print_status()  # Shows what's available

# Auto-prepare if missing
results = project.calculate(
    start="2023-07-01",
    end="2023-07-01",
    auto_prepare=True,  # Generates missing preprocessing
)
```

### Pattern 3: Mixed (Override specific paths)

```python
import solweig

project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",
    walls_dir="shared/walls/",  # Override walls, auto-discover SVF
)

# Uses:
# - shared/walls/ for walls (explicit)
# - data/cache/svf/ for SVF (auto-discovered)
```

### Pattern 4: Persistent Projects

```python
import solweig

# Create and save
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",
)
project.prepare()
project.save()  # Saves to data/cache/project.json

# Later: reload and rerun
project = solweig.Project.load("data/cache/project.json")
results = project.calculate(start="2023-07-02", end="2023-07-02")
```

## API Surface

### Constructor

```python
Project(
    dsm: str | Path,                    # Required: DSM path
    weather: str | Path,                # Required: Weather file
    cdsm: str | Path | None = None,    # Optional: Vegetation
    cache_dir: str | Path | None = None,  # Preprocessing directory
    walls_dir: str | Path | None = None,  # Override walls path
    svf_dir: str | Path | None = None,    # Override SVF path
)
```

### Methods

**`status() -> dict`**
- Returns dictionary of component status
- Keys: "dsm", "cdsm", "weather", "walls", "svf"
- Values: {"ready": bool, "path": str | None, "size": str | None}

**`print_status()`**
- Human-readable status report
- Shows what's ready and what's missing
- Suggests actions (e.g., "Run project.prepare()")

**`prepare(force=False, trans_veg_perc=3.0)`**
- Generate missing preprocessing
- `force=True`: Regenerate even if exists
- `trans_veg_perc`: Vegetation transmissivity for SVF

**`calculate(start, end, **kwargs)`**
- Run SOLWEIG calculation
- `auto_prepare=True`: Auto-run prepare() if needed
- Returns `list[SolweigResult]`
- All `calculate_timeseries()` parameters supported

**`save(path=None)`**
- Save project config to JSON
- Default: `cache_dir/project.json`

**`load(path) -> Project`** (classmethod)
- Load project from JSON config
- Returns configured Project instance

## Examples

### Quick Start

```python
import solweig

# Create project
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",
)

# One-liner: prepare and calculate
results = project.calculate(
    start="2023-07-01",
    end="2023-07-01",
    auto_prepare=True,  # Auto-generates preprocessing
)
```

### Production Workflow

```python
import solweig

# 1. Setup project
project = solweig.Project(
    dsm="data/dsm.tif",
    cdsm="data/cdsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",
)

# 2. Check status
project.print_status()

# 3. Prepare if needed
if not project.status()["walls"]["ready"]:
    project.prepare()

# 4. Calculate multiple periods
for month in range(1, 13):
    start = f"2023-{month:02d}-01"
    end = f"2023-{month:02d}-28"
    results = project.calculate(
        start=start,
        end=end,
        output_dir=f"output/month_{month:02d}/",
    )

# 5. Save config
project.save()
```

### Working with Existing Preprocessing

```python
import solweig

# Point to existing preprocessing
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    walls_dir="/shared/preprocessed/walls/",
    svf_dir="/shared/preprocessed/svf/",
)

# Skip preparation, go straight to calculation
results = project.calculate(
    start="2023-07-01",
    end="2023-07-01",
    auto_prepare=False,  # Error if preprocessing missing
)
```

## Comparison with Lower-Level APIs

### Low-Level (Direct API)

```python
import solweig

# Manual path management
surface, precomputed = solweig.SurfaceData.from_geotiff(
    dsm="data/dsm.tif",
    walls_dir="data/cache/walls/",
    svf_dir="data/cache/svf/",
)

location = solweig.Location.from_dsm_crs("data/dsm.tif")

weather = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2023-07-01",
    end="2023-07-01",
)

results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather,
    precomputed=precomputed,
    output_dir="output/",
)
```

### High-Level (Project API)

```python
import solweig

# Automatic path management
project = solweig.Project(
    dsm="data/dsm.tif",
    weather="data/weather.epw",
    cache_dir="data/cache/",
)

results = project.calculate(
    start="2023-07-01",
    end="2023-07-01",
    output_dir="output/",
    auto_prepare=True,
)
```

**Reduction:** 23 lines → 11 lines (52% less code)

## When to Use Each API

### Use Project API when:
- Running complete workflows start-to-finish
- Managing multiple related calculations
- Working with teams (shareable project configs)
- Want automatic preprocessing management
- Prefer convention over configuration

### Use Direct API when:
- Integrating into larger systems
- Need fine-grained control
- Building custom workflows
- Working with non-standard paths
- Calling specific functions programmatically

### Use Legacy API when:
- Maintaining existing workflows
- Using config file-driven approaches
- Backward compatibility required

## Implementation Notes

### Thread Safety
- Project is NOT thread-safe
- Don't share Project instances across threads
- Create separate instances per thread

### Performance
- Path resolution happens once in `__post_init__()`
- Status checks are lazy (only check when called)
- No caching of loaded data between calls

### Validation
- File existence checked in `status()`
- Missing files error in `calculate()`
- Preprocessing validation in `prepare()`

### Portability
- Use relative paths when possible
- Save/load preserves relative paths
- Move entire directory tree together
