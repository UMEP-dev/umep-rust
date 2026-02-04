# SOLWEIG

High-performance urban microclimate model for computing Mean Radiant Temperature (Tmrt) and thermal comfort indices (UTCI, PET).

**Rust + Python** via [maturin](https://github.com/PyO3/maturin) for performance-critical algorithms.

## Installation

```bash
# Clone and install
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
uv sync                  # Install Python dependencies
maturin develop          # Build Rust extension
```

## Quick Start

```python
import solweig
from datetime import datetime

# Create surface from DSM array
surface = solweig.SurfaceData(dsm=my_dsm_array, pixel_size=1.0)

# Define location and weather
location = solweig.Location(latitude=57.7, longitude=12.0)
weather = solweig.Weather(
    datetime=datetime(2024, 7, 15, 12, 0),
    ta=25.0,        # Air temperature (°C)
    rh=50.0,        # Relative humidity (%)
    global_rad=800.0  # Global radiation (W/m²)
)

# Calculate Tmrt
result = solweig.calculate(surface, location, weather)
print(f"Tmrt: {result.tmrt.mean():.1f}°C")
```

## Loading from GeoTIFFs

```python
import solweig

# Load and prepare surface data (auto-computes walls/SVF)
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",       # Walls/SVF cached here
    cdsm="data/cdsm.tif",       # Optional: vegetation
)

# Load weather from EPW file
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2023-07-01",
    end="2023-07-03",
)

# Calculate timeseries
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    output_dir="output/",
)
```

## Post-Processing (UTCI/PET)

Thermal comfort indices can be computed directly from results:

```python
# Single timestep: compute directly from result
result = solweig.calculate(surface, location, weather)
utci = result.compute_utci(weather)  # Fast polynomial
pet = result.compute_pet(weather)    # Slower iterative solver

# Batch processing: from saved Tmrt files
solweig.compute_utci(tmrt_dir="output/", weather_series=weather_list, output_dir="utci/")
solweig.compute_pet(tmrt_dir="output/", weather_series=weather_list, output_dir="pet/")
```

## Input Validation

Validate inputs before expensive calculations:

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")
    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

## Demos

Complete working examples are in the [demos/](demos/) folder:

- [demos/athens-demo.py](demos/athens-demo.py) - Full workflow with GeoTIFFs
- [demos/solweig_gbg_test.py](demos/solweig_gbg_test.py) - Gothenburg test data

## Documentation

- [Quick Start Guide](docs/getting-started/quick-start.md) - Detailed tutorial
- [Physics Specifications](specs/) - Scientific documentation
- [ROADMAP.md](ROADMAP.md) - Development priorities

## Build & Test

```bash
maturin develop          # Build Rust extension
pytest tests/            # Run tests
poe verify_project       # Full verification (format, lint, test)
```

## Original Code

This package is adapted from the GPLv3-licensed [UMEP-processing](https://github.com/UMEP-dev/UMEP-processing) by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

Licensed under GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

**Citation:**

> Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)
