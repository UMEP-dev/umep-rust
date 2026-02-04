# SOLWEIG

**High-performance urban microclimate model for Mean Radiant Temperature (Tmrt) and thermal comfort indices.**

SOLWEIG calculates spatially distributed Tmrt, UTCI, and PET for urban environments using Digital Surface Models (DSM) and meteorological data.

## Features

- **Fast**: Rust-accelerated core algorithms with GPU support
- **Accurate**: Validated against field measurements
- **Easy**: Simple Python API with sensible defaults
- **Flexible**: Works with GeoTIFFs or numpy arrays

## Quick Example

```python
import solweig
from datetime import datetime

# Create surface from DSM
surface = solweig.SurfaceData(dsm=my_dsm_array, pixel_size=1.0)

# Define location and weather
location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=1)
weather = solweig.Weather(
    datetime=datetime(2024, 7, 15, 12, 0),
    ta=25.0,        # Air temperature (°C)
    rh=50.0,        # Relative humidity (%)
    global_rad=800.0  # Global radiation (W/m²)
)

# Calculate Tmrt
result = solweig.calculate(surface, location, weather)
print(f"Mean Tmrt: {result.tmrt.mean():.1f}°C")

# Compute thermal comfort
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")
```

## Installation

```bash
# Clone and install
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
uv sync                  # Install Python dependencies
maturin develop          # Build Rust extension
```

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install SOLWEIG and run your first calculation

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn how to use SOLWEIG for different scenarios

    [:octicons-arrow-right-24: User Guide](guide/basic-usage.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete reference for all classes and functions

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-flask:{ .lg .middle } **Physics**

    ---

    Scientific documentation of the radiation model

    [:octicons-arrow-right-24: Physics](physics/index.md)

</div>

## Citation

If you use SOLWEIG in your research, please cite:

> Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## License

GNU General Public License v3.0. See [LICENSE](https://github.com/UMEP-dev/solweig/blob/main/LICENSE) for details.
