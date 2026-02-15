# SOLWEIG

**Map how hot it *feels* across a city — pixel by pixel.**

SOLWEIG computes **Mean Radiant Temperature (Tmrt)** and thermal comfort indices (**UTCI**, **PET**) for urban environments. Give it a building height model and weather data, and it produces high-resolution maps showing where people experience heat stress — and where trees, shade, and cool surfaces make a difference.

## Who is this for?

- **Urban planners** comparing street designs, tree planting, or cool-roof strategies
- **Researchers** running controlled microclimate experiments at 1 m resolution
- **Climate service providers** generating heat-risk maps for public health or events
- **Students** learning about urban radiation and thermal comfort

## The 30-second version

```python
import solweig

# Load your building heights and weather
surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather_list = solweig.Weather.from_epw("weather.epw", start="2025-07-01", end="2025-07-03")
location = solweig.Location.from_epw("weather.epw")

# Run — results saved as GeoTIFFs
solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    output_dir="output/",
)
```

That's it. `SurfaceData.prepare()` computes/caches walls and SVF; then `calculate_timeseries()` computes shadows, radiation, and Tmrt.

!!! note "SVF Rule"
    `calculate()` / `calculate_timeseries()` require SVF to already be available on `surface` (or via `precomputed.svf`).
    Use `SurfaceData.prepare(...)` for automatic SVF preparation/caching, or call `surface.compute_svf()` explicitly for in-memory/manual surfaces.

!!! note "Anisotropic Rule"
    If you explicitly set `use_anisotropic_sky=True`, shadow matrices must already be available
    (`surface.shadow_matrices` or `precomputed.shadow_matrices`), typically prepared via
    `SurfaceData.prepare(...)` or `surface.compute_svf()`.

## How it works

SOLWEIG models the complete radiation budget experienced by a person standing outdoors:

1. **Shadows** — Which pixels are shaded by buildings and trees?
2. **Sky View Factor** — How much open sky does each point see?
3. **Surface temperatures** — How hot are the ground and walls?
4. **Radiation balance** — Sum shortwave (sun) and longwave (heat) from all directions
5. **Tmrt** — Convert absorbed radiation into a single "felt temperature"
6. **Thermal comfort** — Optionally derive UTCI or PET indices

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install SOLWEIG and run your first calculation in minutes

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)
    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Common workflows: loading GeoTIFFs, running timeseries, thermal comfort

    [:octicons-arrow-right-24: Basic Usage](guide/basic-usage.md)
    [:octicons-arrow-right-24: Working with GeoTIFFs](guide/geotiffs.md)
    [:octicons-arrow-right-24: Timeseries](guide/timeseries.md)
    [:octicons-arrow-right-24: Thermal Comfort](guide/thermal-comfort.md)

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

If you use SOLWEIG in your research, please cite the original model paper and the UMEP platform:

1. Lindberg F, Holmer B, Thorsson S (2008) SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology* 52, 697–713 [doi:10.1007/s00484-008-0162-7](https://doi.org/10.1007/s00484-008-0162-7)

2. Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) – An integrated tool for city-based climate services. *Environmental Modelling and Software* 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## License

GNU Affero General Public License v3.0. See [LICENSE](https://github.com/UMEP-dev/solweig/blob/main/LICENSE) for details.
