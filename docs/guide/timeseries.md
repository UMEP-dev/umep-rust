# Timeseries Calculations

For multi-timestep simulations (hours, days, or longer), use `calculate()` with a list of weather objects. It properly carries thermal state between timesteps and optionally saves results to disk as they're computed.

## Thermal state management

Ground and wall temperatures depend on accumulated heating from previous hours (thermal inertia). `calculate()` manages this automatically via a `ThermalState` object when given multiple weather timesteps, producing accurate results for ground-level longwave radiation.

```python
results = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
)
```

## Basic timeseries

```python
import solweig

# Load surface
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
)

# Load weather from EPW file
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

# Calculate all timesteps
results = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
)

print(f"Processed {len(results)} timesteps")
```

## Saving results to disk

For long simulations, save results as GeoTIFFs as they're computed rather than keeping them all in memory:

```python
results = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],  # Which outputs to save
)
```

This creates timestamped GeoTIFFs:

```text
output/
├── tmrt/
│   ├── tmrt_20250701_0000.tif
│   ├── tmrt_20250701_0100.tif
│   └── ...
├── shadow/
│   ├── shadow_20250701_0000.tif
│   └── ...
└── run_metadata.json            # All parameters for reproducibility
```

By default, only summary grids are returned (no per-timestep arrays in memory).
Use `timestep_outputs=["tmrt", "shadow"]` to retain specific per-timestep arrays.

## Inspecting results

### Summary report

`report()` returns a human-readable text summary of the run:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
print(summary.report())
```

In Jupyter notebooks, placing `summary` as the last expression in a cell
renders the report automatically (via `_repr_html_`).

### Per-timestep timeseries

`summary.timeseries` contains 1-D arrays of spatial means at each timestep
— useful for understanding how conditions evolved over the simulation:

```python
ts = summary.timeseries
print(ts.datetime)       # timestamps
print(ts.ta)             # air temperature per step
print(ts.tmrt_mean)      # spatial mean Tmrt per step
print(ts.utci_mean)      # spatial mean UTCI per step
print(ts.sun_fraction)   # fraction of sunlit pixels per step
```

### Plotting

`plot()` produces a multi-panel figure showing temperature, radiation,
sun exposure, and meteorological inputs over time:

```python
summary.plot()                                  # interactive display
summary.plot(save_path="output/timeseries.png") # save to file
```

Requires `matplotlib` (`pip install matplotlib`).

## Choose an output strategy

### Strategy A: Stream to disk during computation

Use this for long runs and limited RAM.

```python
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

- Pros: lowest memory use, immediate GeoTIFF outputs, restart-friendly
- Cons: more disk I/O/storage

### Strategy B: Summary only (no file output)

Use this when disk space is tight and you only need summary products.
`TimeseriesSummary` aggregates per-pixel statistics (mean/max/min Tmrt and
UTCI, sun/shade hours, threshold exceedance) incrementally during the loop,
so per-timestep arrays are freed immediately.

```python
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    # No output_dir -> summary-only, minimal memory
)
print(summary.report())
summary.to_geotiff("output/")  # Save summary grids only
```

- Pros: minimal disk usage and memory, automatic aggregation
- Cons: no per-timestep files on disk

## Per-timestep UTCI and PET

UTCI and PET summary grids (mean, max, min, day/night averages) are always
included in the returned `TimeseriesSummary`. To also retain per-timestep
UTCI or PET arrays, include them in `timestep_outputs`:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    timestep_outputs=["tmrt", "utci"],  # per-timestep Tmrt + UTCI
    output_dir="output/",
    outputs=["tmrt", "utci"],           # save both as GeoTIFF files
)
for r in summary.results:
    print(f"UTCI range: {r.utci.min():.1f} – {r.utci.max():.1f}°C")
```

To also save per-timestep files to disk, add `"utci"` or `"pet"` to the
`outputs` parameter. The indices are computed inline during the main loop
(UTCI uses a fast Rust polynomial; PET uses an iterative solver).

## Memory management for long simulations

### Processing in chunks

For very long simulations (weeks or months), process in daily chunks:

```python
for chunk_start in range(0, len(weather_list), 24):
    chunk = weather_list[chunk_start:chunk_start + 24]
    results = solweig.calculate(
        surface=surface,
        location=location,
        weather=chunk,
        output_dir=f"output/",
    )
```

## Performance

| Grid size | Surface prep (SVF) | Per timestep | 72 timesteps |
| --------- | -------------------- | -------------------- | ------------ |
| 100x100 | ~5 s | ~10 ms | ~1 s |
| 200x200 | ~67 s | ~20 ms | ~2 s |
| 500x500 | ~10 min | ~100 ms | ~8 s |

SVF is prepared explicitly (via `SurfaceData.prepare()` or `surface.compute_svf()`). Use a persistent `working_dir` with `prepare()` to avoid recomputing SVF on every run.

## Run metadata

`calculate()` automatically saves a `run_metadata.json` file capturing all parameters used. Load it later for reproducibility:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(f"Version: {metadata['solweig_version']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
print(f"Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
```
