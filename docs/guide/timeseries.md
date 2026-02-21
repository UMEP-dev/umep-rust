# Timeseries Calculations

For multi-timestep simulations (hours, days, or longer), use `calculate()` with a list of weather objects. It properly carries thermal state between timesteps and saves results to disk as they're computed.

## Thermal state management

Ground and wall temperatures depend on accumulated heating from previous hours (thermal inertia). `calculate()` manages this automatically via a `ThermalState` object when given multiple weather timesteps, producing accurate results for ground-level longwave radiation.

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

# Calculate all timesteps — results saved to output/
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

print(f"Processed {len(summary)} timesteps")
```

This creates timestamped GeoTIFFs plus summary grids:

```text
output/
├── tmrt/
│   ├── tmrt_20250701_0000.tif
│   ├── tmrt_20250701_0100.tif
│   └── ...
├── shadow/
│   ├── shadow_20250701_0000.tif
│   └── ...
├── summary/
│   ├── tmrt_mean.tif
│   ├── tmrt_max.tif
│   ├── utci_mean.tif
│   └── ...
└── run_metadata.json            # All parameters for reproducibility
```

If you only need summary statistics (mean/max/min Tmrt, UTCI, sun hours), omit
`outputs` — no per-timestep files are written, but the summary grids are still
saved to `output/summary/`.

```python
summary = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather_list,
    output_dir="output/",  # Summary grids + metadata saved here
)
```

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

## Per-timestep UTCI and PET

UTCI and PET summary grids (mean, max, min, day/night averages) are always
included in the returned `TimeseriesSummary`. To also save per-timestep
UTCI or PET GeoTIFFs, include them in `outputs`:

```python
summary = solweig.calculate(
    surface=surface,
    weather=weather_list,
    output_dir="output/",
    outputs=["tmrt", "utci"],  # per-timestep Tmrt + UTCI saved to disk
)
```

The indices are computed inline during the main loop
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

SVF is prepared automatically by `SurfaceData.prepare()`. For file workflows, use a persistent `working_dir` to avoid recomputing SVF on every run.

## Run metadata

`calculate()` automatically saves a `run_metadata.json` file capturing all parameters used. Load it later for reproducibility:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(f"Version: {metadata['solweig_version']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
print(f"Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
```
