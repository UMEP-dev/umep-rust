# Timeseries Calculations

For multi-timestep simulations (hours, days, or longer), use `calculate_timeseries()`. It properly carries thermal state between timesteps and optionally saves results to disk as they're computed.

## Why not loop over `calculate()`?

Ground and wall temperatures depend on accumulated heating from previous hours (thermal inertia). `calculate_timeseries()` manages this automatically via a `ThermalState` object. Looping over `calculate()` yourself loses this state, producing less accurate results — especially for ground-level longwave radiation.

```python
# Don't do this — loses thermal state between timesteps
for weather in weather_list:
    result = solweig.calculate(surface, location, weather)

# Do this instead
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
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
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
)

print(f"Processed {len(results)} timesteps")
```

## Saving results to disk

For long simulations, save results as GeoTIFFs as they're computed rather than keeping them all in memory:

```python
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
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

When `output_dir` is set, arrays are freed after saving to conserve memory.

## Post-processing thermal comfort

Compute UTCI or PET from saved Tmrt files without re-running the main calculation:

```python
# UTCI — fast polynomial (~1 second for 72 timesteps)
n_utci = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)

# PET — slower iterative solver (~50x slower than UTCI)
n_pet = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
    human=solweig.HumanParams(weight=70, height=1.75),
)
```

This separation means you can:

- Skip thermal comfort entirely if you only need Tmrt
- Recompute for different human parameters without re-running the main model
- Process only a subset of timesteps

## Memory management for long simulations

### Processing in chunks

For very long simulations (weeks or months), process in daily chunks:

```python
for chunk_start in range(0, len(weather_list), 24):
    chunk = weather_list[chunk_start:chunk_start + 24]
    results = solweig.calculate_timeseries(
        surface=surface,
        location=location,
        weather_series=chunk,
        output_dir=f"output/",
    )
```

## Performance

| Grid size | First timestep (SVF) | Subsequent timesteps | 72 timesteps |
| --------- | -------------------- | -------------------- | ------------ |
| 100x100 | ~5 s | ~10 ms | ~1 s |
| 200x200 | ~67 s | ~20 ms | ~2 s |
| 500x500 | ~10 min | ~100 ms | ~8 s |

The first timestep is dominated by SVF computation. Use `SurfaceData.prepare()` with a persistent `working_dir` to avoid recomputing SVF on every run.

## Run metadata

`calculate_timeseries()` automatically saves a `run_metadata.json` file capturing all parameters used. Load it later for reproducibility:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(f"Version: {metadata['solweig_version']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
print(f"Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
```
