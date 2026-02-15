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

## Choose an output strategy

### Strategy A: Stream to disk during computation

Use this for long runs and limited RAM.

```python
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

- Pros: low memory use, immediate GeoTIFF outputs, restart-friendly
- Cons: more disk I/O and storage

### Strategy B: Keep in memory, aggregate, save selected outputs manually

Use this when disk space is tight and you only need summary products.

```python
import numpy as np
import solweig

sum_tmrt = None
count_tmrt = None
max_tmrt = None

# Process in chunks to keep memory bounded
for i in range(0, len(weather_list), 24):
    chunk = weather_list[i : i + 24]
    results = solweig.calculate_timeseries(
        surface=surface,
        location=location,
        weather_series=chunk,
        # No output_dir -> keep arrays in memory
    )

    for result in results:
        tmrt = result.tmrt.astype(np.float32)
        valid = np.isfinite(tmrt)

        if sum_tmrt is None:
            sum_tmrt = np.zeros_like(tmrt, dtype=np.float64)
            count_tmrt = np.zeros_like(tmrt, dtype=np.uint32)
            max_tmrt = np.full_like(tmrt, -np.inf, dtype=np.float32)

        sum_tmrt[valid] += tmrt[valid]
        count_tmrt[valid] += 1
        max_tmrt = np.maximum(max_tmrt, np.nan_to_num(tmrt, nan=-np.inf))

# Build aggregated products
mean_tmrt = np.divide(
    sum_tmrt,
    np.maximum(count_tmrt, 1),
    dtype=np.float64,
).astype(np.float32)
mean_tmrt[count_tmrt == 0] = np.nan
max_tmrt[~np.isfinite(max_tmrt)] = np.nan

# Save only final products
solweig.SolweigResult(tmrt=mean_tmrt).to_geotiff(
    output_dir="summary/",
    outputs=["tmrt"],
    surface=surface,
)
solweig.SolweigResult(tmrt=max_tmrt).to_geotiff(
    output_dir="summary_max/",
    outputs=["tmrt"],
    surface=surface,
)
```

- Pros: minimal disk usage, flexible custom products
- Cons: you must manage aggregation logic explicitly

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

| Grid size | Surface prep (SVF) | Per timestep | 72 timesteps |
| --------- | -------------------- | -------------------- | ------------ |
| 100x100 | ~5 s | ~10 ms | ~1 s |
| 200x200 | ~67 s | ~20 ms | ~2 s |
| 500x500 | ~10 min | ~100 ms | ~8 s |

SVF is prepared explicitly (via `SurfaceData.prepare()` or `surface.compute_svf()`). Use a persistent `working_dir` with `prepare()` to avoid recomputing SVF on every run.

## Run metadata

`calculate_timeseries()` automatically saves a `run_metadata.json` file capturing all parameters used. Load it later for reproducibility:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")
print(f"Version: {metadata['solweig_version']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
print(f"Date range: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
```
