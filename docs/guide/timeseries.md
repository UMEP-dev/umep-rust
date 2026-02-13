# Timeseries Calculations

For multi-timestep simulations, use `calculate_timeseries()` which properly handles thermal state across timesteps.

## Basic Timeseries

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
    start="2024-07-01",
    end="2024-07-03",
)

# Calculate timeseries
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
)

print(f"Processed {len(results)} timesteps")
```

## Auto-Saving Outputs

Save results as they're computed:

```python
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    output_dir="output/",
    outputs=["tmrt", "shadow"],  # Which outputs to save
)
```

Creates files like:
```
output/
├── tmrt_20240701_0000.tif
├── tmrt_20240701_0100.tif
├── shadow_20240701_0000.tif
└── ...
```

## Thermal State

Ground temperature modeling requires previous timestep state (thermal inertia). `calculate_timeseries()` handles this automatically:

```python
# DON'T do this - loses thermal state!
for weather in weather_list:
    result = solweig.calculate(surface, location, weather)

# DO this - preserves thermal state
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
)
```

## Post-Processing

Compute thermal comfort indices from saved Tmrt files:

```python
# UTCI - fast (~1 second for 72 timesteps)
n_utci = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)

# PET - slower (~50× slower than UTCI)
n_pet = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
)
```

## Performance

| Grid Size | First Timestep | Subsequent | 72 Timesteps |
|-----------|----------------|------------|--------------|
| 100×100   | ~5s           | ~10ms      | ~1s          |
| 200×200   | ~67s          | ~20ms      | ~2s          |
| 500×500   | ~10min        | ~100ms     | ~8s          |

!!! tip "Pre-compute SVF"
    For production workflows, pre-compute SVF using `SurfaceData.prepare()` with a persistent `working_dir`.

## Memory Considerations

For long timeseries, results are kept in memory. To reduce memory:

```python
# Option 1: Save to disk, don't keep in memory
results = solweig.calculate_timeseries(
    ...,
    output_dir="output/",  # Results saved to disk
)
# results list still populated, but you can process and discard

# Option 2: Process in chunks
for chunk_start in range(0, len(weather_list), 24):
    chunk = weather_list[chunk_start:chunk_start+24]
    results = solweig.calculate_timeseries(
        surface=surface,
        location=location,
        weather_series=chunk,
        output_dir=f"output/chunk_{chunk_start}/",
    )
```
