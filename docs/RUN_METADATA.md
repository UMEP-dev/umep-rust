# Run Metadata and Provenance

**TL;DR:** Every calculation automatically saves a `run_metadata.json` file that captures all parameters, inputs, and configuration for perfect reproducibility.

---

## What is Run Metadata?

Run metadata is a complete record of all parameters and configuration used in a SOLWEIG calculation. This enables:

1. **Reproducibility** - Re-run the exact same calculation months later
2. **Audit Trail** - Document what parameters were used for publications or reports
3. **Debugging** - Understand why results differ between runs
4. **Archiving** - Save complete experimental setup alongside results

---

## Automatic Metadata Saving

When you use `calculate_timeseries()` with `output_dir` specified, metadata is **automatically saved**:

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01")

# This automatically saves run_metadata.json to output_dir
results = solweig.calculate_timeseries(
    surface, weather,
    human=solweig.HumanParams(weight=70, height=1.65),
    use_anisotropic_sky=True,
    output_dir="output/",  # <-- Triggers automatic metadata saving
)

# Metadata is now saved at: output/run_metadata.json
```

**No extra work needed!** The metadata file is created automatically when `output_dir` is provided.

---

## What's Captured?

The metadata file records everything needed to reproduce a calculation:

### 1. Execution Info
- Timestamp of calculation
- SOLWEIG version used
- Compute backend (CPU/GPU)

### 2. Location
- Latitude, longitude, altitude
- UTC offset

### 3. Model Flags
- `use_anisotropic_sky`: Sky model type
- `conifer`: Evergreen vs deciduous trees
- `use_legacy_kelvin_offset`: Backward compatibility flag

### 4. Human Parameters
- Posture (standing/sitting)
- Absorption coefficients (shortwave/longwave)
- Body metrics (age, weight, height)
- Activity level and clothing insulation

### 5. Physics Parameters
- Whether custom physics file was used
- Path to custom physics file (if any)
- Full physics parameters (if custom)

### 6. Materials Parameters
- Whether materials were used
- Path to materials file (if any)
- Full materials parameters (if used)

### 7. Surface Inputs
- Paths to DSM, CDSM, landcover files
- Bounding box and pixel size
- CRS (coordinate reference system)
- Grid dimensions

### 8. Weather Info
- Path to EPW file or other weather source
- Number of timesteps
- Date range (start and end)

### 9. Outputs
- Output directory path
- List of output types saved

---

## Loading and Inspecting Metadata

Load metadata to inspect or verify parameters:

```python
import solweig

# Load metadata from previous run
metadata = solweig.load_run_metadata("output/run_metadata.json")

# Inspect key parameters
print(f"Calculation performed: {metadata['timestamp']}")
print(f"SOLWEIG version: {metadata['solweig_version']}")
print(f"Location: {metadata['location']['latitude']:.2f}°N")
print(f"Human posture: {metadata['human_params']['posture']}")
print(f"Anisotropic sky: {metadata['model_flags']['use_anisotropic_sky']}")
print(f"Weather period: {metadata['weather']['date_range']}")
```

---

## Manual Metadata Creation

For custom workflows, create metadata manually:

```python
import solweig

# Prepare your calculation
surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw")
location = solweig.Location.from_surface(surface)
human = solweig.HumanParams(weight=70)

# Create metadata
metadata = solweig.create_run_metadata(
    surface=surface,
    location=location,
    weather_series=weather,
    weather_source_path="weather.epw",
    human=human,
    use_anisotropic_sky=True,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

# Save to custom location
solweig.save_run_metadata(metadata, output_dir="custom_dir/", filename="my_metadata.json")
```

---

## Example Metadata File

Here's what a typical `run_metadata.json` looks like:

```json
{
  "timestamp": "2024-07-15T14:30:22.123456",
  "solweig_version": "0.0.1a1",
  "compute_backend": "cpu",
  "location": {
    "latitude": 37.98,
    "longitude": 23.73,
    "altitude": 0.0,
    "utc_offset": 2
  },
  "model_flags": {
    "use_anisotropic_sky": true,
    "conifer": false,
    "use_legacy_kelvin_offset": false
  },
  "human_params": {
    "posture": "standing",
    "abs_k": 0.7,
    "abs_l": 0.95,
    "age": 35,
    "weight": 75,
    "height": 180,
    "activity": 80,
    "clothing": 0.9
  },
  "physics": {
    "custom": false,
    "path": null
  },
  "materials": {
    "used": false,
    "path": null
  },
  "surface": {
    "dsm_path": "/path/to/DSM.tif",
    "cdsm_path": "/path/to/CDSM.tif",
    "land_cover_path": null,
    "bbox": [476800, 4205850, 477200, 4206250],
    "pixel_size": 1.0,
    "crs_wkt": "PROJCS[...]",
    "shape": [400, 400]
  },
  "weather": {
    "source_path": "/path/to/athens_2023.epw",
    "num_timesteps": 72,
    "date_range": ["2023-07-01T00:00:00", "2023-07-03T23:00:00"]
  },
  "outputs": {
    "output_dir": "/path/to/output",
    "outputs": ["tmrt", "shadow"]
  }
}
```

---

## Use Cases

### Research Publications

Document exact parameters for reproducible science:

```python
# Run calculation
results = solweig.calculate_timeseries(
    surface, weather,
    human=solweig.HumanParams(weight=75, height=1.80),
    use_anisotropic_sky=True,
    output_dir="paper_results/",
)

# Metadata is saved automatically - include it in supplementary materials
# Readers can reproduce your exact calculation
```

### Comparing Runs

Compare metadata from different runs to understand differences:

```python
# Load metadata from two runs
meta_run1 = solweig.load_run_metadata("run1/run_metadata.json")
meta_run2 = solweig.load_run_metadata("run2/run_metadata.json")

# Compare key parameters
print("Run 1 posture:", meta_run1['human_params']['posture'])
print("Run 2 posture:", meta_run2['human_params']['posture'])

print("Run 1 sky model:", meta_run1['model_flags']['use_anisotropic_sky'])
print("Run 2 sky model:", meta_run2['model_flags']['use_anisotropic_sky'])
```

### Archival and Documentation

Save complete experimental setup alongside results:

```python
# Your calculation produces:
# output/
#   ├── run_metadata.json      <-- Complete parameter record
#   ├── tmrt_2023-07-01_1200.tif
#   ├── tmrt_2023-07-01_1300.tif
#   └── ...

# Archive the entire directory - everything needed to reproduce the calculation
```

### Debugging

Verify parameters when results seem unexpected:

```python
metadata = solweig.load_run_metadata("output/run_metadata.json")

# Check if anisotropic sky was actually enabled
if not metadata['model_flags']['use_anisotropic_sky']:
    print("Warning: Anisotropic sky was disabled!")

# Check human parameters
if metadata['human_params']['posture'] != 'standing':
    print(f"Note: Results are for {metadata['human_params']['posture']} posture")
```

---

## Custom Physics and Materials

When using custom physics or materials files, the **full parameters are saved** in the metadata:

```python
# Load custom physics
physics = solweig.load_physics("custom_trees.json")

# Calculate with custom physics
results = solweig.calculate_timeseries(
    surface, weather,
    physics=physics,
    output_dir="output/",
)

# Metadata now includes:
# - physics.custom: true
# - physics.path: "custom_trees.json"
# - physics.full_params: {...complete physics parameters...}
```

This ensures the metadata is **self-contained** - you don't need to keep track of the separate physics file.

---

## Backward Compatibility Notes

The metadata system is designed to **complement**, not replace, the old config file approach.

**Old workflow (still supported):**
```python
# Legacy API with config files
SRR = solweig.SolweigRunRust(
    "configsolweig.ini",
    "parametersforsolweig.json"
)
SRR.run()
# No automatic metadata saved
```

**New workflow:**
```python
# Simplified API with automatic metadata
results = solweig.calculate_timeseries(
    surface, weather,
    output_dir="output/",
)
# Metadata automatically saved to output/run_metadata.json
```

The metadata format is JSON-based and **not intended** to be a drop-in replacement for the old `.ini` config format. Instead, it provides a **more complete** record that includes:
- Runtime information (timestamp, version)
- Derived values (auto-extracted location)
- Complete parameter sets (physics, materials)

---

## API Reference

### `create_run_metadata()`

Create a metadata dictionary for a SOLWEIG run.

**Parameters:**
- `surface`: SurfaceData object
- `location`: Location object
- `weather_series`: List of Weather objects (optional)
- `weather_source_path`: Path to EPW file (optional)
- `human`: HumanParams object (optional, uses defaults if None)
- `physics`: Physics parameters from load_physics() (optional)
- `physics_path`: Path to custom physics file (optional)
- `materials`: Materials from load_materials() (optional)
- `materials_path`: Path to materials file (optional)
- `use_anisotropic_sky`: Anisotropic sky flag
- `conifer`: Conifer mode flag
- `output_dir`: Output directory path (optional)
- `outputs`: List of output types (optional)
- `use_legacy_kelvin_offset`: Backward compatibility flag

**Returns:** Dictionary containing complete metadata

---

### `save_run_metadata()`

Save metadata dictionary to JSON file.

**Parameters:**
- `metadata`: Metadata dict from create_run_metadata()
- `output_dir`: Directory to save metadata file
- `filename`: Filename (default: "run_metadata.json")

**Returns:** Path to saved metadata file

---

### `load_run_metadata()`

Load metadata from JSON file.

**Parameters:**
- `metadata_path`: Path to metadata JSON file

**Returns:** Metadata dictionary

---

## Summary

**Automatic:** Metadata is saved automatically when you use `output_dir`

**Complete:** Captures all parameters, inputs, and configuration

**Reproducible:** Contains everything needed to re-run the exact calculation

**Self-contained:** Includes full custom physics/materials (not just paths)

**Future-proof:** Version information enables backward compatibility

**The point:** Perfect reproducibility with zero extra effort.
