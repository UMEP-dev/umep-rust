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

When you use `calculate()` with `output_dir` specified, metadata is **automatically saved**:

```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01")

# This automatically saves run_metadata.json to output_dir
results = solweig.calculate(
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

The metadata file records:

### 1. Execution Info

- `solweig_version`: SOLWEIG version used
- `run_timestamp`: When the calculation was performed

### 2. Grid

- `rows`, `cols`: Grid dimensions
- `pixel_size`: Pixel size in metres
- `crs`: Coordinate reference system

### 3. Location

- `latitude`, `longitude`, `utc_offset`

### 4. Timeseries

- `start`, `end`: Date range (ISO format)
- `timesteps`: Number of timesteps

### 5. Parameters

- `use_anisotropic_sky`: Sky model type
- `conifer`: Evergreen vs deciduous trees

### 6. Outputs

- `directory`: Output directory path
- `variables`: List of output types saved

### 7. Human Parameters (if provided)

- `abs_k`, `abs_l`: Absorption coefficients
- `posture`: Standing or sitting

### 8. Physics and Materials (if custom)

- Full parameter dictionaries when custom physics or materials are provided

---

## Loading and Inspecting Metadata

Load metadata to inspect or verify parameters:

```python
import solweig

# Load metadata from previous run
metadata = solweig.load_run_metadata("output/run_metadata.json")

# Inspect key parameters
print(f"Calculation performed: {metadata['run_timestamp']}")
print(f"SOLWEIG version: {metadata['solweig_version']}")
print(f"Location: {metadata['location']['latitude']:.2f}°N")
print(f"Anisotropic sky: {metadata['parameters']['use_anisotropic_sky']}")
print(f"Weather period: {metadata['timeseries']['start']} to {metadata['timeseries']['end']}")
print(f"Timesteps: {metadata['timeseries']['timesteps']}")
```

If human parameters were provided:

```python
if "human" in metadata:
    print(f"Posture: {metadata['human']['posture']}")
    print(f"abs_k: {metadata['human']['abs_k']}")
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
    weather=weather,
    human=human,
    physics=None,
    materials=None,
    use_anisotropic_sky=True,
    conifer=False,
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
  "solweig_version": "0.0.1a1",
  "run_timestamp": "2024-07-15T14:30:22.123456",
  "grid": {
    "rows": 400,
    "cols": 400,
    "pixel_size": 1.0,
    "crs": "PROJCS[...]"
  },
  "location": {
    "latitude": 37.98,
    "longitude": 23.73,
    "utc_offset": 2
  },
  "timeseries": {
    "start": "2023-07-01T00:00:00",
    "end": "2023-07-03T23:00:00",
    "timesteps": 72
  },
  "parameters": {
    "use_anisotropic_sky": true,
    "conifer": false
  },
  "outputs": {
    "directory": "/path/to/output",
    "variables": ["tmrt", "shadow"]
  },
  "human": {
    "abs_k": 0.7,
    "abs_l": 0.97,
    "posture": "standing"
  }
}
```

---

## Use Cases

### Research Publications

Document exact parameters for reproducible science:

```python
# Run calculation
results = solweig.calculate(
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
if "human" in meta_run1 and "human" in meta_run2:
    print("Run 1 posture:", meta_run1['human']['posture'])
    print("Run 2 posture:", meta_run2['human']['posture'])

print("Run 1 sky model:", meta_run1['parameters']['use_anisotropic_sky'])
print("Run 2 sky model:", meta_run2['parameters']['use_anisotropic_sky'])
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
if not metadata['parameters']['use_anisotropic_sky']:
    print("Warning: Anisotropic sky was disabled!")

# Check human parameters
if "human" in metadata and metadata['human']['posture'] != 'standing':
    print(f"Note: Results are for {metadata['human']['posture']} posture")
```

---

## Custom Physics and Materials

When using custom physics or materials files, the **full parameters are saved** in the metadata:

```python
# Load custom physics
physics = solweig.load_physics("custom_trees.json")

# Calculate with custom physics
results = solweig.calculate(
    surface, weather,
    physics=physics,
    output_dir="output/",
)

# Metadata now includes:
# - physics.full_params: {...complete physics parameters...}
```

This ensures the metadata is **self-contained** - you don't need to keep track of the separate physics file.

---

## API Reference

### `create_run_metadata()`

Create a metadata dictionary for a SOLWEIG run.

**Parameters:**

- `surface`: SurfaceData object
- `location`: Location object
- `weather`: List of Weather objects
- `human`: HumanParams object (or None for defaults)
- `physics`: Physics parameters from load_physics() (or None)
- `materials`: Materials from load_materials() (or None)
- `use_anisotropic_sky`: Whether anisotropic sky model was used
- `conifer`: Whether conifer mode was used
- `output_dir`: Output directory path
- `outputs`: List of output types saved

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
