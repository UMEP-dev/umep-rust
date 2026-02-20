# Parameter Handling - Simple Summary

**TL;DR:** Old API had 58 config options. New API has 3 parameter types, all with defaults. 99% of users never touch them.

---

## The Three Parameter Types

### 1. Human Parameters (Person-Specific)
Who is experiencing the thermal environment?

```python
human = HumanParams(
    abs_k=0.7,        # Shortwave absorption (0-1)
    abs_l=0.97,       # Longwave absorption (0-1)
    posture="standing",  # "standing" or "sitting"
    weight=75,        # kg (for PET post-processing)
    height=1.75,      # m (for PET post-processing)
    age=35,           # years (for PET post-processing)
    activity=80,      # W (for PET post-processing)
    clothing=0.9,     # clo (for PET post-processing)
)
```

**Defaults:** abs_k=0.7, abs_l=0.97, standing, 75 kg, 1.75 m, 35 yo, 80 W, 0.9 clo

**When to customize:** Different body characteristics, sitting posture

---

### 2. Physics Parameters (Site-Independent)
How do vegetation and posture work? (Universal scientific constants)

```python
physics = load_physics("custom_trees.json")  # Optional
```

Contains:
- `Tree_settings`: Transmissivity (0.03), seasonal dates (day 97-300), trunk ratio (0.25)
- `Posture`: Geometry for standing/sitting (Fside, Fup, Fcyl, height)

**Defaults:** Bundled in package (`physics_defaults.json`)

**When to customize:** Different tree species, different seasonal periods

---

### 3. Material Library (Site-Specific)
What is the ground/buildings made of?

```python
materials = load_materials("site_materials.json")  # Required if landcover grid
```

Contains per-landcover-class values:
- Albedo, Emissivity
- Ground temperature model parameters (TmaxLST, Ts_deg, Tstart)
- Wall thermal properties (specific heat, conductivity, density, thickness)

**Defaults:** None (only needed if you have landcover grid)

**When to customize:** You have a landcover classification grid with different surface types

---

## Model Behavior Flags

Two direct parameters control major model behavior:

### `use_anisotropic_sky` (default: follows `ModelConfig`, currently `True`)
- `False` = Simpler isotropic sky model
- `True` = Perez anisotropic sky model
- **When to change:** Research papers, high-accuracy work
If explicitly set to `True`, shadow matrices must already be prepared.

### `conifer` (default: `False`)
- `False` = Deciduous trees (seasonal leaf on/off)
- `True` = Evergreen trees (always have leaves)
- **When to change:** Your site has pine/spruce/fir trees

---

## Usage Patterns

### 99% of users (all defaults)
```python
import solweig

surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="cache/")
weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01")
results = solweig.calculate_timeseries(surface, weather, output_dir="output/")

# All parameters use bundled defaults - nothing to configure!
```

### Custom human parameters (common)
```python
results = solweig.calculate_timeseries(
    surface, weather,
    human=solweig.HumanParams(weight=70, height=1.65, posture="sitting"),
    output_dir="output/",
)
```

### Better accuracy (anisotropic sky)
```python
results = solweig.calculate_timeseries(
    surface, weather,
    use_anisotropic_sky=True,  # <-- Slower, more accurate
    output_dir="output/",
)
```

### Evergreen trees
```python
results = solweig.calculate_timeseries(
    surface, weather,
    conifer=True,  # <-- Always leaf-on
    output_dir="output/",
)
```

### Custom physics (rare)
```python
# Create custom_trees.json with different transmissivity:
# {
#   "Tree_settings": {"Value": {"Transmissivity": 0.05, ...}},
#   "Posture": {"Standing": {...}, "Sitting": {...}}
# }

physics = solweig.load_physics("custom_trees.json")
results = solweig.calculate_timeseries(surface, weather, physics=physics, output_dir="output/")
```

### Landcover material variation (advanced)
```python
# Requires: landcover grid (classification raster with class IDs)
# Requires: materials file with properties per class

materials = solweig.load_materials("site_materials.json")
surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    land_cover="landcover.tif",  # Grid with surface type IDs (0-7, 99-102)
    working_dir="cache/",
)
results = solweig.calculate_timeseries(surface, weather, materials=materials, output_dir="output/")
```

---

## Decision Tree

**Do you need to customize human characteristics?**
- Yes → `human=HumanParams(weight=..., height=..., posture=...)`
- No → Use defaults

**Do you have evergreen trees?**
- Yes → `conifer=True`
- No → Use defaults

**Do you need research-grade accuracy?**
- Yes → `use_anisotropic_sky=True` (slower, more accurate)
- No → Use defaults

**Do you have different tree species or seasonal periods?**
- Yes → Create custom physics file, `physics=load_physics("custom.json")`
- No → Use bundled defaults

**Do you have a landcover grid with different surface materials?**
- Yes → Create materials file, `materials=load_materials("site_materials.json")`
- No → Use uniform defaults

**Everything else?**
- Use defaults!

---

## Conceptual Separation

The three parameter types are **conceptually distinct**:

| Type | What | Example | When Needed |
|------|------|---------|-------------|
| **human** | Person characteristics | Weight, height, absorption | Custom body properties |
| **physics** | Universal constants | Tree transmissivity, posture geometry | Different tree species |
| **materials** | Landcover properties | Albedo per surface type | Spatial material variation |

This separation makes it clear:
- `human` = **WHO** is experiencing the thermal environment
- `physics` = **HOW** vegetation and posture work (universal science)
- `materials` = **WHAT** the ground/buildings are made of (site-specific)

---

## What Happened to Everything Else?

### Now Automatic (28 things)
- Sun position → Computed from datetime + location
- Location → Extracted from DSM file metadata
- Walls → Generated and cached automatically
- SVF → Generated and cached automatically
- Direct/diffuse radiation split → Computed
- Max building height → Computed from DSM
- Many more...

### Now Bundled Defaults (Physics)
Site-independent constants in `physics_defaults.json`:
- Tree transmissivity: 0.03
- Seasonal dates: Day 97-300 (~April-October)
- Trunk ratio: 0.25
- Posture geometry: Standing/sitting projected areas

### Now Bundled Defaults (Human)
Person characteristics:
- Shortwave absorption: 0.7
- Longwave absorption: 0.95
- Posture: Standing
- Weight: 75 kg, Height: 180 cm
- Age: 35, Activity: 80 W
- Clothing: 0.9 clo

**You don't need to think about these unless you want custom values.**

### Advanced: Landcover-Specific (Materials)
Material properties per surface type (asphalt, grass, concrete, etc.):
- Albedo, emissivity, thermal properties
- **Only needed if you have a landcover grid**
- Requires custom `materials.json` file

---

## What If I Need Fine Control?

Three levels of control:

### Level 1: Direct parameters (most users)
```python
calculate_timeseries(
    ...,
    use_anisotropic_sky=True,
    conifer=True,
    human=HumanParams(weight=70),
)
```

### Level 2: Custom physics or materials (advanced)
```python
physics = solweig.load_physics("my_physics.json")
materials = solweig.load_materials("my_materials.json")
calculate_timeseries(..., physics=physics, materials=materials)
```

### Level 3: Manual preprocessing (experts)
```python
solweig.walls.generate_wall_hts(dsm_path="dsm.tif", out_dir="walls/")
solweig.svf.generate_svf(dsm_path="dsm.tif", out_dir="svf/")
surface = solweig.SurfaceData.prepare(dsm="dsm.tif", working_dir="manual/")
```

---

## Backwards Compatibility

The old unified `params.json` file (220 lines with human + physics + materials) is still supported:

```python
# Legacy unified params (still works for backwards compatibility)
params = solweig.load_params("parametersforsolweig.json")
results = solweig.calculate_timeseries(surface, weather, params=params, output_dir="output/")
```

But the new three-parameter model is clearer and more flexible.

---

## Summary

**Before:** 58 configuration options, 2 config files, manual preprocessing

**After:** 3 parameter types (all with defaults), everything else automatic

| Parameter | Purpose | Default | Customization |
|-----------|---------|---------|---------------|
| `human` | Person characteristics | Standing, 75kg, 180cm | `HumanParams(...)` object |
| `physics` | Universal constants | Bundled in package | `load_physics("custom.json")` |
| `materials` | Landcover properties | Not needed if no LC grid | `load_materials("site.json")` |
| `use_anisotropic_sky` | Sky model accuracy | False (faster) | Set to True |
| `conifer` | Tree type | False (deciduous) | Set to True |

**The point:** Start simple. Add complexity only if you need it.
