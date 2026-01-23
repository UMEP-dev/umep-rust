# SOLWEIG API Flow - Current vs Proposed

## CURRENT API (Phase 2 - COMPLETE)

### Method 1: Config File API (Legacy - Fully Working)
```python
# Full runner with config files (existing API - retained for backwards compat)
runner = solweig.SolweigRunRust(
    "config.ini",           # Paths + options
    "params.json",          # Physical parameters
    use_tiled_loading=False,
    tile_size=500,
)
runner.run()
# Results written to output_dir specified in config.ini
```

**Status:** ✅ WORKS - Used in production, must keep for backwards compatibility

---

### Method 2: Direct API (Phase 2 - Partially Working)

#### Current Workflow (Before my changes):
```python
import solweig
from solweig import io
import numpy as np

# STEP 1: MANUALLY load rasters using io module
dsm, dsm_transform, dsm_crs, _ = io.load_raster("dsm.tif")
cdsm, _, _, _ = io.load_raster("cdsm.tif")  # Optional
wall_height, _, _, _ = io.load_raster("walls/wall_hts.tif")
wall_aspect, _, _, _ = io.load_raster("walls/wall_aspects.tif")

# STEP 2: Create SurfaceData manually with numpy arrays
surface = solweig.SurfaceData(
    dsm=dsm,
    cdsm=cdsm,
    wall_height=wall_height,
    wall_aspect=wall_aspect,
    pixel_size=1.0,  # Must extract from transform manually
)

# STEP 3: MANUALLY load precomputed data
svf_arrays = solweig.SvfArrays.from_zip("svf/svfs.zip")
shadow_arrays = solweig.ShadowArrays.from_npz("svf/shadowmats.npz")
precomputed = solweig.PrecomputedData(
    svf=svf_arrays,
    shadow_matrices=shadow_arrays,
)

# STEP 4: MANUALLY create Location from DSM CRS
# (User must do pyproj coordinate transform manually)
from pyproj import Transformer
crs = "EPSG:32633"  # Must extract from DSM manually
transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
center_x = dsm_transform[0] + (dsm.shape[1] / 2) * dsm_transform[1]
center_y = dsm_transform[3] + (dsm.shape[0] / 2) * dsm_transform[5]
lon, lat = transformer.transform(center_x, center_y)
location = solweig.Location(latitude=lat, longitude=lon, utc_offset=2)

# STEP 5: Load weather data (THIS WORKS!)
weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2023-07-01",
    end="2023-07-01",
    hours=[12, 13, 14],
)

# STEP 6: Calculate (THIS WORKS!)
results = []
for weather in weather_list:
    result = solweig.calculate(
        surface=surface,
        location=location,
        weather=weather,
        precomputed=precomputed,
        use_anisotropic_sky=True,
        compute_utci=True,
    )
    results.append(result)

# STEP 7: MANUALLY save results
for i, result in enumerate(results):
    timestamp = weather_list[i].dt
    ts_str = timestamp.strftime("%Y%m%d_%H%M")

    # Must manually call io.save_raster for each output
    io.save_raster(
        f"output/tmrt_{ts_str}.tif",
        result.tmrt,
        dsm_transform,  # Must pass through manually
        dsm_crs,        # Must pass through manually
        no_data_val=np.nan,
    )
    if result.utci is not None:
        io.save_raster(
            f"output/utci_{ts_str}.tif",
            result.utci,
            dsm_transform,
            dsm_crs,
            no_data_val=np.nan,
        )
```

**Status:** ✅ WORKS but very verbose - requires ~50 lines for basic workflow

**What's missing:** Convenience methods to reduce boilerplate

---

## PROPOSED API (Phase 3 - What We're Adding)

### Simplified API (User-Friendly)

```python
import solweig

# STEP 1: Load surface data (NEW from_geotiff method)
surface, precomputed = solweig.SurfaceData.from_geotiff(
    dsm="dsm.tif",
    cdsm="cdsm.tif",  # Optional
    walls_dir="walls/",      # Loads wall_hts.tif and wall_aspects.tif
    svf_dir="svf/",          # Loads svfs.zip and shadowmats.npz
)
# Returns: (SurfaceData, PrecomputedData|None)
# - Auto-extracts pixel_size from geotransform
# - Auto-loads walls and SVF if directories provided

# STEP 2: Load weather (ALREADY EXISTS!)
weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2023-07-01",
    end="2023-07-01",
    hours=[12, 13, 14],
)

# STEP 3: Auto-extract location from DSM (NEW)
location = solweig.Location.from_dsm_crs(
    dsm_path="dsm.tif",
    utc_offset=2,  # User must provide timezone
)
# Alternative: Extract from EPW header
location = weather_list[0].location  # If EPW has location metadata

# STEP 4: Calculate (ALREADY EXISTS!)
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    precomputed=precomputed,
    use_anisotropic_sky=True,
    compute_utci=True,
)

# STEP 5: Save results (NEW to_geotiff method - I ADDED THIS)
for result, weather in zip(results, weather_list):
    result.to_geotiff(
        output_dir="output/",
        timestamp=weather.dt,
        outputs=["tmrt", "utci", "shadow"],
        transform=surface._geotransform,  # Stored during from_geotiff
        crs_wkt=surface._crs_wkt,         # Stored during from_geotiff
    )
```

**Status:** 🚧 PARTIALLY IMPLEMENTED
- `Weather.from_epw()` - ✅ EXISTS
- `calculate_timeseries()` - ✅ EXISTS
- `SurfaceData.from_geotiff()` - ⚠️ I JUST ADDED (needs review)
- `SolweigResult.to_geotiff()` - ⚠️ I JUST ADDED (needs review)
- `Location.from_dsm_crs()` - ❌ NOT IMPLEMENTED
- `PrecomputedData.from_directory()` - ❌ NOT IMPLEMENTED (I referenced it but didn't add it)

---

## SUPER SIMPLIFIED API (4 lines - Ultimate Goal)

```python
import solweig

surface, precomputed = solweig.SurfaceData.from_geotiff(
    dsm="dsm.tif",
    cdsm="cdsm.tif",
    walls_dir="walls/",
    svf_dir="svf/",
)

weather_list = solweig.Weather.from_epw(
    "weather.epw",
    start="2023-07-01",
    end="2023-07-01",
)

# OPTION 1: Auto-detect location from DSM CRS
results = solweig.calculate_from_files(  # NEW unified function
    surface=surface,
    weather=weather_list,
    precomputed=precomputed,
    output_dir="output/",
    outputs=["tmrt", "utci"],
)

# OPTION 2: Provide location manually
location = solweig.Location(latitude=37.9, longitude=23.73, utc_offset=2)
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    precomputed=precomputed,
)
for result, weather in zip(results, weather_list):
    result.to_geotiff("output/", timestamp=weather.dt, outputs=["tmrt", "utci"])
```

---

## ISSUES WITH MY RECENT CHANGES

### What I accidentally did:
1. ✅ Added `SolweigResult.to_geotiff()` - THIS IS GOOD (Phase 3 task 3.3)
2. ⚠️ Added `SurfaceData.from_geotiff()` - INCOMPLETE (Phase 3 task 3.1)
   - Returns tuple `(SurfaceData, PrecomputedData|None)`
   - Stores geotransform/CRS internally (NOT in current dataclass)
   - Calls `PrecomputedData.from_directory()` which DOESN'T EXIST

3. ❌ Wrote Quick Start guide showing API that doesn't exist yet

### What needs fixing:
1. Add `PrecomputedData.from_directory()` class method
2. Add fields to SurfaceData to store geotransform and CRS for later export
3. Add `Location.from_dsm_crs()` class method (or auto-detect in calculate)
4. Update Quick Start guide to match actual working API

---

## DECISION POINT

**Option A: Revert my changes, document CURRENT API**
- Remove `SurfaceData.from_geotiff()` and `to_geotiff()`
- Update Quick Start to show verbose 50-line workflow
- Mark Phase 3 as "not started"

**Option B: Finish what I started (implement Phase 3 tasks 3.1 and 3.3)**
- Keep `SurfaceData.from_geotiff()` and `to_geotiff()`
- Add missing `PrecomputedData.from_directory()`
- Add geotransform/CRS storage to SurfaceData
- Update Quick Start to show working simplified API
- Mark Phase 3 tasks 3.1 and 3.3 as complete

**Option C: Partial revert**
- Keep `to_geotiff()` (simpler, less controversial)
- Remove `from_geotiff()` (more complex, needs more design)
- Document current API + to_geotiff convenience method

---

## RECOMMENDATION: Option B (Finish Phase 3.1 and 3.3)

**Rationale:**
- I'm 70% done already
- These are HIGH priority Phase 3 tasks anyway
- Makes API much more user-friendly (50 lines → 15 lines)
- `Weather.from_epw()` already exists, so having `SurfaceData.from_geotiff()` is consistent
- Users want this (evidenced by it being in the plan as HIGH priority)

**Remaining work:**
1. Add `PrecomputedData.from_directory()` (~20 lines)
2. Store geotransform/CRS in SurfaceData (~10 lines)
3. Fix `from_geotiff()` to use the new `from_directory()` (~5 lines)
4. Add tests for the new methods (~50 lines)
5. Verify Athens demo works end-to-end

**Estimated effort:** 30-60 minutes

---

## USAGE COMPARISON

### Current API (50 lines):
```python
# Load rasters manually
dsm, transform, crs, _ = io.load_raster("dsm.tif")
cdsm, _, _, _ = io.load_raster("cdsm.tif")
walls_h, _, _, _ = io.load_raster("walls/wall_hts.tif")
walls_a, _, _, _ = io.load_raster("walls/wall_aspects.tif")

# Create surface data
surface = SurfaceData(dsm=dsm, cdsm=cdsm, wall_height=walls_h, wall_aspect=walls_a, pixel_size=1.0)

# Load precomputed manually
svf = SvfArrays.from_zip("svf/svfs.zip")
shadows = ShadowArrays.from_npz("svf/shadowmats.npz")
precomputed = PrecomputedData(svf=svf, shadow_matrices=shadows)

# Extract location manually
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
center_x = transform[0] + (dsm.shape[1]/2) * transform[1]
center_y = transform[3] + (dsm.shape[0]/2) * transform[5]
lon, lat = transformer.transform(center_x, center_y)
location = Location(latitude=lat, longitude=lon, utc_offset=2)

# Load weather
weather_list = Weather.from_epw("weather.epw", start="2023-07-01", end="2023-07-01")

# Calculate
results = calculate_timeseries(surface, location, weather_list, precomputed=precomputed)

# Save manually
for result, weather in zip(results, weather_list):
    ts = weather.dt.strftime("%Y%m%d_%H%M")
    io.save_raster(f"output/tmrt_{ts}.tif", result.tmrt, transform, crs)
    io.save_raster(f"output/utci_{ts}.tif", result.utci, transform, crs)
```

### Proposed API (15 lines):
```python
# Load everything with one call
surface, precomputed = SurfaceData.from_geotiff(
    dsm="dsm.tif", cdsm="cdsm.tif", walls_dir="walls/", svf_dir="svf/"
)

# Auto-extract location
location = Location.from_dsm("dsm.tif", utc_offset=2)

# Load weather
weather_list = Weather.from_epw("weather.epw", start="2023-07-01", end="2023-07-01")

# Calculate
results = calculate_timeseries(surface, location, weather_list, precomputed=precomputed)

# Save with one call per timestep
for result, weather in zip(results, weather_list):
    result.to_geotiff("output/", timestamp=weather.dt, outputs=["tmrt", "utci"])
```

**Reduction:** 50 lines → 15 lines (70% reduction in boilerplate!)
