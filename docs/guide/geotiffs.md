# Working with GeoTIFFs

SOLWEIG can load surface data directly from GeoTIFF files using the `SurfaceData.prepare()` factory method.

## Basic Loading

```python
import solweig

surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
)
```

This automatically:

1. Loads the DSM raster
2. Extracts coordinate reference system (CRS)
3. Computes wall heights and aspects
4. Computes Sky View Factor (SVF)
5. Caches results for future use

## Adding Optional Inputs

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    cdsm="data/cdsm.tif",       # Vegetation heights
    dem="data/dem.tif",         # Ground elevation
    land_cover="data/lc.tif",   # Land cover classification
)
```

## Explicit Extent

Crop to a specific bounding box:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    bbox=[476800, 4205850, 477200, 4206250],  # [minx, miny, maxx, maxy]
    pixel_size=1.0,  # Resample to 1m resolution
)
```

## Location from CRS

Extract geographic coordinates from the raster CRS:

```python
surface = solweig.SurfaceData.prepare(dsm="data/dsm.tif", working_dir="cache/")

# Extract centroid location (always specify UTC offset!)
location = solweig.Location.from_surface(surface, utc_offset=2)
```

## Cache Management

The `working_dir` caches expensive computations:

```
cache/
├── walls/
│   ├── wall_hts.tif
│   └── wall_aspects.tif
└── svf/
    └── memmap/
        ├── svf.npy
        ├── svf_north.npy
        └── ...
```

### Force Recomputation

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    force_recompute=True,  # Ignore cache, recompute everything
)
```

### Cache Validation

SOLWEIG automatically validates cached data against current inputs. If the DSM changes, the cache is invalidated and recomputed.

## Saving Results

Save calculation outputs as GeoTIFFs:

```python
result = solweig.calculate(surface, location, weather)

result.save_outputs(
    output_dir="output/",
    transform=surface.transform,
    crs=surface.crs,
)
```

This creates:
- `output/tmrt.tif`
- `output/shadow.tif`
- etc.

## NaN and NoData Handling

SOLWEIG automatically handles NaN (missing) values in surface layers:

- **At load time:** Only negative nodata sentinel values (e.g. -9999) are
  converted to NaN. Zero-valued pixels are preserved as valid data.
- **Before calculation:** `fill_nan()` is called automatically by both
  `preprocess()` and `calculate()`. NaN pixels in DSM/CDSM/TDSM are filled
  with the ground reference (DEM, or DSM if no DEM is provided).
- **Noise clamping:** After filling, surface pixels within 0.1 m of the ground
  reference are collapsed to exactly the ground value, preventing shadow and
  SVF artefacts from resampling jitter.

DEM NaN pixels are never filled — they represent truly missing ground data and
are masked as invalid.

When loading via `SurfaceData.prepare()`, this is all handled automatically.
When constructing `SurfaceData` manually from arrays, `fill_nan()` runs inside
`calculate()` so no extra steps are needed.

## Large Rasters

For rasters larger than available memory, use tiled processing:

```python
result = solweig.calculate_tiled(
    surface=surface,
    location=location,
    weather=weather,
    tile_size=500,  # Process in 500×500 tiles
)
```

See [Timeseries Calculations](timeseries.md) for batch processing workflows.
