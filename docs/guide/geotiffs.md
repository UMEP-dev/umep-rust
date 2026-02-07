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
