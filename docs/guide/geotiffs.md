# Working with GeoTIFFs

Most real-world SOLWEIG projects start from GeoTIFF raster files. This guide covers loading, caching, and saving.

## Loading surface data

`SurfaceData.prepare()` is the recommended way to load GeoTIFFs. It handles everything: loading, NaN filling, wall computation, SVF computation, and caching.

```python
import solweig

surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",           # Required: building/terrain heights
    working_dir="cache/",          # Required: where to cache preprocessing
    cdsm="data/trees.tif",        # Optional: vegetation heights
    dem="data/dem.tif",            # Optional: bare ground elevation
    land_cover="data/lc.tif",     # Optional: surface type classification
)
```

### Cropping to a bounding box

Process only part of a large raster:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    bbox=[476800, 4205850, 477200, 4206250],  # [minx, miny, maxx, maxy]
    pixel_size=1.0,  # Resample to 1 m resolution
)
```

Coordinates are in the DSM's native CRS (e.g. UTM metres).

## What gets cached

The `working_dir` stores expensive preprocessing so subsequent runs are instant:

```text
cache/
├── walls/
│   ├── wall_hts.tif        # Wall heights derived from DSM
│   └── wall_aspects.tif    # Wall compass directions
└── svf/
    └── memmap/
        ├── svf.npy          # Total Sky View Factor
        ├── svf_north.npy    # Directional SVF (4 cardinal directions)
        └── ...              # 15 SVF grids total
```

### Force recomputation

If you change the DSM or want to regenerate everything:

```python
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",
    force_recompute=True,
)
```

SOLWEIG also validates cached data against the current DSM — if the dimensions or extent change, the cache is automatically invalidated.

## Extracting location from CRS

When the DSM has a projected CRS (e.g. UTM), you can extract lat/lon automatically:

```python
location = solweig.Location.from_surface(surface, utc_offset=2)
```

Or from the EPW file, which also includes UTC offset:

```python
location = solweig.Location.from_epw("data/weather.epw")
```

## Saving results as GeoTIFFs

### During timeseries

The simplest approach — results are saved as they're computed:

```python
results = solweig.calculate(
    surface=surface,
    weather=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)
```

Creates files like:

```text
output/
├── tmrt/
│   ├── tmrt_20250701_0000.tif
│   ├── tmrt_20250701_0100.tif
│   └── ...
└── shadow/
    ├── shadow_20250701_0000.tif
    └── ...
```

### Loading saved results

```python
# Load a single output raster
arr, transform, crs, nodata = solweig.io.load_raster("output/tmrt/tmrt_20250701_1200.tif")
```

## NaN and nodata handling

SOLWEIG handles missing data automatically at every stage:

**At load time:** Negative nodata sentinel values (e.g. -9999) are converted to NaN. Zero-valued pixels are preserved as valid data.

**Before calculation:** NaN pixels in DSM, CDSM, and TDSM are filled with the ground reference (DEM if provided, otherwise the DSM itself). Pixels within 0.1 m of the ground reference are clamped to exactly the ground value to prevent shadow artefacts from resampling noise.

DEM NaN pixels are never filled — they represent truly missing ground data.

When using `SurfaceData.prepare()`, this is all handled automatically. When constructing from arrays, `fill_nan()` runs inside `calculate()`.

## Rasterising vector data

Convert tree polygons (GeoDataFrame) to a raster grid:

```python
import geopandas as gpd

trees = gpd.read_file("trees.gpkg")
trees = trees.to_crs(2154)  # Match DSM CRS

cdsm, transform = solweig.io.rasterise_gdf(
    trees,
    geom_col="geometry",
    ht_col="height",
    bbox=[476800, 4205850, 477200, 4206250],
    pixel_size=1.0,
)

# Optionally save to disk
from pyproj import CRS
solweig.io.save_raster(
    "data/cdsm.tif", cdsm, transform.to_gdal(), CRS.from_epsg(2154).to_wkt()
)
```

## Large rasters

For rasters too large to fit in memory, SOLWEIG automatically tiles the computation:

```python
results = solweig.calculate(
    surface=surface,
    location=location,
    weather=weather,
)
```

Tiling is handled internally — no explicit tiling call is needed.
