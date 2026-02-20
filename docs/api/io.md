# I/O Functions

Raster loading, saving, vector rasterisation, and weather file utilities.

All functions are available via `solweig.io` or can be imported directly:

```python
import solweig

data, transform, crs, nodata = solweig.io.load_raster("dsm.tif")
```

---

## load_raster

::: solweig.io.load_raster
    options:
      show_source: false
      heading_level: 3

---

## save_raster

::: solweig.io.save_raster
    options:
      show_source: false
      heading_level: 3

---

## rasterise_gdf

::: solweig.io.rasterise_gdf
    options:
      show_source: false
      heading_level: 3

---

## download_epw

::: solweig.io.download_epw
    options:
      show_source: false
      heading_level: 3

---

## read_epw

::: solweig.io.read_epw
    options:
      show_source: false
      heading_level: 3
