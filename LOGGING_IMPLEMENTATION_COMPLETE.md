# QGIS-Compatible Logging System - COMPLETE ✅

**Date:** January 22, 2026  
**Feature:** Unified logging infrastructure with automatic informative messages

---

## What Was Implemented

Created a QGIS-compatible logging system that automatically logs useful information during key operations, with environment auto-detection for both QGIS and Python environments.

### New Module: `pysrc/solweig/logging.py`

**Features:**
- **Environment auto-detection**: Uses QgsProcessingFeedback in QGIS, Python logging in CLI, fallback to stdout
- **QGIS compatibility**: Integrates with QGIS processing algorithms seamlessly
- **Standard logging levels**: DEBUG, INFO, WARNING, ERROR
- **Global configuration**: Set logging level for all loggers at once

**Usage:**
```python
from solweig.logging import get_logger

logger = get_logger(__name__)
logger.info("Surface data loaded: 400×400 pixels")
logger.debug(f"Using {len(weather_list)} timesteps")
logger.warning("SVF not provided, will compute on-the-fly (slow)")
```

---

## Automatic Logging Added

### 1. `SurfaceData.from_geotiff()`

Logs during data loading:
```
Loading surface data from GeoTIFF files...
  DSM: 400×400 pixels
  Pixel size: 1.00 m
  Layers loaded: DSM, CDSM, walls
  Preprocessing CDSM/TDSM (relative → absolute heights)
  Loaded SVF data: (400, 400)
  Loaded shadow matrices for anisotropic sky
✓ Surface data loaded successfully
```

### 2. `Location.from_dsm_crs()`

Logs extracted coordinates:
```
Extracted location from DSM CRS: 38.0044°N, 23.7397°E (UTC+2)
```

### 3. `Weather.from_epw()`

Logs loaded timesteps:
```
Loaded 3 timesteps from EPW: 2023-07-01 12:00 → 2023-07-01 14:00
```

### 4. `calculate_timeseries()`

**Detailed configuration summary:**
```
============================================================
Starting SOLWEIG timeseries calculation
  Grid size: 400×400 pixels
  Timesteps: 3
  Period: 2023-07-01 12:00 → 2023-07-01 14:00
  Location: 38.00°N, 23.74°E
  Options: anisotropic sky, UTCI, precomputed SVF
  Auto-save: /path/to/output (tmrt, utci)
============================================================
```

**Progress updates:**
```
  Processing timestep 1/3: 2023-07-01 12:00
  Processing timestep 2/3: 2023-07-01 13:00
  Processing timestep 3/3: 2023-07-01 14:00
```

**Summary statistics:**
```
============================================================
✓ Calculation complete: 3 timesteps processed
  Tmrt range: 27.8°C - 65.1°C (mean: 55.8°C)
  UTCI range: 27.7°C - 39.0°C (mean: 35.7°C)
  Files saved: 6 GeoTIFFs in /path/to/output
============================================================
```

---

## Environment Detection

### Python CLI
Uses standard Python `logging` module:
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import solweig
# Logging output goes to stdout with standard formatting
```

### QGIS Processing Algorithm
```python
from qgis.core import QgsProcessingAlgorithm, QgsProcessingFeedback
import solweig.logging as slog

class SolweigAlgorithm(QgsProcessingAlgorithm):
    def processAlgorithm(self, parameters, context, feedback):
        # Set QGIS feedback for all loggers
        slog.set_global_feedback(feedback)
        
        # Now all solweig logging goes to QGIS progress dialog
        surface, precomputed = solweig.SurfaceData.from_geotiff(...)
        # User sees: "Loading surface data from GeoTIFF files..." in QGIS
```

### Fallback Mode
If neither QGIS nor logging is configured:
```python
import solweig
# Logging output goes directly to stdout with simple format
# [INFO] solweig.api: Surface data loaded: 400×400 pixels
```

---

## API Changes

### Added to `pysrc/solweig/api.py`

**Import:**
```python
from .logging import get_logger
logger = get_logger(__name__)
```

**Logging added to:**
1. `SurfaceData.from_geotiff()` - Data loading summary
2. `Location.from_dsm_crs()` - Extracted coordinates
3. `Weather.from_epw()` - Loaded timesteps  
4. `PrecomputedData.from_directory()` - SVF/shadow data
5. `calculate_timeseries()` - Full calculation workflow

---

## Configuration

### Set Global Log Level

```python
import solweig.logging as slog

# Show debug messages
slog.set_global_level(slog.LogLevel.DEBUG)

# Show only warnings and errors
slog.set_global_level(slog.LogLevel.WARNING)
```

### Per-Logger Configuration

```python
from solweig.logging import get_logger

logger = get_logger('my_module', level=logging.DEBUG)
logger.debug("Detailed debugging info")
```

---

## Benefits

1. **User visibility**: Users see what's happening without manual print statements
2. **QGIS integration**: Logging appears in QGIS progress dialogs automatically
3. **Performance monitoring**: See which operations take time
4. **Debugging**: Can enable DEBUG level for troubleshooting
5. **No code changes**: Logging is automatic - users don't need to add print statements
6. **Reduced demo boilerplate**: Athens demo is simpler without manual status prints

---

## Example: Full Workflow Output

```
Loading surface data from GeoTIFF files...
  DSM: 400×400 pixels
  Pixel size: 1.00 m
  Layers loaded: DSM, CDSM, walls
  Loaded SVF data: (400, 400)
  Loaded shadow matrices for anisotropic sky
✓ Surface data loaded successfully

Extracted location from DSM CRS: 38.0044°N, 23.7397°E (UTC+2)

Loaded 3 timesteps from EPW: 2023-07-01 12:00 → 2023-07-01 14:00

============================================================
Starting SOLWEIG timeseries calculation
  Grid size: 400×400 pixels
  Timesteps: 3
  Period: 2023-07-01 12:00 → 2023-07-01 14:00
  Location: 38.00°N, 23.74°E
  Options: anisotropic sky, UTCI, precomputed SVF
  Auto-save: output/ (tmrt, utci)
============================================================
  Processing timestep 1/3: 2023-07-01 12:00
  Processing timestep 2/3: 2023-07-01 13:00  
  Processing timestep 3/3: 2023-07-01 14:00
============================================================
✓ Calculation complete: 3 timesteps processed
  Tmrt range: 27.8°C - 65.1°C (mean: 55.8°C)
  UTCI range: 27.7°C - 39.0°C (mean: 35.7°C)
  Files saved: 6 GeoTIFFs in output/
============================================================
```

---

## Files Modified

1. **[pysrc/solweig/logging.py](pysrc/solweig/logging.py)** - New logging infrastructure (177 lines)
2. **[pysrc/solweig/api.py](pysrc/solweig/api.py)** - Integrated logging in 5 key methods
3. **[demos/athens-demo.py](demos/athens-demo.py)** - Removed manual print statements (now automatic)
4. **[MODERNIZATION_PLAN.md](MODERNIZATION_PLAN.md)** - Marked tasks 3.15 and 3.16 as complete

---

## Design Decisions

1. **Auto-detection over configuration**: Automatically detect environment rather than requiring explicit setup
2. **Informative defaults**: Log useful information at INFO level without being verbose
3. **Structured output**: Use separators and formatting for readability
4. **Summary statistics**: Automatically compute and log ranges/means for results
5. **Progress reporting**: Log every N timesteps for long runs (smart interval calculation)
6. **QGIS compatibility**: Seamless integration with QGIS processing feedback

---

## Next Steps

- ✅ Logging infrastructure complete
- ✅ Integrated into key API methods
- ⏳ Consider adding to preprocessing modules (walls, svf)
- ⏳ Add logging to runner.py for legacy API
- ⏳ Document logging in Quick Start guide

---

## Status

✅ **QGIS-compatible logging system COMPLETE**

Tasks completed:
- Task 3.15: QGIS-compatible logging infrastructure ✅
- Task 3.16: Integrate automatic logging in key operations ✅

This enhances user experience by providing automatic visibility into what SOLWEIG is doing, especially important for:
- Long-running calculations (full year simulations)
- QGIS plugin users (progress in QGIS dialog)
- Debugging and troubleshooting
- Understanding model configuration and results
