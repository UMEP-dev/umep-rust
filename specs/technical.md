# Technical Implementation

Implementation details, performance considerations, and computational requirements.

Normative runtime/API behavior is specified in
[runtime-contract.md](runtime-contract.md). This file focuses on implementation
details rather than public precondition semantics.

## Data Types

### Float32 Precision

All raster calculations use **float32** (single precision) rather than float64:

- **Reason**: GPU compatibility, memory efficiency, sufficient precision
- **Precision**: ~7 significant digits
- **Range**: ±3.4 × 10³⁸

**Properties:**

1. All DSM/CDSM values stored as float32
2. All output rasters are float32
3. Intermediate calculations may use higher precision internally
4. Results should match float64 within 0.1% for typical urban values

### Integer Types

- Shadow masks: uint8 or bool (0/1 values)
- Indices and counts: int32 or int64

## Tiling

Large rasters are processed in tiles to manage memory:

### Tile Properties

1. **Tile size**: Configurable, typically 256×256 to 1024×1024 pixels
2. **Overlap**: Tiles overlap by `max_shadow_reach` to avoid edge artifacts
3. **Seamless output**: Stitched results should be identical to full-raster processing

### Shadow Reach Calculation

Overlap must accommodate the longest possible shadow:

```text
max_shadow_reach = max_building_height / tan(min_sun_altitude)
```

At min_sun_altitude = 5°:
- 50m building → ~572m shadow → 572 pixels at 1m resolution

### Tile Processing Order

1. Tiles can be processed in parallel (independent)
2. Edge pixels use overlap region for context
3. Only interior pixels written to output

## GPU Acceleration

Optional GPU support for shadow and SVF calculations:

### GPU Properties

1. **Automatic fallback**: If GPU unavailable, uses CPU
2. **Equivalent results**: GPU and CPU produce identical outputs (within float32 precision)
3. **Memory management**: Large rasters automatically tiled for GPU memory limits

### GPU-Accelerated Operations

- Shadow casting (ray marching)
- SVF patch visibility checks
- Parallel pixel operations

### CPU-Only Operations

- File I/O
- Coordinate transformations
- Final result assembly

## Coordinate Systems

### Raster Coordinates

- Row 0 = North edge of raster
- Column 0 = West edge of raster
- Increasing row index = moving South
- Increasing column index = moving East

### Sun Position

- Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
- Altitude: 0° = horizon, 90° = zenith

### Geographic Coordinates

- Input rasters should have valid CRS (coordinate reference system)
- WGS84 (EPSG:4326) used for sun position calculations
- Local projected CRS used for distance calculations

## Memory Management

### Typical Memory Usage

| Operation | Memory per megapixel |
| --------- | -------------------- |
| Single raster (float32) | ~4 MB |
| SVF calculation | ~50 MB (multiple arrays) |
| Full SOLWEIG run | ~200 MB |

### Memory Properties

1. Peak memory scales with tile size, not total raster size
2. Intermediate arrays released after use
3. Output written incrementally for large rasters

## Numerical Stability

### Edge Cases

1. **Sun at horizon (altitude ≈ 0°)**: Shadow length approaches infinity
   - Handled by clamping to max_shadow_reach
   - No shadows computed when altitude ≤ 0°

2. **Very tall buildings**: May exceed shadow reach
   - Warning if buildings exceed reasonable height

3. **Flat terrain**: Division by zero avoided
   - max_height = 0 handled gracefully

### NaN Handling

1. Input NaN values indicate no-data (outside study area)
2. NaN propagates through calculations
3. Output NaN indicates invalid/missing result

## Performance Targets

| Operation | Target | Notes |
| --------- | ------ | ----- |
| Shadow (1 timestep) | <1s per megapixel | GPU |
| SVF | <30s per megapixel | GPU |
| Full day (48 timesteps) | <5 min per megapixel | GPU |

## Reproducibility

### Deterministic Results

1. Same inputs → same outputs (bitwise identical)
2. No random number generation in core algorithms
3. Parallel processing order does not affect results

### Version Compatibility

1. Output format stable across minor versions
2. Algorithm changes documented in changelog
3. Regression tests verify consistency
