#!/usr/bin/env python3
"""Memory profiling script for SOLWEIG.

Measures memory usage at various raster sizes to identify bottlenecks
and verify float32 optimization effectiveness.

Usage:
    python scripts/profile_memory.py [--size 1000]
"""

import argparse
import tracemalloc
from datetime import datetime

import numpy as np

# Start tracing before imports to capture module-level allocations
tracemalloc.start()


def format_size(size_bytes: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def create_synthetic_surface(size: int):
    """Create synthetic urban surface for testing."""
    from solweig import Location, SurfaceData

    # Create DSM with some buildings
    np.random.seed(42)
    dsm = np.ones((size, size), dtype=np.float32) * 10.0  # Ground at 10m

    # Add random buildings
    n_buildings = size // 20
    for _ in range(n_buildings):
        x, y = np.random.randint(10, size - 10, 2)
        w, h = np.random.randint(5, 15, 2)
        height = np.random.uniform(15, 40)
        dsm[y : y + h, x : x + w] = height

    # Create vegetation DSM (relative heights)
    cdsm = np.zeros((size, size), dtype=np.float32)
    n_trees = size // 10
    for _ in range(n_trees):
        x, y = np.random.randint(5, size - 5, 2)
        if dsm[y, x] < 12:  # Only place trees on ground
            r = np.random.randint(2, 5)
            h = np.random.uniform(3, 8)
            y1, y2 = max(0, y - r), min(size, y + r)
            x1, x2 = max(0, x - r), min(size, x + r)
            cdsm[y1:y2, x1:x2] = np.maximum(cdsm[y1:y2, x1:x2], h)

    # Create land cover (integer array)
    land_cover = np.ones((size, size), dtype=np.int32) * 5  # Default grass
    land_cover[dsm > 12] = 2  # Buildings (ID 2 in UMEP standard)

    surface = SurfaceData(dsm=dsm, cdsm=cdsm, land_cover=land_cover, pixel_size=1.0)

    location = Location(latitude=37.98, longitude=23.73)  # Athens

    return surface, location


def profile_calculation(size: int) -> dict:
    """Run a single timestep and measure memory usage."""
    from solweig import Weather, calculate

    # Reset tracing
    tracemalloc.reset_peak()

    # Create surface
    surface, location = create_synthetic_surface(size)
    surface.preprocess()

    after_surface = tracemalloc.get_traced_memory()

    # Create weather for noon
    weather = Weather(datetime=datetime(2024, 7, 21, 12, 0), ta=30.0, rh=50.0, global_rad=800.0, ws=2.0)

    # Run calculation
    result = calculate(surface, location, weather)

    after_calc = tracemalloc.get_traced_memory()

    return {
        "size": size,
        "pixels": size * size,
        "surface_current": after_surface[0],
        "surface_peak": after_surface[1],
        "calc_current": after_calc[0],
        "calc_peak": after_calc[1],
        "tmrt_mean": float(np.nanmean(result.tmrt)),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile SOLWEIG memory usage")
    parser.add_argument("--size", type=int, default=500, help="Grid size (default: 500)")
    parser.add_argument("--scale", action="store_true", help="Test multiple sizes")
    args = parser.parse_args()

    print("=" * 60)
    print("SOLWEIG Memory Profiler")
    print("=" * 60)

    if args.scale:
        sizes = [100, 200, 400, 800]
        if args.size > 800:
            sizes.append(args.size)
    else:
        sizes = [args.size]

    results = []
    for size in sizes:
        print(f"\nTesting {size}x{size} grid ({size * size:,} pixels)...")
        try:
            result = profile_calculation(size)
            results.append(result)

            print("  Surface creation:")
            print(f"    Current: {format_size(result['surface_current'])}")
            print(f"    Peak:    {format_size(result['surface_peak'])}")
            print("  After calculation:")
            print(f"    Current: {format_size(result['calc_current'])}")
            print(f"    Peak:    {format_size(result['calc_peak'])}")
            print(f"  Tmrt mean: {result['tmrt_mean']:.1f}°C")

            # Estimate bytes per pixel
            bytes_per_pixel = result["calc_peak"] / (size * size)
            print(f"  Peak memory per pixel: {bytes_per_pixel:.1f} bytes")

        except MemoryError:
            print(f"  MemoryError at size {size}")
            break
        except Exception as e:
            print(f"  Error: {e}")
            break

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Summary (Peak Memory)")
        print("=" * 60)
        print(f"{'Size':>8} {'Pixels':>12} {'Peak Mem':>12} {'Per Pixel':>12}")
        print("-" * 46)
        for r in results:
            bpp = r["calc_peak"] / r["pixels"]
            print(f"{r['size']:>8} {r['pixels']:>12,} {format_size(r['calc_peak']):>12} {bpp:>10.1f} B")

        # Extrapolate to 10k x 10k
        if len(results) >= 2:
            last = results[-1]
            bpp = last["calc_peak"] / last["pixels"]
            est_10k = bpp * 10000 * 10000
            print(f"\nEstimated memory for 10k×10k: {format_size(est_10k)}")


if __name__ == "__main__":
    main()
