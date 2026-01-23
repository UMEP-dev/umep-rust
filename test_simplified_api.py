#!/usr/bin/env python3
"""
Quick test of the simplified SOLWEIG API with Athens demo data.
"""

from pathlib import Path

import solweig

# Paths
input_path = Path("demos/data/athens").absolute()
output_path = Path("temp/athens").absolute()

print("="*60)
print("Testing Simplified SOLWEIG API")
print("="*60)

# Step 1: Load surface data
print("\n1. Loading surface data...")
surface, precomputed = solweig.SurfaceData.from_geotiff(
    dsm=str(input_path / "DSM.tif"),
    cdsm=str(output_path / "CDSM.tif"),
    walls_dir=str(output_path / "walls"),
    svf_dir=str(output_path / "svf"),
)
print(f"   ✓ Surface data loaded: {surface.dsm.shape}")
print(f"   ✓ Precomputed data loaded: {precomputed is not None}")
print(f"   ✓ Geotransform stored: {surface._geotransform is not None}")
print(f"   ✓ CRS stored: {surface._crs_wkt is not None}")

# Step 2: Extract location from DSM CRS
print("\n2. Extracting location from DSM...")
location = solweig.Location.from_dsm_crs(
    str(input_path / "DSM.tif"),
    utc_offset=2,  # Athens is UTC+2
)
print(f"   ✓ Location: {location.latitude:.2f}°N, {location.longitude:.2f}°E")

# Step 3: Load weather data (just one day for quick test)
print("\n3. Loading weather data...")
weather_list = solweig.Weather.from_epw(
    str(input_path / "athens_2023.epw"),
    start="2023-07-01",
    end="2023-07-01",  # Just 1 day for quick test
    hours=[12, 13, 14],  # Just 3 hours for quick test
)
print(f"   ✓ Weather data loaded: {len(weather_list)} timesteps")

# Step 4: Calculate with auto-save
print("\n4. Running SOLWEIG calculation with auto-save...")
output_dir = output_path / "test_output"
results = solweig.calculate_timeseries(
    surface=surface,
    location=location,
    weather_series=weather_list,
    precomputed=precomputed,
    use_anisotropic_sky=True,
    compute_utci=True,
    compute_pet=False,
    output_dir=str(output_dir),  # Auto-save results incrementally
    outputs=["tmrt", "utci", "shadow"],
)
print(f"   ✓ Calculation complete: {len(results)} timesteps")

# Step 5: Print results summary
print("\n5. Results Summary:")
for i, (result, weather) in enumerate(zip(results, weather_list)):
    print(f"   Timestep {weather.datetime.strftime('%Y-%m-%d %H:%M')}:")
    print(f"     Mean Tmrt: {result.tmrt.mean():.1f}°C")
    print(f"     Max Tmrt:  {result.tmrt.max():.1f}°C")
    if result.utci is not None:
        print(f"     Mean UTCI: {result.utci.mean():.1f}°C")

print(f"\n6. Results saved to: {output_dir}")

print("\n" + "="*60)
print("SUCCESS: Simplified API test complete!")
print("="*60)
