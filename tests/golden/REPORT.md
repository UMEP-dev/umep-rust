# UMEP Python vs SOLWEIG Rust — Golden Test Comparison Report

**Generated:** 2026-02-24
**Comparison:** UMEP Python (reference) vs SOLWEIG Rust implementation
**Test data:** Gothenburg demo site (234 x 223 pixels, 1 m resolution)
**Golden tests:** 130/130 pass
**Spatial tests:** 35/35 pass
**Formula agreement:** 12/12 scenarios pass

## Notes

All residuals are within f32 floating-point precision. The largest difference
across all 35 spatial comparisons is 2.20e-02 (PET grid), well within the
0.2 C tolerance for the iterative PET solver. Most components agree to better
than 1e-04.

These tests verify **parity with UMEP Python**, not physical accuracy. Each
test runs a single timestep in isolation, so stateful behaviour across a
multi-hour timeseries (e.g. clearness index carry-forward at night) is not
covered here. Comparison against field observations is in the
[Kronenhuset validation report](../validation/kronenhuset/REPORT.md).

## Spatial Comparison Summary

Each row compares a full 2D output grid between UMEP Python and the Rust
implementation. Max |Diff| is the worst-case pixel-level absolute difference;
Mean Diff is the signed average over all pixels.

| Component                 | Max Diff | Threshold           | Mean Diff | Status |
| ------------------------- | -------: | ------------------- | --------: | ------ |
| shadow_morning_bldg       | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_morning_veg        | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_morning_wall_sh    | 8.99e-06 | 1e-4                |  4.17e-09 | PASS   |
| shadow_morning_wall_sun   | 8.99e-06 | 1e-4                | -4.17e-09 | PASS   |
| shadow_noon_bldg          | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_noon_veg           | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_noon_wall_sh       | 8.33e-06 | 1e-4                | -5.60e-09 | PASS   |
| shadow_noon_wall_sun      | 8.33e-06 | 1e-4                |  5.71e-09 | PASS   |
| shadow_afternoon_bldg     | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_afternoon_veg      | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_afternoon_wall_sh  | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| shadow_afternoon_wall_sun | 0.00e+00 | 1e-4                |  0.00e+00 | PASS   |
| svf_total                 | 2.09e-06 | 1e-4                | -7.97e-07 | PASS   |
| svf_north                 | 1.25e-06 | 1e-4                | -5.83e-07 | PASS   |
| svf_east                  | 1.25e-06 | 1e-4                | -5.39e-07 | PASS   |
| svf_south                 | 1.31e-06 | 1e-4                | -6.68e-07 | PASS   |
| svf_west                  | 1.31e-06 | 1e-4                | -6.60e-07 | PASS   |
| svf_veg                   | 1.09e-02 | 0.02 (1% arch diff) |  3.97e-04 | PASS   |
| gvf_lup                   | 3.05e-04 | 0.1% relative       | -5.67e-05 | PASS   |
| gvf_alb                   | 2.98e-08 | 0.1% relative       |  8.88e-11 | PASS   |
| gvf_norm                  | 5.96e-08 | 0.1% relative       |  3.95e-11 | PASS   |
| radiation_kside_e         | 3.51e-04 | 0.1% relative       |  3.95e-05 | PASS   |
| radiation_kside_s         | 2.14e-04 | 0.1% relative       |  2.96e-05 | PASS   |
| radiation_lside_e         | 4.88e-04 | 0.1% relative       | -5.80e-05 | PASS   |
| radiation_lside_s         | 3.97e-04 | 0.1% relative       | -6.40e-05 | PASS   |
| utci_grid                 | 4.01e-05 | 0.1 C               |  2.77e-07 | PASS   |
| pet_grid                  | 2.20e-02 | 0.2 C               |  4.32e-04 | PASS   |
| tmrt_aniso                | 3.05e-05 | 0.1 C               | -4.59e-06 | PASS   |
| tmrt_iso                  | 3.05e-05 | 0.1 C               | -4.04e-06 | PASS   |
| ground_temp_case1         | 0.00e+00 | 1e-3                |  0.00e+00 | PASS   |
| ground_temp_case2         | 3.05e-05 | 1e-3                | -8.39e-07 | PASS   |
| ground_temp_case3         | 3.05e-05 | 1e-3                | -8.39e-07 | PASS   |
| wall_temp_tg              | 0.00e+00 | 0.1 C               |  0.00e+00 | PASS   |
| sinusoidal_ground_diurnal | 4.17e-06 | 1e-3 C              |  1.47e-07 | PASS   |
| sinusoidal_wall_diurnal   | 4.17e-06 | 1e-3 C              |  1.47e-07 | PASS   |

## Sinusoidal Ground Temperature Model

Compares `compute_ground_temperature()` (Rust) against the UMEP Python
formula from `Solweig_2025a_calc_forprocessing.py` (lines 171–199).

- Ground curve max |diff|: 4.17e-06 (PASS)
- Wall curve max |diff|: 4.17e-06 (PASS)

## Formula Agreement (12 Scenarios)

Ground temperature model tested across 12 representative scenarios spanning
different times of day, sky conditions, land cover types, and wall materials.

**Result:** 12/12 scenarios match within f32 tolerance (atol=1e-4)

| Scenario             | Rust Tg | UMEP Tg |    d Tg | Rust Wall | UMEP Wall |  d Wall |     CI | Status |
| -------------------- | ------: | ------: | ------: | --------: | --------: | ------: | -----: | ------ |
| Noon clear cobble    | 14.9034 | 14.9034 | 9.4e-07 |   14.9034 |   14.9034 | 9.4e-07 | 0.9874 | PASS   |
| Noon clear asphalt   | 19.4607 | 19.4607 | 3.1e-06 |   14.9034 |   14.9034 | 9.4e-07 | 0.9874 | PASS   |
| Afternoon 18h        | 13.6387 | 13.6387 | 2.6e-07 |   13.6387 |   13.6387 | 2.6e-07 | 0.9036 | PASS   |
| Evening 22h          |  7.6906 |  7.6906 | 2.0e-06 |    7.6906 |    7.6906 | 2.0e-06 | 1.0000 | PASS   |
| Before sunrise       |  0.0000 |  0.0000 | 0.0e+00 |    0.0000 |    0.0000 | 0.0e+00 | 1.0000 | PASS   |
| Peak at TmaxLST      | 16.9400 | 16.9400 | 5.3e-07 |   16.9400 |   16.9400 | 5.3e-07 | 1.0000 | PASS   |
| Cloudy CI low        |  5.6150 |  5.6150 | 1.3e-06 |    5.6150 |    5.6150 | 1.3e-06 | 0.3720 | PASS   |
| Wood wall noon       | 14.9034 | 14.9034 | 9.4e-07 |   23.6602 |   23.6602 | 5.3e-07 | 0.9874 | PASS   |
| Brick wall afternoon | 13.6387 | 13.6387 | 2.6e-07 |   14.4921 |   14.4921 | 3.5e-07 | 0.9036 | PASS   |
| Grass morning        |  4.0850 |  4.0850 | 3.8e-08 |    7.6906 |    7.6906 | 8.5e-07 | 1.0000 | PASS   |
| Water                |  0.0000 |  0.0000 | 0.0e+00 |   14.9034 |   14.9034 | 9.4e-07 | 0.9874 | PASS   |
| High lat low sun     |  1.9771 |  1.9771 | 8.0e-08 |    1.9771 |    1.9771 | 8.0e-08 | 1.0000 | PASS   |

## Test Inventory

130 golden tests across 11 modules:

| Module          | Test File                        | Tests | Description                                          |
| --------------- | -------------------------------- | ----: | ---------------------------------------------------- |
| Anisotropic Sky | `test_golden_anisotropic_sky.py` |    16 | Perez et al. (1993) direction-dependent sky model    |
| Ground Temp     | `test_golden_ground_temp.py`     |     6 | TsWaveDelay thermal inertia model                    |
| GVF             | `test_golden_gvf.py`             |    13 | Ground View Factor calculations                      |
| PET             | `test_golden_pet.py`             |    10 | Physiological Equivalent Temperature                 |
| Radiation       | `test_golden_radiation.py`       |    14 | Kside/Lside shortwave/longwave via vegetation module |
| Shadows         | `test_golden_shadows.py`         |     8 | Building and vegetation shadow casting               |
| SVF             | `test_golden_svf.py`             |     8 | Sky View Factor (total, directional, vegetation)     |
| Tmrt            | `test_golden_tmrt.py`            |     6 | Mean Radiant Temperature                             |
| UTCI            | `test_golden_utci.py`            |    12 | Universal Thermal Climate Index                      |
| Wall Geometry   | `test_golden_wall_geometry.py`   |    30 | Wall height/aspect extraction, rotation, dilation    |
| Wall Temp       | `test_golden_walls.py`           |     7 | Ground and wall temperature deviations               |

## Tolerance Settings

| Algorithm        | RTOL | ATOL | Notes                                 |
| ---------------- | ---- | ---- | ------------------------------------- |
| Shadows          | 1e-6 | 1e-6 | Binary masks, high precision required |
| SVF              | 0.01 | 0.02 | 2% tolerance for complex geometry     |
| Radiation        | 1e-4 | 1e-4 | Physical radiation values             |
| UTCI             | 1e-3 | 0.05 | 0.05 C absolute tolerance             |
| PET              | 0.01 | 0.1  | Iterative solver, 0.1 C tolerance     |
| Tmrt             | 1e-4 | 0.01 | 0.01 C absolute tolerance             |
| Anisotropic Sky  | 1e-4 | 0.1  | Complex radiation model               |
| Ground/Wall Temp | —    | 1e-3 | f32 precision                         |
