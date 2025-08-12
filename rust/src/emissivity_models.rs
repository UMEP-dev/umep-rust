// emissivity_models.rs
// Rust implementation of emissivity models, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/emissivity_models.py.

use ndarray::{Array1, Array2};

/// Model 2: Martin & Berdhal, 1984
pub fn model2(sky_patches: &Array2<f32>, esky: f32, _ta: f32) -> (Array1<f32>, Array1<f32>) {
    let deg2rad = std::f32::consts::PI / 180.0;
    let skyalt = sky_patches.column(0).to_owned();
    let skyzen = skyalt.mapv(|x| 90.0 - x);
    let b_c = 0.308;
    let esky_band =
        skyzen.mapv(|z| 1.0 - (1.0 - esky) * ((b_c * (1.7 - (1.0 / (z * deg2rad).cos()))).exp()));
    let p_alt = sky_patches.column(0);
    let mut patch_emissivity = Array1::<f32>::zeros(p_alt.len());
    for (i, &idx) in skyalt.iter().enumerate() {
        let temp_emissivity = esky_band[i];
        for (j, &p) in p_alt.iter().enumerate() {
            if (p - idx).abs() < 1e-8 {
                patch_emissivity[j] = temp_emissivity;
            }
        }
    }
    let sum: f32 = patch_emissivity.sum();
    let patch_emissivity_normalized = patch_emissivity.mapv(|v| v / sum);
    (patch_emissivity_normalized, esky_band)
}
