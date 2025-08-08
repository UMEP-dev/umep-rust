// emissivity_models.rs
// Rust implementation of emissivity models, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/emissivity_models.py.

use ndarray::{Array1, Array2};

/// Model 1: Unsworth & Monteith, 1975
pub fn model1(sky_patches: &Array2<f32>, esky: f32, _ta: f32) -> (Array1<f32>, Array1<f32>) {
    let deg2rad = std::f32::consts::PI / 180.0;
    let skyalt = sky_patches.column(0).to_owned();
    let skyzen = skyalt.mapv(|x| 90.0 - x);
    let cosskyzen = skyzen.mapv(|x| (x * deg2rad).cos());
    let a_c = 0.67;
    let b_c = 0.094;
    let ln_u_prec = esky / b_c - a_c / b_c - 0.5;
    let u_prec = ln_u_prec.exp();
    let owp = cosskyzen.mapv(|c| u_prec / c);
    let log_owp = owp.mapv(|o| o.ln());
    let esky_band = log_owp.mapv(|l| a_c + b_c * l);
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

/// Model 3: Bliss, 1961
pub fn model3(sky_patches: &Array2<f32>, esky: f32, _ta: f32) -> (Array1<f32>, Array1<f32>) {
    let deg2rad = std::f32::consts::PI / 180.0;
    let skyalt = sky_patches.column(0).to_owned();
    let skyzen = skyalt.mapv(|x| 90.0 - x);
    let b_c = 1.8;
    let esky_band = skyzen.mapv(|z| 1.0 - (1.0 - esky).powf(1.0 / (b_c * (z * deg2rad).cos())));
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
