// sunlit_shaded_patches.rs
// Rust implementation of sunlit and shaded patch calculations, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/sunlit_shaded_patches.py.

use ndarray::{ArrayView1, ArrayViewMut1};

/// Calculates whether a patch is sunlit or shaded based on sky view factor, solar altitude, and azimuth.
/// Returns (sunlit_patches, shaded_patches) as boolean arrays.
pub fn shaded_or_sunlit(
    solar_altitude: f32,
    solar_azimuth: f32,
    patch_altitude: &ArrayView1<f32>,
    patch_azimuth: &ArrayView1<f32>,
    asvf: f32,
    sunlit_out: &mut ArrayViewMut1<bool>,
    shaded_out: &mut ArrayViewMut1<bool>,
) {
    let deg2rad = std::f32::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f32::consts::PI;

    for ((&p_alt, &p_azi), (sunlit, shaded)) in patch_altitude
        .iter()
        .zip(patch_azimuth.iter())
        .zip(sunlit_out.iter_mut().zip(shaded_out.iter_mut()))
    {
        let patch_to_sun_azi = (solar_azimuth - p_azi).abs();
        let xi = (patch_to_sun_azi * deg2rad).cos();
        let yi = 2.0 * xi * (solar_altitude * deg2rad).tan();
        let hsvf = asvf.tan();
        let yi_ = if yi > 0.0 { 0.0 } else { yi };
        let tan_delta = hsvf + yi_;
        let sunlit_degrees = tan_delta.atan() * rad2deg;
        *sunlit = sunlit_degrees < p_alt;
        *shaded = sunlit_degrees > p_alt;
    }
}
