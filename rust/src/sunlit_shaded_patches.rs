// sunlit_shaded_patches.rs
// Rust implementation of sunlit and shaded patch calculations, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/sunlit_shaded_patches.py.

use ndarray::Zip;
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
    let patch_to_sun_azi = patch_azimuth.mapv(|p_azi| (solar_azimuth - p_azi).abs());
    let xi = patch_to_sun_azi.mapv(|a| (a * deg2rad).cos());
    let yi = xi.mapv(|x| 2.0 * x * (solar_altitude * deg2rad).tan());
    let hsvf = asvf.tan();
    let yi_ = yi.mapv(|y| if y > 0.0 { 0.0 } else { y });
    let tan_delta = yi_.mapv(|y| hsvf + y);
    let sunlit_degrees = tan_delta.mapv(|t| t.atan() * rad2deg);
    Zip::from(sunlit_out)
        .and(shaded_out)
        .and(patch_altitude)
        .and(&sunlit_degrees)
        .for_each(|sunlit, shaded, &p_alt, &sdeg| {
            *sunlit = sdeg < p_alt;
            *shaded = sdeg > p_alt;
        });
}

/// Calculates whether a single patch is sunlit or shaded.
/// This is a scalar version for use inside pixel-parallel loops.
#[allow(dead_code)]
pub fn shaded_or_sunlit_pixel(
    solar_altitude: f32,
    solar_azimuth: f32,
    patch_altitude: f32,
    patch_azimuth: f32,
    asvf: f32,
) -> (bool, bool) {
    let deg2rad = std::f32::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f32::consts::PI;

    let patch_to_sun_azi = (solar_azimuth - patch_azimuth).abs();
    let xi = (patch_to_sun_azi * deg2rad).cos();
    let yi = 2.0 * xi * (solar_altitude * deg2rad).tan();
    let hsvf = asvf.tan();
    let yi_ = if yi > 0.0 { 0.0 } else { yi };
    let tan_delta = hsvf + yi_;
    let sunlit_degrees = tan_delta.atan() * rad2deg;

    let sunlit = sunlit_degrees < patch_altitude;
    let shaded = sunlit_degrees > patch_altitude;

    (sunlit, shaded)
}
