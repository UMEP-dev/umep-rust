// sunlit_shaded_patches.rs
// Rust implementation of sunlit and shaded patch calculations, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/sunlit_shaded_patches.py.

// Vectorized function removed as it was unused.

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
