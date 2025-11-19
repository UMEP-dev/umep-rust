// patch_radiation.rs
// Rust implementation of patch radiation calculations, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/patch_radiation.py.

use std::f32::consts::PI;

// Vectorized functions removed as they were unused.


pub fn longwave_from_sky_pixel(
    lsky_side: f32,
    lsky_down: f32,
    patch_azimuth: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let deg2rad = PI / 180.0;
    let ldown_sky = lsky_down;
    let lside_sky = lsky_side;

    let least = if !(180.0..=360.0).contains(&patch_azimuth) {
        lside_sky * ((90.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lsouth = if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
        lside_sky * ((180.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lwest = if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
        lside_sky * ((270.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lnorth = if !(90.0..=270.0).contains(&patch_azimuth) {
        lside_sky * ((0.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    (lside_sky, ldown_sky, least, lsouth, lwest, lnorth)
}

// Vectorized functions removed as they were unused.


pub fn longwave_from_veg_pixel(
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_altitude: f32,
    patch_azimuth: f32,
    ewall: f32,
    ta: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let sbc = 5.67051e-8;
    let deg2rad = PI / 180.0;
    let vegetation_surface = (ewall * sbc * (ta + 273.15).powi(4)) / PI;
    let cos_alt = (patch_altitude * deg2rad).cos();

    let lside_veg = vegetation_surface * steradian * angle_of_incidence;
    let ldown_veg = vegetation_surface * steradian * angle_of_incidence_h;

    let least = if !(180.0..=360.0).contains(&patch_azimuth) {
        vegetation_surface * steradian * cos_alt * ((90.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lsouth = if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
        vegetation_surface * steradian * cos_alt * ((180.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lwest = if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
        vegetation_surface * steradian * cos_alt * ((270.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lnorth = if !(90.0..=270.0).contains(&patch_azimuth) {
        vegetation_surface * steradian * cos_alt * ((0.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    (lside_veg, ldown_veg, least, lsouth, lwest, lnorth)
}

// Vectorized functions removed as they were unused.


pub fn longwave_from_buildings_pixel(
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
    sunlit_patch: bool,
    shaded_patch: bool,
    azimuth_difference: f32,
    solar_altitude: f32,
    ewall: f32,
    ta: f32,
    tgwall: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let sbc = 5.67051e-8;
    let deg2rad = PI / 180.0;
    let sunlit_surface = (ewall * sbc * (ta + tgwall + 273.15).powi(4)) / PI;
    let shaded_surface = (ewall * sbc * (ta + 273.15).powi(4)) / PI;

    let (lside_sun, lside_sh, ldown_sun, ldown_sh, mask_val) =
        if (azimuth_difference > 90.0) && (azimuth_difference < 270.0) && (solar_altitude > 0.0) {
            let sunlit = if sunlit_patch { 1.0 } else { 0.0 };
            let shaded = if shaded_patch { 1.0 } else { 0.0 };
            let lside_sun = sunlit * sunlit_surface * steradian * angle_of_incidence;
            let lside_sh = shaded * shaded_surface * steradian * angle_of_incidence;
            let ldown_sun = sunlit * sunlit_surface * steradian * angle_of_incidence_h;
            let ldown_sh = shaded * shaded_surface * steradian * angle_of_incidence_h;
            let mask_val = sunlit * sunlit_surface + shaded * shaded_surface;
            (lside_sun, lside_sh, ldown_sun, ldown_sh, mask_val)
        } else {
            let lside_sun = 0.0;
            let lside_sh = shaded_surface * steradian * angle_of_incidence;
            let ldown_sun = 0.0;
            let ldown_sh = shaded_surface * steradian * angle_of_incidence_h;
            let mask_val = shaded_surface;
            (lside_sun, lside_sh, ldown_sun, ldown_sh, mask_val)
        };

    let least = if !(180.0..=360.0).contains(&patch_azimuth) {
        mask_val * steradian * angle_of_incidence * ((90.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lsouth = if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
        mask_val * steradian * angle_of_incidence * ((180.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lwest = if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
        mask_val * steradian * angle_of_incidence * ((270.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lnorth = if !(90.0..=270.0).contains(&patch_azimuth) {
        mask_val * steradian * angle_of_incidence * ((0.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };

    (
        lside_sun, lside_sh, ldown_sun, ldown_sh, least, lsouth, lwest, lnorth,
    )
}

// Vectorized functions removed as they were unused.


pub fn longwave_from_buildings_wall_scheme_pixel(
    voxel_table: ndarray::ArrayView2<f32>,
    voxel_map_val: usize,
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let deg2rad = PI / 180.0;

    let patch_radiation = if voxel_map_val > 0 {
        // This is a simplification. In a real scenario, you'd have a more robust
        // way to map voxel_map_val to the correct row in voxel_table.
        // Assuming voxel_map_val is a 1-based index corresponding to the row.
        voxel_table.row(voxel_map_val - 1)[1]
    } else {
        0.0
    };

    let lside = patch_radiation * steradian * angle_of_incidence;
    let ldown = patch_radiation * steradian * angle_of_incidence_h;
    let lside_sh = 0.0;
    let ldown_sh = 0.0;

    let least = if !(180.0..=360.0).contains(&patch_azimuth) {
        patch_radiation * steradian * angle_of_incidence * ((90.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lsouth = if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
        patch_radiation * steradian * angle_of_incidence * ((180.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lwest = if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
        patch_radiation * steradian * angle_of_incidence * ((270.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lnorth = if !(90.0..=270.0).contains(&patch_azimuth) {
        patch_radiation * steradian * angle_of_incidence * ((0.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };

    (
        lside, lside_sh, ldown, ldown_sh, least, lsouth, lwest, lnorth,
    )
}

// Vectorized functions removed as they were unused.


pub fn reflected_longwave_pixel(
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
    ldown_sky: f32,
    lup: f32,
    ewall: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let deg2rad = PI / 180.0;
    let reflected_radiation = (ldown_sky + lup) * (1.0 - ewall) * 0.5 / PI;
    let lside_ref = reflected_radiation * steradian * angle_of_incidence;
    let ldown_ref = reflected_radiation * steradian * angle_of_incidence_h;
    let least = if !(180.0..=360.0).contains(&patch_azimuth) {
        reflected_radiation
            * steradian
            * angle_of_incidence
            * ((90.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lsouth = if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
        reflected_radiation
            * steradian
            * angle_of_incidence
            * ((180.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lwest = if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
        reflected_radiation
            * steradian
            * angle_of_incidence
            * ((270.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };
    let lnorth = if !(90.0..=270.0).contains(&patch_azimuth) {
        reflected_radiation
            * steradian
            * angle_of_incidence
            * ((0.0 - patch_azimuth) * deg2rad).cos()
    } else {
        0.0
    };

    (lside_ref, ldown_ref, least, lsouth, lwest, lnorth)
}
