// patch_radiation.rs
// Rust implementation of patch radiation calculations, ported from the original Python source.
// This file is intended to match the structure and intent of the Python version in pysrc/umepr/patch_radiation.py.

use ndarray::{Array1, Array2};
use std::f32::consts::PI;

pub fn shortwave_from_sky(
    sky: &Array2<f32>,
    angle_of_incidence: &Array2<f32>,
    lum_chi: &Array2<f32>,
    steradian: &Array2<f32>,
) -> Array2<f32> {
    sky * lum_chi * angle_of_incidence * steradian
}

pub fn longwave_from_sky(
    sky: &Array2<f32>,
    lsky_side: f32,
    lsky_down: f32,
    patch_azimuth: f32,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let deg2rad = PI / 180.0;
    let shape = sky.raw_dim();
    let mut ldown_sky = Array2::<f32>::zeros(shape.clone());
    let mut lside_sky = Array2::<f32>::zeros(shape.clone());
    let mut least = Array2::<f32>::zeros(shape.clone());
    let mut lsouth = Array2::<f32>::zeros(shape.clone());
    let mut lwest = Array2::<f32>::zeros(shape.clone());
    let mut lnorth = Array2::<f32>::zeros(shape.clone());
    for ((i, j), &skyval) in sky.indexed_iter() {
        ldown_sky[(i, j)] = skyval * lsky_down;
        lside_sky[(i, j)] = skyval * lsky_side;
        if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
            least[(i, j)] = skyval * lsky_side * ((90.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
            lsouth[(i, j)] = skyval * lsky_side * ((180.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
            lwest[(i, j)] = skyval * lsky_side * ((270.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
            lnorth[(i, j)] = skyval * lsky_side * ((0.0 - patch_azimuth) * deg2rad).cos();
        }
    }
    (lside_sky, ldown_sky, least, lsouth, lwest, lnorth)
}

pub fn longwave_from_veg(
    vegetation: &Array2<f32>,
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_altitude: f32,
    patch_azimuth: f32,
    ewall: f32,
    ta: f32,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let sbc = 5.67051e-8;
    let deg2rad = PI / 180.0;
    let shape = vegetation.raw_dim();
    let mut lside_veg = Array2::<f32>::zeros(shape.clone());
    let mut ldown_veg = Array2::<f32>::zeros(shape.clone());
    let mut least = Array2::<f32>::zeros(shape.clone());
    let mut lsouth = Array2::<f32>::zeros(shape.clone());
    let mut lwest = Array2::<f32>::zeros(shape.clone());
    let mut lnorth = Array2::<f32>::zeros(shape.clone());
    let vegetation_surface = (ewall * sbc * (ta + 273.15).powi(4)) / PI;
    for ((i, j), &veg) in vegetation.indexed_iter() {
        lside_veg[(i, j)] = vegetation_surface * steradian * angle_of_incidence * veg;
        ldown_veg[(i, j)] = vegetation_surface * steradian * angle_of_incidence_h * veg;
        if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
            least[(i, j)] = vegetation_surface
                * steradian
                * (patch_altitude * deg2rad).cos()
                * veg
                * ((90.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
            lsouth[(i, j)] = vegetation_surface
                * steradian
                * (patch_altitude * deg2rad).cos()
                * veg
                * ((180.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
            lwest[(i, j)] = vegetation_surface
                * steradian
                * (patch_altitude * deg2rad).cos()
                * veg
                * ((270.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
            lnorth[(i, j)] = vegetation_surface
                * steradian
                * (patch_altitude * deg2rad).cos()
                * veg
                * ((0.0 - patch_azimuth) * deg2rad).cos();
        }
    }
    (lside_veg, ldown_veg, least, lsouth, lwest, lnorth)
}

pub fn longwave_from_buildings(
    building: &Array2<f32>,
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
    sunlit_patches: &Array2<bool>,
    shaded_patches: &Array2<bool>,
    azimuth_difference: f32,
    solar_altitude: f32,
    ewall: f32,
    ta: f32,
    tgwall: f32,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let sbc = 5.67051e-8;
    let deg2rad = PI / 180.0;
    let shape = building.raw_dim();
    let mut lside_sun = Array2::<f32>::zeros(shape.clone());
    let mut lside_sh = Array2::<f32>::zeros(shape.clone());
    let mut ldown_sun = Array2::<f32>::zeros(shape.clone());
    let mut ldown_sh = Array2::<f32>::zeros(shape.clone());
    let mut least = Array2::<f32>::zeros(shape.clone());
    let mut lsouth = Array2::<f32>::zeros(shape.clone());
    let mut lwest = Array2::<f32>::zeros(shape.clone());
    let mut lnorth = Array2::<f32>::zeros(shape.clone());
    let sunlit_surface = (ewall * sbc * (ta + tgwall + 273.15).powi(4)) / PI;
    let shaded_surface = (ewall * sbc * (ta + 273.15).powi(4)) / PI;
    if (azimuth_difference > 90.0) && (azimuth_difference < 270.0) && (solar_altitude > 0.0) {
        for ((i, j), &bldg) in building.indexed_iter() {
            let sunlit = sunlit_patches[(i, j)] as u8 as f32;
            let shaded = shaded_patches[(i, j)] as u8 as f32;
            lside_sun[(i, j)] = sunlit_surface * sunlit * steradian * angle_of_incidence * bldg;
            lside_sh[(i, j)] = shaded_surface * shaded * steradian * angle_of_incidence * bldg;
            ldown_sun[(i, j)] = sunlit_surface * sunlit * steradian * angle_of_incidence_h * bldg;
            ldown_sh[(i, j)] = shaded_surface * shaded * steradian * angle_of_incidence_h * bldg;
            if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
                least[(i, j)] = (sunlit_surface * sunlit + shaded_surface * shaded)
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((90.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
                lsouth[(i, j)] = (sunlit_surface * sunlit + shaded_surface * shaded)
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((180.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
                lwest[(i, j)] = (sunlit_surface * sunlit + shaded_surface * shaded)
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((270.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
                lnorth[(i, j)] = (sunlit_surface * sunlit + shaded_surface * shaded)
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((0.0 - patch_azimuth) * deg2rad).cos();
            }
        }
    } else {
        for ((i, j), &bldg) in building.indexed_iter() {
            lside_sh[(i, j)] = shaded_surface * steradian * angle_of_incidence * bldg;
            lside_sun[(i, j)] = 0.0;
            ldown_sh[(i, j)] = shaded_surface * steradian * angle_of_incidence_h * bldg;
            ldown_sun[(i, j)] = 0.0;
            if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
                least[(i, j)] = shaded_surface
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((90.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
                lsouth[(i, j)] = shaded_surface
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((180.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
                lwest[(i, j)] = shaded_surface
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((270.0 - patch_azimuth) * deg2rad).cos();
            }
            if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
                lnorth[(i, j)] = shaded_surface
                    * steradian
                    * angle_of_incidence
                    * bldg
                    * ((0.0 - patch_azimuth) * deg2rad).cos();
            }
        }
    }
    (
        lside_sun, lside_sh, ldown_sun, ldown_sh, least, lsouth, lwest, lnorth,
    )
}

pub fn longwave_from_buildings_wall_scheme(
    voxel_maps: &Array2<f32>,
    voxel_table: &Array2<f32>,
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let deg2rad = PI / 180.0;
    let shape = voxel_maps.raw_dim();
    let mut lside = Array2::<f32>::zeros(shape.clone());
    let mut lside_sh = Array2::<f32>::zeros(shape.clone());
    let mut ldown = Array2::<f32>::zeros(shape.clone());
    let mut ldown_sh = Array2::<f32>::zeros(shape.clone());
    let mut least = Array2::<f32>::zeros(shape.clone());
    let mut lsouth = Array2::<f32>::zeros(shape.clone());
    let mut lwest = Array2::<f32>::zeros(shape.clone());
    let mut lnorth = Array2::<f32>::zeros(shape.clone());

    // Build a map from voxel ID to LongwaveRadiation (assume col 0 = ID, col 1 = LongwaveRadiation)
    use std::collections::HashMap;
    let mut id_to_longwave = HashMap::new();
    for row in voxel_table.rows() {
        let id = row[0] as i32;
        let val = row[1];
        id_to_longwave.insert(id, val);
    }

    // For each voxel, get the longwave value (0 if not found or id==0)
    let mut patch_radiation = Array2::<f32>::zeros(shape.clone());
    for ((i, j), &vid) in voxel_maps.indexed_iter() {
        let vid_i32 = vid.round() as i32;
        if vid_i32 == 0 {
            patch_radiation[(i, j)] = 0.0;
        } else {
            patch_radiation[(i, j)] = *id_to_longwave.get(&vid_i32).unwrap_or(&0.0);
        }
    }

    // Lside = patch_radiation * steradian * angle_of_incidence
    for ((i, j), &pr) in patch_radiation.indexed_iter() {
        lside[(i, j)] = pr * steradian * angle_of_incidence;
        ldown[(i, j)] = pr * steradian * angle_of_incidence_h;
        // Cardinal directions
        if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
            least[(i, j)] =
                pr * steradian * angle_of_incidence * ((90.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
            lsouth[(i, j)] =
                pr * steradian * angle_of_incidence * ((180.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
            lwest[(i, j)] =
                pr * steradian * angle_of_incidence * ((270.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
            lnorth[(i, j)] =
                pr * steradian * angle_of_incidence * ((0.0 - patch_azimuth) * deg2rad).cos();
        }
    }
    // lside_sh, ldown_sh remain zero arrays (no direct analog in Python for wallScheme)
    (
        lside, lside_sh, ldown, ldown_sh, least, lsouth, lwest, lnorth,
    )
}

pub fn reflected_longwave(
    reflecting_surface: &Array2<f32>,
    steradian: f32,
    angle_of_incidence: f32,
    angle_of_incidence_h: f32,
    patch_azimuth: f32,
    ldown_sky: &Array2<f32>,
    lup: &Array2<f32>,
    ewall: f32,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let deg2rad = PI / 180.0;
    let shape = reflecting_surface.raw_dim();
    let mut lside_ref = Array2::<f32>::zeros(shape.clone());
    let mut ldown_ref = Array2::<f32>::zeros(shape.clone());
    let mut least = Array2::<f32>::zeros(shape.clone());
    let mut lsouth = Array2::<f32>::zeros(shape.clone());
    let mut lwest = Array2::<f32>::zeros(shape.clone());
    let mut lnorth = Array2::<f32>::zeros(shape.clone());
    // (Ldown_sky + Lup) * (1 - ewall) * 0.5 / np.pi
    let reflected_radiation = ldown_sky + lup;
    let reflected_radiation = reflected_radiation.mapv(|v| (v * (1.0 - ewall) * 0.5) / PI);
    for ((i, j), &surf) in reflecting_surface.indexed_iter() {
        lside_ref[(i, j)] = reflected_radiation[(i, j)] * steradian * angle_of_incidence * surf;
        ldown_ref[(i, j)] = reflected_radiation[(i, j)] * steradian * angle_of_incidence_h * surf;
        if patch_azimuth > 360.0 || patch_azimuth < 180.0 {
            least[(i, j)] = reflected_radiation[(i, j)]
                * steradian
                * angle_of_incidence
                * surf
                * ((90.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 90.0 && patch_azimuth < 270.0 {
            lsouth[(i, j)] = reflected_radiation[(i, j)]
                * steradian
                * angle_of_incidence
                * surf
                * ((180.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 180.0 && patch_azimuth < 360.0 {
            lwest[(i, j)] = reflected_radiation[(i, j)]
                * steradian
                * angle_of_incidence
                * surf
                * ((270.0 - patch_azimuth) * deg2rad).cos();
        }
        if patch_azimuth > 270.0 || patch_azimuth < 90.0 {
            lnorth[(i, j)] = reflected_radiation[(i, j)]
                * steradian
                * angle_of_incidence
                * surf
                * ((0.0 - patch_azimuth) * deg2rad).cos();
        }
    }
    (lside_ref, ldown_ref, least, lsouth, lwest, lnorth)
}

pub fn patch_steradians(l_patches: &Array2<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
    let deg2rad = PI / 180.0;
    let patch_altitude = l_patches.column(0).to_owned();
    // Find unique altitudes and their counts (mimic np.unique(..., return_counts=True))
    let mut skyalt: Vec<f32> = Vec::new();
    let mut skyalt_c: Vec<usize> = Vec::new();
    for &alt in patch_altitude.iter() {
        if let Some(idx) = skyalt.iter().position(|&x| (x - alt).abs() < 1e-6) {
            skyalt_c[idx] += 1;
        } else {
            skyalt.push(alt);
            skyalt_c.push(1);
        }
    }
    let mut steradian = Array1::<f32>::zeros(patch_altitude.len());
    for (i, &alt) in patch_altitude.iter().enumerate() {
        // Find count for this altitude
        let idx = skyalt.iter().position(|&x| (x - alt).abs() < 1e-6).unwrap();
        let count = skyalt_c[idx] as f32;
        if count > 1.0 {
            steradian[i] = ((360.0 / count) * deg2rad)
                * (((alt + patch_altitude[0]) * deg2rad).sin()
                    - ((alt - patch_altitude[0]) * deg2rad).sin());
        } else {
            let prev = if i > 0 {
                patch_altitude[i - 1]
            } else {
                patch_altitude[0]
            };
            steradian[i] = ((360.0 / count) * deg2rad)
                * ((alt * deg2rad).sin() - ((prev + patch_altitude[0]) * deg2rad).sin());
        }
    }
    (steradian, Array1::from_vec(skyalt), patch_altitude)
}

// Additional functions (longwave_from_veg, longwave_from_buildings, etc.) can be ported similarly as needed.
