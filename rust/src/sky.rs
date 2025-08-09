use crate::{emissivity_models, patch_radiation, sunlit_shaded_patches};
use ndarray::{s, Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67051e-8; // Stefan-Boltzmann constant

#[pyclass]
pub struct SkyResult {
    #[pyo3(get)]
    pub ldown: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sky: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_sun: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lside_ref: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub least: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lnorth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lsouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub keast: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub ksouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub knorth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside_i: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside_d: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kside: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub steradians: Py<PyArray1<f32>>,
    #[pyo3(get)]
    pub skyalt: Py<PyArray1<f32>>,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn anisotropic_sky(
    py: Python,
    shmat: PyReadonlyArray3<f32>,
    vegshmat: PyReadonlyArray3<f32>,
    vbshvegshmat: PyReadonlyArray3<f32>,
    solar_altitude: f32,
    solar_azimuth: f32,
    asvf: PyReadonlyArray2<f32>,
    cyl: bool,
    esky: f32,
    l_patches: PyReadonlyArray2<f32>,
    wall_scheme: bool,
    voxel_table: Option<PyReadonlyArray2<f32>>,
    voxel_maps: Option<PyReadonlyArray3<f32>>,
    steradians: PyReadonlyArray1<f32>,
    ta: f32,
    tgwall: f32,
    ewall: f32,
    lup: PyReadonlyArray2<f32>,
    rad_i: f32,
    rad_d: f32,
    _rad_g: f32,
    lv: PyReadonlyArray2<f32>,
    albedo: f32,
    _anisotropic_diffuse: bool,
    _diffsh: PyReadonlyArray3<f32>,
    shadow: PyReadonlyArray2<f32>,
    kup_e: PyReadonlyArray2<f32>,
    kup_s: PyReadonlyArray2<f32>,
    kup_w: PyReadonlyArray2<f32>,
    kup_n: PyReadonlyArray2<f32>,
    _current_step: i32,
) -> PyResult<Py<SkyResult>> {
    // Convert PyReadonlyArray to ArrayView for easier manipulation
    let shmat = shmat.as_array();
    let vegshmat = vegshmat.as_array();
    let vbshvegshmat = vbshvegshmat.as_array();
    let asvf = asvf.as_array();
    let l_patches = l_patches.as_array();
    let voxel_table = voxel_table.as_ref().map(|v| v.as_array());
    let voxel_maps = voxel_maps.as_ref().map(|v| v.as_array());
    let steradians = steradians.as_array();
    let lup = lup.as_array();
    let lv = lv.as_array();
    let shadow = shadow.as_array();
    let kup_e = kup_e.as_array();
    let kup_s = kup_s.as_array();
    let kup_w = kup_w.as_array();
    let kup_n = kup_n.as_array();

    let rows = shmat.shape()[0];
    let cols = shmat.shape()[1];
    let n_patches = l_patches.shape()[0];

    // ...existing code...

    // Output arrays
    let mut keast = Array2::<f32>::zeros((rows, cols));
    let mut kwest = Array2::<f32>::zeros((rows, cols));
    let mut knorth = Array2::<f32>::zeros((rows, cols));
    let mut ksouth = Array2::<f32>::zeros((rows, cols));

    let mut ldown_ref = Array2::<f32>::zeros((rows, cols));

    let mut lside_ref = Array2::<f32>::zeros((rows, cols));

    let mut least = Array2::<f32>::zeros((rows, cols));
    let mut lwest = Array2::<f32>::zeros((rows, cols));
    let mut lnorth = Array2::<f32>::zeros((rows, cols));
    let mut lsouth = Array2::<f32>::zeros((rows, cols));

    // Patch altitudes and azimuths
    let patch_altitude = l_patches.column(0).to_owned();
    let patch_azimuth = l_patches.column(1).to_owned();

    // Calculate unique altitudes and their counts (skyalt, skyalt_c)
    let mut skyalt_vec: Vec<f32> = patch_altitude.iter().cloned().collect();
    skyalt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    skyalt_vec.dedup();
    let skyalt = Array1::<f32>::from(skyalt_vec.clone());
    let mut skyalt_c = Vec::with_capacity(skyalt_vec.len());
    for &val in &skyalt_vec {
        let count = patch_altitude
            .iter()
            .filter(|&&x| (x - val).abs() < 1e-6)
            .count();
        skyalt_c.push(count as i32);
    }

    // Emissivity model selection (model2)
    let (_patch_emissivity_normalized, esky_band) =
        emissivity_models::model2(&l_patches.mapv(|v| v as f32), esky, ta);

    // Longwave based on spectral flux density (divide by pi)
    let mut ldown_patch = Array1::<f32>::zeros(n_patches);
    let mut lside_patch = Array1::<f32>::zeros(n_patches);
    let mut lnormal_patch = Array1::<f32>::zeros(n_patches);
    let deg2rad = PI / 180.0;
    for &temp_altitude in skyalt.iter() {
        // Find indices where patch_altitude == temp_altitude (within tolerance)
        let indices: Vec<usize> = patch_altitude
            .iter()
            .enumerate()
            .filter(|(_, &x)| (x - temp_altitude).abs() < 1e-6)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        // Use first index for esky_band (since all with the same altitude have the same value)
        let temp_emissivity = esky_band[indices[0]];
        let ta_k = ta + 273.15;
        let lval = (temp_emissivity * SBC * ta_k.powi(4)) / PI;
        for &i in &indices {
            let s = steradians[i];
            ldown_patch[i] = lval * s * (patch_altitude[i] * deg2rad).sin();
            lside_patch[i] = lval * s * (patch_altitude[i] * deg2rad).cos();
            lnormal_patch[i] = lval * s;
        }
    }

    // Shortwave normalization
    let mut lum_chi = Array1::<f32>::zeros(n_patches);
    if solar_altitude > 0.0 {
        let patch_luminance = lv.column(2);
        let mut rad_tot = 0.0;
        for i in 0..n_patches {
            rad_tot += patch_luminance[i] * steradians[i] * (patch_altitude[i] * deg2rad).sin();
        }
        for i in 0..n_patches {
            lum_chi[i] = (patch_luminance[i] * rad_d) / rad_tot;
        }
    }

    let (
        lside_sky,
        ldown_sky,
        lside_veg,
        ldown_veg,
        lside_sun,
        lside_sh,
        ldown_sun,
        ldown_sh,
        kside_d,
        kref_sun,
        kref_sh,
        kref_veg,
        least_patch_sum,
        lsouth_patch_sum,
        lwest_patch_sum,
        lnorth_patch_sum,
    ) = (0..n_patches)
        .into_par_iter()
        .map(|i| {
            let mut lside_sky_p = Array2::<f32>::zeros((rows, cols));
            let mut ldown_sky_p = Array2::<f32>::zeros((rows, cols));
            let mut lside_veg_p = Array2::<f32>::zeros((rows, cols));
            let mut ldown_veg_p = Array2::<f32>::zeros((rows, cols));
            let mut lside_sun_p = Array2::<f32>::zeros((rows, cols));
            let mut lside_sh_p = Array2::<f32>::zeros((rows, cols));
            let mut ldown_sun_p = Array2::<f32>::zeros((rows, cols));
            let mut ldown_sh_p = Array2::<f32>::zeros((rows, cols));
            let mut kside_d_p = Array2::<f32>::zeros((rows, cols));
            let mut kref_sun_p = Array2::<f32>::zeros((rows, cols));
            let mut kref_sh_p = Array2::<f32>::zeros((rows, cols));
            let mut kref_veg_p = Array2::<f32>::zeros((rows, cols));
            let mut least_p = Array2::<f32>::zeros((rows, cols));
            let mut lsouth_p = Array2::<f32>::zeros((rows, cols));
            let mut lwest_p = Array2::<f32>::zeros((rows, cols));
            let mut lnorth_p = Array2::<f32>::zeros((rows, cols));

            let temp_sky_bool = shmat.slice(s![.., .., i]).mapv(|v| v == 1.0)
                & vegshmat.slice(s![.., .., i]).mapv(|v| v == 1.0);
            let temp_vegsh_bool = vegshmat.slice(s![.., .., i]).mapv(|v| v == 0.0)
                | vbshvegshmat.slice(s![.., .., i]).mapv(|v| v == 0.0);
            let temp_vbsh_bool = (shmat.slice(s![.., .., i]).mapv(|v| 1.0 - v)
                * vbshvegshmat.slice(s![.., .., i]))
            .mapv(|v| v == 1.0);
            let temp_sh_bool = temp_vbsh_bool.clone();
            let mut temp_sh_w_bool = Array2::<bool>::default((rows, cols));
            let mut temp_sh_roof_bool = Array2::<bool>::default((rows, cols));
            if wall_scheme {
                temp_sh_w_bool = temp_sh_bool.clone()
                    & voxel_maps
                        .as_ref()
                        .unwrap()
                        .slice(s![.., .., i])
                        .mapv(|v| v > 0.0);
                temp_sh_roof_bool = temp_sh_bool.clone()
                    & voxel_maps
                        .as_ref()
                        .unwrap()
                        .slice(s![.., .., i])
                        .mapv(|v| v == 0.0);
            }
            // Convert all boolean masks to f32 for patch_radiation.rs compatibility
            let temp_sky = temp_sky_bool.mapv(|v| if v { 1.0 } else { 0.0 });
            let temp_vegsh = temp_vegsh_bool.mapv(|v| if v { 1.0 } else { 0.0 });
            let temp_sh = temp_sh_bool.mapv(|v| if v { 1.0 } else { 0.0 });
            let temp_sh_w = temp_sh_w_bool.mapv(|v| if v { 1.0 } else { 0.0 });
            let temp_sh_roof = temp_sh_roof_bool.mapv(|v| if v { 1.0 } else { 0.0 });

            // Estimate sunlit and shaded patches for this patch (per-pixel)
            let mut sunlit_patches = Array2::<bool>::default((rows, cols));
            let mut shaded_patches = Array2::<bool>::default((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    let mut sunlit = ndarray::Array1::from_elem(1, false);
                    let mut shaded = ndarray::Array1::from_elem(1, false);
                    sunlit_shaded_patches::shaded_or_sunlit(
                        solar_altitude,
                        solar_azimuth,
                        &ndarray::Array1::from_elem(1, patch_altitude[i]).view(),
                        &ndarray::Array1::from_elem(1, patch_azimuth[i]).view(),
                        asvf[[r, c]],
                        &mut sunlit.view_mut(),
                        &mut shaded.view_mut(),
                    );
                    sunlit_patches[[r, c]] = sunlit[0];
                    shaded_patches[[r, c]] = shaded[0];
                }
            }

            if cyl {
                let angle_of_incidence = (patch_altitude[i] * deg2rad).cos();
                let angle_of_incidence_h = (patch_altitude[i] * deg2rad).sin();

                // Longwave from sky
                let (
                    lside_sky_temp,
                    ldown_sky_temp,
                    least_temp,
                    lsouth_temp,
                    lwest_temp,
                    lnorth_temp,
                ) = patch_radiation::longwave_from_sky(
                    &temp_sky,
                    lside_patch[i],
                    ldown_patch[i],
                    patch_azimuth[i],
                );
                lside_sky_p = lside_sky_temp;
                ldown_sky_p = ldown_sky_temp;
                least_p.zip_mut_with(&least_temp, |a, &b| *a += b);
                lsouth_p.zip_mut_with(&lsouth_temp, |a, &b| *a += b);
                lwest_p.zip_mut_with(&lwest_temp, |a, &b| *a += b);
                lnorth_p.zip_mut_with(&lnorth_temp, |a, &b| *a += b);

                // Longwave from vegetation
                let (
                    lside_veg_temp,
                    ldown_veg_temp,
                    least_temp,
                    lsouth_temp,
                    lwest_temp,
                    lnorth_temp,
                ) = patch_radiation::longwave_from_veg(
                    &temp_vegsh,
                    steradians[i],
                    angle_of_incidence,
                    angle_of_incidence_h,
                    patch_altitude[i],
                    patch_azimuth[i],
                    ewall,
                    ta,
                );
                lside_veg_p = lside_veg_temp;
                ldown_veg_p = ldown_veg_temp;
                least_p.zip_mut_with(&least_temp, |a, &b| *a += b);
                lsouth_p.zip_mut_with(&lsouth_temp, |a, &b| *a += b);
                lwest_p.zip_mut_with(&lwest_temp, |a, &b| *a += b);
                lnorth_p.zip_mut_with(&lnorth_temp, |a, &b| *a += b);

                // Longwave from buildings
                if !wall_scheme {
                    let azimuth_difference = (solar_azimuth - patch_azimuth[i]).abs();
                    let (
                        lside_sun_temp,
                        lside_sh_temp,
                        ldown_sun_temp,
                        ldown_sh_temp,
                        least_temp,
                        lsouth_temp,
                        lwest_temp,
                        lnorth_temp,
                    ) = patch_radiation::longwave_from_buildings(
                        &temp_sh,
                        steradians[i],
                        angle_of_incidence,
                        angle_of_incidence_h,
                        patch_azimuth[i],
                        &sunlit_patches,
                        &shaded_patches,
                        azimuth_difference,
                        solar_altitude,
                        ewall,
                        ta,
                        tgwall,
                    );
                    lside_sun_p = lside_sun_temp;
                    lside_sh_p = lside_sh_temp;
                    ldown_sun_p = ldown_sun_temp;
                    ldown_sh_p = ldown_sh_temp;
                    least_p.zip_mut_with(&least_temp, |a, &b| *a += b);
                    lsouth_p.zip_mut_with(&lsouth_temp, |a, &b| *a += b);
                    lwest_p.zip_mut_with(&lwest_temp, |a, &b| *a += b);
                    lnorth_p.zip_mut_with(&lnorth_temp, |a, &b| *a += b);
                } else {
                    let (
                        lside_sun_temp,
                        lside_sh_temp,
                        ldown_sun_temp,
                        ldown_sh_temp,
                        least_temp,
                        lsouth_temp,
                        lwest_temp,
                        lnorth_temp,
                    ) = patch_radiation::longwave_from_buildings_wall_scheme(
                        &temp_sh_w,
                        &voxel_table.as_ref().unwrap().to_owned(),
                        steradians[i],
                        angle_of_incidence,
                        angle_of_incidence_h,
                        patch_azimuth[i],
                    );
                    let azimuth_difference = (solar_azimuth - patch_azimuth[i]).abs();
                    let (
                        lside_sun_r_temp,
                        lside_sh_r_temp,
                        ldown_sun_r_temp,
                        ldown_sh_r_temp,
                        least_r_temp,
                        lsouth_r_temp,
                        lwest_r_temp,
                        lnorth_r_temp,
                    ) = patch_radiation::longwave_from_buildings(
                        &temp_sh_roof,
                        steradians[i],
                        angle_of_incidence,
                        angle_of_incidence_h,
                        patch_azimuth[i],
                        &sunlit_patches,
                        &shaded_patches,
                        azimuth_difference,
                        solar_altitude,
                        ewall,
                        ta,
                        tgwall,
                    );
                    lside_sun_p = &lside_sun_temp + &lside_sun_r_temp;
                    lside_sh_p = &lside_sh_temp + &lside_sh_r_temp;
                    ldown_sun_p = &ldown_sun_temp + &ldown_sun_r_temp;
                    ldown_sh_p = &ldown_sh_temp + &ldown_sh_r_temp;
                    least_p.zip_mut_with(&(&least_temp + &least_r_temp), |a, &b| *a += b);
                    lsouth_p.zip_mut_with(&(&lsouth_temp + &lsouth_r_temp), |a, &b| *a += b);
                    lwest_p.zip_mut_with(&(&lwest_temp + &lwest_r_temp), |a, &b| *a += b);
                    lnorth_p.zip_mut_with(&(&lnorth_temp + &lnorth_r_temp), |a, &b| *a += b);
                }

                // Shortwave from sky
                if solar_altitude > 0.0 {
                    kside_d_p.zip_mut_with(&temp_sky, |kd, &ts| {
                        *kd += ts * lum_chi[i] * angle_of_incidence * steradians[i]
                    });
                    // Reflected shortwave
                    let sunlit_surface =
                        (albedo * (rad_i * (solar_altitude * deg2rad).cos()) + (rad_d * 0.5)) / PI;
                    let shaded_surface = (albedo * rad_d * 0.5) / PI;
                    kref_veg_p.zip_mut_with(&temp_vegsh, |kv, &tv| {
                        *kv += tv * shaded_surface * steradians[i] * angle_of_incidence
                    });
                    for r in 0..rows {
                        for c in 0..cols {
                            if sunlit_patches[[r, c]] {
                                kref_sun_p[[r, c]] += temp_sh[[r, c]]
                                    * sunlit_surface
                                    * steradians[i]
                                    * angle_of_incidence;
                            }
                            if shaded_patches[[r, c]] {
                                kref_sh_p[[r, c]] += temp_sh[[r, c]]
                                    * shaded_surface
                                    * steradians[i]
                                    * angle_of_incidence;
                            }
                        }
                    }
                }
            }
            (
                lside_sky_p,
                ldown_sky_p,
                lside_veg_p,
                ldown_veg_p,
                lside_sun_p,
                lside_sh_p,
                ldown_sun_p,
                ldown_sh_p,
                kside_d_p,
                kref_sun_p,
                kref_sh_p,
                kref_veg_p,
                least_p,
                lsouth_p,
                lwest_p,
                lnorth_p,
            )
        })
        .reduce(
            || {
                (
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                    Array2::<f32>::zeros((rows, cols)),
                )
            },
            |mut a, b| {
                a.0.zip_mut_with(&b.0, |x, &y| *x += y);
                a.1.zip_mut_with(&b.1, |x, &y| *x += y);
                a.2.zip_mut_with(&b.2, |x, &y| *x += y);
                a.3.zip_mut_with(&b.3, |x, &y| *x += y);
                a.4.zip_mut_with(&b.4, |x, &y| *x += y);
                a.5.zip_mut_with(&b.5, |x, &y| *x += y);
                a.6.zip_mut_with(&b.6, |x, &y| *x += y);
                a.7.zip_mut_with(&b.7, |x, &y| *x += y);
                a.8.zip_mut_with(&b.8, |x, &y| *x += y);
                a.9.zip_mut_with(&b.9, |x, &y| *x += y);
                a.10.zip_mut_with(&b.10, |x, &y| *x += y);
                a.11.zip_mut_with(&b.11, |x, &y| *x += y);
                a.12.zip_mut_with(&b.12, |x, &y| *x += y);
                a.13.zip_mut_with(&b.13, |x, &y| *x += y);
                a.14.zip_mut_with(&b.14, |x, &y| *x += y);
                a.15.zip_mut_with(&b.15, |x, &y| *x += y);
                a
            },
        );

    least.zip_mut_with(&least_patch_sum, |a, &b| *a += b);
    lsouth.zip_mut_with(&lsouth_patch_sum, |a, &b| *a += b);
    lwest.zip_mut_with(&lwest_patch_sum, |a, &b| *a += b);
    lnorth.zip_mut_with(&lnorth_patch_sum, |a, &b| *a += b);

    // Calculate reflected longwave in each patch
    for i in 0..n_patches {
        let angle_of_incidence = (patch_altitude[i] * deg2rad).cos();
        let angle_of_incidence_h = (patch_altitude[i] * deg2rad).sin();
        let temp_sh_bool = shmat.slice(s![.., .., i]).mapv(|v| v == 0.0)
            | vegshmat.slice(s![.., .., i]).mapv(|v| v == 0.0)
            | vbshvegshmat.slice(s![.., .., i]).mapv(|v| v == 0.0);
        let temp_sh = temp_sh_bool.mapv(|v| if v { 1.0 } else { 0.0 });
        let (lside_ref_temp, ldown_ref_temp, least_temp, lsouth_temp, lwest_temp, lnorth_temp) =
            patch_radiation::reflected_longwave(
                &temp_sh,
                steradians[i],
                angle_of_incidence,
                angle_of_incidence_h,
                patch_azimuth[i],
                &ldown_sky,
                &lup.to_owned(),
                ewall,
            );
        lside_ref = &lside_ref + &lside_ref_temp;
        ldown_ref = &ldown_ref + &ldown_ref_temp;
        least = &least + &least_temp;
        lsouth = &lsouth + &lsouth_temp;
        lwest = &lwest + &lwest_temp;
        lnorth = &lnorth + &lnorth_temp;
    }

    // Sum of all Lside components (sky, vegetation, sunlit and shaded buildings, reflected)
    let lside = &lside_sky + &lside_veg + &lside_sh + &lside_sun + &lside_ref;
    let ldown = &ldown_sky + &ldown_veg + &ldown_sh + &ldown_sun + &ldown_ref;

    // Direct radiation
    let mut kside_i = Array2::<f32>::zeros((rows, cols));
    if cyl {
        kside_i = &shadow * rad_i * (solar_altitude * deg2rad).cos();
    }
    let mut kside = Array2::<f32>::zeros((rows, cols));
    if solar_altitude > 0.0 {
        kside = &kside_i + &kside_d + &kref_sun + &kref_sh + &kref_veg;
        keast = &kup_e * 0.5;
        kwest = &kup_w * 0.5;
        knorth = &kup_n * 0.5;
        ksouth = &kup_s * 0.5;
    }

    let result = SkyResult {
        ldown: ldown.into_pyarray(py).unbind(),
        lside: lside.into_pyarray(py).unbind(),
        lside_sky: lside_sky.into_pyarray(py).unbind(),
        lside_veg: lside_veg.into_pyarray(py).unbind(),
        lside_sh: lside_sh.into_pyarray(py).unbind(),
        lside_sun: lside_sun.into_pyarray(py).unbind(),
        lside_ref: lside_ref.into_pyarray(py).unbind(),
        least: least.into_pyarray(py).unbind(),
        lwest: lwest.into_pyarray(py).unbind(),
        lnorth: lnorth.into_pyarray(py).unbind(),
        lsouth: lsouth.into_pyarray(py).unbind(),
        keast: keast.into_pyarray(py).unbind(),
        ksouth: ksouth.into_pyarray(py).unbind(),
        kwest: kwest.into_pyarray(py).unbind(),
        knorth: knorth.into_pyarray(py).unbind(),
        kside_i: kside_i.into_pyarray(py).unbind(),
        kside_d: kside_d.into_pyarray(py).unbind(),
        kside: kside.into_pyarray(py).unbind(),
        steradians: steradians.mapv(|v| v as f32).into_pyarray(py).unbind(),
        skyalt: skyalt.into_pyarray(py).unbind(),
    };
    Ok(Py::new(py, result)?)
}
