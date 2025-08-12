use crate::{emissivity_models, patch_radiation, sunlit_shaded_patches};
use ndarray::{Array1, Array2};
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

// Struct to hold the accumulated radiation values for a single pixel
#[derive(Clone, Copy)]
struct PixelResult {
    lside_sky: f32,
    ldown_sky: f32,
    lside_veg: f32,
    ldown_veg: f32,
    lside_sun: f32,
    lside_sh: f32,
    ldown_sun: f32,
    ldown_sh: f32,
    kside_d: f32,
    kref_sun: f32,
    kref_sh: f32,
    kref_veg: f32,
    least: f32,
    lsouth: f32,
    lwest: f32,
    lnorth: f32,
    lside_ref: f32,
    ldown_ref: f32,
}

impl PixelResult {
    fn new() -> Self {
        Self {
            lside_sky: 0.0,
            ldown_sky: 0.0,
            lside_veg: 0.0,
            ldown_veg: 0.0,
            lside_sun: 0.0,
            lside_sh: 0.0,
            ldown_sun: 0.0,
            ldown_sh: 0.0,
            kside_d: 0.0,
            kref_sun: 0.0,
            kref_sh: 0.0,
            kref_veg: 0.0,
            least: 0.0,
            lsouth: 0.0,
            lwest: 0.0,
            lnorth: 0.0,
            lside_ref: 0.0,
            ldown_ref: 0.0,
        }
    }
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

    // Output arrays
    let mut lside_sky = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sky = Array2::<f32>::zeros((rows, cols));
    let mut lside_veg = Array2::<f32>::zeros((rows, cols));
    let mut ldown_veg = Array2::<f32>::zeros((rows, cols));
    let mut lside_sun = Array2::<f32>::zeros((rows, cols));
    let mut lside_sh = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sun = Array2::<f32>::zeros((rows, cols));
    let mut ldown_sh = Array2::<f32>::zeros((rows, cols));
    let mut kside_d = Array2::<f32>::zeros((rows, cols));
    let mut kref_sun = Array2::<f32>::zeros((rows, cols));
    let mut kref_sh = Array2::<f32>::zeros((rows, cols));
    let mut kref_veg = Array2::<f32>::zeros((rows, cols));
    let mut least = Array2::<f32>::zeros((rows, cols));
    let mut lwest = Array2::<f32>::zeros((rows, cols));
    let mut lnorth = Array2::<f32>::zeros((rows, cols));
    let mut lsouth = Array2::<f32>::zeros((rows, cols));
    let mut lside_ref = Array2::<f32>::zeros((rows, cols));
    let mut ldown_ref = Array2::<f32>::zeros((rows, cols));

    // Patch altitudes and azimuths
    let patch_altitude = l_patches.column(0).to_owned();
    let patch_azimuth = l_patches.column(1).to_owned();

    // Calculate unique altitudes for returning from function
    let mut skyalt_vec: Vec<f32> = patch_altitude.iter().cloned().collect();
    skyalt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    skyalt_vec.dedup();
    let skyalt = Array1::<f32>::from(skyalt_vec);

    let deg2rad = PI / 180.0;

    // Shortwave normalization
    let mut lum_chi = Array1::<f32>::zeros(n_patches);
    if solar_altitude > 0.0 {
        let patch_luminance = lv.column(2);
        let mut rad_tot = 0.0;
        for i in 0..n_patches {
            rad_tot += patch_luminance[i] * steradians[i] * (patch_altitude[i] * deg2rad).sin();
        }
        lum_chi = patch_luminance.mapv(|lum| (lum * rad_d) / rad_tot);
    }

    // Precompute emissivity per patch
    let (_patch_emissivity_normalized, esky_band) =
        emissivity_models::model2(&l_patches.to_owned(), esky, ta);

    // Create a flat list of pixel indices to parallelize over
    let pixel_indices: Vec<(usize, usize)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .collect();

    // Main parallel computation over pixels
    let pixel_results: Vec<PixelResult> = pixel_indices
        .into_par_iter()
        .map(|(r, c)| {
            let mut pres = PixelResult::new();
            let pixel_asvf = asvf[[r, c]];

            for i in 0..n_patches {
                let p_alt = patch_altitude[i];
                let p_azi = patch_azimuth[i];
                let steradian = steradians[i];

                let temp_sky = shmat[[r, c, i]] == 1.0 && vegshmat[[r, c, i]] == 1.0;
                let temp_vegsh = vegshmat[[r, c, i]] == 0.0 || vbshvegshmat[[r, c, i]] == 0.0;
                let temp_sh = (1.0 - shmat[[r, c, i]]) * vbshvegshmat[[r, c, i]] == 1.0;

                if cyl {
                    let angle_of_incidence = (p_alt * deg2rad).cos();
                    let angle_of_incidence_h = (p_alt * deg2rad).sin();

                    // Longwave from sky
                    if temp_sky {
                        let temp_emissivity = esky_band[i];
                        let ta_k = ta + 273.15;
                        let lval = (temp_emissivity * SBC * ta_k.powi(4)) / PI;
                        let lside_patch = lval * steradian * angle_of_incidence;
                        let ldown_patch = lval * steradian * angle_of_incidence_h;

                        let (ls, ld, le, lso, lw, ln) = patch_radiation::longwave_from_sky_pixel(
                            lside_patch,
                            ldown_patch,
                            p_azi,
                        );
                        pres.lside_sky += ls;
                        pres.ldown_sky += ld;
                        pres.least += le;
                        pres.lsouth += lso;
                        pres.lwest += lw;
                        pres.lnorth += ln;
                    }

                    // Longwave from vegetation
                    if temp_vegsh {
                        let (ls, ld, le, lso, lw, ln) = patch_radiation::longwave_from_veg_pixel(
                            steradian,
                            angle_of_incidence,
                            angle_of_incidence_h,
                            p_alt,
                            p_azi,
                            ewall,
                            ta,
                        );
                        pres.lside_veg += ls;
                        pres.ldown_veg += ld;
                        pres.least += le;
                        pres.lsouth += lso;
                        pres.lwest += lw;
                        pres.lnorth += ln;
                    }

                    // Longwave from buildings
                    if temp_sh {
                        let (sunlit_patch, shaded_patch) =
                            sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                solar_altitude,
                                solar_azimuth,
                                p_alt,
                                p_azi,
                                pixel_asvf,
                            );

                        if !wall_scheme {
                            let azimuth_difference = (solar_azimuth - p_azi).abs();
                            let (lside_sun, lside_sh, ldown_sun, ldown_sh, le, lso, lw, ln) =
                                patch_radiation::longwave_from_buildings_pixel(
                                    steradian,
                                    angle_of_incidence,
                                    angle_of_incidence_h,
                                    p_azi,
                                    sunlit_patch,
                                    shaded_patch,
                                    azimuth_difference,
                                    solar_altitude,
                                    ewall,
                                    ta,
                                    tgwall,
                                );
                            pres.lside_sun += lside_sun;
                            pres.lside_sh += lside_sh;
                            pres.ldown_sun += ldown_sun;
                            pres.ldown_sh += ldown_sh;
                            pres.least += le;
                            pres.lsouth += lso;
                            pres.lwest += lw;
                            pres.lnorth += ln;
                        } else {
                            let voxel_map_val = voxel_maps.as_ref().unwrap()[[r, c, i]];
                            if voxel_map_val > 0.0 {
                                // Wall
                                let (lside_sun, lside_sh, ldown_sun, ldown_sh, le, lso, lw, ln) =
                                    patch_radiation::longwave_from_buildings_wall_scheme_pixel(
                                        voxel_table.as_ref().unwrap().view(),
                                        voxel_map_val as usize,
                                        steradian,
                                        angle_of_incidence,
                                        angle_of_incidence_h,
                                        p_azi,
                                    );
                                pres.lside_sun += lside_sun;
                                pres.lside_sh += lside_sh;
                                pres.ldown_sun += ldown_sun;
                                pres.ldown_sh += ldown_sh;
                                pres.least += le;
                                pres.lsouth += lso;
                                pres.lwest += lw;
                                pres.lnorth += ln;
                            } else {
                                // Roof
                                let azimuth_difference = (solar_azimuth - p_azi).abs();
                                let (lside_sun, lside_sh, ldown_sun, ldown_sh, le, lso, lw, ln) =
                                    patch_radiation::longwave_from_buildings_pixel(
                                        steradian,
                                        angle_of_incidence,
                                        angle_of_incidence_h,
                                        p_azi,
                                        sunlit_patch,
                                        shaded_patch,
                                        azimuth_difference,
                                        solar_altitude,
                                        ewall,
                                        ta,
                                        tgwall,
                                    );
                                pres.lside_sun += lside_sun;
                                pres.lside_sh += lside_sh;
                                pres.ldown_sun += ldown_sun;
                                pres.ldown_sh += ldown_sh;
                                pres.least += le;
                                pres.lsouth += lso;
                                pres.lwest += lw;
                                pres.lnorth += ln;
                            }
                        }
                    }

                    // Shortwave from sky
                    if solar_altitude > 0.0 {
                        if temp_sky {
                            pres.kside_d += lum_chi[i] * angle_of_incidence * steradian;
                        }
                        let sunlit_surface = (albedo * (rad_i * (solar_altitude * deg2rad).cos())
                            + (rad_d * 0.5))
                            / PI;
                        let shaded_surface = (albedo * rad_d * 0.5) / PI;
                        if temp_vegsh {
                            pres.kref_veg += shaded_surface * steradian * angle_of_incidence;
                        }
                        if temp_sh {
                            let (sunlit_patch, shaded_patch) =
                                sunlit_shaded_patches::shaded_or_sunlit_pixel(
                                    solar_altitude,
                                    solar_azimuth,
                                    p_alt,
                                    p_azi,
                                    pixel_asvf,
                                );
                            if sunlit_patch {
                                pres.kref_sun += sunlit_surface * steradian * angle_of_incidence;
                            }
                            if shaded_patch {
                                pres.kref_sh += shaded_surface * steradian * angle_of_incidence;
                            }
                        }
                    }
                }
            }

            // Reflected longwave calculation (loop over patches again for this pixel)
            let mut pres_with_reflection = pres;
            for i in 0..n_patches {
                let p_alt = patch_altitude[i];
                let p_azi = patch_azimuth[i];
                let steradian = steradians[i];
                let temp_sh = shmat[[r, c, i]] == 0.0
                    || vegshmat[[r, c, i]] == 0.0
                    || vbshvegshmat[[r, c, i]] == 0.0;

                if temp_sh {
                    let angle_of_incidence = (p_alt * deg2rad).cos();
                    let angle_of_incidence_h = (p_alt * deg2rad).sin();
                    let (lsr, ldr, le, lso, lw, ln) = patch_radiation::reflected_longwave_pixel(
                        steradian,
                        angle_of_incidence,
                        angle_of_incidence_h,
                        p_azi,
                        pres.ldown_sky,
                        lup[[r, c]],
                        ewall,
                    );
                    pres_with_reflection.lside_ref += lsr;
                    pres_with_reflection.ldown_ref += ldr;
                    pres_with_reflection.least += le;
                    pres_with_reflection.lsouth += lso;
                    pres_with_reflection.lwest += lw;
                    pres_with_reflection.lnorth += ln;
                }
            }
            pres_with_reflection
        })
        .collect();

    // Populate the final 2D arrays from the results
    for (idx, pres) in pixel_results.into_iter().enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        lside_sky[[r, c]] = pres.lside_sky;
        ldown_sky[[r, c]] = pres.ldown_sky;
        lside_veg[[r, c]] = pres.lside_veg;
        ldown_veg[[r, c]] = pres.ldown_veg;
        lside_sun[[r, c]] = pres.lside_sun;
        lside_sh[[r, c]] = pres.lside_sh;
        ldown_sun[[r, c]] = pres.ldown_sun;
        ldown_sh[[r, c]] = pres.ldown_sh;
        kside_d[[r, c]] = pres.kside_d;
        kref_sun[[r, c]] = pres.kref_sun;
        kref_sh[[r, c]] = pres.kref_sh;
        kref_veg[[r, c]] = pres.kref_veg;
        least[[r, c]] = pres.least;
        lsouth[[r, c]] = pres.lsouth;
        lwest[[r, c]] = pres.lwest;
        lnorth[[r, c]] = pres.lnorth;
        lside_ref[[r, c]] = pres.lside_ref;
        ldown_ref[[r, c]] = pres.ldown_ref;
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
    let mut keast = Array2::<f32>::zeros((rows, cols));
    let mut kwest = Array2::<f32>::zeros((rows, cols));
    let mut knorth = Array2::<f32>::zeros((rows, cols));
    let mut ksouth = Array2::<f32>::zeros((rows, cols));

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
