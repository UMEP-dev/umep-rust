//! Fused timestep pipeline — single FFI entrance/exit per timestep.
//!
//! Orchestrates: shadows → ground_temp → GVF → thermal_delay → radiation → Tmrt
//! All intermediate arrays stay as ndarray::Array2<f32> — never cross FFI boundary.
//!
//! Supports both isotropic and anisotropic (Perez) sky models.

use ndarray::{Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::ground::{
    compute_ground_temperature_pure, ts_wave_delay_batch_pure, GroundTempResult,
};
use crate::gvf::{gvf_calc_pure, gvf_calc_with_cache, GvfResultPure};
use crate::gvf_geometry::{precompute_gvf_geometry, GvfGeometryCache};
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};
use crate::sky::{anisotropic_sky_pure, cylindric_wedge_pure_masked, weighted_patch_sum_pure};
use crate::tmrt::compute_tmrt_pure;
use crate::vegetation::{kside_veg_isotropic_pure, lside_veg_pure};

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67e-8;
const KELVIN_OFFSET: f32 = 273.15;

// ── Input structs (created once in Python, passed by reference) ───────────

/// Weather scalars for a single timestep.
#[pyclass]
#[derive(Clone)]
pub struct WeatherScalars {
    #[pyo3(get, set)]
    pub sun_azimuth: f32,
    #[pyo3(get, set)]
    pub sun_altitude: f32,
    #[pyo3(get, set)]
    pub sun_zenith: f32,
    #[pyo3(get, set)]
    pub ta: f32,
    #[pyo3(get, set)]
    pub rh: f32,
    #[pyo3(get, set)]
    pub global_rad: f32,
    #[pyo3(get, set)]
    pub direct_rad: f32,
    #[pyo3(get, set)]
    pub diffuse_rad: f32,
    #[pyo3(get, set)]
    pub altmax: f32,
    #[pyo3(get, set)]
    pub clearness_index: f32,
    #[pyo3(get, set)]
    pub dectime: f32,
    #[pyo3(get, set)]
    pub snup: f32,
    #[pyo3(get, set)]
    pub rad_g0: f32,
    #[pyo3(get, set)]
    pub zen_deg: f32,
    #[pyo3(get, set)]
    pub psi: f32,
    #[pyo3(get, set)]
    pub is_daytime: bool,
}

#[pymethods]
impl WeatherScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sun_azimuth: f32,
        sun_altitude: f32,
        sun_zenith: f32,
        ta: f32,
        rh: f32,
        global_rad: f32,
        direct_rad: f32,
        diffuse_rad: f32,
        altmax: f32,
        clearness_index: f32,
        dectime: f32,
        snup: f32,
        rad_g0: f32,
        zen_deg: f32,
        psi: f32,
        is_daytime: bool,
    ) -> Self {
        Self {
            sun_azimuth,
            sun_altitude,
            sun_zenith,
            ta,
            rh,
            global_rad,
            direct_rad,
            diffuse_rad,
            altmax,
            clearness_index,
            dectime,
            snup,
            rad_g0,
            zen_deg,
            psi,
            is_daytime,
        }
    }
}

/// Human body parameters.
#[pyclass]
#[derive(Clone)]
pub struct HumanScalars {
    #[pyo3(get, set)]
    pub height: f32,
    #[pyo3(get, set)]
    pub abs_k: f32,
    #[pyo3(get, set)]
    pub abs_l: f32,
    #[pyo3(get, set)]
    pub is_standing: bool,
}

#[pymethods]
impl HumanScalars {
    #[new]
    pub fn new(height: f32, abs_k: f32, abs_l: f32, is_standing: bool) -> Self {
        Self {
            height,
            abs_k,
            abs_l,
            is_standing,
        }
    }
}

/// Configuration scalars (constant across timesteps).
#[pyclass]
#[derive(Clone)]
pub struct ConfigScalars {
    #[pyo3(get, set)]
    pub pixel_size: f32,
    #[pyo3(get, set)]
    pub max_height: f32,
    #[pyo3(get, set)]
    pub albedo_wall: f32,
    #[pyo3(get, set)]
    pub emis_wall: f32,
    #[pyo3(get, set)]
    pub tgk_wall: f32,
    #[pyo3(get, set)]
    pub tstart_wall: f32,
    #[pyo3(get, set)]
    pub tmaxlst_wall: f32,
    #[pyo3(get, set)]
    pub use_veg: bool,
    #[pyo3(get, set)]
    pub has_walls: bool,
    #[pyo3(get, set)]
    pub conifer: bool,
    #[pyo3(get, set)]
    pub use_anisotropic: bool,
}

#[pymethods]
impl ConfigScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pixel_size: f32,
        max_height: f32,
        albedo_wall: f32,
        emis_wall: f32,
        tgk_wall: f32,
        tstart_wall: f32,
        tmaxlst_wall: f32,
        use_veg: bool,
        has_walls: bool,
        conifer: bool,
        use_anisotropic: bool,
    ) -> Self {
        Self {
            pixel_size,
            max_height,
            albedo_wall,
            emis_wall,
            tgk_wall,
            tstart_wall,
            tmaxlst_wall,
            use_veg,
            has_walls,
            conifer,
            use_anisotropic,
        }
    }
}

// ── Output struct ──────────────────────────────────────────────────────────

/// Result from a single fused timestep.
#[pyclass]
pub struct TimestepResult {
    #[pyo3(get)]
    pub tmrt: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub shadow: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kdown: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub kup: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub ldown: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lup: Py<PyArray2<f32>>,
    // Updated thermal state arrays (Python extracts and passes back next timestep)
    #[pyo3(get)]
    pub timeadd: f32,
    #[pyo3(get)]
    pub tgmap1: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgout1: Py<PyArray2<f32>>,
}

// ── Radiation helpers (ported from Python physics) ─────────────────────────

/// Compute sky emissivity (Jonsson et al. 2006).
#[inline]
fn compute_esky(ta: f32, rh: f32) -> f32 {
    let ta_k = ta + KELVIN_OFFSET;
    let ea = 6.107 * 10.0_f32.powf((7.5 * ta) / (237.3 + ta)) * (rh / 100.0);
    let msteg = 46.5 * (ea / ta_k);
    1.0 - (1.0 + msteg) * (-((1.2 + 3.0 * msteg) as f32).sqrt()).exp()
}

/// Compute Kup (ground-reflected shortwave) — Kup_veg_2015a.
///
/// Returns (kup, kup_e, kup_s, kup_w, kup_n) as owned arrays.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn compute_kup(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    altitude: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_b: f32,
    f_sh: ArrayView2<f32>,
    gvfalb: ArrayView2<f32>,
    gvfalb_e: ArrayView2<f32>,
    gvfalb_s: ArrayView2<f32>,
    gvfalb_w: ArrayView2<f32>,
    gvfalb_n: ArrayView2<f32>,
    gvfalbnosh: ArrayView2<f32>,
    gvfalbnosh_e: ArrayView2<f32>,
    gvfalbnosh_s: ArrayView2<f32>,
    gvfalbnosh_w: ArrayView2<f32>,
    gvfalbnosh_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let rad_i_sin_alt = rad_i * (altitude * PI / 180.0).sin();

    // common_term = radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)
    let shape = svfbuveg.dim();
    let mut common_term = Array2::<f32>::zeros(shape);
    let mut kup = Array2::<f32>::zeros(shape);
    let mut kup_e = Array2::<f32>::zeros(shape);
    let mut kup_s = Array2::<f32>::zeros(shape);
    let mut kup_w = Array2::<f32>::zeros(shape);
    let mut kup_n = Array2::<f32>::zeros(shape);

    // Compute in a single pass for cache efficiency
    let ncols = shape.1;
    for idx in 0..shape.0 * shape.1 {
        let r = idx / ncols;
        let c = idx % ncols;
        if valid[[r, c]] == 0 {
            kup[[r, c]] = f32::NAN;
            kup_e[[r, c]] = f32::NAN;
            kup_s[[r, c]] = f32::NAN;
            kup_w[[r, c]] = f32::NAN;
            kup_n[[r, c]] = f32::NAN;
            continue;
        }
        let sv = svfbuveg[[r, c]];
        let fsh = f_sh[[r, c]];
        let ct = rad_d * sv + albedo_b * (1.0 - sv) * (rad_g * (1.0 - fsh) + rad_d * fsh);
        common_term[[r, c]] = ct;
        kup[[r, c]] = gvfalb[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh[[r, c]];
        kup_e[[r, c]] = gvfalb_e[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_e[[r, c]];
        kup_s[[r, c]] = gvfalb_s[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_s[[r, c]];
        kup_w[[r, c]] = gvfalb_w[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_w[[r, c]];
        kup_n[[r, c]] = gvfalb_n[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_n[[r, c]];
    }

    (kup, kup_e, kup_s, kup_w, kup_n)
}

/// Compute Ldown (downwelling longwave) — Jonsson et al. 2006.
#[allow(clippy::too_many_arguments)]
fn compute_ldown(
    esky: f32,
    ta: f32,
    tg_wall: f32,
    svf: ArrayView2<f32>,
    svf_veg: ArrayView2<f32>,
    svf_aveg: ArrayView2<f32>,
    emis_wall: f32,
    ci: f32,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let ta_k = ta + KELVIN_OFFSET;
    let ta_k4 = ta_k.powi(4);
    let tg_wall_k4 = (ta + tg_wall + KELVIN_OFFSET).powi(4);
    let shape = svf.dim();
    let mut ldown = Array2::<f32>::zeros(shape);
    let ncols = shape.1;

    for idx in 0..shape.0 * shape.1 {
        let r = idx / ncols;
        let c = idx % ncols;
        if valid[[r, c]] == 0 {
            ldown[[r, c]] = f32::NAN;
            continue;
        }
        let sv = svf[[r, c]];
        let sv_veg = svf_veg[[r, c]];
        let sv_aveg = svf_aveg[[r, c]];

        let val = (sv + sv_veg - 1.0) * esky * SBC * ta_k4
            + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
            + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
            + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * esky * SBC * ta_k4;

        if ci < 0.95 {
            let c_cloud = 1.0 - ci;
            let val_cloudy = (sv + sv_veg - 1.0) * SBC * ta_k4
                + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
                + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
                + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * SBC * ta_k4;
            ldown[[r, c]] = val * (1.0 - c_cloud) + val_cloudy * c_cloud;
        } else {
            ldown[[r, c]] = val;
        }
    }

    ldown
}

/// Compute Kdown (downwelling shortwave).
#[allow(clippy::too_many_arguments)]
fn compute_kdown(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    shadow: ArrayView2<f32>,
    sin_alt: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_wall: f32,
    f_sh: ArrayView2<f32>,
    drad: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let shape = shadow.dim();
    let mut kdown = Array2::<f32>::zeros(shape);
    let ncols = shape.1;

    for idx in 0..shape.0 * shape.1 {
        let r = idx / ncols;
        let c = idx % ncols;
        if valid[[r, c]] == 0 {
            kdown[[r, c]] = f32::NAN;
            continue;
        }
        kdown[[r, c]] = rad_i * shadow[[r, c]] * sin_alt
            + drad[[r, c]]
            + albedo_wall * (1.0 - svfbuveg[[r, c]]) * (rad_g * (1.0 - f_sh[[r, c]]) + rad_d * f_sh[[r, c]]);
    }

    kdown
}

// ── GVF Geometry Cache (opaque handle for Python) ─────────────────────────

/// Opaque handle to a precomputed GVF geometry cache.
///
/// Created once per DSM via `precompute_gvf_cache()`, then passed to
/// `compute_timestep()` on subsequent calls to skip building ray-tracing.
#[pyclass]
pub struct PyGvfGeometryCache {
    pub(crate) inner: GvfGeometryCache,
}

/// Precompute GVF geometry cache for a given set of surface arrays.
///
/// This runs the building ray-trace once (18 azimuths, parallelized).
/// The returned cache is passed to `compute_timestep()` to skip geometry
/// on subsequent timesteps with the same DSM.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn precompute_gvf_cache(
    buildings: PyReadonlyArray2<f32>,
    wall_asp: PyReadonlyArray2<f32>,
    wall_ht: PyReadonlyArray2<f32>,
    alb_grid: PyReadonlyArray2<f32>,
    pixel_size: f32,
    human_height: f32,
    wall_albedo: f32,
) -> PyResult<PyGvfGeometryCache> {
    let first_ht = human_height.round().max(1.0);
    let second_ht = human_height * 20.0;

    let cache = precompute_gvf_geometry(
        buildings.as_array(),
        wall_asp.as_array(),
        wall_ht.as_array(),
        alb_grid.as_array(),
        pixel_size,
        first_ht,
        second_ht,
        wall_albedo,
    );

    Ok(PyGvfGeometryCache { inner: cache })
}

// ── Main fused timestep function ───────────────────────────────────────────

/// Compute a single daytime timestep entirely in Rust.
///
/// All intermediate arrays stay as ndarray::Array2<f32> — only the final
/// results cross back to Python.
///
/// Parameters are grouped into structs to keep the signature manageable:
/// - weather: Per-timestep scalars (sun position, temperature, radiation)
/// - human: Body parameters (height, posture, absorptivities)
/// - config: Constants (pixel_size, wall materials)
/// - Surface/SVF arrays: Borrowed from Python (zero-copy on input)
/// - Thermal state: Carried forward between timesteps
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_timestep(
    py: Python,
    // Scalar parameter structs
    weather: &WeatherScalars,
    human: &HumanScalars,
    config: &ConfigScalars,
    // Optional GVF geometry cache (skip building ray-tracing if provided)
    gvf_cache: Option<&PyGvfGeometryCache>,
    // Surface arrays (constant across timesteps, borrowed)
    dsm: PyReadonlyArray2<f32>,
    cdsm: Option<PyReadonlyArray2<f32>>,
    tdsm: Option<PyReadonlyArray2<f32>>,
    bush: Option<PyReadonlyArray2<f32>>,
    wall_ht: Option<PyReadonlyArray2<f32>>,
    wall_asp: Option<PyReadonlyArray2<f32>>,
    // SVF arrays (constant across timesteps, borrowed)
    svf: PyReadonlyArray2<f32>,
    svf_n: PyReadonlyArray2<f32>,
    svf_e: PyReadonlyArray2<f32>,
    svf_s: PyReadonlyArray2<f32>,
    svf_w: PyReadonlyArray2<f32>,
    svf_veg: PyReadonlyArray2<f32>,
    svf_veg_n: PyReadonlyArray2<f32>,
    svf_veg_e: PyReadonlyArray2<f32>,
    svf_veg_s: PyReadonlyArray2<f32>,
    svf_veg_w: PyReadonlyArray2<f32>,
    svf_aveg: PyReadonlyArray2<f32>,
    svf_aveg_n: PyReadonlyArray2<f32>,
    svf_aveg_e: PyReadonlyArray2<f32>,
    svf_aveg_s: PyReadonlyArray2<f32>,
    svf_aveg_w: PyReadonlyArray2<f32>,
    svfbuveg: PyReadonlyArray2<f32>,
    svfalfa: PyReadonlyArray2<f32>,
    // Land cover property grids (constant across timesteps, borrowed)
    alb_grid: PyReadonlyArray2<f32>,
    emis_grid: PyReadonlyArray2<f32>,
    tgk_grid: PyReadonlyArray2<f32>,
    tstart_grid: PyReadonlyArray2<f32>,
    tmaxlst_grid: PyReadonlyArray2<f32>,
    // Buildings mask for GVF
    buildings: PyReadonlyArray2<f32>,
    lc_grid: Option<PyReadonlyArray2<f32>>,
    // Anisotropic sky inputs (None for isotropic)
    shmat: Option<PyReadonlyArray3<u8>>,
    vegshmat: Option<PyReadonlyArray3<u8>>,
    vbshmat: Option<PyReadonlyArray3<u8>>,
    l_patches: Option<PyReadonlyArray2<f32>>,
    steradians: Option<PyReadonlyArray1<f32>>,
    lv: Option<PyReadonlyArray2<f32>>,
    asvf: Option<PyReadonlyArray2<f32>>,
    esky_aniso: Option<f32>,
    // Thermal state (mutable, updated each timestep)
    firstdaytime: i32,
    timeadd: f32,
    timestep_dec: f32,
    tgmap1: PyReadonlyArray2<f32>,
    tgmap1_e: PyReadonlyArray2<f32>,
    tgmap1_s: PyReadonlyArray2<f32>,
    tgmap1_w: PyReadonlyArray2<f32>,
    tgmap1_n: PyReadonlyArray2<f32>,
    tgout1: PyReadonlyArray2<f32>,
    // Valid pixel mask (1=valid, 0=NaN/nodata — skip computation for invalid pixels)
    valid_mask: PyReadonlyArray2<u8>,
) -> PyResult<TimestepResult> {
    // Borrow all arrays (zero-copy from numpy)
    let valid_v = valid_mask.as_array();
    let dsm_v = dsm.as_array();
    let cdsm_v = cdsm.as_ref().map(|a| a.as_array());
    let tdsm_v = tdsm.as_ref().map(|a| a.as_array());
    let bush_v = bush.as_ref().map(|a| a.as_array());
    let wall_ht_v = wall_ht.as_ref().map(|a| a.as_array());
    let wall_asp_v = wall_asp.as_ref().map(|a| a.as_array());
    let svf_v = svf.as_array();
    let svf_n_v = svf_n.as_array();
    let svf_e_v = svf_e.as_array();
    let svf_s_v = svf_s.as_array();
    let svf_w_v = svf_w.as_array();
    let svf_veg_v = svf_veg.as_array();
    let svf_veg_n_v = svf_veg_n.as_array();
    let svf_veg_e_v = svf_veg_e.as_array();
    let svf_veg_s_v = svf_veg_s.as_array();
    let svf_veg_w_v = svf_veg_w.as_array();
    let svf_aveg_v = svf_aveg.as_array();
    let svf_aveg_n_v = svf_aveg_n.as_array();
    let svf_aveg_e_v = svf_aveg_e.as_array();
    let svf_aveg_s_v = svf_aveg_s.as_array();
    let svf_aveg_w_v = svf_aveg_w.as_array();
    let svfbuveg_v = svfbuveg.as_array();
    let svfalfa_v = svfalfa.as_array();
    let alb_grid_v = alb_grid.as_array();
    let emis_grid_v = emis_grid.as_array();
    let tgk_grid_v = tgk_grid.as_array();
    let tstart_grid_v = tstart_grid.as_array();
    let tmaxlst_grid_v = tmaxlst_grid.as_array();
    let buildings_v = buildings.as_array();
    let lc_grid_v = lc_grid.as_ref().map(|a| a.as_array());
    let tgmap1_v = tgmap1.as_array();
    let tgmap1_e_v = tgmap1_e.as_array();
    let tgmap1_s_v = tgmap1_s.as_array();
    let tgmap1_w_v = tgmap1_w.as_array();
    let tgmap1_n_v = tgmap1_n.as_array();
    let tgout1_v = tgout1.as_array();

    // Borrow anisotropic arrays (if provided)
    let shmat_v = shmat.as_ref().map(|a| a.as_array());
    let vegshmat_v = vegshmat.as_ref().map(|a| a.as_array());
    let vbshmat_v = vbshmat.as_ref().map(|a| a.as_array());
    let l_patches_v = l_patches.as_ref().map(|a| a.as_array());
    let steradians_v = steradians.as_ref().map(|a| a.as_array());
    let lv_v = lv.as_ref().map(|a| a.as_array());
    let asvf_v = asvf.as_ref().map(|a| a.as_array());

    let shape = dsm_v.dim();

    // Wall aspect in radians for shadows
    let wall_asp_rad: Option<Array2<f32>> = wall_asp_v.map(|a| a.mapv(|d| d * PI / 180.0));
    let wall_asp_rad_view = wall_asp_rad.as_ref().map(|a| a.view());

    // ── Step 1: Shadows ──────────────────────────────────────────────────
    let shadow_result: ShadowingResultRust = calculate_shadows_rust(
        weather.sun_azimuth,
        weather.sun_altitude,
        config.pixel_size,
        config.max_height,
        dsm_v,
        if config.use_veg { cdsm_v } else { None },
        if config.use_veg { tdsm_v } else { None },
        if config.use_veg { bush_v } else { None },
        if config.has_walls { wall_ht_v } else { None },
        if config.has_walls { wall_asp_rad_view } else { None },
        None, // walls_scheme
        None, // aspect_scheme
        3.0,  // min_sun_altitude
    );

    // Combine shadows with vegetation transmissivity
    let bldg_sh = &shadow_result.bldg_sh;
    let shadow = if config.use_veg {
        let veg_sh = &shadow_result.veg_sh;
        bldg_sh - &((1.0 - veg_sh) * (1.0 - weather.psi))
    } else {
        bldg_sh.clone()
    };
    let shadow_f32 = shadow.mapv(|v| v as f32);

    let wallsun = shadow_result
        .wall_sun
        .unwrap_or_else(|| Array2::zeros(shape));

    // ── Step 2: Ground Temperature ───────────────────────────────────────
    let ground: GroundTempResult = compute_ground_temperature_pure(
        weather.sun_altitude,
        weather.altmax,
        weather.dectime,
        weather.snup,
        weather.global_rad,
        weather.rad_g0,
        weather.zen_deg,
        tgk_grid_v,
        tstart_grid_v,
        tmaxlst_grid_v,
        config.tgk_wall,
        config.tstart_wall,
        config.tmaxlst_wall,
    );

    // ── Step 3: GVF ─────────────────────────────────────────────────────
    let first = {
        let h = human.height.round();
        if h == 0.0 { 1.0 } else { h }
    };
    let second = (human.height * 20.0).round();

    let gvf: GvfResultPure = if config.has_walls {
        if let Some(cache) = gvf_cache {
            // Use cached geometry — thermal-only pass
            gvf_calc_with_cache(
                &cache.inner,
                wallsun.view(),
                buildings_v,
                shadow_f32.view(),
                ground.tg.view(),
                ground.tg_wall,
                weather.ta,
                emis_grid_v,
                config.emis_wall,
                alb_grid_v,
                SBC,
                config.albedo_wall,
                weather.ta, // twater = ta
                lc_grid_v,
                lc_grid_v.is_some(),
            )
        } else {
            // Full GVF (first timestep or no cache)
            let wh = wall_ht_v.unwrap();
            gvf_calc_pure(
                wallsun.view(),
                wh,
                buildings_v,
                config.pixel_size,
                shadow_f32.view(),
                first,
                second,
                wall_asp_v.unwrap(),
                ground.tg.view(),
                ground.tg_wall,
                weather.ta,
                emis_grid_v,
                config.emis_wall,
                alb_grid_v,
                SBC,
                config.albedo_wall,
                weather.ta, // twater = ta
                lc_grid_v,
                lc_grid_v.is_some(),
            )
        }
    } else {
        // Simplified GVF (no walls) - compute inline
        let gvf_simple = 1.0 - &svf_v;
        let tg_with_shadow = &ground.tg * &shadow_f32;
        // Lup = emis × SBC × (Ta + Tg_shadow + 273.15)^4
        let lup_simple = {
            let ncols = shape.1;
            let mut arr = Array2::<f32>::zeros(shape);
            for idx in 0..shape.0 * shape.1 {
                let r = idx / ncols;
                let c = idx % ncols;
                if valid_v[[r, c]] == 0 {
                    arr[[r, c]] = f32::NAN;
                    continue;
                }
                let t = weather.ta + tg_with_shadow[[r, c]] + KELVIN_OFFSET;
                arr[[r, c]] = emis_grid_v[[r, c]] * SBC * t.powi(4);
            }
            arr
        };
        let gvfalb_simple = &alb_grid_v * &gvf_simple;

        GvfResultPure {
            gvf_lup: lup_simple.clone(),
            gvfalb: gvfalb_simple.clone(),
            gvfalbnosh: alb_grid_v.to_owned(),
            gvf_lup_e: lup_simple.clone(),
            gvfalb_e: gvfalb_simple.clone(),
            gvfalbnosh_e: alb_grid_v.to_owned(),
            gvf_lup_s: lup_simple.clone(),
            gvfalb_s: gvfalb_simple.clone(),
            gvfalbnosh_s: alb_grid_v.to_owned(),
            gvf_lup_w: lup_simple.clone(),
            gvfalb_w: gvfalb_simple.clone(),
            gvfalbnosh_w: alb_grid_v.to_owned(),
            gvf_lup_n: lup_simple.clone(),
            gvfalb_n: gvfalb_simple,
            gvfalbnosh_n: alb_grid_v.to_owned(),
            gvf_sum: Array2::zeros(shape),
            gvf_norm: Array2::ones(shape),
        }
    };

    // ── Step 4: Thermal Delay ────────────────────────────────────────────
    let tg_temp = (&ground.tg * &shadow_f32 + weather.ta).mapv(|v| v as f32);

    let delay = ts_wave_delay_batch_pure(
        gvf.gvf_lup.view(),
        gvf.gvf_lup_e.view(),
        gvf.gvf_lup_s.view(),
        gvf.gvf_lup_w.view(),
        gvf.gvf_lup_n.view(),
        tg_temp.view(),
        firstdaytime,
        timeadd,
        timestep_dec,
        tgmap1_v,
        tgmap1_e_v,
        tgmap1_s_v,
        tgmap1_w_v,
        tgmap1_n_v,
        tgout1_v,
    );

    // ── Step 5: Radiation ─────────────────────────────────────────────────
    let esky = compute_esky(weather.ta, weather.rh);
    let sin_alt = (weather.sun_altitude * PI / 180.0).sin();
    let rad_i = weather.direct_rad;
    let rad_d = weather.diffuse_rad;
    let rad_g = weather.global_rad;
    let psi = weather.psi;
    let cyl = human.is_standing;

    // F_sh (cylindric wedge shadow fraction) — shared by both paths
    let zen_rad = weather.sun_zenith * PI / 180.0;
    let f_sh = cylindric_wedge_pure_masked(zen_rad, svfalfa_v, Some(valid_v));

    // Kup — shared by both paths
    let (kup, kup_e, kup_s, kup_w, kup_n) = compute_kup(
        rad_i,
        rad_d,
        rad_g,
        weather.sun_altitude,
        svfbuveg_v,
        config.albedo_wall,
        f_sh.view(),
        gvf.gvfalb.view(),
        gvf.gvfalb_e.view(),
        gvf.gvfalb_s.view(),
        gvf.gvfalb_w.view(),
        gvf.gvfalb_n.view(),
        gvf.gvfalbnosh.view(),
        gvf.gvfalbnosh_e.view(),
        gvf.gvfalbnosh_s.view(),
        gvf.gvfalbnosh_w.view(),
        gvf.gvfalbnosh_n.view(),
        valid_v,
    );

    // Branch: anisotropic vs isotropic
    let use_aniso = config.use_anisotropic
        && shmat_v.is_some()
        && l_patches_v.is_some()
        && steradians_v.is_some()
        && lv_v.is_some()
        && asvf_v.is_some();

    let (kdown, ldown, kside_knorth, kside_keast, kside_ksouth, kside_kwest,
         lside_lnorth, lside_least, lside_lsouth, lside_lwest,
         kside_total, lside_total) = if use_aniso {
        // === Anisotropic sky ===
        let shmat_a = shmat_v.unwrap();
        let vegshmat_a = vegshmat_v.unwrap();
        let vbshmat_a = vbshmat_v.unwrap();
        let l_patches_a = l_patches_v.unwrap();
        let steradians_a = steradians_v.unwrap();
        let lv_a = lv_v.unwrap();
        let asvf_a = asvf_v.unwrap();
        let esky_a = esky_aniso.unwrap_or(esky);

        // drad via weighted_patch_sum on diffsh
        // diffsh = shmat - (1 - vegshmat) * (1 - psi)  (u8 -> f32 inline)
        let n_patches = shmat_a.shape()[2];
        let mut diffsh = Array3::<f32>::zeros((shape.0, shape.1, n_patches));
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                if valid_v[[r, c]] == 0 {
                    continue; // Leave as zeros — NaN set by downstream functions
                }
                for i in 0..n_patches {
                    let sh = shmat_a[[r, c, i]] as f32 / 255.0;
                    let vsh = vegshmat_a[[r, c, i]] as f32 / 255.0;
                    diffsh[[r, c, i]] = sh - (1.0 - vsh) * (1.0 - psi);
                }
            }
        }
        let lv_col2 = lv_a.column(2);
        let ani_lum = weighted_patch_sum_pure(diffsh.view(), lv_col2);
        let drad = ani_lum.mapv(|v| v * rad_d);

        // Ldown base (isotropic Jonsson formula — needed for lside_veg)
        let ldown_base = compute_ldown(
            esky,
            weather.ta,
            ground.tg_wall,
            svf_v,
            svf_veg_v,
            svf_aveg_v,
            config.emis_wall,
            weather.clearness_index,
            valid_v,
        );

        // lside_veg with anisotropic=true (returns lup * 0.5 for each direction)
        let lside = lside_veg_pure(
            svf_s_v,
            svf_w_v,
            svf_n_v,
            svf_e_v,
            svf_veg_e_v,
            svf_veg_s_v,
            svf_veg_w_v,
            svf_veg_n_v,
            svf_aveg_e_v,
            svf_aveg_s_v,
            svf_aveg_w_v,
            svf_aveg_n_v,
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            ground.tg_wall,
            SBC,
            config.emis_wall,
            ldown_base.view(),
            esky,
            0.0, // t
            f_sh.view(),
            weather.clearness_index,
            delay.lup_e.view(),
            delay.lup_s.view(),
            delay.lup_w.view(),
            delay.lup_n.view(),
            true, // anisotropic
            Some(valid_v),
        );

        // Full anisotropic sky calculation (ldown, kside, lside totals)
        let ani = anisotropic_sky_pure(
            shmat_a,
            vegshmat_a,
            vbshmat_a,
            weather.sun_altitude,
            weather.sun_azimuth,
            esky_a,
            weather.ta,
            cyl,
            false, // wall_scheme
            config.albedo_wall,
            ground.tg_wall,
            config.emis_wall,
            rad_i,
            rad_d,
            asvf_a,
            l_patches_a,
            steradians_a,
            delay.lup.view(),
            lv_a,
            shadow_f32.view(),
            kup_e.view(),
            kup_s.view(),
            kup_w.view(),
            kup_n.view(),
            None, // voxel_table
            None, // voxel_maps
            Some(valid_v),
        );

        // Kdown (shared formula, but with anisotropic drad)
        let kdown = compute_kdown(
            rad_i,
            rad_d,
            rad_g,
            shadow_f32.view(),
            sin_alt,
            svfbuveg_v,
            config.albedo_wall,
            f_sh.view(),
            drad.view(),
            valid_v,
        );

        // From anisotropic: ldown from ani_sky, lside from lside_veg, kside from ani_sky
        (
            kdown,
            ani.ldown,
            ani.knorth,
            ani.keast,
            ani.ksouth,
            ani.kwest,
            lside.lnorth,
            lside.least,
            lside.lsouth,
            lside.lwest,
            ani.kside,
            ani.lside,
        )
    } else {
        // === Isotropic sky ===

        // drad (isotropic diffuse)
        let drad = svfbuveg_v.mapv(|sv| rad_d * sv);

        // Ldown
        let ldown = compute_ldown(
            esky,
            weather.ta,
            ground.tg_wall,
            svf_v,
            svf_veg_v,
            svf_aveg_v,
            config.emis_wall,
            weather.clearness_index,
            valid_v,
        );

        // kside_veg (isotropic)
        let kside = kside_veg_isotropic_pure(
            rad_i,
            rad_d,
            rad_g,
            shadow_f32.view(),
            svf_s_v,
            svf_w_v,
            svf_n_v,
            svf_e_v,
            svf_veg_e_v,
            svf_veg_s_v,
            svf_veg_w_v,
            svf_veg_n_v,
            weather.sun_azimuth,
            weather.sun_altitude,
            psi,
            0.0, // t (instrument offset)
            config.albedo_wall,
            f_sh.view(),
            kup_e.view(),
            kup_s.view(),
            kup_w.view(),
            kup_n.view(),
            cyl,
            Some(valid_v),
        );

        // lside_veg (isotropic)
        let lside = lside_veg_pure(
            svf_s_v,
            svf_w_v,
            svf_n_v,
            svf_e_v,
            svf_veg_e_v,
            svf_veg_s_v,
            svf_veg_w_v,
            svf_veg_n_v,
            svf_aveg_e_v,
            svf_aveg_s_v,
            svf_aveg_w_v,
            svf_aveg_n_v,
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            ground.tg_wall,
            SBC,
            config.emis_wall,
            ldown.view(),
            esky,
            0.0, // t
            f_sh.view(),
            weather.clearness_index,
            delay.lup_e.view(),
            delay.lup_s.view(),
            delay.lup_w.view(),
            delay.lup_n.view(),
            false, // isotropic
            Some(valid_v),
        );

        // Kdown
        let kdown = compute_kdown(
            rad_i,
            rad_d,
            rad_g,
            shadow_f32.view(),
            sin_alt,
            svfbuveg_v,
            config.albedo_wall,
            f_sh.view(),
            drad.view(),
            valid_v,
        );

        // Isotropic: kside_total = kside_i, lside_total = zeros
        (
            kdown,
            ldown,
            kside.knorth,
            kside.keast,
            kside.ksouth,
            kside.kwest,
            lside.lnorth,
            lside.least,
            lside.lsouth,
            lside.lwest,
            kside.kside_i,
            Array2::<f32>::zeros(shape),
        )
    };

    // ── Step 6: Tmrt ─────────────────────────────────────────────────────
    let tmrt = compute_tmrt_pure(
        kdown.view(),
        kup.view(),
        ldown.view(),
        delay.lup.view(),
        kside_knorth.view(),
        kside_keast.view(),
        kside_ksouth.view(),
        kside_kwest.view(),
        lside_lnorth.view(),
        lside_least.view(),
        lside_lsouth.view(),
        lside_lwest.view(),
        kside_total.view(),
        lside_total.view(),
        human.abs_k,
        human.abs_l,
        human.is_standing,
        use_aniso,
    );

    // ── Convert final outputs to PyArrays ────────────────────────────────
    Ok(TimestepResult {
        tmrt: tmrt.into_pyarray(py).unbind(),
        shadow: shadow_f32.into_pyarray(py).unbind(),
        kdown: kdown.into_pyarray(py).unbind(),
        kup: kup.into_pyarray(py).unbind(),
        ldown: ldown.into_pyarray(py).unbind(),
        lup: delay.lup.into_pyarray(py).unbind(),
        timeadd: delay.timeadd,
        tgmap1: delay.tgmap1.into_pyarray(py).unbind(),
        tgmap1_e: delay.tgmap1_e.into_pyarray(py).unbind(),
        tgmap1_s: delay.tgmap1_s.into_pyarray(py).unbind(),
        tgmap1_w: delay.tgmap1_w.into_pyarray(py).unbind(),
        tgmap1_n: delay.tgmap1_n.into_pyarray(py).unbind(),
        tgout1: delay.tgout1.into_pyarray(py).unbind(),
    })
}
