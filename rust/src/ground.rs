use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Ground temperature calculation parameters
const PI: f32 = std::f32::consts::PI;

/// Pure result type for ground temperature (no PyO3 dependency).
pub(crate) struct GroundTempResult {
    pub tg: Array2<f32>,
    pub tg_wall: f32,
    pub ci_tg: f32,
}

/// Pure-ndarray implementation of ground temperature calculation.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ground_temperature_pure(
    sun_altitude: f32,
    altmax: f32,
    dectime: f32,
    snup: f32,
    global_rad: f32,
    rad_g0: f32,
    zen_deg: f32,
    tgk_grid: ArrayView2<f32>,
    tstart_grid: ArrayView2<f32>,
    tmaxlst_grid: ArrayView2<f32>,
    tgk_wall: f32,
    tstart_wall: f32,
    tmaxlst_wall: f32,
) -> GroundTempResult {
    let shape = tgk_grid.dim();

    // Temperature amplitude based on max sun altitude (per-pixel from land cover)
    let tgamp = &tgk_grid * altmax + &tstart_grid;

    // Wall temperature amplitude
    let tgamp_wall = tgk_wall * altmax + tstart_wall;

    // Phase calculation for ground (per-pixel)
    let snup_frac = snup / 24.0;
    let tmaxlst_frac = &tmaxlst_grid / 24.0;
    let tmaxlst_wall_frac = tmaxlst_wall / 24.0;

    let mut tg = Array2::<f32>::zeros(shape);

    tg.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let row = idx / shape.1;
            let col = idx % shape.1;

            let tgamp_val = tgamp[[row, col]];
            if !tgamp_val.is_finite() {
                *out = f32::NAN;
                return;
            }
            let tmaxlst_frac_val = tmaxlst_frac[[row, col]];

            if dectime > snup_frac {
                let denom = tmaxlst_frac_val - snup_frac;
                let denom = if denom > 0.0 { denom } else { 1.0 };
                let phase = (dectime - snup_frac) / denom;
                *out = tgamp_val * (phase * PI / 2.0).sin();
            } else {
                *out = 0.0;
            }
        });

    // Wall phase (scalar)
    let tg_wall = if dectime > snup_frac && tmaxlst_wall_frac > snup_frac {
        let denom_wall = tmaxlst_wall_frac - snup_frac;
        let denom_wall = if denom_wall > 0.0 { denom_wall } else { 1.0 };
        let phase_wall = (dectime - snup_frac) / denom_wall;
        tgamp_wall * (phase_wall * PI / 2.0).sin()
    } else {
        0.0
    };

    // CI_TgG correction for non-clear conditions
    let ci_tg = if sun_altitude > 0.0 && rad_g0 > 0.0 {
        let corr = if zen_deg > 0.0 && zen_deg < 90.0 {
            0.1473 * (90.0 - zen_deg).ln() + 0.3454
        } else {
            0.3454
        };
        let mut ci = (global_rad / rad_g0) + (1.0 - corr);
        ci = ci.min(1.0);
        if ci.is_infinite() || ci.is_nan() {
            1.0
        } else {
            ci
        }
    } else {
        1.0
    };

    // Apply clearness correction
    tg.par_mapv_inplace(|v| (v * ci_tg).max(0.0));
    let tg_wall_corrected = (tg_wall * ci_tg).max(0.0);

    GroundTempResult {
        tg,
        tg_wall: tg_wall_corrected,
        ci_tg,
    }
}

/// Pure result type for batched thermal delay (no PyO3 dependency).
pub(crate) struct TsWaveDelayBatchPureResult {
    pub lup: Array2<f32>,
    pub lup_e: Array2<f32>,
    pub lup_s: Array2<f32>,
    pub lup_w: Array2<f32>,
    pub lup_n: Array2<f32>,
    pub tg_out: Array2<f32>,
    pub timeadd: f32,
    pub tgmap1: Array2<f32>,
    pub tgmap1_e: Array2<f32>,
    pub tgmap1_s: Array2<f32>,
    pub tgmap1_w: Array2<f32>,
    pub tgmap1_n: Array2<f32>,
    pub tgout1: Array2<f32>,
}

/// Pure-ndarray implementation of batched thermal delay.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ts_wave_delay_batch_pure(
    lup: ArrayView2<f32>,
    lup_e: ArrayView2<f32>,
    lup_s: ArrayView2<f32>,
    lup_w: ArrayView2<f32>,
    lup_n: ArrayView2<f32>,
    tg_temp: ArrayView2<f32>,
    firstdaytime: i32,
    timeadd: f32,
    timestepdec: f32,
    tgmap1: ArrayView2<f32>,
    tgmap1_e: ArrayView2<f32>,
    tgmap1_s: ArrayView2<f32>,
    tgmap1_w: ArrayView2<f32>,
    tgmap1_n: ArrayView2<f32>,
    tgout1: ArrayView2<f32>,
) -> TsWaveDelayBatchPureResult {
    let mut tgmap1_arr = tgmap1.to_owned();
    let mut tgmap1_e_arr = tgmap1_e.to_owned();
    let mut tgmap1_s_arr = tgmap1_s.to_owned();
    let mut tgmap1_w_arr = tgmap1_w.to_owned();
    let mut tgmap1_n_arr = tgmap1_n.to_owned();
    let mut tgout1_arr = tgout1.to_owned();

    // First morning: reset previous temperatures
    if firstdaytime == 1 {
        tgmap1_arr.assign(&lup);
        tgmap1_e_arr.assign(&lup_e);
        tgmap1_s_arr.assign(&lup_s);
        tgmap1_w_arr.assign(&lup_w);
        tgmap1_n_arr.assign(&lup_n);
        tgout1_arr.assign(&tg_temp);
    }

    let threshold = 59.0 / 1440.0;
    let decay_constant = -33.27f32;

    if timeadd >= threshold {
        let weight1 = (decay_constant * timeadd).exp();
        let new_tgmap1 = &lup * (1.0 - weight1) + &tgmap1_arr * weight1;
        let new_tgmap1_e = &lup_e * (1.0 - weight1) + &tgmap1_e_arr * weight1;
        let new_tgmap1_s = &lup_s * (1.0 - weight1) + &tgmap1_s_arr * weight1;
        let new_tgmap1_w = &lup_w * (1.0 - weight1) + &tgmap1_w_arr * weight1;
        let new_tgmap1_n = &lup_n * (1.0 - weight1) + &tgmap1_n_arr * weight1;
        let new_tgout1 = &tg_temp * (1.0 - weight1) + &tgout1_arr * weight1;

        let new_timeadd = if timestepdec > threshold {
            timestepdec
        } else {
            0.0
        };

        TsWaveDelayBatchPureResult {
            lup: new_tgmap1.clone(),
            lup_e: new_tgmap1_e.clone(),
            lup_s: new_tgmap1_s.clone(),
            lup_w: new_tgmap1_w.clone(),
            lup_n: new_tgmap1_n.clone(),
            tg_out: new_tgout1.clone(),
            timeadd: new_timeadd,
            tgmap1: new_tgmap1,
            tgmap1_e: new_tgmap1_e,
            tgmap1_s: new_tgmap1_s,
            tgmap1_w: new_tgmap1_w,
            tgmap1_n: new_tgmap1_n,
            tgout1: new_tgout1,
        }
    } else {
        let new_timeadd = timeadd + timestepdec;
        let weight1 = (decay_constant * new_timeadd).exp();
        let out_lup = &lup * (1.0 - weight1) + &tgmap1_arr * weight1;
        let out_lup_e = &lup_e * (1.0 - weight1) + &tgmap1_e_arr * weight1;
        let out_lup_s = &lup_s * (1.0 - weight1) + &tgmap1_s_arr * weight1;
        let out_lup_w = &lup_w * (1.0 - weight1) + &tgmap1_w_arr * weight1;
        let out_lup_n = &lup_n * (1.0 - weight1) + &tgmap1_n_arr * weight1;
        let out_tg = &tg_temp * (1.0 - weight1) + &tgout1_arr * weight1;

        TsWaveDelayBatchPureResult {
            lup: out_lup,
            lup_e: out_lup_e,
            lup_s: out_lup_s,
            lup_w: out_lup_w,
            lup_n: out_lup_n,
            tg_out: out_tg,
            timeadd: new_timeadd,
            tgmap1: tgmap1_arr,
            tgmap1_e: tgmap1_e_arr,
            tgmap1_s: tgmap1_s_arr,
            tgmap1_w: tgmap1_w_arr,
            tgmap1_n: tgmap1_n_arr,
            tgout1: tgout1_arr,
        }
    }
}

/// Calculate ground and wall temperature deviations from air temperature.
///
/// Implements the SOLWEIG TgMaps model with land-cover-specific parameterization.
/// Temperature amplitude depends on max sun altitude and land cover type.
/// Clearness index correction accounts for reduced heating under cloudy skies.
///
/// Parameters:
/// - ta: Air temperature (°C)
/// - sun_altitude: Sun altitude/elevation (degrees)
/// - altmax: Maximum sun altitude for the day (degrees)
/// - dectime: Decimal time (fraction of day, 0-1)
/// - snup: Sunrise time (hours, 0-24)
/// - global_rad: Global horizontal radiation (W/m²)
/// - rad_g0: Clear sky global horizontal radiation (W/m²)
/// - zen_deg: Solar zenith angle (degrees)
/// - alb_grid: Albedo per pixel (0-1) from land cover properties
/// - emis_grid: Emissivity per pixel (0-1) from land cover properties
/// - tgk_grid: TgK parameter per pixel (temperature gain coefficient)
/// - tstart_grid: Tstart parameter per pixel (temperature baseline offset)
/// - tmaxlst_grid: TmaxLST parameter per pixel (hour of maximum temperature, 0-24)
/// - tgk_wall: Optional wall TgK parameter (default: 0.37, cobblestone)
/// - tstart_wall: Optional wall Tstart parameter (default: -3.41, cobblestone)
/// - tmaxlst_wall: Optional wall TmaxLST parameter (default: 15.0, cobblestone)
///
/// Returns tuple:
/// - tg: Ground temperature deviation from air temperature (K)
/// - tg_wall: Wall temperature deviation from air temperature (K)
/// - ci_tg: Clearness index correction factor (0-1)
#[pyfunction]
#[pyo3(signature = (
    _ta, sun_altitude, altmax, dectime, snup, global_rad, rad_g0, zen_deg,
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid,
    tgk_wall=None, tstart_wall=None, tmaxlst_wall=None,
))]
pub fn compute_ground_temperature<'py>(
    py: Python<'py>,
    _ta: f32,
    sun_altitude: f32,
    altmax: f32,
    dectime: f32,
    snup: f32,
    global_rad: f32,
    rad_g0: f32,
    zen_deg: f32,
    alb_grid: PyReadonlyArray2<'py, f32>,
    emis_grid: PyReadonlyArray2<'py, f32>,
    tgk_grid: PyReadonlyArray2<'py, f32>,
    tstart_grid: PyReadonlyArray2<'py, f32>,
    tmaxlst_grid: PyReadonlyArray2<'py, f32>,
    tgk_wall: Option<f32>,
    tstart_wall: Option<f32>,
    tmaxlst_wall: Option<f32>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    f32,
    f32,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let alb_arr = alb_grid.as_array();
    let emis_arr = emis_grid.as_array();

    let result = compute_ground_temperature_pure(
        sun_altitude,
        altmax,
        dectime,
        snup,
        global_rad,
        rad_g0,
        zen_deg,
        tgk_grid.as_array(),
        tstart_grid.as_array(),
        tmaxlst_grid.as_array(),
        tgk_wall.unwrap_or(0.37),
        tstart_wall.unwrap_or(-3.41),
        tmaxlst_wall.unwrap_or(15.0),
    );

    let tg_py = PyArray2::from_owned_array(py, result.tg);
    let alb_py = PyArray2::from_owned_array(py, alb_arr.to_owned());
    let emis_py = PyArray2::from_owned_array(py, emis_arr.to_owned());

    Ok((tg_py, result.tg_wall, result.ci_tg, alb_py, emis_py))
}

/// Apply thermal delay to ground temperature using TsWaveDelay model.
///
/// The thermal delay model simulates ground temperature response to changing
/// radiation conditions using an exponential decay function with a decay constant
/// of 33.27 day⁻¹ (time constant ≈ 43 minutes).
///
/// Parameters:
/// - gvfLup: Current radiative equilibrium temperature (2D array)
/// - firstdaytime: True (1) if first timestep after sunrise, False (0) otherwise
/// - timeadd: Time since last full update (fraction of day)
/// - timestepdec: Current timestep duration (fraction of day)
/// - Tgmap1: Previous delayed temperature (2D array)
///
/// Returns tuple:
/// - Lup: Temperature with thermal inertia applied (2D array)
/// - timeadd: Updated time accumulator (fraction of day)
/// - Tgmap1: Updated previous temperature for next iteration (2D array)
#[pyfunction]
pub fn ts_wave_delay<'py>(
    py: Python<'py>,
    gvf_lup: PyReadonlyArray2<'py, f32>,
    firstdaytime: i32,
    timeadd: f32,
    timestepdec: f32,
    tgmap1: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, Bound<'py, PyArray2<f32>>)> {
    let gvf_lup_arr = gvf_lup.as_array();
    let mut tgmap1_arr = tgmap1.as_array().to_owned();

    let tgmap0 = &gvf_lup_arr; // current timestep

    // First morning: reset previous temperature
    if firstdaytime == 1 {
        tgmap1_arr.assign(tgmap0);
    }

    let threshold = 59.0 / 1440.0; // ~59 minutes threshold
    let decay_constant = -33.27f32;

    let (lup, new_timeadd, new_tgmap1) = if timeadd >= threshold {
        // More or equal to 59 min
        let weight1 = (decay_constant * timeadd).exp();
        let new_tgmap1 = tgmap0 * (1.0 - weight1) + &tgmap1_arr * weight1;
        let lup = new_tgmap1.clone();

        let new_timeadd = if timestepdec > threshold {
            timestepdec
        } else {
            0.0
        };

        (lup, new_timeadd, new_tgmap1)
    } else {
        // Accumulate time
        let new_timeadd = timeadd + timestepdec;
        let weight1 = (decay_constant * new_timeadd).exp();
        let lup = tgmap0 * (1.0 - weight1) + &tgmap1_arr * weight1;

        (lup, new_timeadd, tgmap1_arr.clone())
    };

    let lup_py = PyArray2::from_owned_array(py, lup);
    let tgmap1_py = PyArray2::from_owned_array(py, new_tgmap1);

    Ok((lup_py, new_timeadd, tgmap1_py))
}

/// Result struct for batched thermal delay
#[pyclass]
pub struct TsWaveDelayBatchResult {
    /// Delayed lup (center)
    #[pyo3(get)]
    pub lup: Py<PyArray2<f32>>,
    /// Delayed lup_e (east)
    #[pyo3(get)]
    pub lup_e: Py<PyArray2<f32>>,
    /// Delayed lup_s (south)
    #[pyo3(get)]
    pub lup_s: Py<PyArray2<f32>>,
    /// Delayed lup_w (west)
    #[pyo3(get)]
    pub lup_w: Py<PyArray2<f32>>,
    /// Delayed lup_n (north)
    #[pyo3(get)]
    pub lup_n: Py<PyArray2<f32>>,
    /// Delayed ground temperature
    #[pyo3(get)]
    pub tg_out: Py<PyArray2<f32>>,
    /// Updated time accumulator
    #[pyo3(get)]
    pub timeadd: f32,
    /// Updated tgmap1 (center)
    #[pyo3(get)]
    pub tgmap1: Py<PyArray2<f32>>,
    /// Updated tgmap1_e (east)
    #[pyo3(get)]
    pub tgmap1_e: Py<PyArray2<f32>>,
    /// Updated tgmap1_s (south)
    #[pyo3(get)]
    pub tgmap1_s: Py<PyArray2<f32>>,
    /// Updated tgmap1_w (west)
    #[pyo3(get)]
    pub tgmap1_w: Py<PyArray2<f32>>,
    /// Updated tgmap1_n (north)
    #[pyo3(get)]
    pub tgmap1_n: Py<PyArray2<f32>>,
    /// Updated tgout1 (ground temperature)
    #[pyo3(get)]
    pub tgout1: Py<PyArray2<f32>>,
}

/// Apply thermal delay to ground temperature for all 6 directional components.
///
/// Batched version of ts_wave_delay that processes lup, lup_e/s/w/n, and tg_temp
/// in a single FFI call, reducing Python/Rust crossing overhead from 6 calls to 1.
///
/// Parameters:
/// - lup, lup_e, lup_s, lup_w, lup_n: Current radiative equilibrium for each direction
/// - tg_temp: Ground temperature (tg * shadow + ta)
/// - firstdaytime: True (1) if first timestep after sunrise, False (0) otherwise
/// - timeadd: Time since last full update (fraction of day)
/// - timestepdec: Current timestep duration (fraction of day)
/// - tgmap1, tgmap1_e, tgmap1_s, tgmap1_w, tgmap1_n: Previous delayed temperatures
/// - tgout1: Previous delayed ground temperature
///
/// Returns TsWaveDelayBatchResult with all delayed outputs and updated state
#[pyfunction]
pub fn ts_wave_delay_batch<'py>(
    py: Python<'py>,
    lup: PyReadonlyArray2<'py, f32>,
    lup_e: PyReadonlyArray2<'py, f32>,
    lup_s: PyReadonlyArray2<'py, f32>,
    lup_w: PyReadonlyArray2<'py, f32>,
    lup_n: PyReadonlyArray2<'py, f32>,
    tg_temp: PyReadonlyArray2<'py, f32>,
    firstdaytime: i32,
    timeadd: f32,
    timestepdec: f32,
    tgmap1: PyReadonlyArray2<'py, f32>,
    tgmap1_e: PyReadonlyArray2<'py, f32>,
    tgmap1_s: PyReadonlyArray2<'py, f32>,
    tgmap1_w: PyReadonlyArray2<'py, f32>,
    tgmap1_n: PyReadonlyArray2<'py, f32>,
    tgout1: PyReadonlyArray2<'py, f32>,
) -> PyResult<TsWaveDelayBatchResult> {
    let result = ts_wave_delay_batch_pure(
        lup.as_array(),
        lup_e.as_array(),
        lup_s.as_array(),
        lup_w.as_array(),
        lup_n.as_array(),
        tg_temp.as_array(),
        firstdaytime,
        timeadd,
        timestepdec,
        tgmap1.as_array(),
        tgmap1_e.as_array(),
        tgmap1_s.as_array(),
        tgmap1_w.as_array(),
        tgmap1_n.as_array(),
        tgout1.as_array(),
    );

    Ok(TsWaveDelayBatchResult {
        lup: PyArray2::from_owned_array(py, result.lup).unbind(),
        lup_e: PyArray2::from_owned_array(py, result.lup_e).unbind(),
        lup_s: PyArray2::from_owned_array(py, result.lup_s).unbind(),
        lup_w: PyArray2::from_owned_array(py, result.lup_w).unbind(),
        lup_n: PyArray2::from_owned_array(py, result.lup_n).unbind(),
        tg_out: PyArray2::from_owned_array(py, result.tg_out).unbind(),
        timeadd: result.timeadd,
        tgmap1: PyArray2::from_owned_array(py, result.tgmap1).unbind(),
        tgmap1_e: PyArray2::from_owned_array(py, result.tgmap1_e).unbind(),
        tgmap1_s: PyArray2::from_owned_array(py, result.tgmap1_s).unbind(),
        tgmap1_w: PyArray2::from_owned_array(py, result.tgmap1_w).unbind(),
        tgmap1_n: PyArray2::from_owned_array(py, result.tgmap1_n).unbind(),
        tgout1: PyArray2::from_owned_array(py, result.tgout1).unbind(),
    })
}
