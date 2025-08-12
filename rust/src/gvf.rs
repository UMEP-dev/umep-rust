use ndarray::{Array1, Array2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;

#[pyclass]
pub struct GvfResult {
    #[pyo3(get)]
    pub gvf_lup: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_lup_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalb_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvfalbnosh_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_sum: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub gvf_norm: Py<PyArray2<f32>>,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
pub fn gvf_calc(
    py: Python,
    wallsun: PyReadonlyArray2<f32>,
    walls: PyReadonlyArray2<f32>,
    buildings: PyReadonlyArray2<f32>,
    scale: f32,
    shadow: PyReadonlyArray2<f32>,
    first: f32,
    second: f32,
    dirwalls: PyReadonlyArray2<f32>,
    tg: PyReadonlyArray2<f32>,
    tgwall: f32,
    ta: f32,
    emis_grid: PyReadonlyArray2<f32>,
    ewall: f32,
    alb_grid: PyReadonlyArray2<f32>,
    sbc: f32,
    albedo_b: f32,
    twater: f32,
    lc_grid: Option<PyReadonlyArray2<f32>>,
    landcover: bool,
) -> PyResult<Py<GvfResult>> {
    let wallsun = wallsun.as_array();
    let walls = walls.as_array();
    let buildings = buildings.as_array();
    let shadow = shadow.as_array();
    let dirwalls = dirwalls.as_array();
    let tg = tg.as_array();
    let emis_grid = emis_grid.as_array();
    let alb_grid = alb_grid.as_array();
    let lc_grid_arr = lc_grid.as_ref().map(|arr| arr.as_array());

    let (rows, cols) = (buildings.shape()[0], buildings.shape()[1]);
    let azimuth_a: Array1<f32> = Array1::range(5.0, 359.0, 20.0);
    let num_azimuths = azimuth_a.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;
    const SUNWALL_TOL: f32 = 1e-6;

    // Sunwall mask
    let mut sunwall_mask = Array2::<f32>::zeros((rows, cols));
    Zip::from(&mut sunwall_mask)
        .and(&wallsun)
        .and(&walls)
        .and(&buildings)
        .par_for_each(|mask, &wsun, &wall, &bldg| {
            if wall > 0.0 && bldg > 0.0 {
                let ratio = (wsun / wall) * bldg;
                if (ratio - 1.0).abs() < SUNWALL_TOL {
                    *mask = 1.0;
                }
            }
        });
    let dirwalls_rad = dirwalls.mapv(|d| d * PI / 180.0);

    struct Accum {
        lup: Array2<f32>,
        alb: Array2<f32>,
        albnosh: Array2<f32>,
        sum: Array2<f32>,
        lup_e: Array2<f32>,
        alb_e: Array2<f32>,
        albnosh_e: Array2<f32>,
        lup_s: Array2<f32>,
        alb_s: Array2<f32>,
        albnosh_s: Array2<f32>,
        lup_w: Array2<f32>,
        alb_w: Array2<f32>,
        albnosh_w: Array2<f32>,
        lup_n: Array2<f32>,
        alb_n: Array2<f32>,
        albnosh_n: Array2<f32>,
    }
    let init_accum = || Accum {
        lup: Array2::zeros((rows, cols)),
        alb: Array2::zeros((rows, cols)),
        albnosh: Array2::zeros((rows, cols)),
        sum: Array2::zeros((rows, cols)),
        lup_e: Array2::zeros((rows, cols)),
        alb_e: Array2::zeros((rows, cols)),
        albnosh_e: Array2::zeros((rows, cols)),
        lup_s: Array2::zeros((rows, cols)),
        alb_s: Array2::zeros((rows, cols)),
        albnosh_s: Array2::zeros((rows, cols)),
        lup_w: Array2::zeros((rows, cols)),
        alb_w: Array2::zeros((rows, cols)),
        albnosh_w: Array2::zeros((rows, cols)),
        lup_n: Array2::zeros((rows, cols)),
        alb_n: Array2::zeros((rows, cols)),
        albnosh_n: Array2::zeros((rows, cols)),
    };

    let accum = azimuth_a
        .par_iter()
        .fold(init_accum, |mut a, &azimuth| {
            let (_gvf, gvf_lup_i, gvfalb_i, gvfalbnosh_i, gvf2_i) = crate::sun::sun_on_surface(
                azimuth,
                scale,
                buildings,
                shadow,
                sunwall_mask.view(),
                first,
                second,
                dirwalls_rad.view(),
                walls,
                tg,
                tgwall,
                ta,
                emis_grid,
                ewall,
                alb_grid,
                sbc,
                albedo_b,
                twater,
                lc_grid_arr.as_ref().map(|a| a.view()),
                landcover,
            );
            a.lup.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
            a.alb.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
            a.albnosh.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            a.sum.zip_mut_with(&gvf2_i, |x, &y| *x += y);
            if (0.0..180.0).contains(&azimuth) {
                a.lup_e.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_e.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_e.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if (90.0..270.0).contains(&azimuth) {
                a.lup_s.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_s.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_s.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if (180.0..360.0).contains(&azimuth) {
                a.lup_w.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_w.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_w.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            if azimuth >= 270.0 || azimuth < 90.0 {
                a.lup_n.zip_mut_with(&gvf_lup_i, |x, &y| *x += y);
                a.alb_n.zip_mut_with(&gvfalb_i, |x, &y| *x += y);
                a.albnosh_n.zip_mut_with(&gvfalbnosh_i, |x, &y| *x += y);
            }
            a
        })
        .reduce(init_accum, |mut a, b| {
            a.lup.zip_mut_with(&b.lup, |x, &y| *x += y);
            a.alb.zip_mut_with(&b.alb, |x, &y| *x += y);
            a.albnosh.zip_mut_with(&b.albnosh, |x, &y| *x += y);
            a.sum.zip_mut_with(&b.sum, |x, &y| *x += y);
            a.lup_e.zip_mut_with(&b.lup_e, |x, &y| *x += y);
            a.alb_e.zip_mut_with(&b.alb_e, |x, &y| *x += y);
            a.albnosh_e.zip_mut_with(&b.albnosh_e, |x, &y| *x += y);
            a.lup_s.zip_mut_with(&b.lup_s, |x, &y| *x += y);
            a.alb_s.zip_mut_with(&b.alb_s, |x, &y| *x += y);
            a.albnosh_s.zip_mut_with(&b.albnosh_s, |x, &y| *x += y);
            a.lup_w.zip_mut_with(&b.lup_w, |x, &y| *x += y);
            a.alb_w.zip_mut_with(&b.alb_w, |x, &y| *x += y);
            a.albnosh_w.zip_mut_with(&b.albnosh_w, |x, &y| *x += y);
            a.lup_n.zip_mut_with(&b.lup_n, |x, &y| *x += y);
            a.alb_n.zip_mut_with(&b.alb_n, |x, &y| *x += y);
            a.albnosh_n.zip_mut_with(&b.albnosh_n, |x, &y| *x += y);
            a
        });

    // Extract totals
    let ta_kelvin_pow4 = (ta + 273.15).powi(4);
    let emis_add = emis_grid.mapv(|e| e * sbc * ta_kelvin_pow4);
    let scale_all = 1.0 / num_azimuths;
    let scale_half = 1.0 / num_azimuths_half;
    let gvf_lup = accum.lup.mapv(|v| v * scale_all) + &emis_add;
    let gvfalb = accum.alb.mapv(|v| v * scale_all);
    let gvfalbnosh = accum.albnosh.mapv(|v| v * scale_all);
    let gvf_lup_e = accum.lup_e.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_e = accum.alb_e.mapv(|v| v * scale_half);
    let gvfalbnosh_e = accum.albnosh_e.mapv(|v| v * scale_half);
    let gvf_lup_s = accum.lup_s.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_s = accum.alb_s.mapv(|v| v * scale_half);
    let gvfalbnosh_s = accum.albnosh_s.mapv(|v| v * scale_half);
    let gvf_lup_w = accum.lup_w.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_w = accum.alb_w.mapv(|v| v * scale_half);
    let gvfalbnosh_w = accum.albnosh_w.mapv(|v| v * scale_half);
    let gvf_lup_n = accum.lup_n.mapv(|v| v * scale_half) + &emis_add;
    let gvfalb_n = accum.alb_n.mapv(|v| v * scale_half);
    let gvfalbnosh_n = accum.albnosh_n.mapv(|v| v * scale_half);
    let gvf_sum = accum.sum; // raw sum
    let mut gvf_norm = gvf_sum.mapv(|v| v * scale_all);
    Zip::from(&mut gvf_norm)
        .and(&buildings)
        .for_each(|norm, &b| {
            if b == 0.0 {
                *norm = 1.0;
            }
        });

    Py::new(
        py,
        GvfResult {
            gvf_lup: gvf_lup.into_pyarray(py).unbind(),
            gvfalb: gvfalb.into_pyarray(py).unbind(),
            gvfalbnosh: gvfalbnosh.into_pyarray(py).unbind(),
            gvf_lup_e: gvf_lup_e.into_pyarray(py).unbind(),
            gvfalb_e: gvfalb_e.into_pyarray(py).unbind(),
            gvfalbnosh_e: gvfalbnosh_e.into_pyarray(py).unbind(),
            gvf_lup_s: gvf_lup_s.into_pyarray(py).unbind(),
            gvfalb_s: gvfalb_s.into_pyarray(py).unbind(),
            gvfalbnosh_s: gvfalbnosh_s.into_pyarray(py).unbind(),
            gvf_lup_w: gvf_lup_w.into_pyarray(py).unbind(),
            gvfalb_w: gvfalb_w.into_pyarray(py).unbind(),
            gvfalbnosh_w: gvfalbnosh_w.into_pyarray(py).unbind(),
            gvf_lup_n: gvf_lup_n.into_pyarray(py).unbind(),
            gvfalb_n: gvfalb_n.into_pyarray(py).unbind(),
            gvfalbnosh_n: gvfalbnosh_n.into_pyarray(py).unbind(),
            gvf_sum: gvf_sum.into_pyarray(py).unbind(),
            gvf_norm: gvf_norm.into_pyarray(py).unbind(),
        },
    )
}
