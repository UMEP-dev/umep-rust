use ndarray::{Array1, Array2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sun::sun_on_surface;

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
    lc_grid: PyReadonlyArray2<f32>,
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
    let lc_grid = lc_grid.as_array();

    let (rows, cols) = (buildings.shape()[0], buildings.shape()[1]);

    let azimuth_a: Array1<f32> = Array1::range(5.0, 359.0, 20.0);

    let mut sunwall = Array2::from_elem((rows, cols), 0.0);
    Zip::from(&mut sunwall)
        .and(&wallsun)
        .and(&walls)
        .and(&buildings)
        .par_for_each(|sw, &ws, &w, &b| {
            if w > 0.0 {
                *sw = if (ws / w * b) == 1.0 { 1.0 } else { 0.0 };
            }
        });

    let dirwalls_rad = dirwalls.mapv(|x| x * PI / 180.0);

    struct SunResult {
        azimuth: f32,
        gvf_lup: Array2<f32>,
        gvfalb: Array2<f32>,
        gvfalbnosh: Array2<f32>,
        gvf_sum: Array2<f32>,
    }

    let sun_results: Vec<SunResult> = azimuth_a
        .par_iter()
        .map(|&azimuth| {
            let (_, gvf_lupi, gvfalbi, gvfalbnoshi, gvf2) = sun_on_surface(
                azimuth,
                scale,
                buildings,
                shadow,
                sunwall.view(),
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
                lc_grid,
                landcover,
            );
            SunResult {
                azimuth,
                gvf_lup: gvf_lupi,
                gvfalb: gvfalbi,
                gvfalbnosh: gvfalbnoshi,
                gvf_sum: gvf2,
            }
        })
        .collect();

    // Helper to sum fields for a filter
    fn sum_field<F>(results: &[SunResult], field: F, rows: usize, cols: usize) -> Array2<f32>
    where
        F: Fn(&SunResult) -> &Array2<f32> + Sync + Send,
    {
        if results.is_empty() {
            return Array2::zeros((rows, cols));
        }
        results
            .par_iter()
            .fold(
                || Array2::zeros((rows, cols)),
                |mut acc, r| {
                    acc.zip_mut_with(field(r), |a, &b| *a += b);
                    acc
                },
            )
            .reduce(
                || Array2::zeros((rows, cols)),
                |mut a, b| {
                    a.zip_mut_with(&b, |x, &y| *x += y);
                    a
                },
            )
    }

    // All
    let gvf_lup = sum_field(&sun_results, |r| &r.gvf_lup, rows, cols);
    let gvfalb = sum_field(&sun_results, |r| &r.gvfalb, rows, cols);
    let gvfalbnosh = sum_field(&sun_results, |r| &r.gvfalbnosh, rows, cols);
    let gvf_sum = sum_field(&sun_results, |r| &r.gvf_sum, rows, cols);

    struct DirectionalSums {
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

    let init_sums = || DirectionalSums {
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

    let sums = sun_results
        .par_iter()
        .fold(init_sums, |mut acc, r| {
            let azimuth = r.azimuth;
            // East
            if azimuth >= 0.0 && azimuth < 180.0 {
                acc.lup_e.zip_mut_with(&r.gvf_lup, |a, &b| *a += b);
                acc.alb_e.zip_mut_with(&r.gvfalb, |a, &b| *a += b);
                acc.albnosh_e.zip_mut_with(&r.gvfalbnosh, |a, &b| *a += b);
            }
            // South
            if azimuth >= 90.0 && azimuth < 270.0 {
                acc.lup_s.zip_mut_with(&r.gvf_lup, |a, &b| *a += b);
                acc.alb_s.zip_mut_with(&r.gvfalb, |a, &b| *a += b);
                acc.albnosh_s.zip_mut_with(&r.gvfalbnosh, |a, &b| *a += b);
            }
            // West
            if azimuth >= 180.0 && azimuth < 360.0 {
                acc.lup_w.zip_mut_with(&r.gvf_lup, |a, &b| *a += b);
                acc.alb_w.zip_mut_with(&r.gvfalb, |a, &b| *a += b);
                acc.albnosh_w.zip_mut_with(&r.gvfalbnosh, |a, &b| *a += b);
            }
            // North
            if azimuth >= 270.0 || azimuth < 90.0 {
                acc.lup_n.zip_mut_with(&r.gvf_lup, |a, &b| *a += b);
                acc.alb_n.zip_mut_with(&r.gvfalb, |a, &b| *a += b);
                acc.albnosh_n.zip_mut_with(&r.gvfalbnosh, |a, &b| *a += b);
            }
            acc
        })
        .reduce(init_sums, |mut a, b| {
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

    let gvf_lup_e = sums.lup_e;
    let gvfalb_e = sums.alb_e;
    let gvfalbnosh_e = sums.albnosh_e;

    let gvf_lup_s = sums.lup_s;
    let gvfalb_s = sums.alb_s;
    let gvfalbnosh_s = sums.albnosh_s;

    let gvf_lup_w = sums.lup_w;
    let gvfalb_w = sums.alb_w;
    let gvfalbnosh_w = sums.albnosh_w;

    let gvf_lup_n = sums.lup_n;
    let gvfalb_n = sums.alb_n;
    let gvfalbnosh_n = sums.albnosh_n;

    let num_azimuths = azimuth_a.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;

    let ta_kelvin_pow4 = (ta + 273.15).powi(4);
    let emis_add = &emis_grid * (sbc * ta_kelvin_pow4);

    let gvf_lup = gvf_lup / num_azimuths + &emis_add;
    let gvfalb = gvfalb / num_azimuths;
    let gvfalbnosh = gvfalbnosh / num_azimuths;

    let gvf_lup_e = gvf_lup_e / num_azimuths_half + &emis_add;
    let gvf_lup_s = gvf_lup_s / num_azimuths_half + &emis_add;
    let gvf_lup_w = gvf_lup_w / num_azimuths_half + &emis_add;
    let gvf_lup_n = gvf_lup_n / num_azimuths_half + &emis_add;

    let gvfalb_e = gvfalb_e / num_azimuths_half;
    let gvfalb_s = gvfalb_s / num_azimuths_half;
    let gvfalb_w = gvfalb_w / num_azimuths_half;
    let gvfalb_n = gvfalb_n / num_azimuths_half;

    let gvfalbnosh_e = gvfalbnosh_e / num_azimuths_half;
    let gvfalbnosh_s = gvfalbnosh_s / num_azimuths_half;
    let gvfalbnosh_w = gvfalbnosh_w / num_azimuths_half;
    let gvfalbnosh_n = gvfalbnosh_n / num_azimuths_half;

    let mut gvf_norm = gvf_sum.clone() / num_azimuths;
    Zip::from(&mut gvf_norm)
        .and(buildings)
        .par_for_each(|norm, &bldg| {
            if bldg == 0.0 {
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
