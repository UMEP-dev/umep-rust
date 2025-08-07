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

    use std::sync::Arc;
    struct SunResult {
        azimuth: f32,
        gvf_lup: Array2<f32>,
        gvfalb: Array2<f32>,
        gvfalbnosh: Array2<f32>,
        gvf_sum: Array2<f32>,
    }

    let sun_results: Vec<Arc<SunResult>> = azimuth_a
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
            Arc::new(SunResult {
                azimuth,
                gvf_lup: gvf_lupi,
                gvfalb: gvfalbi,
                gvfalbnosh: gvfalbnoshi,
                gvf_sum: gvf2,
            })
        })
        .collect();

    // Helper to sum fields for a filter
    fn sum_field<F>(results: &[Arc<SunResult>], field: F) -> Array2<f32>
    where
        F: Fn(&SunResult) -> &Array2<f32>,
    {
        let mut acc = field(&results[0]).clone();
        for r in &results[1..] {
            acc.zip_mut_with(field(r), |a, &b| *a += b);
        }
        acc
    }

    // All
    let gvf_lup = sum_field(&sun_results, |r| &r.gvf_lup);
    let gvfalb = sum_field(&sun_results, |r| &r.gvfalb);
    let gvfalbnosh = sum_field(&sun_results, |r| &r.gvfalbnosh);
    let gvf_sum = sum_field(&sun_results, |r| &r.gvf_sum);

    // Directional subsets
    let east: Vec<_> = sun_results
        .iter()
        .filter(|r| r.azimuth >= 0.0 && r.azimuth < 180.0)
        .cloned()
        .collect();
    let south: Vec<_> = sun_results
        .iter()
        .filter(|r| r.azimuth >= 90.0 && r.azimuth < 270.0)
        .cloned()
        .collect();
    let west: Vec<_> = sun_results
        .iter()
        .filter(|r| r.azimuth >= 180.0 && r.azimuth < 360.0)
        .cloned()
        .collect();
    let north: Vec<_> = sun_results
        .iter()
        .filter(|r| r.azimuth >= 270.0 || r.azimuth < 90.0)
        .cloned()
        .collect();

    let gvf_lup_e = if !east.is_empty() {
        sum_field(&east, |r| &r.gvf_lup)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalb_e = if !east.is_empty() {
        sum_field(&east, |r| &r.gvfalb)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalbnosh_e = if !east.is_empty() {
        sum_field(&east, |r| &r.gvfalbnosh)
    } else {
        Array2::zeros((rows, cols))
    };

    let gvf_lup_s = if !south.is_empty() {
        sum_field(&south, |r| &r.gvf_lup)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalb_s = if !south.is_empty() {
        sum_field(&south, |r| &r.gvfalb)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalbnosh_s = if !south.is_empty() {
        sum_field(&south, |r| &r.gvfalbnosh)
    } else {
        Array2::zeros((rows, cols))
    };

    let gvf_lup_w = if !west.is_empty() {
        sum_field(&west, |r| &r.gvf_lup)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalb_w = if !west.is_empty() {
        sum_field(&west, |r| &r.gvfalb)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalbnosh_w = if !west.is_empty() {
        sum_field(&west, |r| &r.gvfalbnosh)
    } else {
        Array2::zeros((rows, cols))
    };

    let gvf_lup_n = if !north.is_empty() {
        sum_field(&north, |r| &r.gvf_lup)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalb_n = if !north.is_empty() {
        sum_field(&north, |r| &r.gvfalb)
    } else {
        Array2::zeros((rows, cols))
    };
    let gvfalbnosh_n = if !north.is_empty() {
        sum_field(&north, |r| &r.gvfalbnosh)
    } else {
        Array2::zeros((rows, cols))
    };

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
