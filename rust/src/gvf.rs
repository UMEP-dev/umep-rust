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
    let num_azimuths = azimuth_a.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;

    // Struct to hold results for a single pixel
    struct PixelResult {
        r: usize,
        c: usize,
        gvf_lup: f32,
        gvfalb: f32,
        gvfalbnosh: f32,
        gvf_sum: f32,
        gvf_lup_e: f32,
        gvfalb_e: f32,
        gvfalbnosh_e: f32,
        gvf_lup_s: f32,
        gvfalb_s: f32,
        gvfalbnosh_s: f32,
        gvf_lup_w: f32,
        gvfalb_w: f32,
        gvfalbnosh_w: f32,
        gvf_lup_n: f32,
        gvfalb_n: f32,
        gvfalbnosh_n: f32,
    }

    // Create a flat list of pixel indices to parallelize over
    let pixel_indices: Vec<(usize, usize)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .collect();

    // Main parallel computation over pixels
    let pixel_results: Vec<PixelResult> = pixel_indices
        .into_par_iter()
        .map(|(r, c)| {
            let building = buildings[(r, c)];
            let wall = walls[(r, c)];
            let wall_aspect = dirwalls[(r, c)] * PI / 180.0;
            let wall_ht = wall;
            let shadow_val = shadow[(r, c)];
            let sunwall_val = if wall > 0.0 {
                let ws = wallsun[(r, c)];
                if (ws / wall * building) == 1.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let tground = tg[(r, c)];
            let emis = emis_grid[(r, c)];
            let alb = alb_grid[(r, c)];
            let lc = lc_grid[(r, c)];

            let mut sum_lup = 0.0;
            let mut sum_alb = 0.0;
            let mut sum_albnosh = 0.0;
            let mut sum_gvf2 = 0.0;

            let mut sum_lup_e = 0.0;
            let mut sum_alb_e = 0.0;
            let mut sum_albnosh_e = 0.0;
            let mut sum_lup_s = 0.0;
            let mut sum_alb_s = 0.0;
            let mut sum_albnosh_s = 0.0;
            let mut sum_lup_w = 0.0;
            let mut sum_alb_w = 0.0;
            let mut sum_albnosh_w = 0.0;
            let mut sum_lup_n = 0.0;
            let mut sum_alb_n = 0.0;
            let mut sum_albnosh_n = 0.0;

            for &azimuth in azimuth_a.iter() {
                let (_, gvfLup, gvfalb_val, gvfalbnosh_val, gvf2) =
                    crate::sun::sun_on_surface_pixel(
                        azimuth,
                        scale,
                        building,
                        shadow_val,
                        sunwall_val,
                        first,
                        second,
                        wall_aspect,
                        wall_ht,
                        tground,
                        tgwall,
                        ta,
                        emis,
                        ewall,
                        alb,
                        sbc,
                        albedo_b,
                        twater,
                        lc,
                        landcover,
                    );
                sum_lup += gvfLup;
                sum_alb += gvfalb_val;
                sum_albnosh += gvfalbnosh_val;
                sum_gvf2 += gvf2;

                // Directional sums
                if (0.0..180.0).contains(&azimuth) {
                    sum_lup_e += gvfLup;
                    sum_alb_e += gvfalb_val;
                    sum_albnosh_e += gvfalbnosh_val;
                }
                if (90.0..270.0).contains(&azimuth) {
                    sum_lup_s += gvfLup;
                    sum_alb_s += gvfalb_val;
                    sum_albnosh_s += gvfalbnosh_val;
                }
                if (180.0..360.0).contains(&azimuth) {
                    sum_lup_w += gvfLup;
                    sum_alb_w += gvfalb_val;
                    sum_albnosh_w += gvfalbnosh_val;
                }
                if azimuth >= 270.0 || azimuth < 90.0 {
                    sum_lup_n += gvfLup;
                    sum_alb_n += gvfalb_val;
                    sum_albnosh_n += gvfalbnosh_val;
                }
            }

            let ta_kelvin_pow4 = (ta + 273.15).powi(4);
            let emis_add = emis * (sbc * ta_kelvin_pow4);

            PixelResult {
                r,
                c,
                gvf_lup: sum_lup / num_azimuths + emis_add,
                gvfalb: sum_alb / num_azimuths,
                gvfalbnosh: sum_albnosh / num_azimuths,
                gvf_sum: sum_gvf2 / num_azimuths,
                gvf_lup_e: sum_lup_e / num_azimuths_half + emis_add,
                gvfalb_e: sum_alb_e / num_azimuths_half,
                gvfalbnosh_e: sum_albnosh_e / num_azimuths_half,
                gvf_lup_s: sum_lup_s / num_azimuths_half + emis_add,
                gvfalb_s: sum_alb_s / num_azimuths_half,
                gvfalbnosh_s: sum_albnosh_s / num_azimuths_half,
                gvf_lup_w: sum_lup_w / num_azimuths_half + emis_add,
                gvfalb_w: sum_alb_w / num_azimuths_half,
                gvfalbnosh_w: sum_albnosh_w / num_azimuths_half,
                gvf_lup_n: sum_lup_n / num_azimuths_half + emis_add,
                gvfalb_n: sum_alb_n / num_azimuths_half,
                gvfalbnosh_n: sum_albnosh_n / num_azimuths_half,
            }
        })
        .collect();

    // Prepare output arrays
    let mut gvf_lup = Array2::<f32>::zeros((rows, cols));
    let mut gvfalb = Array2::<f32>::zeros((rows, cols));
    let mut gvfalbnosh = Array2::<f32>::zeros((rows, cols));
    let mut gvf_sum = Array2::<f32>::zeros((rows, cols));
    let mut gvf_lup_e = Array2::<f32>::zeros((rows, cols));
    let mut gvfalb_e = Array2::<f32>::zeros((rows, cols));
    let mut gvfalbnosh_e = Array2::<f32>::zeros((rows, cols));
    let mut gvf_lup_s = Array2::<f32>::zeros((rows, cols));
    let mut gvfalb_s = Array2::<f32>::zeros((rows, cols));
    let mut gvfalbnosh_s = Array2::<f32>::zeros((rows, cols));
    let mut gvf_lup_w = Array2::<f32>::zeros((rows, cols));
    let mut gvfalb_w = Array2::<f32>::zeros((rows, cols));
    let mut gvfalbnosh_w = Array2::<f32>::zeros((rows, cols));
    let mut gvf_lup_n = Array2::<f32>::zeros((rows, cols));
    let mut gvfalb_n = Array2::<f32>::zeros((rows, cols));
    let mut gvfalbnosh_n = Array2::<f32>::zeros((rows, cols));

    // Populate output arrays from results
    for result in pixel_results {
        gvf_lup[(result.r, result.c)] = result.gvf_lup;
        gvfalb[(result.r, result.c)] = result.gvfalb;
        gvfalbnosh[(result.r, result.c)] = result.gvfalbnosh;
        gvf_sum[(result.r, result.c)] = result.gvf_sum;
        gvf_lup_e[(result.r, result.c)] = result.gvf_lup_e;
        gvfalb_e[(result.r, result.c)] = result.gvfalb_e;
        gvfalbnosh_e[(result.r, result.c)] = result.gvfalbnosh_e;
        gvf_lup_s[(result.r, result.c)] = result.gvf_lup_s;
        gvfalb_s[(result.r, result.c)] = result.gvfalb_s;
        gvfalbnosh_s[(result.r, result.c)] = result.gvfalbnosh_s;
        gvf_lup_w[(result.r, result.c)] = result.gvf_lup_w;
        gvfalb_w[(result.r, result.c)] = result.gvfalb_w;
        gvfalbnosh_w[(result.r, result.c)] = result.gvfalbnosh_w;
        gvf_lup_n[(result.r, result.c)] = result.gvf_lup_n;
        gvfalb_n[(result.r, result.c)] = result.gvfalb_n;
        gvfalbnosh_n[(result.r, result.c)] = result.gvfalbnosh_n;
    }

    let mut gvf_norm = gvf_sum.clone();
    Zip::from(&mut gvf_norm)
        .and(&buildings)
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
