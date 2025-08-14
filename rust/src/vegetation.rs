use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Result container for lside_veg_v2022a direction-wise longwave fluxes.
#[pyclass]
pub struct LsideVegResult {
    #[pyo3(get)]
    pub least: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lsouth: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lwest: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub lnorth: Py<PyArray2<f32>>,
}

/// Vectorized Rust port of Python `Lside_veg_v2022a` operating on grid arrays.
/// Returns a `LsideVegResult` pyclass with four 2D arrays (least, lsouth, lwest, lnorth).
#[pyfunction]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn lside_veg(
    py: Python,
    svfS: PyReadonlyArray2<f32>,
    svfW: PyReadonlyArray2<f32>,
    svfN: PyReadonlyArray2<f32>,
    svfE: PyReadonlyArray2<f32>,
    svfEveg: PyReadonlyArray2<f32>,
    svfSveg: PyReadonlyArray2<f32>,
    svfWveg: PyReadonlyArray2<f32>,
    svfNveg: PyReadonlyArray2<f32>,
    svfEaveg: PyReadonlyArray2<f32>,
    svfSaveg: PyReadonlyArray2<f32>,
    svfWaveg: PyReadonlyArray2<f32>,
    svfNaveg: PyReadonlyArray2<f32>,
    azimuth: f32,
    altitude: f32,
    Ta: f32,
    Tw: f32,
    SBC: f32,
    ewall: f32,
    Ldown: PyReadonlyArray2<f32>,
    esky: f32,
    t: f32,
    F_sh: PyReadonlyArray2<f32>,
    CI: f32,
    LupE: PyReadonlyArray2<f32>,
    LupS: PyReadonlyArray2<f32>,
    LupW: PyReadonlyArray2<f32>,
    LupN: PyReadonlyArray2<f32>,
    anisotropic_longwave: bool,
) -> PyResult<Py<LsideVegResult>> {
    // Borrow arrays
    let svfS = svfS.as_array();
    let svfW = svfW.as_array();
    let svfN = svfN.as_array();
    let svfE = svfE.as_array();
    let svfEveg = svfEveg.as_array();
    let svfSveg = svfSveg.as_array();
    let svfWveg = svfWveg.as_array();
    let svfNveg = svfNveg.as_array();
    let svfEaveg = svfEaveg.as_array();
    let svfSaveg = svfSaveg.as_array();
    let svfWaveg = svfWaveg.as_array();
    let svfNaveg = svfNaveg.as_array();
    let Ldown = Ldown.as_array();
    let LupE = LupE.as_array();
    let LupS = LupS.as_array();
    let LupW = LupW.as_array();
    let LupN = LupN.as_array();
    let F_sh = F_sh.as_array();

    // Shape validation (all must match shape of svfE)
    let shape = svfE.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let vikttot: f32 = 4.4897;
    let TaK = Ta + 273.15;
    let TaK_pow4 = TaK.powi(4);
    // F_sh is per-cell; scaling to -1..1 handled inside loop per original Python (2*F_sh -1). No global scalar.
    let c = 1.0 - CI;
    let Lsky_allsky = esky * SBC * TaK_pow4 * (1.0 - c) + c * SBC * TaK_pow4;
    let altitude_day = altitude > 0.0;

    let sun_east = azimuth > (180.0 - t) && azimuth <= (360.0 - t);
    let sun_south = azimuth <= (90.0 - t) || azimuth > (270.0 - t);
    let sun_west = azimuth > (360.0 - t) || azimuth <= (180.0 - t);
    let sun_north = azimuth > (90.0 - t) && azimuth <= (270.0 - t);

    // Precompute azimuth temperature offsets (constant per grid)
    let temp_e = TaK + Tw * ((azimuth - 180.0 + t) * std::f32::consts::PI / 180.0).sin();
    let temp_s = TaK + Tw * ((azimuth - 270.0 + t) * std::f32::consts::PI / 180.0).sin();
    let temp_w = TaK + Tw * ((azimuth + t) * std::f32::consts::PI / 180.0).sin();
    let temp_n = TaK + Tw * ((azimuth - 90.0 + t) * std::f32::consts::PI / 180.0).sin();

    // Polynomial from Lvikt_veg
    #[inline]
    fn poly(x: f32) -> f32 {
        63.227 * x.powi(6) - 161.51 * x.powi(5) + 156.91 * x.powi(4) - 70.424 * x.powi(3)
            + 16.773 * x.powi(2)
            - 0.4863 * x
    }

    // Pre-allocate flat Vecs for each direction
    let npix = rows * cols;
    let mut least_vec = vec![0.0f32; npix];
    let mut lsouth_vec = vec![0.0f32; npix];
    let mut lwest_vec = vec![0.0f32; npix];
    let mut lnorth_vec = vec![0.0f32; npix];

    use rayon::prelude::*;
    least_vec
        .par_iter_mut()
        .zip(lsouth_vec.par_iter_mut())
        .zip(lwest_vec.par_iter_mut())
        .zip(lnorth_vec.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (((least, lsouth), lwest), lnorth))| {
            let r = idx / cols;
            let c = idx % cols;
            let compute = |svf: f32,
                           svfveg: f32,
                           svfaveg: f32,
                           lup: f32,
                           sun_cond: bool,
                           temp_wall: f32|
             -> f32 {
                if anisotropic_longwave {
                    return lup * 0.5;
                }
                let svfalfa = (1.0 - svf).clamp(0.0, 1.0).sqrt().asin();
                let poly_svf_val = poly(svf);
                let poly_svfaveg_val = poly(svfaveg);
                let viktonlywall = (vikttot - poly_svf_val) / vikttot;
                let viktaveg = (vikttot - poly_svfaveg_val) / vikttot;
                let viktwall = viktonlywall - viktaveg;
                let svfvegbu = svfveg + svf - 1.0;
                let poly_svfvegbu = poly(svfvegbu);
                let viktsky = poly_svfvegbu / vikttot;
                let viktrefl = (vikttot - poly_svfvegbu) / vikttot;
                let viktveg = viktrefl - viktwall;
                let fsh_val = F_sh[(r, c)];
                let ldown_val = Ldown[(r, c)];
                let f_sh_scaled = 2.0 * fsh_val - 1.0;
                let (lwallsun, lwallsh) = if altitude_day {
                    let alfa_b = svfalfa.atan();
                    let beta_b = (svfalfa * f_sh_scaled).tan().atan();
                    let betasun = ((alfa_b - beta_b) / 2.0) + beta_b;
                    if sun_cond {
                        let lwallsun = SBC
                            * ewall
                            * temp_wall.powi(4)
                            * viktwall
                            * (1.0 - f_sh_scaled)
                            * betasun.cos()
                            * 0.5;
                        let lwallsh = SBC * ewall * TaK_pow4 * viktwall * f_sh_scaled * 0.5;
                        (lwallsun, lwallsh)
                    } else {
                        let lwallsun = 0.0;
                        let lwallsh = SBC * ewall * TaK_pow4 * viktwall * 0.5;
                        (lwallsun, lwallsh)
                    }
                } else {
                    (0.0, SBC * ewall * TaK_pow4 * viktwall * 0.5)
                };
                let lsky = ((svf + svfveg - 1.0) * Lsky_allsky) * viktsky * 0.5;
                let lveg = SBC * ewall * TaK_pow4 * viktveg * 0.5;
                let lground = lup * 0.5;
                let lrefl = (ldown_val + lup) * viktrefl * (1.0 - ewall) * 0.5;
                lsky + lwallsun + lwallsh + lveg + lground + lrefl
            };
            *least = compute(
                svfE[(r, c)],
                svfEveg[(r, c)],
                svfEaveg[(r, c)],
                LupE[(r, c)],
                sun_east,
                temp_e,
            );
            *lsouth = compute(
                svfS[(r, c)],
                svfSveg[(r, c)],
                svfSaveg[(r, c)],
                LupS[(r, c)],
                sun_south,
                temp_s,
            );
            *lwest = compute(
                svfW[(r, c)],
                svfWveg[(r, c)],
                svfWaveg[(r, c)],
                LupW[(r, c)],
                sun_west,
                temp_w,
            );
            *lnorth = compute(
                svfN[(r, c)],
                svfNveg[(r, c)],
                svfNaveg[(r, c)],
                LupN[(r, c)],
                sun_north,
                temp_n,
            );
        });

    // Convert flat Vecs to Array2s
    let least = Array2::from_shape_vec((rows, cols), least_vec).unwrap();
    let lsouth = Array2::from_shape_vec((rows, cols), lsouth_vec).unwrap();
    let lwest = Array2::from_shape_vec((rows, cols), lwest_vec).unwrap();
    let lnorth = Array2::from_shape_vec((rows, cols), lnorth_vec).unwrap();

    Py::new(
        py,
        LsideVegResult {
            least: least.into_pyarray(py).unbind(),
            lsouth: lsouth.into_pyarray(py).unbind(),
            lwest: lwest.into_pyarray(py).unbind(),
            lnorth: lnorth.into_pyarray(py).unbind(),
        },
    )
}
