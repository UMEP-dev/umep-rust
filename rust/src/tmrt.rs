use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Physical constants
const SBC: f32 = 5.67e-8; // Stefan-Boltzmann constant (W/m²/K⁴)
const KELVIN_OFFSET: f32 = 273.15; // Kelvin to Celsius conversion

/// View factors for standing posture
const F_UP_STANDING: f32 = 0.06;
const F_SIDE_STANDING: f32 = 0.22;
const F_CYL_STANDING: f32 = 0.28;

/// View factors for sitting posture
const F_UP_SITTING: f32 = 0.166666;
const F_SIDE_SITTING: f32 = 0.166666;
const F_CYL_SITTING: f32 = 0.20;

/// Parameters for Tmrt calculation.
///
/// Groups scalar parameters to reduce function signature complexity.
#[pyclass]
#[derive(Clone)]
pub struct TmrtParams {
    /// Shortwave absorption coefficient (0.70 for clothed human)
    #[pyo3(get, set)]
    pub abs_k: f32,
    /// Longwave absorption coefficient (0.97 for clothed human)
    #[pyo3(get, set)]
    pub abs_l: f32,
    /// True for standing posture, False for sitting
    #[pyo3(get, set)]
    pub is_standing: bool,
    /// Whether anisotropic sky model was used
    #[pyo3(get, set)]
    pub use_anisotropic_sky: bool,
}

#[pymethods]
impl TmrtParams {
    #[new]
    pub fn new(abs_k: f32, abs_l: f32, is_standing: bool, use_anisotropic_sky: bool) -> Self {
        Self {
            abs_k,
            abs_l,
            is_standing,
            use_anisotropic_sky,
        }
    }
}

/// Pure-ndarray implementation of Tmrt calculation.
/// Callable from pipeline.rs (fused path) or from the PyO3 wrapper (modular path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_tmrt_pure(
    kdown: ArrayView2<f32>,
    kup: ArrayView2<f32>,
    ldown: ArrayView2<f32>,
    lup: ArrayView2<f32>,
    kside_n: ArrayView2<f32>,
    kside_e: ArrayView2<f32>,
    kside_s: ArrayView2<f32>,
    kside_w: ArrayView2<f32>,
    lside_n: ArrayView2<f32>,
    lside_e: ArrayView2<f32>,
    lside_s: ArrayView2<f32>,
    lside_w: ArrayView2<f32>,
    kside_total: ArrayView2<f32>,
    lside_total: ArrayView2<f32>,
    abs_k: f32,
    abs_l: f32,
    is_standing: bool,
    use_anisotropic_sky: bool,
) -> Array2<f32> {
    let shape = kdown.dim();

    // Select view factors based on posture
    let (f_up, f_side, f_cyl) = if is_standing {
        (F_UP_STANDING, F_SIDE_STANDING, F_CYL_STANDING)
    } else {
        (F_UP_SITTING, F_SIDE_SITTING, F_CYL_SITTING)
    };

    // Allocate output array
    let mut tmrt = Array2::<f32>::zeros(shape);

    // Compute Tmrt element-wise in parallel
    tmrt.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let row = idx / shape.1;
            let col = idx % shape.1;

            // Extract radiation components at this pixel
            let kdown_val = kdown[[row, col]];
            let kup_val = kup[[row, col]];
            let ldown_val = ldown[[row, col]];
            let lup_val = lup[[row, col]];
            let kside_n_val = kside_n[[row, col]];
            let kside_e_val = kside_e[[row, col]];
            let kside_s_val = kside_s[[row, col]];
            let kside_w_val = kside_w[[row, col]];
            let lside_n_val = lside_n[[row, col]];
            let lside_e_val = lside_e[[row, col]];
            let lside_s_val = lside_s[[row, col]];
            let lside_w_val = lside_w[[row, col]];
            let kside_total_val = kside_total[[row, col]];
            let lside_total_val = lside_total[[row, col]];

            // Compute absorbed radiation
            let k_absorbed = if use_anisotropic_sky {
                abs_k * (kside_total_val * f_cyl
                    + (kdown_val + kup_val) * f_up
                    + (kside_n_val + kside_e_val + kside_s_val + kside_w_val) * f_side)
            } else {
                abs_k * (kside_total_val * f_cyl
                    + (kdown_val + kup_val) * f_up
                    + (kside_n_val + kside_e_val + kside_s_val + kside_w_val) * f_side)
            };

            let l_absorbed = if use_anisotropic_sky {
                abs_l * ((ldown_val + lup_val) * f_up
                    + lside_total_val * f_cyl
                    + (lside_n_val + lside_e_val + lside_s_val + lside_w_val) * f_side)
            } else {
                abs_l
                    * ((ldown_val + lup_val) * f_up
                        + (lside_n_val + lside_e_val + lside_s_val + lside_w_val) * f_side)
            };

            // Total absorbed radiation (Sstr)
            let sstr = k_absorbed + l_absorbed;

            // Convert to Tmrt using Stefan-Boltzmann law
            // Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
            let tmrt_val = (sstr / (abs_l * SBC)).sqrt().sqrt() - KELVIN_OFFSET;

            // Clip to physically reasonable range
            *out = tmrt_val.clamp(-50.0, 80.0);
        });

    tmrt
}

/// Compute Mean Radiant Temperature (Tmrt) from radiation budget.
///
/// Tmrt represents the uniform temperature of an imaginary enclosure where
/// the radiant heat exchange with the human body equals that in the actual
/// non-uniform radiant environment.
///
/// Parameters:
/// - kdown/kup: Downwelling/upwelling shortwave radiation (W/m²)
/// - ldown/lup: Downwelling/upwelling longwave radiation (W/m²)
/// - kside_n/e/s/w: Directional shortwave radiation (W/m²)
/// - lside_n/e/s/w: Directional longwave radiation (W/m²)
/// - kside_total/lside_total: Total radiation on vertical surface (W/m²)
/// - params: TmrtParams with absorption coefficients and posture settings
///
/// Returns:
/// - Tmrt array in degrees Celsius, clipped to [-50, 80]
///
/// Formula:
///     Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
///     where Sstr = absorbed shortwave + absorbed longwave
///
/// Reference:
///     Lindberg et al. (2008): "SOLWEIG 1.0 - modelling spatial variations
///     of 3D radiant fluxes and mean radiant temperature in complex urban settings"
#[pyfunction]
pub fn compute_tmrt<'py>(
    py: Python<'py>,
    kdown: PyReadonlyArray2<'py, f32>,
    kup: PyReadonlyArray2<'py, f32>,
    ldown: PyReadonlyArray2<'py, f32>,
    lup: PyReadonlyArray2<'py, f32>,
    kside_n: PyReadonlyArray2<'py, f32>,
    kside_e: PyReadonlyArray2<'py, f32>,
    kside_s: PyReadonlyArray2<'py, f32>,
    kside_w: PyReadonlyArray2<'py, f32>,
    lside_n: PyReadonlyArray2<'py, f32>,
    lside_e: PyReadonlyArray2<'py, f32>,
    lside_s: PyReadonlyArray2<'py, f32>,
    lside_w: PyReadonlyArray2<'py, f32>,
    kside_total: PyReadonlyArray2<'py, f32>,
    lside_total: PyReadonlyArray2<'py, f32>,
    params: &TmrtParams,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let result = compute_tmrt_pure(
        kdown.as_array(),
        kup.as_array(),
        ldown.as_array(),
        lup.as_array(),
        kside_n.as_array(),
        kside_e.as_array(),
        kside_s.as_array(),
        kside_w.as_array(),
        lside_n.as_array(),
        lside_e.as_array(),
        lside_s.as_array(),
        lside_w.as_array(),
        kside_total.as_array(),
        lside_total.as_array(),
        params.abs_k,
        params.abs_l,
        params.is_standing,
        params.use_anisotropic_sky,
    );
    Ok(PyArray2::from_owned_array(py, result))
}
