use ndarray::Array2;
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

/// Internal implementation of Tmrt calculation.
#[allow(clippy::too_many_arguments)]
fn compute_tmrt_impl<'py>(
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
    abs_k: f32,
    abs_l: f32,
    is_standing: bool,
    use_anisotropic_sky: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Convert to ndarray
    let kdown_arr = kdown.as_array();
    let kup_arr = kup.as_array();
    let ldown_arr = ldown.as_array();
    let lup_arr = lup.as_array();
    let kside_n_arr = kside_n.as_array();
    let kside_e_arr = kside_e.as_array();
    let kside_s_arr = kside_s.as_array();
    let kside_w_arr = kside_w.as_array();
    let lside_n_arr = lside_n.as_array();
    let lside_e_arr = lside_e.as_array();
    let lside_s_arr = lside_s.as_array();
    let lside_w_arr = lside_w.as_array();
    let kside_total_arr = kside_total.as_array();
    let lside_total_arr = lside_total.as_array();

    let shape = kdown_arr.dim();

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
            let kdown_val = kdown_arr[[row, col]];
            let kup_val = kup_arr[[row, col]];
            let ldown_val = ldown_arr[[row, col]];
            let lup_val = lup_arr[[row, col]];
            let kside_n_val = kside_n_arr[[row, col]];
            let kside_e_val = kside_e_arr[[row, col]];
            let kside_s_val = kside_s_arr[[row, col]];
            let kside_w_val = kside_w_arr[[row, col]];
            let lside_n_val = lside_n_arr[[row, col]];
            let lside_e_val = lside_e_arr[[row, col]];
            let lside_s_val = lside_s_arr[[row, col]];
            let lside_w_val = lside_w_arr[[row, col]];
            let kside_total_val = kside_total_arr[[row, col]];
            let lside_total_val = lside_total_arr[[row, col]];

            // Compute absorbed radiation
            let k_absorbed = if use_anisotropic_sky {
                // Anisotropic model formula (cyl=1, aniso=1)
                // Uses full directional radiation with cylindrical projection
                abs_k * (kside_total_val * f_cyl // Anisotropic shortwave on vertical body surface
                    + (kdown_val + kup_val) * f_up // Downwelling + upwelling on top/bottom
                    + (kside_n_val + kside_e_val + kside_s_val + kside_w_val) * f_side)
                // Directional from 4 sides
            } else {
                // Isotropic model: use only direct beam on vertical (kside_total = kside_i)
                abs_k * (kside_total_val * f_cyl // Direct beam on vertical body surface
                    + (kdown_val + kup_val) * f_up // Downwelling + upwelling on top/bottom
                    + (kside_n_val + kside_e_val + kside_s_val + kside_w_val) * f_side)
                // Diffuse from 4 sides
            };

            let l_absorbed = if use_anisotropic_sky {
                abs_l * ((ldown_val + lup_val) * f_up
                    + lside_total_val * f_cyl // Anisotropic longwave on vertical surface
                    + (lside_n_val + lside_e_val + lside_s_val + lside_w_val) * f_side)
            } else {
                // Isotropic longwave: no lside_total term (only directional components)
                abs_l
                    * ((ldown_val + lup_val) * f_up
                        + (lside_n_val + lside_e_val + lside_s_val + lside_w_val) * f_side)
            };

            // Total absorbed radiation (Sstr)
            let sstr = k_absorbed + l_absorbed;

            // Convert to Tmrt using Stefan-Boltzmann law
            // Tmrt = (Sstr / (abs_l × SBC))^0.25 - 273.15
            // Using sqrt(sqrt(x)) for fourth root
            let tmrt_val = (sstr / (abs_l * SBC)).sqrt().sqrt() - KELVIN_OFFSET;

            // Clip to physically reasonable range
            *out = tmrt_val.clamp(-50.0, 80.0);
        });

    // Convert to PyArray
    let tmrt_py = PyArray2::from_owned_array(py, tmrt);
    Ok(tmrt_py)
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
    compute_tmrt_impl(
        py,
        kdown,
        kup,
        ldown,
        lup,
        kside_n,
        kside_e,
        kside_s,
        kside_w,
        lside_n,
        lside_e,
        lside_s,
        lside_w,
        kside_total,
        lside_total,
        params.abs_k,
        params.abs_l,
        params.is_standing,
        params.use_anisotropic_sky,
    )
}
