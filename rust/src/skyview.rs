use core::f32;
use ndarray::{Array2, Array3, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Import the correct result struct from shadowing
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};

// Correction factor applied in finalize step
const LAST_ANNULUS_CORRECTION: f32 = 3.0459e-4;

// Struct to hold patch configurations

pub struct PatchInfo {
    pub altitude: f32,
    pub azimuth: f32,
    pub azimuth_patches: f32,
    pub azimuth_patches_aniso: f32,
    pub annulino_start: i32,
    pub annulino_end: i32,
}

fn create_patches(option: u8) -> Vec<PatchInfo> {
    let (annulino, altitudes, azi_starts, azimuth_patches) = match option {
        1 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![30, 30, 24, 24, 18, 12, 6, 1],
        ),
        2 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![31, 30, 28, 24, 19, 13, 7, 1],
        ),
        3 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![62, 60, 56, 48, 38, 26, 14, 2],
        ),
        4 => (
            vec![0, 4, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0],
            vec![62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 2],
        ),
        _ => panic!("Unsupported patch option: {}", option),
    };

    // Iterate over the patch configurations and create PatchInfo instances
    let mut patches: Vec<PatchInfo> = Vec::new();
    for i in 0..altitudes.len() {
        let azimuth_interval = 360.0 / azimuth_patches[i] as f32;
        for j in 0..azimuth_patches[i] as usize {
            // Calculate azimuth based on the start and interval
            // Use rem_euclid to ensure azimuth is within [0, 360)
            let azimuth =
                (azi_starts[i] as f32 + j as f32 * azimuth_interval as f32).rem_euclid(360.0);
            patches.push(PatchInfo {
                altitude: altitudes[i] as f32,
                azimuth,
                azimuth_patches: azimuth_patches[i] as f32,
                // Calculate anisotropic azimuth patches (ceil(interval/2))
                azimuth_patches_aniso: (azimuth_patches[i] as f32 / 2.0).ceil(),
                annulino_start: annulino[i] + 1, // Start from the next annulino degree to avoid overlap
                annulino_end: annulino[i + 1],
            });
        }
    }
    patches
}

// Structure to hold SVF results for Python
#[pyclass]
pub struct SvfResult {
    #[pyo3(get)]
    pub svf: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_blocks_bldg_sh_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub bldg_sh_matrix: Py<PyArray3<f32>>,
    #[pyo3(get)]
    pub veg_sh_matrix: Py<PyArray3<f32>>,
    #[pyo3(get)]
    pub veg_blocks_bldg_sh_matrix: Py<PyArray3<f32>>,
}

// Intermediate (pure Rust) SVF result used to avoid holding the GIL during compute
pub struct SvfIntermediate {
    pub svf: Array2<f32>,
    pub svf_n: Array2<f32>,
    pub svf_e: Array2<f32>,
    pub svf_s: Array2<f32>,
    pub svf_w: Array2<f32>,
    pub svf_veg: Array2<f32>,
    pub svf_veg_n: Array2<f32>,
    pub svf_veg_e: Array2<f32>,
    pub svf_veg_s: Array2<f32>,
    pub svf_veg_w: Array2<f32>,
    pub svf_veg_blocks_bldg_sh: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_n: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_e: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_s: Array2<f32>,
    pub svf_veg_blocks_bldg_sh_w: Array2<f32>,
    pub bldg_sh_matrix: Array3<f32>,
    pub veg_sh_matrix: Array3<f32>,
    pub veg_blocks_bldg_sh_matrix: Array3<f32>,
}

impl SvfIntermediate {
    /// Create a zero-initialized SvfIntermediate with the given dimensions.
    pub fn zeros(num_rows: usize, num_cols: usize, total_patches: usize) -> Self {
        let shape2 = (num_rows, num_cols);
        let shape3 = (num_rows, num_cols, total_patches);

        SvfIntermediate {
            svf: Array2::<f32>::zeros(shape2),
            svf_n: Array2::<f32>::zeros(shape2),
            svf_e: Array2::<f32>::zeros(shape2),
            svf_s: Array2::<f32>::zeros(shape2),
            svf_w: Array2::<f32>::zeros(shape2),
            svf_veg: Array2::<f32>::zeros(shape2),
            svf_veg_n: Array2::<f32>::zeros(shape2),
            svf_veg_e: Array2::<f32>::zeros(shape2),
            svf_veg_s: Array2::<f32>::zeros(shape2),
            svf_veg_w: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_n: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_e: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_s: Array2::<f32>::zeros(shape2),
            svf_veg_blocks_bldg_sh_w: Array2::<f32>::zeros(shape2),
            bldg_sh_matrix: Array3::<f32>::zeros(shape3),
            veg_sh_matrix: Array3::<f32>::zeros(shape3),
            veg_blocks_bldg_sh_matrix: Array3::<f32>::zeros(shape3),
        }
    }
}

fn prepare_bushes(vegdem: ArrayView2<f32>, vegdem2: ArrayView2<f32>) -> Array2<f32> {
    // Allocate output array with same shape as input
    let mut bush_areas = Array2::<f32>::zeros(vegdem.raw_dim());
    // Fill bush_areas in place, no unnecessary clones
    Zip::from(&mut bush_areas)
        .and(&vegdem)
        .and(&vegdem2)
        .for_each(|bush, &v1, &v2| {
            *bush = if v2 > 0.0 { 0.0 } else { v1 };
        });
    bush_areas
}

// --- Main Calculation Function ---
// Calculate SVF with 153 patches (equivalent to Python's svfForProcessing153)
// Internal implementation that supports an optional progress counter
fn calculate_svf_inner(
    dsm_owned: Array2<f32>,
    vegdem_owned: Array2<f32>,
    vegdem2_owned: Array2<f32>,
    scale: f32,
    usevegdem: bool,
    max_local_dsm_ht: f32,
    patch_option: u8,
    min_sun_elev_deg: Option<f32>,
    progress_counter: Option<Arc<AtomicUsize>>,
) -> PyResult<SvfIntermediate> {
    // Convert owned arrays to views for internal processing
    let dsm_f32 = dsm_owned.view();
    let vegdem_f32 = vegdem_owned.view();
    let vegdem2_f32 = vegdem2_owned.view(); // Keep f32 version for finalize step

    let num_rows = dsm_f32.nrows();
    let num_cols = dsm_f32.ncols();

    // Prepare bushes
    let bush_f32 = prepare_bushes(vegdem_f32.view(), vegdem2_f32.view());

    // Create sky patches (use patch_option argument)
    let patches = create_patches(patch_option);
    let total_patches = patches.len(); // Needed for 3D array dimensions

    // Create a single intermediate result and allocate all arrays there
    let mut inter = SvfIntermediate::zeros(num_rows, num_cols, total_patches);

    // Process patches sequentially: compute shadows (may be parallel internally),
    // immediately write shadow slices, then compute the per-patch contribution
    // using local parallelism (row-chunked) and merge into accumulator.
    for (patch_idx, patch) in patches.iter().enumerate() {
        let dsm_view = dsm_f32.view();
        // Only pass vegetation views if usevegdem is true, otherwise pass None
        let (vegdem_view, vegdem2_view, bush_view) = if usevegdem {
            (
                Some(vegdem_f32.view()),
                Some(vegdem2_f32.view()),
                Some(bush_f32.view()),
            )
        } else {
            (None, None, None)
        };
        // Calculate shadows for this patch
        let shadow_result: ShadowingResultRust = calculate_shadows_rust(
            patch.azimuth,
            patch.altitude,
            scale,
            max_local_dsm_ht,
            dsm_view,
            vegdem_view,
            vegdem2_view,
            bush_view,
            None,
            None,
            None,
            None,
            min_sun_elev_deg.unwrap_or(5.0_f32),
        );
        // --- Assign the shadow slices into the 3D matrices ---
        inter
            .bldg_sh_matrix
            .slice_mut(ndarray::s![.., .., patch_idx])
            .assign(&shadow_result.bldg_sh);
        if usevegdem {
            inter
                .veg_sh_matrix
                .slice_mut(ndarray::s![.., .., patch_idx])
                .assign(&shadow_result.veg_sh);
            inter
                .veg_blocks_bldg_sh_matrix
                .slice_mut(ndarray::s![.., .., patch_idx])
                .assign(&shadow_result.veg_blocks_bldg_sh);
        }

        // --- Per-patch vectorized accumulation (per-pixel) ---
        // --- Algorithmic block: Patch/annulus loop, weights, and accumulation ---
        let n = 90.0;
        let common_w_factor = (1.0 / (2.0 * PI)) * (PI / (2.0 * n)).sin();
        let steprad_iso = (360.0 / patch.azimuth_patches) * (PI / 180.0);
        let steprad_aniso = (360.0 / patch.azimuth_patches_aniso) * (PI / 180.0);

        for annulus_idx in patch.annulino_start..=patch.annulino_end {
            let annulus = 91.0 - annulus_idx as f32;
            let sin_term = ((PI * (2.0 * annulus - 1.0)) / (2.0 * n)).sin();
            let common_w_part = common_w_factor * sin_term;

            let weight_iso = steprad_iso * common_w_part;
            let weight_aniso = steprad_aniso * common_w_part;

            // Precompute directional anisotropic weights for this patch
            let weight_e = if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                weight_aniso
            } else {
                0.0
            };
            let weight_s = if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                weight_aniso
            } else {
                0.0
            };
            let weight_w = if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                weight_aniso
            } else {
                0.0
            };
            let weight_n = if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                weight_aniso
            } else {
                0.0
            };

            // Accumulate building shadows (parallel, SIMD-friendly)
            Zip::from(&shadow_result.bldg_sh)
                .and(&mut inter.svf)
                .and(&mut inter.svf_e)
                .and(&mut inter.svf_s)
                .and(&mut inter.svf_w)
                .and(&mut inter.svf_n)
                .par_for_each(|&b, svf, svf_e, svf_s, svf_w, svf_n| {
                    *svf += weight_iso * b;
                    *svf_e += weight_e * b;
                    *svf_s += weight_s * b;
                    *svf_w += weight_w * b;
                    *svf_n += weight_n * b;
                });

            if usevegdem {
                // Accumulate vegetation shadows
                Zip::from(&shadow_result.veg_sh)
                    .and(&mut inter.svf_veg)
                    .and(&mut inter.svf_veg_e)
                    .and(&mut inter.svf_veg_s)
                    .and(&mut inter.svf_veg_w)
                    .and(&mut inter.svf_veg_n)
                    .par_for_each(|&veg, svf_v, svf_v_e, svf_v_s, svf_v_w, svf_v_n| {
                        *svf_v += weight_iso * veg;
                        *svf_v_e += weight_e * veg;
                        *svf_v_s += weight_s * veg;
                        *svf_v_w += weight_w * veg;
                        *svf_v_n += weight_n * veg;
                    });

                // Accumulate veg-blocks-building shadows
                Zip::from(&shadow_result.veg_blocks_bldg_sh)
                    .and(&mut inter.svf_veg_blocks_bldg_sh)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_e)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_s)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_w)
                    .and(&mut inter.svf_veg_blocks_bldg_sh_n)
                    .par_for_each(
                        |&veg_bldg, svf_v_b, svf_v_be, svf_v_bs, svf_v_bw, svf_v_bn| {
                            *svf_v_b += weight_iso * veg_bldg;
                            *svf_v_be += weight_e * veg_bldg;
                            *svf_v_bs += weight_s * veg_bldg;
                            *svf_v_bw += weight_w * veg_bldg;
                            *svf_v_bn += weight_n * veg_bldg;
                        },
                    );
            } // end if usevegdem
        } // end annulus loop

        // Update progress counter after this patch is fully processed
        if let Some(ref counter) = progress_counter {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    } // end patch loop

    // Finalize: apply last-annulus correction and clamp values, same semantics as the previous finalize
    inter.svf_s += LAST_ANNULUS_CORRECTION;
    inter.svf_w += LAST_ANNULUS_CORRECTION;

    inter.svf.mapv_inplace(|x| x.min(1.0));
    inter.svf_n.mapv_inplace(|x| x.min(1.0));
    inter.svf_e.mapv_inplace(|x| x.min(1.0));
    inter.svf_s.mapv_inplace(|x| x.min(1.0));
    inter.svf_w.mapv_inplace(|x| x.min(1.0));

    if usevegdem {
        // Create correction array for veg components
        let last_veg = Array2::from_shape_fn((num_rows, num_cols), |(row_idx, col_idx)| {
            if vegdem2_f32[[row_idx, col_idx]] == 0.0 {
                LAST_ANNULUS_CORRECTION
            } else {
                0.0
            }
        });

        inter.svf_veg_s += &last_veg;
        inter.svf_veg_w += &last_veg;
        inter.svf_veg_blocks_bldg_sh_s += &last_veg;
        inter.svf_veg_blocks_bldg_sh_w += &last_veg;

        inter.svf_veg.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_n.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_e.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_s.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_w.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_blocks_bldg_sh.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_blocks_bldg_sh_n.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_blocks_bldg_sh_e.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_blocks_bldg_sh_s.mapv_inplace(|x| x.min(1.0));
        inter.svf_veg_blocks_bldg_sh_w.mapv_inplace(|x| x.min(1.0));
    }

    Ok(inter)
}

// Convert SvfIntermediate into Python SvfResult under the GIL
fn svf_intermediate_to_py(py: Python, inter: SvfIntermediate) -> PyResult<Py<SvfResult>> {
    Py::new(
        py,
        SvfResult {
            svf: inter.svf.into_pyarray(py).unbind(),
            svf_north: inter.svf_n.into_pyarray(py).unbind(),
            svf_east: inter.svf_e.into_pyarray(py).unbind(),
            svf_south: inter.svf_s.into_pyarray(py).unbind(),
            svf_west: inter.svf_w.into_pyarray(py).unbind(),
            svf_veg: inter.svf_veg.into_pyarray(py).unbind(),
            svf_veg_north: inter.svf_veg_n.into_pyarray(py).unbind(),
            svf_veg_east: inter.svf_veg_e.into_pyarray(py).unbind(),
            svf_veg_south: inter.svf_veg_s.into_pyarray(py).unbind(),
            svf_veg_west: inter.svf_veg_w.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh: inter.svf_veg_blocks_bldg_sh.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_north: inter.svf_veg_blocks_bldg_sh_n.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_east: inter.svf_veg_blocks_bldg_sh_e.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_south: inter.svf_veg_blocks_bldg_sh_s.into_pyarray(py).unbind(),
            svf_veg_blocks_bldg_sh_west: inter.svf_veg_blocks_bldg_sh_w.into_pyarray(py).unbind(),
            bldg_sh_matrix: inter.bldg_sh_matrix.into_pyarray(py).unbind(),
            veg_sh_matrix: inter.veg_sh_matrix.into_pyarray(py).unbind(),
            veg_blocks_bldg_sh_matrix: inter.veg_blocks_bldg_sh_matrix.into_pyarray(py).unbind(),
        },
    )
}

// Keep existing pyfunction wrapper for backward compatibility (ignores progress)
#[pyfunction]
pub fn calculate_svf(
    py: Python,
    dsm_py: PyReadonlyArray2<f32>,
    vegdem_py: PyReadonlyArray2<f32>,
    vegdem2_py: PyReadonlyArray2<f32>,
    scale: f32,
    usevegdem: bool,
    max_local_dsm_ht: f32,
    patch_option: Option<u8>, // New argument for patch option
    min_sun_elev_deg: Option<f32>,
    _progress_callback: Option<PyObject>,
) -> PyResult<Py<SvfResult>> {
    let patch_option = patch_option.unwrap_or(2);
    // Copy Python arrays into owned Rust arrays so computation can run without the GIL
    let dsm_owned = dsm_py.as_array().to_owned();
    let vegdem_owned = vegdem_py.as_array().to_owned();
    let vegdem2_owned = vegdem2_py.as_array().to_owned();
    let inter = py.allow_threads(|| {
        calculate_svf_inner(
            dsm_owned,
            vegdem_owned,
            vegdem2_owned,
            scale,
            usevegdem,
            max_local_dsm_ht,
            patch_option,
            min_sun_elev_deg,
            None,
        )
    })?;
    svf_intermediate_to_py(py, inter)
}

// New pyclass runner that exposes a progress() method and a calculate_svf that updates an internal counter
#[pyclass]
pub struct SkyviewRunner {
    progress: Arc<AtomicUsize>,
}

#[pymethods]
impl SkyviewRunner {
    #[new]
    pub fn new() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::SeqCst)
    }

    pub fn calculate_svf(
        &self,
        py: Python,
        dsm_py: PyReadonlyArray2<f32>,
        vegdem_py: PyReadonlyArray2<f32>,
        vegdem2_py: PyReadonlyArray2<f32>,
        scale: f32,
        usevegdem: bool,
        max_local_dsm_ht: f32,
        patch_option: Option<u8>,
        min_sun_elev_deg: Option<f32>,
    ) -> PyResult<Py<SvfResult>> {
        let patch_option = patch_option.unwrap_or(2);
        // reset progress
        self.progress.store(0, Ordering::SeqCst);
        // Copy arrays to owned buffers and run without the GIL so progress can be polled
        let dsm_owned = dsm_py.as_array().to_owned();
        let vegdem_owned = vegdem_py.as_array().to_owned();
        let vegdem2_owned = vegdem2_py.as_array().to_owned();
        let inter = py.allow_threads(|| {
            calculate_svf_inner(
                dsm_owned,
                vegdem_owned,
                vegdem2_owned,
                scale,
                usevegdem,
                max_local_dsm_ht,
                patch_option,
                min_sun_elev_deg,
                Some(self.progress.clone()),
            )
        })?;
        svf_intermediate_to_py(py, inter)
    }
}
