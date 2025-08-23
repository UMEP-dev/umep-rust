use core::f32;
use ndarray::{Array, Array2, Array3, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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

// Internal structure for accumulating contributions during parallel processing
#[derive(Clone)]
struct PatchContribution {
    num_rows: usize,
    num_cols: usize,
    svf: Array2<f32>,
    svf_n: Array2<f32>,
    svf_e: Array2<f32>,
    svf_s: Array2<f32>,
    svf_w: Array2<f32>,
    svf_veg: Array2<f32>,
    svf_veg_n: Array2<f32>,
    svf_veg_e: Array2<f32>,
    svf_veg_s: Array2<f32>,
    svf_veg_w: Array2<f32>,
    svf_veg_blocks_bldg_sh: Array2<f32>,
    svf_veg_blocks_bldg_sh_n: Array2<f32>,
    svf_veg_blocks_bldg_sh_e: Array2<f32>,
    svf_veg_blocks_bldg_sh_s: Array2<f32>,
    svf_veg_blocks_bldg_sh_w: Array2<f32>,
}

impl PatchContribution {
    // Create a new contribution object initialized with zeros
    // Always initialize all arrays, regardless of usevegdem
    fn zeros(num_rows: usize, num_cols: usize) -> Self {
        let zero_array = || Array2::zeros((num_rows, num_cols));
        Self {
            num_rows,
            num_cols,
            svf: zero_array(),
            svf_n: zero_array(),
            svf_e: zero_array(),
            svf_s: zero_array(),
            svf_w: zero_array(),
            svf_veg: zero_array(),
            svf_veg_n: zero_array(),
            svf_veg_e: zero_array(),
            svf_veg_s: zero_array(),
            svf_veg_w: zero_array(),
            svf_veg_blocks_bldg_sh: zero_array(),
            svf_veg_blocks_bldg_sh_n: zero_array(),
            svf_veg_blocks_bldg_sh_e: zero_array(),
            svf_veg_blocks_bldg_sh_s: zero_array(),
            svf_veg_blocks_bldg_sh_w: zero_array(),
        }
    }

    // Combine two contributions (used in reduce step)
    fn combine(mut self, other: Self) -> Self {
        self.svf += &other.svf;
        self.svf_n += &other.svf_n;
        self.svf_e += &other.svf_e;
        self.svf_s += &other.svf_s;
        self.svf_w += &other.svf_w;
        // Always combine veg arrays as they are always initialized
        self.svf_veg += &other.svf_veg;
        self.svf_veg_n += &other.svf_veg_n;
        self.svf_veg_e += &other.svf_veg_e;
        self.svf_veg_s += &other.svf_veg_s;
        self.svf_veg_w += &other.svf_veg_w;
        self.svf_veg_blocks_bldg_sh += &other.svf_veg_blocks_bldg_sh;
        self.svf_veg_blocks_bldg_sh_n += &other.svf_veg_blocks_bldg_sh_n;
        self.svf_veg_blocks_bldg_sh_e += &other.svf_veg_blocks_bldg_sh_e;
        self.svf_veg_blocks_bldg_sh_s += &other.svf_veg_blocks_bldg_sh_s;
        self.svf_veg_blocks_bldg_sh_w += &other.svf_veg_blocks_bldg_sh_w;
        self
    }

    // Finalize the results and convert to an intermediate pure-Rust result
    fn finalize_intermediate(
        mut self,
        usevegdem: bool,
        vegdem2: ArrayView2<f32>,
        bldg_sh_matrix: Array3<f32>,
        veg_sh_matrix: Array3<f32>,
        veg_blocks_bldg_sh_matrix: Array3<f32>,
    ) -> SvfIntermediate {
        // Apply correction factors (matching Python code)
        self.svf_s += LAST_ANNULUS_CORRECTION;
        self.svf_w += LAST_ANNULUS_CORRECTION;

        // Ensure no values exceed 1.0
        self.svf.mapv_inplace(|x| x.min(1.0));
        self.svf_n.mapv_inplace(|x| x.min(1.0));
        self.svf_e.mapv_inplace(|x| x.min(1.0));
        self.svf_s.mapv_inplace(|x| x.min(1.0));
        self.svf_w.mapv_inplace(|x| x.min(1.0));

        // Process veg arrays if needed
        if usevegdem {
            // Create correction array for veg components
            let last_veg =
                Array::from_shape_fn((self.num_rows, self.num_cols), |(row_idx, col_idx)| {
                    if vegdem2[[row_idx, col_idx]] == 0.0 {
                        LAST_ANNULUS_CORRECTION
                    } else {
                        0.0
                    }
                });

            // Apply corrections
            self.svf_veg_s += &last_veg;
            self.svf_veg_w += &last_veg;
            self.svf_veg_blocks_bldg_sh_s += &last_veg;
            self.svf_veg_blocks_bldg_sh_w += &last_veg;

            // Ensure no values exceed 1.0
            self.svf_veg.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_n.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_e.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_s.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_w.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_blocks_bldg_sh.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_blocks_bldg_sh_n.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_blocks_bldg_sh_e.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_blocks_bldg_sh_s.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_blocks_bldg_sh_w.mapv_inplace(|x| x.min(1.0));
        }

        SvfIntermediate {
            svf: self.svf,
            svf_n: self.svf_n,
            svf_e: self.svf_e,
            svf_s: self.svf_s,
            svf_w: self.svf_w,
            svf_veg: self.svf_veg,
            svf_veg_n: self.svf_veg_n,
            svf_veg_e: self.svf_veg_e,
            svf_veg_s: self.svf_veg_s,
            svf_veg_w: self.svf_veg_w,
            svf_veg_blocks_bldg_sh: self.svf_veg_blocks_bldg_sh,
            svf_veg_blocks_bldg_sh_n: self.svf_veg_blocks_bldg_sh_n,
            svf_veg_blocks_bldg_sh_e: self.svf_veg_blocks_bldg_sh_e,
            svf_veg_blocks_bldg_sh_s: self.svf_veg_blocks_bldg_sh_s,
            svf_veg_blocks_bldg_sh_w: self.svf_veg_blocks_bldg_sh_w,
            bldg_sh_matrix,
            veg_sh_matrix,
            veg_blocks_bldg_sh_matrix,
        }
    }
}

// --- Helper Functions ---
fn calculate_max_local_dsm_ht(dsm: ArrayView2<f32>, scale: f32) -> f32 {
    // Sliding-window size in meters (assumption). Use 100m by default.
    const LOCAL_WINDOW_M: f32 = 100.0;
    if !(scale.is_finite()) || scale <= 0.0 {
        return 0.0;
    }
    // Convert window radius from meters to pixels
    let radius = ((LOCAL_WINDOW_M / scale).ceil() as usize).max(0);
    if radius == 0 {
        return 0.0;
    }
    let (num_rows, num_cols) = (dsm.nrows(), dsm.ncols());
    // Parallel per-pixel implementation:
    // - Iterate over the flattened pixel indices in parallel
    // - Compute local min/max for each pixel's square window and return the range
    let total_pixels = num_rows.saturating_mul(num_cols);
    let ranges: Vec<f32> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let r = idx / num_cols;
            let c = idx % num_cols;

            let r0 = if r >= radius { r - radius } else { 0 };
            let r1 = (r + radius).min(num_rows - 1);
            let c0 = if c >= radius { c - radius } else { 0 };
            let c1 = (c + radius).min(num_cols - 1);

            let mut local_range = f32::NEG_INFINITY;
            let val: f32 = dsm[[r, c]];
            for rr in r0..=r1 {
                for cc in c0..=c1 {
                    let dv = dsm[[rr, cc]];
                    let nv = if dv.is_finite() { dv } else { 0.0 };
                    if val - nv > local_range {
                        local_range = val - nv;
                    }
                }
            }
            if local_range.is_finite() {
                (local_range).max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    // Keep only finite values (should already be finite) to be safe
    let mut finite_ranges: Vec<f32> = ranges.into_iter().filter(|v| v.is_finite()).collect();

    let final_value = if finite_ranges.is_empty() {
        0.0
    } else {
        let idx = (((finite_ranges.len() - 1) as f64) * 0.99).floor() as usize;
        // Use comparator for f32 partial ordering
        finite_ranges.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap());
        finite_ranges[idx]
    };

    eprintln!(
        "[umep-rust] percentile_of_local_ranges(99)={:.3}",
        final_value
    );

    final_value
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
    dsm_py: PyReadonlyArray2<f32>,
    vegdem_py: PyReadonlyArray2<f32>,
    vegdem2_py: PyReadonlyArray2<f32>,
    scale: f32,
    usevegdem: bool,
    patch_option: u8,
    min_sun_elev_deg: Option<f32>,
    max_shadow_length: Option<f32>,
    progress_counter: Option<Arc<AtomicUsize>>,
) -> PyResult<SvfIntermediate> {
    // Get array views from Python arrays
    let dsm_f32 = dsm_py.as_array();
    let vegdem_f32 = vegdem_py.as_array();
    let vegdem2_f32 = vegdem2_py.as_array(); // Keep f32 version for finalize step

    let num_rows = dsm_f32.nrows();
    let num_cols = dsm_f32.ncols();

    // Calculate maximum height for shadow calculations (local sliding-window)
    let max_local_dsm_ht = calculate_max_local_dsm_ht(dsm_f32, scale);

    // Prepare bushes
    let bush_f32 = prepare_bushes(vegdem_f32.view(), vegdem2_f32.view());

    // Create sky patches (use patch_option argument)
    let patches = create_patches(patch_option);
    let total_patches = patches.len(); // Needed for 3D array dimensions

    // Initialize 3D arrays to store shadow maps for each patch
    let bldg_sh_matrix = Arc::new(Mutex::new(Array::zeros((
        num_rows,
        num_cols,
        total_patches,
    ))));
    let veg_sh_matrix = Arc::new(Mutex::new(Array::zeros((
        num_rows,
        num_cols,
        total_patches,
    ))));
    let veg_blocks_bldg_sh_matrix = Arc::new(Mutex::new(Array::zeros((
        num_rows,
        num_cols,
        total_patches,
    ))));

    // Use parallel iterator with reduce to avoid collecting all results in memory
    let final_contribution = patches
        .par_iter()
        .enumerate()
        .map(|(patch_idx, patch)| {
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
                min_sun_elev_deg.unwrap_or(6.0_f32),
                max_shadow_length.unwrap_or(1000.0_f32),
            );

            // --- Calculate SVF contribution for this patch ---
            let mut contribution = PatchContribution::zeros(num_rows, num_cols);
            let bldg_sh_view = shadow_result.bldg_sh.view();

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

                contribution.svf.scaled_add(weight_iso, &bldg_sh_view);

                if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                    contribution.svf_e.scaled_add(weight_aniso, &bldg_sh_view);
                }
                if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                    contribution.svf_s.scaled_add(weight_aniso, &bldg_sh_view);
                }
                if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                    contribution.svf_w.scaled_add(weight_aniso, &bldg_sh_view);
                }
                if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                    contribution.svf_n.scaled_add(weight_aniso, &bldg_sh_view);
                }

                if usevegdem {
                    let veg_sh_view = shadow_result.veg_sh.view();
                    let veg_blocks_bldg_sh_view = shadow_result.veg_blocks_bldg_sh.view();

                    contribution.svf_veg.scaled_add(weight_iso, &veg_sh_view);
                    contribution
                        .svf_veg_blocks_bldg_sh
                        .scaled_add(weight_iso, &veg_blocks_bldg_sh_view);

                    if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                        contribution
                            .svf_veg_e
                            .scaled_add(weight_aniso, &veg_sh_view);
                        contribution
                            .svf_veg_blocks_bldg_sh_e
                            .scaled_add(weight_aniso, &veg_blocks_bldg_sh_view);
                    }
                    if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                        contribution
                            .svf_veg_s
                            .scaled_add(weight_aniso, &veg_sh_view);
                        contribution
                            .svf_veg_blocks_bldg_sh_s
                            .scaled_add(weight_aniso, &veg_blocks_bldg_sh_view);
                    }
                    if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                        contribution
                            .svf_veg_w
                            .scaled_add(weight_aniso, &veg_sh_view);
                        contribution
                            .svf_veg_blocks_bldg_sh_w
                            .scaled_add(weight_aniso, &veg_blocks_bldg_sh_view);
                    }
                    if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                        contribution
                            .svf_veg_n
                            .scaled_add(weight_aniso, &veg_sh_view);
                        contribution
                            .svf_veg_blocks_bldg_sh_n
                            .scaled_add(weight_aniso, &veg_blocks_bldg_sh_view);
                    }
                }
            }

            // Assign the shadow maps to the correct slice in the 3D arrays using Mutex for thread safety
            {
                let mut bldg_lock = bldg_sh_matrix.lock().unwrap();
                bldg_lock
                    .slice_mut(ndarray::s![.., .., patch_idx])
                    .assign(&shadow_result.bldg_sh);
            }
            if usevegdem {
                let mut veg_lock = veg_sh_matrix.lock().unwrap();
                veg_lock
                    .slice_mut(ndarray::s![.., .., patch_idx])
                    .assign(&shadow_result.veg_sh);
                let mut veg_blocks_lock = veg_blocks_bldg_sh_matrix.lock().unwrap();
                veg_blocks_lock
                    .slice_mut(ndarray::s![.., .., patch_idx])
                    .assign(&shadow_result.veg_blocks_bldg_sh);
            }

            // Update progress counter if provided (cheap atomic op)
            if let Some(ref counter) = progress_counter {
                counter.fetch_add(1, Ordering::SeqCst);
            }

            contribution
        })
        .reduce(
            || PatchContribution::zeros(num_rows, num_cols),
            |a, b| a.combine(b),
        );

    // Unwrap the matrices from Arc<Mutex<...>>
    let bldg_sh_matrix = Arc::try_unwrap(bldg_sh_matrix)
        .unwrap()
        .into_inner()
        .unwrap();
    let veg_sh_matrix = Arc::try_unwrap(veg_sh_matrix)
        .unwrap()
        .into_inner()
        .unwrap();
    let veg_blocks_bldg_sh_matrix = Arc::try_unwrap(veg_blocks_bldg_sh_matrix)
        .unwrap()
        .into_inner()
        .unwrap();

    // Finalize and return an intermediate result - pass the populated 3D arrays
    Ok(final_contribution.finalize_intermediate(
        usevegdem,
        vegdem2_f32,
        bldg_sh_matrix,
        veg_sh_matrix,
        veg_blocks_bldg_sh_matrix,
    ))
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
    patch_option: Option<u8>, // New argument for patch option
    min_sun_elev_deg: Option<f32>,
    max_shadow_length: Option<f32>,
    _progress_callback: Option<PyObject>,
) -> PyResult<Py<SvfResult>> {
    let patch_option = patch_option.unwrap_or(2);
    let inter = calculate_svf_inner(
        dsm_py,
        vegdem_py,
        vegdem2_py,
        scale,
        usevegdem,
        patch_option,
        Some(min_sun_elev_deg.unwrap_or(6.0_f32)),
        Some(max_shadow_length.unwrap_or(1000.0_f32)),
        None,
    )?;
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
        patch_option: Option<u8>,
        min_sun_elev_deg: Option<f32>,
        max_shadow_length: Option<f32>,
    ) -> PyResult<Py<SvfResult>> {
        let patch_option = patch_option.unwrap_or(2);
        // reset progress
        self.progress.store(0, Ordering::SeqCst);
        let inter = calculate_svf_inner(
            dsm_py,
            vegdem_py,
            vegdem2_py,
            scale,
            usevegdem,
            patch_option,
            Some(min_sun_elev_deg.unwrap_or(6.0_f32)),
            Some(max_shadow_length.unwrap_or(1000.0_f32)),
            Some(self.progress.clone()),
        )?;
        svf_intermediate_to_py(py, inter)
    }
}
