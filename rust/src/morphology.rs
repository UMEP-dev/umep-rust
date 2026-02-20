//! Morphological operations (binary dilation).
//!
//! Replaces the pure-Python implementation in `physics/morphology.py`
//! with an optimized Rust version using slice-based shift-and-OR.

use ndarray::{s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Binary dilation — pure Rust, no PyO3.
///
/// Uses the same shift-and-OR strategy as numpy: for each active position
/// in the structuring element, shift the entire grid and OR into the result.
/// This is cache-friendly and SIMD-vectorizable via ndarray's Zip.
pub(crate) fn binary_dilation_pure(
    input: ArrayView2<u8>,
    structure: ArrayView2<u8>,
    iterations: usize,
) -> Array2<u8> {
    let (rows, cols) = input.dim();
    let (sr, sc) = structure.dim();
    let offset_r = (sr / 2) as i32;
    let offset_c = (sc / 2) as i32;

    // Collect active structuring element offsets (relative to center)
    let offsets: Vec<(i32, i32)> = (0..sr)
        .flat_map(|dr| {
            (0..sc).filter_map(move |dc| {
                if structure[[dr, dc]] != 0 {
                    Some((dr as i32 - offset_r, dc as i32 - offset_c))
                } else {
                    None
                }
            })
        })
        .collect();

    let mut current = input.to_owned();

    for _ in 0..iterations {
        let mut new_result = Array2::<u8>::zeros((rows, cols));

        for &(dr, dc) in &offsets {
            // Compute overlapping ranges for source and destination
            let (src_r, dst_r, h) = shift_range(dr, rows);
            let (src_c, dst_c, w) = shift_range(dc, cols);

            if h == 0 || w == 0 {
                continue;
            }

            // Slice-based OR: cache-friendly, SIMD-vectorizable
            Zip::from(new_result.slice_mut(s![dst_r..dst_r + h, dst_c..dst_c + w]))
                .and(current.slice(s![src_r..src_r + h, src_c..src_c + w]))
                .for_each(|dst, &src| *dst |= src);
        }

        current = new_result;
    }

    current
}

/// Compute source start, destination start, and length for a shift offset.
#[inline]
fn shift_range(offset: i32, size: usize) -> (usize, usize, usize) {
    let n = size as i32;
    if offset >= 0 {
        // Shift right/down: source starts at 0, dest starts at offset
        let len = (n - offset) as usize;
        (0, offset as usize, len)
    } else {
        // Shift left/up: source starts at -offset, dest starts at 0
        let len = (n + offset) as usize;
        ((-offset) as usize, 0, len)
    }
}

/// Binary dilation (PyO3 wrapper).
///
/// Args:
///     input: 2D array (uint8, 0/1).
///     structure: 3×3 structuring element (uint8, 0/1).
///     iterations: Number of dilation passes.
///
/// Returns:
///     Dilated 2D array (uint8, 0/1).
#[pyfunction]
pub fn binary_dilation(
    py: Python<'_>,
    input: PyReadonlyArray2<u8>,
    structure: PyReadonlyArray2<u8>,
    iterations: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let input_v = input.as_array();
    let struct_v = structure.as_array();

    let result = binary_dilation_pure(input_v, struct_v, iterations);
    Ok(result.into_pyarray(py).unbind())
}
