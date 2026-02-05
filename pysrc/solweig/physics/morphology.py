"""
Pure numpy implementations of morphological operations.

Replaces scipy.ndimage functions to eliminate the scipy dependency,
making the package lighter for QGIS plugin distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotate_array(
    array: NDArray[np.floating],
    angle: float,
    order: int = 1,
    reshape: bool = False,
    mode: str = "nearest",
) -> NDArray[np.floating]:
    """
    Rotate a 2D array by the given angle (in degrees).

    Pure numpy implementation replacing scipy.ndimage.interpolation.rotate.

    Args:
        array: 2D input array to rotate.
        angle: Rotation angle in degrees (counter-clockwise).
        order: Interpolation order (0=nearest, 1=bilinear).
        reshape: If True, output shape is adjusted to contain the whole rotated array.
                 If False (default), output has same shape as input.
        mode: How to handle boundaries ('nearest', 'constant').

    Returns:
        Rotated array.
    """
    if reshape:
        raise NotImplementedError("reshape=True not implemented")

    rows, cols = array.shape
    # scipy uses pixel-centered coordinates: center is at (n-1)/2 for n pixels
    center_y, center_x = (rows - 1) / 2, (cols - 1) / 2

    # Convert angle to radians
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Create output array
    output = np.zeros_like(array)

    # Create coordinate grids
    y_indices, x_indices = np.mgrid[0:rows, 0:cols]

    # Translate to center, rotate, translate back (inverse mapping)
    # For each output pixel, find the corresponding input pixel
    x_centered = x_indices - center_x
    y_centered = y_indices - center_y

    # Inverse rotation to find source coordinates
    # scipy.ndimage.rotate uses counter-clockwise in image coordinates (y pointing down)
    # For inverse mapping, we apply the transpose of the rotation matrix
    src_x = cos_t * x_centered - sin_t * y_centered + center_x
    src_y = sin_t * x_centered + cos_t * y_centered + center_y

    if order == 0:
        # Nearest neighbor interpolation
        src_x_int = np.round(src_x).astype(np.int32)
        src_y_int = np.round(src_y).astype(np.int32)

        # Clip to valid range
        src_x_int = np.clip(src_x_int, 0, cols - 1)
        src_y_int = np.clip(src_y_int, 0, rows - 1)

        output = array[src_y_int, src_x_int]

    elif order == 1:
        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Clip coordinates
        x0_clipped = np.clip(x0, 0, cols - 1)
        x1_clipped = np.clip(x1, 0, cols - 1)
        y0_clipped = np.clip(y0, 0, rows - 1)
        y1_clipped = np.clip(y1, 0, rows - 1)

        # Weights
        wx = src_x - x0
        wy = src_y - y0
        wx = np.clip(wx, 0, 1)
        wy = np.clip(wy, 0, 1)

        # Bilinear interpolation
        output = (
            array[y0_clipped, x0_clipped] * (1 - wx) * (1 - wy)
            + array[y0_clipped, x1_clipped] * wx * (1 - wy)
            + array[y1_clipped, x0_clipped] * (1 - wx) * wy
            + array[y1_clipped, x1_clipped] * wx * wy
        )
    else:
        raise ValueError(f"order must be 0 or 1, got {order}")

    return output.astype(array.dtype)


def binary_dilation(
    input_array: NDArray[np.bool_],
    structure: NDArray[np.bool_] | None = None,
    iterations: int = 1,
) -> NDArray[np.bool_]:
    """
    Perform binary dilation on a 2D boolean array.

    Pure numpy implementation replacing scipy.ndimage.binary_dilation.

    Args:
        input_array: 2D boolean array to dilate.
        structure: Structuring element (3x3 boolean array).
                   If None, uses 8-connectivity (all neighbors).
        iterations: Number of times to apply dilation.

    Returns:
        Dilated boolean array.
    """
    if structure is None:
        # Default: 8-connectivity (3x3 all ones)
        structure = np.ones((3, 3), dtype=bool)

    result = input_array.copy()

    for _ in range(iterations):
        # Pad the array
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        new_result = np.zeros_like(result)

        # Apply structuring element
        rows, cols = result.shape
        struct_rows, struct_cols = structure.shape
        offset_r = struct_rows // 2
        offset_c = struct_cols // 2

        for dr in range(struct_rows):
            for dc in range(struct_cols):
                if structure[dr, dc]:
                    shifted = padded[
                        1 + dr - offset_r : 1 + rows + dr - offset_r,
                        1 + dc - offset_c : 1 + cols + dc - offset_c,
                    ]
                    new_result |= shifted

        result = new_result

    return result


def generate_binary_structure(rank: int, connectivity: int) -> NDArray[np.bool_]:
    """
    Generate a binary structuring element for morphological operations.

    Pure numpy implementation replacing scipy.ndimage.generate_binary_structure.

    Args:
        rank: Number of dimensions (must be 2).
        connectivity: 1 for 4-connectivity (cross), 2 for 8-connectivity (square).

    Returns:
        3x3 boolean structuring element.
    """
    if rank != 2:
        raise ValueError(f"Only rank=2 supported, got {rank}")

    if connectivity == 1:
        # 4-connectivity (cross pattern)
        return np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
    elif connectivity == 2:
        # 8-connectivity (all neighbors)
        return np.ones((3, 3), dtype=bool)
    else:
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
