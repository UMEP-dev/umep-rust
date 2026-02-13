// Shadow float32 -> bitpacked matrix update shader.
//
// For each patch dispatch, this shader reads the 3 shadow output buffers and
// sets a single bit (patch-specific) in bitpacked shadow matrices.
//
// Layout of packed_output (byte-addressed):
//   matrix 0: bldg_sh bits     [0 .. matrix_bytes)
//   matrix 1: veg_sh bits      [matrix_bytes .. 2*matrix_bytes)      (if has_veg)
//   matrix 2: vbsh bits        [2*matrix_bytes .. 3*matrix_bytes)    (if has_veg)
//
// matrix_bytes = total_pixels * n_pack, where n_pack = ceil(n_patches / 8)
// Each matrix byte stores up to 8 patch bits for one pixel.
// Storage buffer uses u32 words; bytes are read/updated via atomic bit operations.

struct U8PackParams {
    total_pixels: u32,
    cols: u32,
    rows: u32,
    n_pack: u32,
    matrix_words: u32,
    has_veg: u32,
    patch_byte_idx: u32,
    patch_bit_mask: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: U8PackParams;
@group(0) @binding(1) var<storage, read> bldg_sh: array<f32>;
@group(0) @binding(2) var<storage, read> veg_sh: array<f32>;
@group(0) @binding(3) var<storage, read> veg_blocks_bldg_sh: array<f32>;
@group(0) @binding(4) var<storage, read_write> packed_output: array<atomic<u32>>;

fn set_shadow_bit(matrix_base_words: u32, pixel_idx: u32) {
    let byte_idx = pixel_idx * params.n_pack + params.patch_byte_idx;
    let word_idx = matrix_base_words + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    let bit = (params.patch_bit_mask & 0xFFu) << shift;
    atomicOr(&packed_output[word_idx], bit);
}

@compute @workgroup_size(16, 16, 1)
fn shadow_to_bitpack(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= params.cols || y >= params.rows) {
        return;
    }
    let idx = y * params.cols + x;
    if (idx >= params.total_pixels) {
        return;
    }

    // Matrix 0: building shadow
    if (bldg_sh[idx] >= 0.5) {
        set_shadow_bit(0u, idx);
    }

    if (params.has_veg == 1u) {
        let matrix_words = params.matrix_words;
        // Matrix 1: vegetation shadow
        if (veg_sh[idx] >= 0.5) {
            set_shadow_bit(matrix_words, idx);
        }
        // Matrix 2: veg-blocks-building shadow
        if (veg_blocks_bldg_sh[idx] >= 0.5) {
            set_shadow_bit(2u * matrix_words, idx);
        }
    }
}
