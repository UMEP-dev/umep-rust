// Shadow float32 → uint8 quantization shader.
//
// Reads 3 float32 shadow output buffers and packs them into a compact u32 array
// where each u32 holds 4 consecutive uint8 values (little-endian byte order).
//
// Layout of packed_output:
//   [0..Q)       bldg_sh packed     (Q = ceil(total_pixels / 4))
//   [Q..2Q)      veg_sh packed      (only if has_veg)
//   [2Q..3Q)     vbsh packed        (only if has_veg)

struct U8PackParams {
    total_pixels: u32,
    num_quads: u32,       // ceil(total_pixels / 4)
    has_veg: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: U8PackParams;
@group(0) @binding(1) var<storage, read> bldg_sh: array<f32>;
@group(0) @binding(2) var<storage, read> veg_sh: array<f32>;
@group(0) @binding(3) var<storage, read> veg_blocks_bldg_sh: array<f32>;
@group(0) @binding(4) var<storage, read_write> packed_output: array<u32>;

@compute @workgroup_size(256)
fn shadow_to_u8(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_quads) {
        return;
    }

    let base = idx * 4u;
    let n = params.total_pixels;
    let q = params.num_quads;

    // Pack 4 consecutive building shadow float32 values into one u32 as 4×uint8
    var bldg_packed: u32 = 0u;
    for (var i = 0u; i < 4u; i++) {
        let px = base + i;
        if (px < n) {
            let val = u32(clamp(bldg_sh[px], 0.0, 1.0) * 255.0);
            bldg_packed |= (val & 0xFFu) << (i * 8u);
        }
    }
    packed_output[idx] = bldg_packed;

    if (params.has_veg == 1u) {
        // Pack vegetation shadow
        var veg_packed: u32 = 0u;
        for (var i = 0u; i < 4u; i++) {
            let px = base + i;
            if (px < n) {
                let val = u32(clamp(veg_sh[px], 0.0, 1.0) * 255.0);
                veg_packed |= (val & 0xFFu) << (i * 8u);
            }
        }
        packed_output[idx + q] = veg_packed;

        // Pack veg-blocks-building shadow
        var vbsh_packed: u32 = 0u;
        for (var i = 0u; i < 4u; i++) {
            let px = base + i;
            if (px < n) {
                let val = u32(clamp(veg_blocks_bldg_sh[px], 0.0, 1.0) * 255.0);
                vbsh_packed |= (val & 0xFFu) << (i * 8u);
            }
        }
        packed_output[idx + 2u * q] = vbsh_packed;
    }
}
