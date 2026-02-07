// SVF accumulation shader — runs after shadow propagation for each sky patch.
//
// Reads the 3 shadow output buffers (bldg_sh, veg_sh, veg_blocks_bldg_sh)
// and accumulates weighted values into a packed SVF data buffer.
//
// Layout of svf_data (15 arrays × total_pixels, all contiguous):
//   [0..N)       svf         (isotropic building)
//   [N..2N)      svf_n       (north)
//   [2N..3N)     svf_e       (east)
//   [3N..4N)     svf_s       (south)
//   [4N..5N)     svf_w       (west)
//   [5N..6N)     svf_veg     (isotropic vegetation)
//   [6N..7N)     svf_veg_n
//   [7N..8N)     svf_veg_e
//   [8N..9N)     svf_veg_s
//   [9N..10N)    svf_veg_w
//   [10N..11N)   svf_aveg    (isotropic veg-blocks-bldg)
//   [11N..12N)   svf_aveg_n
//   [12N..13N)   svf_aveg_e
//   [13N..14N)   svf_aveg_s
//   [14N..15N)   svf_aveg_w

struct SvfAccumParams {
    total_pixels: u32,
    weight_iso: f32,
    weight_n: f32,
    weight_e: f32,
    weight_s: f32,
    weight_w: f32,
    has_veg: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: SvfAccumParams;
@group(0) @binding(1) var<storage, read> bldg_sh: array<f32>;
@group(0) @binding(2) var<storage, read> veg_sh: array<f32>;
@group(0) @binding(3) var<storage, read> veg_blocks_bldg_sh: array<f32>;
@group(0) @binding(4) var<storage, read_write> svf_data: array<f32>;

@compute @workgroup_size(256)
fn accumulate_svf(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.total_pixels) {
        return;
    }

    let n = params.total_pixels;
    let b = bldg_sh[idx];

    // Accumulate building shadow into SVF (5 directional components)
    svf_data[idx]          += params.weight_iso * b;  // svf
    svf_data[idx + n]      += params.weight_n * b;    // svf_n
    svf_data[idx + 2u * n] += params.weight_e * b;    // svf_e
    svf_data[idx + 3u * n] += params.weight_s * b;    // svf_s
    svf_data[idx + 4u * n] += params.weight_w * b;    // svf_w

    if (params.has_veg == 1u) {
        let v = veg_sh[idx];

        // Accumulate vegetation shadow into SVF
        svf_data[idx + 5u * n]  += params.weight_iso * v;  // svf_veg
        svf_data[idx + 6u * n]  += params.weight_n * v;    // svf_veg_n
        svf_data[idx + 7u * n]  += params.weight_e * v;    // svf_veg_e
        svf_data[idx + 8u * n]  += params.weight_s * v;    // svf_veg_s
        svf_data[idx + 9u * n]  += params.weight_w * v;    // svf_veg_w

        let a = veg_blocks_bldg_sh[idx];

        // Accumulate veg-blocks-building shadow into SVF
        svf_data[idx + 10u * n] += params.weight_iso * a;  // svf_aveg
        svf_data[idx + 11u * n] += params.weight_n * a;    // svf_aveg_n
        svf_data[idx + 12u * n] += params.weight_e * a;    // svf_aveg_e
        svf_data[idx + 13u * n] += params.weight_s * a;    // svf_aveg_s
        svf_data[idx + 14u * n] += params.weight_w * a;    // svf_aveg_w
    }
}
