// WGSL compute shader for shadow propagation
// This implements the ray-marching shadow algorithm on GPU

struct ShadowParams {
    rows: u32,
    cols: u32,
    azimuth_rad: f32,
    altitude_rad: f32,
    sin_azimuth: f32,
    cos_azimuth: f32,
    tan_azimuth: f32,
    tan_altitude_by_scale: f32,
    scale: f32,
    max_index: f32,
    max_local_dsm_ht: f32,
    has_veg: u32,
    has_walls: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: ShadowParams;
@group(0) @binding(1) var<storage, read> dsm: array<f32>;
@group(0) @binding(2) var<storage, read_write> bldg_shadow: array<f32>;
@group(0) @binding(3) var<storage, read_write> propagated_bldg_height: array<f32>;
@group(0) @binding(4) var<storage, read> veg_canopy_dsm: array<f32>;
@group(0) @binding(5) var<storage, read> veg_trunk_dsm: array<f32>;
@group(0) @binding(6) var<storage, read> bush: array<f32>;
@group(0) @binding(7) var<storage, read_write> veg_shadow: array<f32>;
@group(0) @binding(8) var<storage, read_write> propagated_veg_height: array<f32>;
@group(0) @binding(9) var<storage, read_write> veg_blocks_bldg_shadow: array<f32>;
@group(0) @binding(10) var<storage, read> walls: array<f32>;
@group(0) @binding(11) var<storage, read> aspect: array<f32>;
@group(0) @binding(12) var<storage, read_write> wall_sh: array<f32>;
@group(0) @binding(13) var<storage, read_write> wall_sun: array<f32>;
@group(0) @binding(14) var<storage, read_write> wall_sh_veg: array<f32>;
@group(0) @binding(15) var<storage, read_write> face_sh: array<f32>;
@group(0) @binding(16) var<storage, read_write> face_sun: array<f32>;

const PI: f32 = 3.14159265359;
const PI_OVER_4: f32 = 0.78539816339;
const THREE_PI_OVER_4: f32 = 2.35619449019;
const FIVE_PI_OVER_4: f32 = 3.92699081699;
const SEVEN_PI_OVER_4: f32 = 5.49778714378;

fn get_index(row: i32, col: i32) -> i32 {
    return row * i32(params.cols) + col;
}

fn in_bounds(row: i32, col: i32) -> bool {
    return row >= 0 && row < i32(params.rows) && col >= 0 && col < i32(params.cols);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_row = i32(global_id.y);
    let target_col = i32(global_id.x);
    
    // Bounds check
    if (target_row >= i32(params.rows) || target_col >= i32(params.cols)) {
        return;
    }
    
    let target_idx = get_index(target_row, target_col);
    let target_dsm_height = dsm[target_idx];
    
    // Skip NaN/invalid pixels (NaN != NaN is true)
    if (target_dsm_height != target_dsm_height) {
        bldg_shadow[target_idx] = 0.0;
        propagated_bldg_height[target_idx] = target_dsm_height;
        if (params.has_veg != 0u) {
            veg_shadow[target_idx] = 0.0;
            propagated_veg_height[target_idx] = 0.0;
            veg_blocks_bldg_shadow[target_idx] = 0.0;
        }
        return;
    }
    
    // Initialize building shadow
    var is_bldg_shadowed = 0.0;
    var max_propagated_bldg_height = target_dsm_height;
    
    // Initialize vegetation shadow (if enabled)
    var veg_sh_value = 0.0;
    var max_propagated_veg_height = 0.0;
    var veg_blocks_bldg_count = 0.0;
    
    if (params.has_veg != 0u) {
        let bush_val = bush[target_idx];
        if (bush_val > 1.0) {
            veg_sh_value = 1.0;
        }
        max_propagated_veg_height = veg_canopy_dsm[target_idx];
    }
    
    // Determine step direction based on azimuth
    let azimuth = params.azimuth_rad;
    let use_sin_stepping = (azimuth >= PI_OVER_4 && azimuth < THREE_PI_OVER_4) ||
                           (azimuth >= FIVE_PI_OVER_4 && azimuth < SEVEN_PI_OVER_4);
    
    let sign_sin = sign(params.sin_azimuth);
    let sign_cos = sign(params.cos_azimuth);
    let ds_sin = abs(1.0 / params.sin_azimuth);
    let ds_cos = abs(1.0 / params.cos_azimuth);
    
    // Ray march from shadow source
    var index = 1.0;
    var prev_dz = 0.0;
    
    while (index <= params.max_index) {
        var dx: f32;
        var dy: f32;
        var ds: f32;
        
        if (use_sin_stepping) {
            dy = sign_sin * index;
            dx = -1.0 * sign_cos * round(abs(index / params.tan_azimuth));
            ds = ds_sin;
        } else {
            dy = sign_sin * round(abs(index * params.tan_azimuth));
            dx = -1.0 * sign_cos * index;
            ds = ds_cos;
        }
        
        let dz = (ds * index) * params.tan_altitude_by_scale;
        
        // Check if dz exceeds maximum height
        if (dz > params.max_local_dsm_ht) {
            break;
        }
        
        // Calculate source pixel position
        let source_row = target_row + i32(round(dx));
        let source_col = target_col + i32(round(dy));
        
        // Bounds check for source
        if (!in_bounds(source_row, source_col)) {
            index += 1.0;
            prev_dz = dz;
            continue;
        }
        
        let source_idx = get_index(source_row, source_col);
        let source_height = dsm[source_idx];
        
        // Skip NaN source pixels (NaN != NaN is true)
        if (source_height != source_height) {
            index += 1.0;
            prev_dz = dz;
            continue;
        }
        
        // Building shadow propagation
        let shifted_bldg_height = source_height - dz;
        
        // Update maximum propagated building height at target
        if (shifted_bldg_height > max_propagated_bldg_height) {
            max_propagated_bldg_height = shifted_bldg_height;
        }
        
        // Check if propagated height exceeds target DSM height
        if (shifted_bldg_height > target_dsm_height) {
            is_bldg_shadowed = 1.0;
        }
        
        // Vegetation shadow propagation (if enabled)
        if (params.has_veg != 0u) {
            let source_veg_canopy = veg_canopy_dsm[source_idx];
            let source_veg_trunk = veg_trunk_dsm[source_idx];
            
            // Skip NaN vegetation
            if (source_veg_canopy == source_veg_canopy && source_veg_trunk == source_veg_trunk) {
                let shifted_veg_canopy = source_veg_canopy - dz;
                let shifted_veg_trunk = source_veg_trunk - dz;
                let prev_shifted_veg_canopy = source_veg_canopy - prev_dz;
                let prev_shifted_veg_trunk = source_veg_trunk - prev_dz;
                
                // Update maximum propagated vegetation height
                if (shifted_veg_canopy > max_propagated_veg_height) {
                    max_propagated_veg_height = shifted_veg_canopy;
                }
                
                // Pergola shadow logic
                var cond_count = 0.0;
                if (shifted_veg_canopy > target_dsm_height) { cond_count += 1.0; }
                if (shifted_veg_trunk > target_dsm_height) { cond_count += 1.0; }
                if (prev_shifted_veg_canopy > target_dsm_height) { cond_count += 1.0; }
                if (prev_shifted_veg_trunk > target_dsm_height) { cond_count += 1.0; }
                
                if (cond_count > 0.0 && cond_count < 4.0) {
                    veg_sh_value = max(veg_sh_value, 1.0);
                }
                
                // Accumulate only when vegetation is not blocked by buildings at this iteration
                if (veg_sh_value > 0.0 && is_bldg_shadowed == 0.0) {
                    veg_blocks_bldg_count += 1.0;
                }
            }
        }
        
        prev_dz = dz;
        index += 1.0;
    }
    
    // Write building shadow results (inverted: 1 = sunlit, 0 = shadow)
    bldg_shadow[target_idx] = 1.0 - is_bldg_shadowed;
    propagated_bldg_height[target_idx] = max_propagated_bldg_height;
    
    // Write vegetation shadow results (if enabled)
    if (params.has_veg != 0u) {
        // Remove vegetation shadow where building shadow exists
        if (veg_sh_value > 0.0 && is_bldg_shadowed > 0.0) {
            veg_sh_value = 0.0;
        }
        
        // Invert vegetation shadow (1 = sunlit, 0 = shadow)
        if (veg_sh_value > 0.0) {
            veg_shadow[target_idx] = 0.0;
        } else {
            veg_shadow[target_idx] = 1.0;
        }
        
        // Process veg_blocks_bldg_shadow - represents veg that blocks the sky but NOT already blocked by buildings
        // 1. Convert count to binary (> 0 => 1) - this uses ALL veg shadow hits
        var vbs_value = select(0.0, 1.0, veg_blocks_bldg_count > 0.0);
        // 2. Subtract visible veg shadow mask (areas where veg is NOT blocked by buildings)
        // Use the FINAL veg_sh_value (after building shadow clearing)
        let veg_sh_mask = select(0.0, 1.0, veg_sh_value > 0.0);
        vbs_value = vbs_value - veg_sh_mask;
        // 3. Invert without clamping to mirror Python logic exactly
        veg_blocks_bldg_shadow[target_idx] = 1.0 - vbs_value;
        
        // Calculate propagated vegetation height above DSM
        let final_veg_mask = 1.0 - veg_shadow[target_idx];
        propagated_veg_height[target_idx] = max(max_propagated_veg_height - target_dsm_height, 0.0) * final_veg_mask;
    }
}

// Wall shadow computation - runs as a separate pass after main shadow computation
@compute @workgroup_size(8, 8, 1)
fn compute_wall_shadows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = i32(global_id.y);
    let col = i32(global_id.x);
    
    if (row >= i32(params.rows) || col >= i32(params.cols)) {
        return;
    }
    
    let idx = get_index(row, col);
    let wall_height = walls[idx];
    let aspect_val = aspect[idx];
    let dsm_height = dsm[idx];
    let prop_bldg_h = propagated_bldg_height[idx];
    
    // Wall mask: 1 if wall exists, 0 otherwise
    let wall_mask = select(0.0, 1.0, wall_height > 0.0);
    
    // Calculate face shadow based on aspect
    let azimuth = params.azimuth_rad;
    let azimuth_low = azimuth - (PI / 2.0);
    let azimuth_high = azimuth + (PI / 2.0);
    let TAU = 2.0 * PI;
    
    var face_shadow = 0.0;
    
    if (azimuth_low >= 0.0 && azimuth_high < TAU) {
        // Normal case: no wrapping
        if (aspect_val < azimuth_low || aspect_val >= azimuth_high) {
            face_shadow = 1.0;
        } else {
            face_shadow = 0.0;
        }
        // Adjust with wall mask
        face_shadow = face_shadow - wall_mask + 1.0;
    } else if (azimuth_low < 0.0 && azimuth_high <= TAU) {
        // Wrap at low end
        let azimuth_low_wrapped = azimuth_low + TAU;
        if (aspect_val > azimuth_low_wrapped || aspect_val <= azimuth_high) {
            face_shadow = -1.0;
        } else {
            face_shadow = 0.0;
        }
        face_shadow = face_shadow + 1.0;
    } else {
        // Wrap at high end
        let azimuth_high_wrapped = azimuth_high - TAU;
        if (aspect_val > azimuth_low || aspect_val <= azimuth_high_wrapped) {
            face_shadow = -1.0;
        } else {
            face_shadow = 0.0;
        }
        face_shadow = face_shadow + 1.0;
    }
    
    face_sh[idx] = face_shadow;
    
    // Calculate building shadow volume height
    let bldg_sh_vol_height = prop_bldg_h - dsm_height;
    
    // Calculate sunlit wall face
    // face_sun = 1 if: (face_shadow + wall_exists_flag) == 1.0 && wall_exists
    // This means: wall exists AND face is NOT in shadow (face_shadow == 0)
    let wall_exists = wall_height > 0.0;
    let wall_exists_flag = select(0.0, 1.0, wall_exists);
    let sum_check = face_shadow + wall_exists_flag;
    let face_sun_val = select(0.0, 1.0, abs(sum_check - 1.0) < 0.001 && wall_exists);
    face_sun[idx] = face_sun_val;
    
    // Calculate sunlit wall height
    var wall_sun_height = wall_height - bldg_sh_vol_height;
    wall_sun_height = max(wall_sun_height, 0.0);
    
    // Zero out sunlit height if face is in shadow
    if (abs(face_shadow - 1.0) < 0.001) {
        wall_sun_height = 0.0;
    }
    wall_sun[idx] = wall_sun_height;
    
    // Calculate shadowed wall height
    let wall_sh_height = wall_height - wall_sun_height;
    wall_sh[idx] = wall_sh_height;
    
    // Calculate vegetation shadow on walls (if vegetation enabled)
    if (params.has_veg != 0u) {
        let prop_veg_h = propagated_veg_height[idx];
        var veg_sh_wall = prop_veg_h * wall_mask;
        veg_sh_wall = veg_sh_wall - wall_sh_height;
        veg_sh_wall = max(veg_sh_wall, 0.0);
        
        // Clamp to wall height
        if (veg_sh_wall > wall_height) {
            veg_sh_wall = wall_height;
        }
        
        wall_sh_veg[idx] = veg_sh_wall;
        
        // Adjust wall_sun by removing vegetation shadow
        var adjusted_wall_sun = wall_sun_height - veg_sh_wall;
        
        // Zero out veg shadow if wall_sun becomes negative
        if (adjusted_wall_sun < 0.0) {
            wall_sh_veg[idx] = 0.0;
        } else {
            wall_sun[idx] = max(adjusted_wall_sun, 0.0);
        }
    }
}
