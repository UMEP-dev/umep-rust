//! GPU-accelerated cached GVF thermal accumulation.
//!
//! Fuses the `sun_on_surface_cached() × 18 azimuths` loop from
//! `gvf_calc_with_cache()` onto a single GPU compute dispatch. Each thread
//! handles one pixel and iterates over all azimuths internally.
//!
//! Shares `Arc<wgpu::Device>` and `Arc<wgpu::Queue>` with `ShadowGpuContext`.

use ndarray::{Array2, ArrayView2};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::gvf_geometry::GvfGeometryCache;

/// Ensures mapped staging buffers are always unmapped on scope exit.
struct MappedBufferGuard<'a> {
    buffer: &'a wgpu::Buffer,
}

impl<'a> MappedBufferGuard<'a> {
    fn new(buffer: &'a wgpu::Buffer) -> Self {
        Self { buffer }
    }
}

impl Drop for MappedBufferGuard<'_> {
    fn drop(&mut self) {
        self.buffer.unmap();
    }
}

// ── Uniform buffer (must match Params in gvf_cached.wgsl) ────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GvfParams {
    rows: u32,
    cols: u32,
    num_azimuths: u32,
    max_steps: u32,
    first: f32,
    second: f32,
    lwall: f32,
    wall_albedo: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AzimuthMetaGpu {
    dir_mask: u32,
    shift_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

const NUM_OUTPUT_CHANNELS: usize = 10;

// ── Cached GPU buffers ───────────────────────────────────────────────────

struct CachedBuffers {
    rows: usize,
    cols: usize,
    num_azimuths: usize,
    max_steps: usize,
    // Bind group 0: params + azimuth meta + shifts
    params_buffer: wgpu::Buffer,
    azimuth_meta_buffer: wgpu::Buffer,
    shifts_buffer: wgpu::Buffer,
    // Bind group 1: geometry (cached per DSM)
    blocking_distance_buffer: wgpu::Buffer,
    facesh_buffer: wgpu::Buffer,
    // Bind group 2: per-timestep inputs + output
    lup_buffer: wgpu::Buffer,
    albshadow_buffer: wgpu::Buffer,
    sunwall_mask_buffer: wgpu::Buffer,
    outputs_buffer: wgpu::Buffer,
    // Staging for readback
    staging_buffer: wgpu::Buffer,
    // Bind groups
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    bind_group_2: wgpu::BindGroup,
    // Track whether geometry has been uploaded
    geometry_uploaded: bool,
    readback_inflight: bool,
}

// ── Public context ───────────────────────────────────────────────────────

pub struct GvfGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_compute_workgroups_per_dimension: u32,
    pipeline: wgpu::ComputePipeline,
    bg_layout_0: wgpu::BindGroupLayout,
    bg_layout_1: wgpu::BindGroupLayout,
    bg_layout_2: wgpu::BindGroupLayout,
    cached: Mutex<Option<CachedBuffers>>,
}

/// Raw GPU output — 10 accumulated arrays before scaling/baseline.
pub struct GvfGpuResult {
    pub lup: Array2<f32>,
    pub alb: Array2<f32>,
    pub lup_e: Array2<f32>,
    pub alb_e: Array2<f32>,
    pub lup_s: Array2<f32>,
    pub alb_s: Array2<f32>,
    pub lup_w: Array2<f32>,
    pub alb_w: Array2<f32>,
    pub lup_n: Array2<f32>,
    pub alb_n: Array2<f32>,
}

/// In-flight GPU dispatch token.
pub struct GvfGpuPending {
    rows: usize,
    cols: usize,
    total_pixels: usize,
    staging_size: u64,
    submission_index: wgpu::SubmissionIndex,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl GvfGpuContext {
    /// Create a new context, sharing device/queue from the shadow GPU context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GVF Cached Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gvf_cached.wgsl").into()),
        });

        let bg_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG0 Layout"),
            entries: &Self::bg0_layout_entries(),
        });
        let bg_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG1 Layout"),
            entries: &Self::bg1_layout_entries(),
        });
        let bg_layout_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG2 Layout"),
            entries: &Self::bg2_layout_entries(),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GVF Pipeline Layout"),
            bind_group_layouts: &[&bg_layout_0, &bg_layout_1, &bg_layout_2],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GVF Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let max_compute_workgroups_per_dimension =
            device.limits().max_compute_workgroups_per_dimension;

        Self {
            device,
            queue,
            max_compute_workgroups_per_dimension,
            pipeline,
            bg_layout_0,
            bg_layout_1,
            bg_layout_2,
            cached: Mutex::new(None),
        }
    }

    /// Upload cached geometry from a GvfGeometryCache. Call once per DSM.
    pub fn upload_geometry(&self, cache: &GvfGeometryCache) -> Result<(), String> {
        let num_azimuths = cache.azimuths.len();
        if num_azimuths == 0 {
            return Err("No azimuths in GVF geometry cache".to_string());
        }
        let (rows, cols) = (
            cache.azimuths[0].blocking_distance.nrows(),
            cache.azimuths[0].blocking_distance.ncols(),
        );
        let max_steps = cache.second as usize;

        let mut buf_cache = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock GVF GPU cache: {}", e))?;

        self.ensure_buffers_locked(&mut buf_cache, rows, cols, num_azimuths, max_steps);
        let buffers = buf_cache
            .as_mut()
            .ok_or_else(|| "GVF GPU buffers missing after allocation".to_string())?;

        // Upload azimuth metadata
        let mut meta_data = Vec::with_capacity(num_azimuths);
        for (i, geom) in cache.azimuths.iter().enumerate() {
            let az = geom.azimuth_deg;
            let mut dir_mask = 0u32;
            if (0.0..180.0).contains(&az) {
                dir_mask |= 1;
            } // E
            if (90.0..270.0).contains(&az) {
                dir_mask |= 2;
            } // S
            if (180.0..360.0).contains(&az) {
                dir_mask |= 4;
            } // W
            if !(90.0..270.0).contains(&az) {
                dir_mask |= 8;
            } // N

            meta_data.push(AzimuthMetaGpu {
                dir_mask,
                shift_offset: (i * max_steps) as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }
        self.queue.write_buffer(
            &buffers.azimuth_meta_buffer,
            0,
            bytemuck::cast_slice(&meta_data),
        );

        // Upload shifts: flatten all azimuths' shifts into one buffer
        // Each shift is (dx, dy) as [i32; 2]
        let total_shifts = num_azimuths * max_steps;
        let mut shift_data = vec![[0i32; 2]; total_shifts];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            for (n, &(dx, dy)) in geom.shifts.iter().enumerate() {
                shift_data[i * max_steps + n] = [dx as i32, dy as i32];
            }
        }
        self.queue.write_buffer(
            &buffers.shifts_buffer,
            0,
            bytemuck::cast_slice(&shift_data),
        );

        // Upload blocking_distance: [az × rows × cols] as u32
        let total_geom_pixels = num_azimuths * rows * cols;
        let mut bd_data = vec![0u32; total_geom_pixels];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            let offset = i * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    bd_data[offset + r * cols + c] = geom.blocking_distance[[r, c]] as u32;
                }
            }
        }
        self.queue.write_buffer(
            &buffers.blocking_distance_buffer,
            0,
            bytemuck::cast_slice(&bd_data),
        );

        // Upload facesh: [az × rows × cols] as f32
        let mut facesh_data = vec![0.0f32; total_geom_pixels];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            let offset = i * rows * cols;
            if let Some(slice) = geom.facesh.as_slice() {
                facesh_data[offset..offset + rows * cols].copy_from_slice(slice);
            } else {
                for r in 0..rows {
                    for c in 0..cols {
                        facesh_data[offset + r * cols + c] = geom.facesh[[r, c]];
                    }
                }
            }
        }
        self.queue.write_buffer(
            &buffers.facesh_buffer,
            0,
            bytemuck::cast_slice(&facesh_data),
        );

        buffers.geometry_uploaded = true;
        Ok(())
    }

    /// Begin GPU dispatch for one timestep. Returns a pending token.
    pub fn dispatch_begin(
        &self,
        lup: ArrayView2<f32>,
        albshadow: ArrayView2<f32>,
        sunwall_mask: ArrayView2<f32>,
        first: f32,
        second: f32,
        lwall: f32,
        wall_albedo: f32,
    ) -> Result<GvfGpuPending, String> {
        let rows = lup.nrows();
        let cols = lup.ncols();
        let total_pixels = rows * cols;

        let mut cache = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock GVF GPU cache: {}", e))?;

        let buffers = cache
            .as_mut()
            .ok_or_else(|| "GVF GPU buffers not allocated".to_string())?;

        if !buffers.geometry_uploaded {
            return Err("GVF geometry not uploaded — call upload_geometry() first".to_string());
        }
        if buffers.readback_inflight {
            return Err("GVF GPU readback already in flight".to_string());
        }
        if buffers.rows != rows || buffers.cols != cols {
            return Err(format!(
                "Grid size mismatch: buffers {}x{} vs input {}x{}",
                buffers.rows, buffers.cols, rows, cols
            ));
        }

        buffers.readback_inflight = true;

        // Upload uniform params
        let params = GvfParams {
            rows: rows as u32,
            cols: cols as u32,
            num_azimuths: buffers.num_azimuths as u32,
            max_steps: buffers.max_steps as u32,
            first,
            second,
            lwall,
            wall_albedo,
        };
        self.queue
            .write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

        // Upload per-timestep inputs
        Self::write_2d_f32(&self.queue, &buffers.lup_buffer, &lup);
        Self::write_2d_f32(&self.queue, &buffers.albshadow_buffer, &albshadow);
        Self::write_2d_f32(&self.queue, &buffers.sunwall_mask_buffer, &sunwall_mask);

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GVF Cached Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GVF Cached Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group_0, &[]);
            pass.set_bind_group(1, &buffers.bind_group_1, &[]);
            pass.set_bind_group(2, &buffers.bind_group_2, &[]);

            let (wg_x, wg_y) = self.checked_workgroups_2d(rows, cols, 8, 8, "GVF cached")?;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output to staging
        let output_bytes = (total_pixels * NUM_OUTPUT_CHANNELS * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &buffers.outputs_buffer,
            0,
            &buffers.staging_buffer,
            0,
            output_bytes,
        );

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = buffers.staging_buffer.slice(..output_bytes);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        Ok(GvfGpuPending {
            rows,
            cols,
            total_pixels,
            staging_size: output_bytes,
            submission_index,
            map_rx: receiver,
        })
    }

    /// Complete an in-flight dispatch and read back the 10 output arrays.
    pub fn dispatch_end(&self, pending: GvfGpuPending) -> Result<GvfGpuResult, String> {
        let result = (|| {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(pending.submission_index),
                    timeout: None,
                })
                .map_err(|e| format!("GPU poll failed: {:?}", e))?;

            pending
                .map_rx
                .recv()
                .map_err(|e| format!("Channel recv failed: {}", e))?
                .map_err(|e| format!("Failed to map staging buffer: {:?}", e))?;

            let cache = self
                .cached
                .lock()
                .map_err(|e| format!("Failed to lock GVF buffer cache: {}", e))?;
            let buffers = cache
                .as_ref()
                .ok_or_else(|| "GVF GPU buffers missing".to_string())?;
            let buffer_slice = buffers.staging_buffer.slice(..pending.staging_size);
            let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
            let data = buffer_slice.get_mapped_range();
            let all_f32: &[f32] = bytemuck::cast_slice(&data);

            let n = pending.total_pixels;
            let extract = |ch: usize| -> Result<Array2<f32>, String> {
                Array2::from_shape_vec(
                    (pending.rows, pending.cols),
                    all_f32[ch * n..(ch + 1) * n].to_vec(),
                )
                .map_err(|e| format!("GVF output channel {}: {}", ch, e))
            };

            Ok(GvfGpuResult {
                lup: extract(0)?,
                alb: extract(1)?,
                lup_e: extract(2)?,
                alb_e: extract(3)?,
                lup_s: extract(4)?,
                alb_s: extract(5)?,
                lup_w: extract(6)?,
                alb_w: extract(7)?,
                lup_n: extract(8)?,
                alb_n: extract(9)?,
            })
        })();

        if let Ok(mut cache) = self.cached.lock() {
            if let Some(buffers) = cache.as_mut() {
                buffers.readback_inflight = false;
            }
        }

        result
    }

    // ── Buffer management ────────────────────────────────────────────────

    fn ensure_buffers_locked(
        &self,
        cache: &mut Option<CachedBuffers>,
        rows: usize,
        cols: usize,
        num_azimuths: usize,
        max_steps: usize,
    ) {
        if let Some(ref c) = *cache {
            if c.rows == rows
                && c.cols == cols
                && c.num_azimuths == num_azimuths
                && c.max_steps == max_steps
            {
                return;
            }
        }

        let total_pixels = rows * cols;
        let pixel_bytes = (total_pixels * 4) as u64;
        let geom_pixels = num_azimuths * total_pixels;
        let geom_f32_bytes = (geom_pixels * 4) as u64;
        let geom_u32_bytes = (geom_pixels * 4) as u64;
        let total_shifts = num_azimuths * max_steps;
        let shifts_bytes = (total_shifts * 8) as u64; // vec2<i32> = 8 bytes
        let meta_bytes = (num_azimuths * std::mem::size_of::<AzimuthMetaGpu>()) as u64;
        let output_bytes = (total_pixels * NUM_OUTPUT_CHANNELS * 4) as u64;

        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let input = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let output = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        // Bind group 0: params + azimuth meta + shifts
        let params_buffer = make(
            "GVF Params",
            std::mem::size_of::<GvfParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let azimuth_meta_buffer = make("GVF AzimuthMeta", meta_bytes, input);
        let shifts_buffer = make("GVF Shifts", shifts_bytes, input);

        // Bind group 1: geometry
        let blocking_distance_buffer = make("GVF BlockingDist", geom_u32_bytes, input);
        let facesh_buffer = make("GVF Facesh", geom_f32_bytes, input);

        // Bind group 2: per-timestep + output
        let lup_buffer = make("GVF Lup", pixel_bytes, input);
        let albshadow_buffer = make("GVF Albshadow", pixel_bytes, input);
        let sunwall_mask_buffer = make("GVF SunwallMask", pixel_bytes, input);
        let outputs_buffer = make("GVF Outputs", output_bytes, output);

        let staging_buffer = make(
            "GVF Staging",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF BG0"),
            layout: &self.bg_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: azimuth_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shifts_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF BG1"),
            layout: &self.bg_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocking_distance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: facesh_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF BG2"),
            layout: &self.bg_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lup_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: albshadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sunwall_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: outputs_buffer.as_entire_binding(),
                },
            ],
        });

        *cache = Some(CachedBuffers {
            rows,
            cols,
            num_azimuths,
            max_steps,
            params_buffer,
            azimuth_meta_buffer,
            shifts_buffer,
            blocking_distance_buffer,
            facesh_buffer,
            lup_buffer,
            albshadow_buffer,
            sunwall_mask_buffer,
            outputs_buffer,
            staging_buffer,
            bind_group_0,
            bind_group_1,
            bind_group_2,
            geometry_uploaded: false,
            readback_inflight: false,
        });
    }

    fn checked_workgroups_2d(
        &self,
        rows: usize,
        cols: usize,
        workgroup_x: u32,
        workgroup_y: u32,
        label: &str,
    ) -> Result<(u32, u32), String> {
        let wg_x = (cols as u32).div_ceil(workgroup_x);
        let wg_y = (rows as u32).div_ceil(workgroup_y);
        let limit = self.max_compute_workgroups_per_dimension;
        if wg_x > limit || wg_y > limit {
            return Err(format!(
                "{} dispatch exceeds GPU workgroup limit {}: ({}, {}) for {}x{} grid",
                label, limit, wg_x, wg_y, rows, cols
            ));
        }
        Ok((wg_x, wg_y))
    }

    fn write_2d_f32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView2<f32>) {
        if let Some(slice) = arr.as_slice() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        } else {
            let contiguous: Vec<f32> = arr.iter().copied().collect();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&contiguous));
        }
    }

    // ── Bind group layouts ───────────────────────────────────────────────

    fn bg0_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            // @binding(0) params: uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(1) azimuth_meta: storage read
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) shifts: storage read
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }

    fn bg1_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        vec![
            storage_ro(0), // blocking_distance
            storage_ro(1), // facesh
        ]
    }

    fn bg2_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        vec![
            storage_ro(0), // lup
            storage_ro(1), // albshadow
            storage_ro(2), // sunwall_mask
            storage_rw(3), // outputs
        ]
    }
}
