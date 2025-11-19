use ndarray::Array2;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Result struct for GPU shadow calculations
pub struct GpuShadowResult {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Option<Array2<f32>>,
    pub veg_blocks_bldg_sh: Option<Array2<f32>>,
    pub propagated_veg_height: Option<Array2<f32>>,
    pub wall_sh: Option<Array2<f32>>,
    pub wall_sun: Option<Array2<f32>>,
    pub wall_sh_veg: Option<Array2<f32>>,
    pub face_sh: Option<Array2<f32>>,
    pub face_sun: Option<Array2<f32>>,
}

/// GPU context for shadow calculations - maintains GPU resources across multiple calls
pub struct ShadowGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    wall_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ShadowGpuContext {
    /// Initialize GPU context - call once at startup
    pub async fn new() -> Result<Self, String> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find suitable GPU adapter: {:?}", e))?;

        // Request higher limits for storage buffers
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16; // We need 16 storage buffers

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shadow Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow_propagation.wgsl").into()),
        });

        // Create bind group layout for all shadow types
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[
                // Binding 0: Params buffer (uniforms)
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
                // Binding 1: DSM input (read-only)
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
                // Binding 2: Building shadow output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Propagated building height buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 4-9: Vegetation inputs and outputs
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 10-16: Wall inputs and outputs
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Propagation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let wall_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wall Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_wall_shadows"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            wall_pipeline,
            bind_group_layout,
        })
    }

    /// Compute all shadows (building, vegetation, walls) on GPU
    #[allow(clippy::too_many_arguments)]
    pub fn compute_all_shadows(
        &self,
        dsm: &Array2<f32>,
        veg_canopy_dsm_opt: Option<&Array2<f32>>,
        veg_trunk_dsm_opt: Option<&Array2<f32>>,
        bush_opt: Option<&Array2<f32>>,
        walls_opt: Option<&Array2<f32>>,
        aspect_opt: Option<&Array2<f32>>,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
    ) -> Result<GpuShadowResult, String> {
        let (rows, cols) = dsm.dim();
        let total_pixels = rows * cols;

        // Check if vegetation inputs are provided
        let has_veg =
            veg_canopy_dsm_opt.is_some() && veg_trunk_dsm_opt.is_some() && bush_opt.is_some();
        let has_walls = walls_opt.is_some() && aspect_opt.is_some();

        // Convert DSM to contiguous f32 slice
        let dsm_data: Vec<f32> = if dsm.is_standard_layout() {
            dsm.as_slice().unwrap().to_vec()
        } else {
            dsm.iter().copied().collect()
        };

        // Prepare vegetation data or create dummy buffers
        let veg_canopy_data: Vec<f32> = if let Some(veg_canopy) = veg_canopy_dsm_opt {
            if veg_canopy.is_standard_layout() {
                veg_canopy.as_slice().unwrap().to_vec()
            } else {
                veg_canopy.iter().copied().collect()
            }
        } else {
            vec![0.0; total_pixels]
        };

        let veg_trunk_data: Vec<f32> = if let Some(veg_trunk) = veg_trunk_dsm_opt {
            if veg_trunk.is_standard_layout() {
                veg_trunk.as_slice().unwrap().to_vec()
            } else {
                veg_trunk.iter().copied().collect()
            }
        } else {
            vec![0.0; total_pixels]
        };

        let bush_data: Vec<f32> = if let Some(bush) = bush_opt {
            if bush.is_standard_layout() {
                bush.as_slice().unwrap().to_vec()
            } else {
                bush.iter().copied().collect()
            }
        } else {
            vec![0.0; total_pixels]
        };

        // Prepare wall data or create dummy buffers
        let walls_data: Vec<f32> = if let Some(walls) = walls_opt {
            if walls.is_standard_layout() {
                walls.as_slice().unwrap().to_vec()
            } else {
                walls.iter().copied().collect()
            }
        } else {
            vec![0.0; total_pixels]
        };

        let aspect_data: Vec<f32> = if let Some(aspect) = aspect_opt {
            if aspect.is_standard_layout() {
                aspect.as_slice().unwrap().to_vec()
            } else {
                aspect.iter().copied().collect()
            }
        } else {
            vec![0.0; total_pixels]
        };

        // Precompute trigonometric values
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let sin_azimuth = azimuth_rad.sin();
        let cos_azimuth = azimuth_rad.cos();
        let tan_azimuth = azimuth_rad.tan();
        let tan_altitude_by_scale = altitude_rad.tan() / scale;
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let max_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_index = (max_reach_m / scale).ceil();

        // Create uniform buffer with parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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

        let params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth,
            cos_azimuth,
            tan_azimuth,
            tan_altitude_by_scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: if has_walls { 1 } else { 0 },
            _padding: 0,
        };

        // Create GPU buffers
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shadow Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Binding 1: DSM
        let dsm_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DSM Buffer"),
                contents: bytemuck::cast_slice(&dsm_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Binding 2: Building shadow output
        let bldg_shadow_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Building Shadow Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Binding 3: Propagated building height
        let propagated_bldg_height_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Propagated Building Height Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(
            &propagated_bldg_height_buffer,
            0,
            bytemuck::cast_slice(&dsm_data),
        );

        // Bindings 4-6: Vegetation input buffers
        let veg_canopy_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Veg Canopy Buffer"),
                contents: bytemuck::cast_slice(&veg_canopy_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let veg_trunk_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Veg Trunk Buffer"),
                contents: bytemuck::cast_slice(&veg_trunk_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let bush_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bush Buffer"),
                contents: bytemuck::cast_slice(&bush_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Bindings 7-9: Vegetation output buffers
        let veg_shadow_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Veg Shadow Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let propagated_veg_height_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Propagated Veg Height Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if has_veg {
            self.queue.write_buffer(
                &propagated_veg_height_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
        }

        let veg_blocks_bldg_shadow_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Veg Blocks Bldg Shadow Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bindings 10-11: Wall input buffers
        let walls_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Walls Buffer"),
                contents: bytemuck::cast_slice(&walls_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let aspect_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Aspect Buffer"),
                contents: bytemuck::cast_slice(&aspect_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Bindings 12-16: Wall output buffers
        let wall_sh_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Shadow Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let wall_sun_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Sun Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let wall_sh_veg_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Shadow Veg Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let face_sh_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Face Shadow Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let face_sun_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Face Sun Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group with all buffers
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dsm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: propagated_bldg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: veg_canopy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: veg_trunk_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bush_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: propagated_veg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: walls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: aspect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wall_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wall_sun_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wall_sh_veg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: face_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: face_sun_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit compute passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shadow Compute Encoder"),
            });

        let workgroup_size_x = 8;
        let workgroup_size_y = 8;
        let num_workgroups_x = (cols as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroups_y = (rows as u32 + workgroup_size_y - 1) / workgroup_size_y;

        // First pass: Main shadow propagation
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Second pass: Wall shadows (if enabled)
        if has_walls {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wall Shadow Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.wall_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Create staging buffers and copy results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size * 10, // Large enough for multiple outputs
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy building shadow
        encoder.copy_buffer_to_buffer(&bldg_shadow_buffer, 0, &staging_buffer, 0, buffer_size);

        // Copy vegetation outputs if enabled
        let veg_offset = buffer_size;
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &veg_shadow_buffer,
                0,
                &staging_buffer,
                veg_offset,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &veg_blocks_bldg_shadow_buffer,
                0,
                &staging_buffer,
                veg_offset + buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &propagated_veg_height_buffer,
                0,
                &staging_buffer,
                veg_offset + buffer_size * 2,
                buffer_size,
            );
        }

        // Copy wall outputs if enabled
        let wall_offset = buffer_size * 4;
        if has_walls {
            encoder.copy_buffer_to_buffer(
                &wall_sh_buffer,
                0,
                &staging_buffer,
                wall_offset,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &wall_sun_buffer,
                0,
                &staging_buffer,
                wall_offset + buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &wall_sh_veg_buffer,
                0,
                &staging_buffer,
                wall_offset + buffer_size * 2,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &face_sh_buffer,
                0,
                &staging_buffer,
                wall_offset + buffer_size * 3,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &face_sun_buffer,
                0,
                &staging_buffer,
                wall_offset + buffer_size * 4,
                buffer_size,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        // Read back all results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let all_data: &[f32] = bytemuck::cast_slice(&data);

        // Extract building shadow
        let bldg_sh = Array2::from_shape_vec((rows, cols), all_data[..total_pixels].to_vec())
            .map_err(|e| format!("Failed to create building shadow array: {}", e))?;

        // Extract vegetation results if enabled
        let (veg_sh, veg_blocks_bldg_sh, propagated_veg_height) = if has_veg {
            let veg_offset_px = total_pixels;
            let veg_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px..veg_offset_px + total_pixels].to_vec(),
            )
            .ok();
            let veg_blocks = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px + total_pixels..veg_offset_px + total_pixels * 2].to_vec(),
            )
            .ok();
            let prop_veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px + total_pixels * 2..veg_offset_px + total_pixels * 3]
                    .to_vec(),
            )
            .ok();
            (veg_sh, veg_blocks, prop_veg)
        } else {
            (None, None, None)
        };

        // Extract wall results if enabled
        let (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun) = if has_walls {
            let wall_offset_px = total_pixels * 4;
            let wall_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px..wall_offset_px + total_pixels].to_vec(),
            )
            .ok();
            let wall_sun = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels..wall_offset_px + total_pixels * 2].to_vec(),
            )
            .ok();
            let wall_sh_veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 2..wall_offset_px + total_pixels * 3]
                    .to_vec(),
            )
            .ok();
            let face_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 3..wall_offset_px + total_pixels * 4]
                    .to_vec(),
            )
            .ok();
            let face_sun = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 4..wall_offset_px + total_pixels * 5]
                    .to_vec(),
            )
            .ok();
            (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun)
        } else {
            (None, None, None, None, None)
        };

        drop(data);
        staging_buffer.unmap();

        Ok(GpuShadowResult {
            bldg_sh,
            veg_sh,
            veg_blocks_bldg_sh,
            propagated_veg_height,
            wall_sh,
            wall_sun,
            wall_sh_veg,
            face_sh,
            face_sun,
        })
    }
}

/// Synchronous wrapper that blocks on async GPU initialization
pub fn create_shadow_gpu_context() -> Result<ShadowGpuContext, String> {
    pollster::block_on(ShadowGpuContext::new())
}
