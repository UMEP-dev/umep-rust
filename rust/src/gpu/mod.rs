// GPU acceleration module for shadow propagation
#[cfg(feature = "gpu")]
pub mod shadow_gpu;

#[cfg(feature = "gpu")]
pub use shadow_gpu::{create_shadow_gpu_context, GpuShadowResult, ShadowGpuContext};
