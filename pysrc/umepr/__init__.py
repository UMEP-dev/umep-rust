"""
UMEP-Rust: Urban Multi-scale Environmental Predictor (Rust implementation)
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .rustalgos import GPU_ENABLED, shadowing

    # Export GPU functions at package level
    __all__ = ["GPU_ENABLED", "shadowing"]

    # Enable GPU by default if available
    if GPU_ENABLED:
        shadowing.enable_gpu()
        logger.info("GPU acceleration enabled by default")
    else:
        logger.info("GPU support not compiled in this build")

except ImportError as e:
    # If rustalgos is not available or GPU feature not compiled
    logger.warning(f"Failed to import rustalgos GPU functions: {e}")
    GPU_ENABLED = False
    __all__ = ["GPU_ENABLED"]
