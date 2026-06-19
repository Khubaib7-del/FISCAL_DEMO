import cv2
import numpy as np

def execute_upscale(image_matrix: np.ndarray) -> np.ndarray:
    """Upscales dynamically to target a standard short-edge size (e.g., 1200px) and applies an Unsharp Mask."""
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    height, width = image_matrix.shape[:2]
    min_dim = min(height, width)
    target_short = 1200
    
    # Calculate scale factor to reach target short edge
    if min_dim < target_short:
        scale_factor = target_short / min_dim
        # Limit scale factor to a max of 3.0x to prevent excessive memory blowup
        scale_factor = min(3.0, scale_factor)
    else:
        # If already larger than target short edge, no upscale needed
        scale_factor = 1.0

    if scale_factor == 1.0:
        return image_matrix

    new_w = int(width * scale_factor)
    new_h = int(height * scale_factor)

    # 1. High-fidelity interpolation resize expansion
    upscaled = cv2.resize(image_matrix, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Tightened Unsharp Mask
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.5)
    sharpened_matrix = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    
    return sharpened_matrix
