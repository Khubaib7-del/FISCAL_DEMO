import cv2
import numpy as np

def execute_upscale(image_matrix: np.ndarray) -> np.ndarray:
    """Upscales by 2x and applies a tightened Unsharp Mask (3x3, sigma 0.5)."""
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    height, width = image_matrix.shape[:2]
    if max(height, width) >= 4000:
        import logging
        logging.getLogger(__name__).debug(f"Upscale: Skipped — image resolution exceeded guard ({width}x{height})")
        return image_matrix

    # 1. High-fidelity interpolation resize expansion
    upscaled = cv2.resize(image_matrix, (int(width * 2), int(height * 2)), interpolation=cv2.INTER_CUBIC)

    # 2. Tightened Unsharp Mask
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.5)
    sharpened_matrix = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    
    return sharpened_matrix
