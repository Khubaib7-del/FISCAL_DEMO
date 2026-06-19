import cv2
import numpy as np

def execute_denoise(image_matrix: np.ndarray) -> np.ndarray:
    """
    Stage 5: High-frequency noise patch stripping.
    Supports both Grayscale and Color (BGR) denoising.
    """
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    is_color = len(image_matrix.shape) == 3
    
    if is_color:
        # Specialized color denoising to preserve original stamps/ink colors
        return cv2.fastNlMeansDenoisingColored(
            image_matrix, 
            None, 
            h=7, hColor=7,
            templateWindowSize=7, 
            searchWindowSize=21
        )
    else:
        # Standard luminance-only denoising
        return cv2.fastNlMeansDenoising(
            image_matrix, 
            None, 
            h=7, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
