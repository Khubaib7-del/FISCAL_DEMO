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
    gray = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY) if is_color else image_matrix
    
    # Fast noise level estimation: average absolute difference from small Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = cv2.absdiff(gray, blurred)
    noise_est = np.mean(diff)
    
    # 1. Skip Denoising if noise is negligible
    if noise_est < 1.8:
        return image_matrix

    # 2. Moderate Noise: Run fast NLM (smaller search window)
    if noise_est < 3.8:
        if is_color:
            return cv2.fastNlMeansDenoisingColored(
                image_matrix, None, h=4, hColor=4,
                templateWindowSize=5, searchWindowSize=11
            )
        else:
            return cv2.fastNlMeansDenoising(
                image_matrix, None, h=4,
                templateWindowSize=5, searchWindowSize=11
            )

    # 3. High Noise: Run full strength NLM (deep search window)
    if is_color:
        return cv2.fastNlMeansDenoisingColored(
            image_matrix, None, h=7, hColor=7,
            templateWindowSize=7, searchWindowSize=21
        )
    else:
        return cv2.fastNlMeansDenoising(
            image_matrix, None, h=7,
            templateWindowSize=7, searchWindowSize=21
        )
