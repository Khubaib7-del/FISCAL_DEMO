import cv2
import numpy as np

def execute_denoise(image_matrix: np.ndarray) -> np.ndarray:
    """
    Stage 5: High-frequency noise patch stripping.
    Replaces the wide bilateral filter with Fast NLM to stop soft gray transitions 
    along character borders that cause downstream ink-bloating.
    """
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    # h=7 strips sensor noise grain cleanly without creating wide blurred tracking halos
    return cv2.fastNlMeansDenoising(
        image_matrix, 
        None, 
        h=7, 
        templateWindowSize=7, 
        searchWindowSize=21
    )
