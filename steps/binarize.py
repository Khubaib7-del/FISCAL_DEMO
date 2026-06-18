import cv2
import numpy as np

def execute_binarize(image_matrix: np.ndarray) -> np.ndarray:
    """
    Stage 6: Fine-Stroke Adaptive Gaussian Binarization.
    Switches from large-area Mean thresholding to a tightly localized Gaussian window 
    to eliminate heavy-ink bolding and preserve fine character definitions.
    """
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    # If the matrix happens to still be 3-channel, protect the thresholding call
    if len(image_matrix.shape) == 3:
        gray_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    else:
        gray_matrix = image_matrix.copy()

    # 1. Background Texture Smoothing (Removes paper grain)
    # This prepares the image for a cleaner adaptive threshold by reducing micro-noise.
    gray_matrix = cv2.GaussianBlur(gray_matrix, (3, 3), 0)

    # CRITICAL MECHANICAL PARAMETERS:
    # 1. cv2.ADAPTIVE_THRESH_GAUSSIAN_C weights pixels closest to the center.
    # 2. blockSize=41 provides a broader context to ignore local illumination variations.
    # 3. C=30 aggressively thins character strokes to eliminate 'heavy ink' effects.
    binary = cv2.adaptiveThreshold(
        gray_matrix, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 41, 30
    )

    # NOISE STRIPPING: Remove small orphaned black pixels
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
