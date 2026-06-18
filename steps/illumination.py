import cv2
import numpy as np

def execute_illumination_correction(gray_matrix: np.ndarray) -> np.ndarray:
    """Applies CLAHE, 51x51 morphological division, and a 0.75 Gamma lift."""
    if gray_matrix is None or gray_matrix.size == 0:
        return gray_matrix

    # Step 1: Very subtle CLAHE to avoid noise amplification
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    output_matrix = clahe.apply(gray_matrix)

    # Step 2: Smoothing for better background estimation
    dilated = cv2.dilate(output_matrix, np.ones((7, 7), np.uint8))
    bg_estimate = cv2.medianBlur(dilated, 21)
    
    # Step 3: Division for baseline normalization
    normalized = cv2.divide(output_matrix, bg_estimate, scale=255)
    
    # Step 4: Soft Intensity Stretch (Avoid binary-look, keep some shades)
    # We remove the hard thresholding at 230 to keep it 'natural'
    output_matrix = cv2.normalize(normalized, None, 10, 255, cv2.NORM_MINMAX)

    return output_matrix
