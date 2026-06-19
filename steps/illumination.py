import cv2
import numpy as np

def execute_illumination_correction(image_matrix: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE and Background Normalization to the Lightness channel 
    while preserving the original A/B color channels.
    """
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    # Step 1: Detect if input is Grayscale or Color
    is_color = len(image_matrix.shape) == 3
    
    if is_color:
        # Convert to LAB to isolate Lightness
        lab = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        target = l_channel
    else:
        target = image_matrix

    # Step 2: Subtle CLAHE on Lightness
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(target)

    # Step 3: Background Smoothing for normalization
    dilated = cv2.dilate(enhanced_l, np.ones((7, 7), np.uint8))
    bg_estimate = cv2.medianBlur(dilated, 21)
    
    # Step 4: Division Normalization
    normalized = cv2.divide(enhanced_l, bg_estimate, scale=255)
    
    # Step 5: Full Intensity Stretch [0, 255]
    final_l = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    if is_color:
        # Reconstruct LAB and convert back to BGR
        final_lab = cv2.merge((final_l, a, b))
        return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    else:
        return final_l
