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

    # Step 2: Adaptive CLAHE on Lightness
    std_dev = np.std(target)
    if std_dev < 20.0:
        clip_limit = 2.0
    elif std_dev < 40.0:
        clip_limit = 1.5
    else:
        clip_limit = 1.0
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(target)

    # Step 3: Dynamic Background Smoothing for normalization
    # Kernels scale with image size to prevent character washout on high-res images
    min_dim = min(enhanced_l.shape[:2])
    blur_k = int(min_dim * 0.02) * 2 + 1
    blur_k = max(9, blur_k)
    
    dilation_k = max(3, (blur_k // 3) | 1)
    
    dilated = cv2.dilate(enhanced_l, np.ones((dilation_k, dilation_k), np.uint8))
    bg_estimate = cv2.medianBlur(dilated, blur_k)
    
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
