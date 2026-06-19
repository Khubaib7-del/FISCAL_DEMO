import cv2
import numpy as np

def execute_binarize(image_matrix: np.ndarray) -> np.ndarray:
    """
    Stage 5: CamScanner-style 'Magic Color' Binarization.
    Whitens the document background while preserving and enhancing original text and stamp colors.
    """
    if image_matrix is None or image_matrix.size == 0:
        return image_matrix

    is_color = len(image_matrix.shape) == 3
    gray = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY) if is_color else image_matrix

    # 1. Background Texture Smoothing (Removes paper grain)
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Compute dynamic block size and subtract constant C
    min_dim = min(gray.shape[:2])
    block_size = int(min_dim * 0.035)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, block_size)
    
    std_dev = np.std(gray)
    c_val = 5 if std_dev < 30.0 else 10
    
    # 3. Get adaptive threshold mask (background is 255, text/foreground is 0)
    binary_mask = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c_val
    )

    # 4. Construct 'Magic Color' Output
    # Slight contrast boost (alpha=1.15, beta=-15) on foreground to enhance colored stamps/text
    enhanced = cv2.convertScaleAbs(image_matrix, alpha=1.15, beta=-15)
    
    if is_color:
        # Create a 3-channel version of the binary mask
        mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])
        # White background, original enhanced colors for text/stamps
        output = np.where(mask_3ch == 255, 255, enhanced)
    else:
        # White background, original enhanced grayscale levels for text/stamps
        output = np.where(binary_mask == 255, 255, enhanced)

    return output.astype(np.uint8)
