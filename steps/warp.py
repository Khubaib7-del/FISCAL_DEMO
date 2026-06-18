import cv2
import numpy as np
import logging

def execute_warp(image_matrix: np.ndarray, corners: list = None) -> np.ndarray:
    """
    Stage 2: Precision Perspective & Leveling.
    Round 3: Implements Text-Content Margin Anchoring to fix paper curl 
    and progressive trapezoid distortion.
    """
    try:
        if image_matrix is None or image_matrix.size == 0:
            return image_matrix

        # 1. Manual User Corners (Priority)
        if corners is not None and len(corners) == 4:
            return apply_perspective_warp(image_matrix, np.float32(corners))
        
        # 2a. Document Hull Discovery (Standard Quad Detection)
        doc_pts = detect_document_hull_v2(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Document boundary detected via Quad Discovery.")
            warped = apply_perspective_warp(image_matrix, doc_pts)
            return apply_line_deskew(warped)

        # 2b. Blob boundary detection (Dark background fallback)
        doc_pts = detect_receipt_bounds(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Receipt boundary detected via blob bounds.")
            warped = apply_perspective_warp(image_matrix, doc_pts)
            return apply_line_deskew(warped)

        # 2c. NEW: Text-content boundary corners (handles curl + trapezoid)
        # This anchors the perspective transform to the actual start/end of text columns.
        doc_pts = detect_text_boundary_corners(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Using text-content boundary for perspective correction.")
            return apply_perspective_warp(image_matrix, doc_pts)

        # 3. Last resort: Projection-profile only deskew
        logging.info("Warp: All point detection failed. Falling back to projection-profile deskew.")
        return apply_line_deskew(image_matrix)

    except Exception as e:
        logging.error(f"Warp Layer Failure: {str(e)}")
        return image_matrix

def detect_document_hull_v2(image):
    """Adaptive quad detection using two-pass Canny thresholds."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    target_h = 1000
    scale = target_h / h
    resized = cv2.resize(gray, (int(w * scale), target_h))

    best_approx = None

    for low, high in [(30, 150), (10, 80)]:
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        edged = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.dilate(edged, kernel, iterations=2)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(c) / (target_h * int(w * scale))

        if 0.05 < area_ratio < 0.98:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                best_approx = approx.squeeze().astype(np.float32) * (1.0 / scale)
                break 

    return best_approx

def detect_receipt_bounds(image):
    """Fallback for receipt blobs. Skips if coverage > 85%."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(c) / (h * w)

    if area_ratio < 0.05 or area_ratio > 0.85:
        return None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def detect_text_boundary_corners(image):
    """
    Finds the 4 corners of the receipt by fitting lines to the 
    left and right text edges at top and bottom halves.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Use adaptive threshold for robust text-edge detection in shadows
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 41, 15)
    
    left_pts, right_pts = [], []
    step = max(1, h // 100)  # finer sampling

    for row in range(int(h * 0.05), int(h * 0.95), step):
        dark_cols = np.where(binary[row, :] > 0)[0]
        if len(dark_cols) < 15:
            continue
        left_pts.append([float(np.min(dark_cols)), float(row)])
        right_pts.append([float(np.max(dark_cols)), float(row)])

    if len(left_pts) < 10 or len(right_pts) < 10:
        return None

    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)

    def fit_line_endpoint(pts, y_target):
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return float(np.polyval(coeffs, y_target))

    # Ultra-wide padding (50% of width) to recover full margins
    est_width = np.median(right_pts[:, 0] - left_pts[:, 0])
    padding = est_width * 0.50 

    tl_x = fit_line_endpoint(left_pts[:len(left_pts)//2],  h * 0.01) - padding
    tr_x = fit_line_endpoint(right_pts[:len(right_pts)//2], h * 0.01) + padding
    bl_x = fit_line_endpoint(left_pts[len(left_pts)//2:],  h * 0.99) - padding
    br_x = fit_line_endpoint(right_pts[len(right_pts)//2:], h * 0.99) + padding

    # Clamp to image boundaries
    tl_x, bl_x = max(0, tl_x), max(0, bl_x)
    tr_x, br_x = min(w, tr_x), min(w, br_x)

    # Sanity check: valid quad coverage
    top_width = tr_x - tl_x
    bot_width = br_x - bl_x
    if top_width < w * 0.1 or bot_width < w * 0.1:
        return None
    if top_width > w * 1.1 or bot_width > w * 1.1:
        return None

    corners = np.array([
        [tl_x, h * 0.02],
        [tr_x, h * 0.02],
        [br_x, h * 0.98],
        [bl_x, h * 0.98],
    ], dtype=np.float32)

    logging.info(f"Warp: Text boundary corners found (TL={tl_x:.0f}, TR={tr_x:.0f})")
    return corners

def apply_perspective_warp(image, src_pts):
    """4-point perspective transform with a content-safe 2% margin."""
    rect = np.zeros((4, 2), dtype="float32")
    s = src_pts.sum(axis=1)
    rect[0] = src_pts[np.argmin(s)]
    rect[2] = src_pts[np.argmax(s)]
    diff = np.diff(src_pts, axis=1)
    rect[1] = src_pts[np.argmin(diff)]
    rect[3] = src_pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    
    # CONTENT-SAFE: Reduce crop to 2% (avoids cutting text on content-anchored quads)
    margin_w = int(0.02 * maxWidth)
    margin_h = int(0.02 * maxHeight)
    if margin_w > 0 and margin_h > 0:
        warped = warped[margin_h:-margin_h, margin_w:-margin_w]
        
    return warped

def apply_line_deskew(image):
    """Fine-grained deskew via middle-region projection profile variance."""
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    y_start, y_end = int(h * 0.25), int(h * 0.75)
    region = gray[y_start:y_end, :]
    rh, rw = region.shape

    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0

    for angle in np.linspace(-5.0, 5.0, 200):
        M = cv2.getRotationMatrix2D((rw // 2, rh // 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score = float(np.var(np.sum(rotated, axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)

    if abs(best_angle) < 0.1:
        return image

    logging.info(f"Warp: Base leveling by {best_angle:.3f}°")
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
