import cv2
import numpy as np
import logging

def execute_warp(image_matrix: np.ndarray, corners: list = None) -> np.ndarray:
    """
    Stage 2: Precision Perspective & Leveling.
    Implements multi-tiered boundary detection (Paper Blob, Document Hull, and Text-margin Fallback).
    """
    try:
        if image_matrix is None or image_matrix.size == 0:
            return image_matrix

        # 1. Manual User Corners (Priority)
        if corners is not None and len(corners) == 4:
            return apply_perspective_warp(image_matrix, np.float32(corners), margin_ratio=0.02, shrink_ratio=0.025)
        
        # 2a. Paper Blob Discovery (Adaptive threshold sweep + Otsu dark-bg fallback)
        doc_pts = detect_paper_blob_corners(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Receipt boundary detected via paper blob bounds.")
            # margin_ratio=0.01: blob corners are tight to paper edge, minimal crop needed
            # shrink_ratio=0.0: do not pre-shrink the quad — blob detection is already precise
            warped = apply_perspective_warp(image_matrix, doc_pts, margin_ratio=0.01, shrink_ratio=0.0)
            return apply_line_deskew(warped)

        # 2b. Document Hull Discovery (Canny-based Quad Fallback)
        doc_pts = detect_document_hull_v2(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Document boundary detected via Quad Discovery.")
            warped = apply_perspective_warp(image_matrix, doc_pts, margin_ratio=0.02, shrink_ratio=0.025)
            return apply_line_deskew(warped)

        # 2c. Text-content boundary corners (handles cut-off edges)
        # This anchors the perspective transform to the actual start/end of text columns.
        doc_pts = detect_text_boundary_corners(image_matrix)
        if doc_pts is not None:
            logging.info("Warp: Using text-content boundary for perspective correction.")
            return apply_perspective_warp(image_matrix, doc_pts, margin_ratio=0.02, shrink_ratio=0.0)

        # 3. Last resort: Projection-profile only deskew
        logging.info("Warp: All point detection failed. Falling back to projection-profile deskew.")
        return apply_line_deskew(image_matrix)

    except Exception as e:
        logging.error(f"Warp Layer Failure: {str(e)}")
        return image_matrix

def detect_document_hull_v2(image):
    """Adaptive quad detection using two-pass dynamic Canny thresholds."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Scale dynamically to maintain high precision on larger images
    target_h = 1500 if h >= 2000 else 1000
    scale = target_h / h
    resized = cv2.resize(gray, (int(w * scale), target_h))

    # Dynamic Canny thresholds based on image median intensity
    median_val = np.median(resized)
    low_thresh = int(max(10, 0.66 * median_val))
    high_thresh = int(min(255, 1.33 * median_val))
    
    threshold_pairs = [(low_thresh, high_thresh), (low_thresh // 2, high_thresh // 2)]
    best_approx = None

    # Dynamic morphological kernel size based on resized dimensions
    k_size = int(min(resized.shape[:2]) * 0.02)
    if k_size % 2 == 0:
        k_size += 1
    k_size = max(5, k_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    for low, high in threshold_pairs:
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        edged = cv2.Canny(blurred, low, high)
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
            elif 5 <= len(approx) <= 8:
                # Curled paper / rounded corners fallback: compute convex hull and force-approximate to quad
                hull = cv2.convexHull(approx)
                peri_hull = cv2.arcLength(hull, True)
                approx_quad = cv2.approxPolyDP(hull, 0.02 * peri_hull, True)
                if len(approx_quad) == 4:
                    best_approx = approx_quad.squeeze().astype(np.float32) * (1.0 / scale)
                    break

    return best_approx

def detect_paper_blob_corners(image):
    """
    Finds physical receipt corners via bilateral threshold sweep.
    Falls back to Otsu for dark-background images (e.g. receipt on wooden table).
    Returns None if no valid receipt boundary found.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Resize to 400px long edge for fast threshold sweeping
    scale = 400.0 / max(h, w)
    small_h, small_w = int(h * scale), int(w * scale)
    small = cv2.resize(gray, (small_w, small_h))
    blurred_small = cv2.bilateralFilter(small, 9, 75, 75)

    best_thresh = None
    max_ratio = 0.0
    prev_ratio = 0.0
    best_c = None

    # Sweep from bright to dark — first valid blob wins
    for t in range(245, 95, -2):
        _, thresh = cv2.threshold(blurred_small, t, 255, cv2.THRESH_BINARY)

        # Clear 5px border to prevent blob merging with image edge
        border_px = 5
        thresh[0:border_px, :] = 0
        thresh[small_h - border_px:small_h, :] = 0
        thresh[:, 0:border_px] = 0
        thresh[:, small_w - border_px:small_w] = 0

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            prev_ratio = 0.0
            continue

        c = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(c) / (small_h * small_w)

        # Dynamic jump detection to prevent background table bleed
        if prev_ratio > 0.35:
            jump_limit = 0.16 if prev_ratio < 0.60 else 0.10
            if (area_ratio - prev_ratio) > jump_limit:
                logging.info(f"Warp: Dynamic jump detected ({area_ratio:.3f} from {prev_ratio:.3f}). Stopping sweep.")
                break

        # FIXED: upper limit raised from 0.82 → 0.92
        # Receipts filling 80-92% of frame are still valid crops
        if 0.10 <= area_ratio <= 0.92 and area_ratio > max_ratio:
            max_ratio = area_ratio
            best_thresh = t
            best_c = c

        prev_ratio = area_ratio

    # Otsu fallback for dark-background images (e.g. receipt on wooden table)
    # Only fires when the sweep above found nothing
    if best_thresh is None:
        otsu_val, _ = cv2.threshold(blurred_small, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_val < 210:  # Only use Otsu when background is genuinely dark
            best_thresh = int(otsu_val)
            logging.info(f"Warp: Bilateral sweep failed. "
                         f"Using Otsu threshold {best_thresh} for dark-background receipt.")
            
            # Re-generate threshold mask at Otsu value and find contour
            _, thresh = cv2.threshold(blurred_small, best_thresh, 255, cv2.THRESH_BINARY)
            border_px = 5
            thresh[0:border_px, :] = 0
            thresh[small_h - border_px:small_h, :] = 0
            thresh[:, 0:border_px] = 0
            thresh[:, small_w - border_px:small_w] = 0
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best_c = max(cnts, key=cv2.contourArea)

    if best_thresh is None or best_c is None:
        logging.info("Warp: No valid receipt blob found in threshold sweep.")
        return None

    # Final sanity check — reject if blob covers entire image (no background separation)
    final_ratio = cv2.contourArea(best_c) / (small_h * small_w)
    if final_ratio < 0.05 or final_ratio > 0.97:
        logging.info(f"Warp: Blob coverage {final_ratio:.2f} out of valid range. Skipping.")
        return None

    # Extract 4 corners using sum/diff method
    pts = best_c.reshape(-1, 2)
    corners = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    corners[0] = pts[np.argmin(s)]    # Top-Left
    corners[2] = pts[np.argmax(s)]    # Bottom-Right
    diff = np.diff(pts, axis=1)
    corners[1] = pts[np.argmin(diff)] # Top-Right
    corners[3] = pts[np.argmax(diff)] # Bottom-Left

    # Scale corners back to full resolution
    corners_full = corners / scale

    logging.info(f"Warp: Blob corners found — "
                 f"TL={corners_full[0]} TR={corners_full[1]} "
                 f"BR={corners_full[2]} BL={corners_full[3]} "
                 f"coverage={final_ratio:.2f}")
    return corners_full

def detect_text_boundary_corners(image):
    """
    Finds the 4 corners of the receipt by locating the bounds of the text content
    vertically and horizontally, fitting lines, and applying standard margins.
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

    # Topmost and bottommost text rows (removes empty scanner borders at top/bottom)
    min_y = np.min(left_pts[:, 1])
    max_y = np.max(left_pts[:, 1])
    content_h = max_y - min_y
    y_padding = content_h * 0.03 # 3% height padding
    
    top_y = max(0, min_y - y_padding)
    bottom_y = min(h, max_y + y_padding)

    def fit_line_endpoint(pts, y_target):
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return float(np.polyval(coeffs, y_target))

    # Standard margin padding (8% of average text column width)
    est_width = np.median(right_pts[:, 0] - left_pts[:, 0])
    padding = est_width * 0.08 

    tl_x = fit_line_endpoint(left_pts[:len(left_pts)//2],  top_y) - padding
    tr_x = fit_line_endpoint(right_pts[:len(right_pts)//2], top_y) + padding
    bl_x = fit_line_endpoint(left_pts[len(left_pts)//2:],  bottom_y) - padding
    br_x = fit_line_endpoint(right_pts[len(right_pts)//2:], bottom_y) + padding

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
        [tl_x, top_y],
        [tr_x, top_y],
        [br_x, bottom_y],
        [bl_x, bottom_y],
    ], dtype=np.float32)

    logging.info(f"Warp: Text boundary corners found (TL={tl_x:.0f}, TR={tr_x:.0f})")
    return corners

def apply_perspective_warp(image, src_pts, margin_ratio=0.02, shrink_ratio=0.0):
    """4-point perspective transform with dynamic content-safe margin and pre-warp quad shrinking."""
    if shrink_ratio > 0.0:
        center = np.mean(src_pts, axis=0)
        src_pts = np.array([pt + shrink_ratio * (center - pt) for pt in src_pts], dtype=np.float32)

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
    
    # CONTENT-SAFE: Reduce crop to target margin ratio (avoids cutting text on content-anchored quads)
    if margin_ratio > 0.0:
        margin_w = int(margin_ratio * maxWidth)
        margin_h = int(margin_ratio * maxHeight)
        if margin_w > 0 and margin_h > 0:
            warped = warped[margin_h:-margin_h, margin_w:-margin_w]
        
    return warped

def apply_line_deskew(image):
    """Fine-grained deskew via middle-region projection profile variance."""
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # CHANGED: 15%-85% instead of 25%-75%
    # Gives more text rows to measure from, handles larger tilts and short receipts
    y_start, y_end = int(h * 0.15), int(h * 0.85)
    region = gray[y_start:y_end, :]
    rh, rw = region.shape

    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    # Expanded range (±20.0°) to handle aggressive phone photo angles
    for angle in np.linspace(-20.0, 20.0, 400):
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
