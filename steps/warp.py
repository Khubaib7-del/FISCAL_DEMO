import cv2
import numpy as np
import logging

def execute_warp(image_matrix: np.ndarray, corners: list = None) -> np.ndarray:
    """
    Stage 2: Precision Straightening & Alignment.
    Specifically designed to fix tilted text documents by detecting both 
    paper boundaries and dominant text-line angles.
    """
    try:
        if image_matrix is None or image_matrix.size == 0:
            return image_matrix

        # 1. Manual User Corners
        if corners is not None and len(corners) == 4:
            return apply_perspective_warp(image_matrix, np.float32(corners))
        
        # 2. Document Hull Detection (Perspective)
        doc_pts = detect_document_hull(image_matrix)
        if doc_pts is not None:
            h, w = image_matrix.shape[:2]
            area_ratio = cv2.contourArea(doc_pts) / (w * h)
            # Only use warp if it finds a reasonable sub-region (not the whole frame)
            if 0.1 < area_ratio < 0.95:
                logging.info(f"Warp: Document boundary detected ({area_ratio:.2f}). Applying perspective fix.")
                return apply_perspective_warp(image_matrix, doc_pts)

        # 3. Text-Line Alignment (Rotation)
        # We always try to level the text lines regardless of boundary detection
        return apply_line_deskew(image_matrix)

    except Exception as e:
        logging.error(f"Warp Layer Error: {str(e)}")
        return image_matrix

def detect_document_hull(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    target_h = 800
    scale = target_h / h
    resized = cv2.resize(gray, (int(w * scale), target_h))
    
    # Use Otsu + Morphological Cleanup to find paper
    _, thresh = cv2.threshold(cv2.GaussianBlur(resized, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.squeeze().astype(np.float32) * (1.0 / scale)
    return None

def apply_perspective_warp(image, src_pts):
    # Sort points: [top-left, top-right, bottom-right, bottom-left]
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
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

def apply_line_deskew(image):
    """Detects dominant line angles using Hough Transform and rotates to level."""
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use probabilistic Hough transform to find lines
    # minLineLength=w // 4 ensures we only look at reasonably long text segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=w // 4, maxLineGap=20)
    
    if lines is None:
        return image
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # We only care about horizontal-ish lines
        if -45 < angle < 45:
            angles.append(angle)
            
    if not angles:
        return image
        
    median_angle = np.median(angles)
    if abs(median_angle) < 0.1:
        return image
        
    logging.info(f"Warp: Text alignment detected at {median_angle:.2f} degrees. Rotating.")
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
