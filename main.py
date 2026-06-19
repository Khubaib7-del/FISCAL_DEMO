import os
import cv2
import numpy as np
import logging
from PIL import Image
from pdf2image import convert_from_path

# Import the 5-stage modular processing pipeline steps
from steps import (
    execute_warp,
    execute_denoise,
    execute_illumination_correction,
    execute_upscale,
    execute_binarize
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DocumentPreprocessingLayer:
    def __init__(self):
        logger.info("Initializing VLM-Optimized Preprocessing Layer (Production Refined).")

    def _is_already_enhanced(self, matrix: np.ndarray) -> bool:
        """Heuristic: If >85%% of pixels are already near-black or near-white, skip enhancement."""
        # Calculate histogram: focus on 0-20 and 235-255 ranges
        bin_counts = np.bincount(matrix.ravel(), minlength=256)
        extreme_pixels = np.sum(bin_counts[:20]) + np.sum(bin_counts[235:])
        ratio = extreme_pixels / matrix.size
        return ratio > 0.85

    def _check_orientation_drift(self, matrix: np.ndarray):
        """Simple projection profile check for 90-degree rotations."""
        h, w = matrix.shape
        # Sample center region
        region = matrix[h//4:3*h//4, w//4:3*w//4]
        h_var = np.var(np.mean(region, axis=1)) # Horizontal rows
        v_var = np.var(np.mean(region, axis=0)) # Vertical columns
        if v_var > h_var * 1.5:
             logger.warning("Orientation: Content mass suggests image may be rotated 90/270 degrees.")

    def process_and_save_stages(self, matrix: np.ndarray, filename: str, corners: list = None):
        """Executes the spec sequence while preserving color for VLM recognition."""
        try:
            name, ext = os.path.splitext(filename)
            
            # --- STAGE 0: INTEGRITY CHECK ---
            processed = matrix.copy()
            
            # Check for 90-degree rotations
            gray_for_check = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
            self._check_orientation_drift(gray_for_check)

            # Check if image is already a clean scan/CamScanner-enhanced
            already_clean = self._is_already_enhanced(gray_for_check)
            if already_clean:
                logger.info(f"Integrity: {filename} detected as already enhanced. Skipping redundant stages.")

            # --- STAGE 1: ILLUMINATION ---
            if not already_clean:
                processed = execute_illumination_correction(processed)
                cv2.imwrite(os.path.join("output/stage1_illumination", f"{name}_illumination{ext}"), processed)
            
            # --- STAGE 2: WARP (Always run for perspective correction) ---
            processed = execute_warp(processed, corners=corners)
            cv2.imwrite(os.path.join("output/stage2_warp", f"{name}_warp{ext}"), processed)
            
            # --- STAGE 3: DENOISE ---
            if not already_clean:
                processed = execute_denoise(processed)
                cv2.imwrite(os.path.join("output/stage3_denoise", f"{name}_denoise{ext}"), processed)
            
            # --- STAGE 4: UPSCALE ---
            # This grayscale upscaled image is the IDEAL output for VLM (PaddleOCR-VL-1.6)
            processed = execute_upscale(processed)
            cv2.imwrite(os.path.join("output/stage4_upscale", f"{name}_upscale{ext}"), processed)
            
            # --- STAGE 5: BINARIZE (Debug/Visual ONLY) ---
            # Binarization is kept for visual QA, but not passed as the primary production return.
            debug_binary = execute_binarize(processed)
            cv2.imwrite(os.path.join("output/stage5_binarize", f"{name}_binarize{ext}"), debug_binary)
            
            logger.info(f"Success: {filename} preprocessed for VLM Ingestion.")
            return processed # Return grayscale for VLM
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            return None

    def process_directory(self, input_dir: str):
        # Create output folders if they don't exist
        for folder in ["output/stage1_illumination", "output/stage2_warp", "output/stage3_denoise", "output/stage4_upscale", "output/stage5_binarize"]:
            os.makedirs(folder, exist_ok=True)

        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        if not files:
            logger.warning(f"No valid assets found in {input_dir}")
            return
        logger.info(f"Batch Start: Processing {len(files)} assets...")
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            if filename.lower().endswith(".pdf"):
                try:
                    pages = convert_from_path(file_path, dpi=200)
                    for i, page in enumerate(pages):
                        page_filename = f"{os.path.splitext(filename)[0]}_page_{i+1}.jpg"
                        matrix = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                        self.process_and_save_stages(matrix, page_filename)
                except Exception as e:
                    logger.error(f"PDF Error ({filename}): {str(e)}")
            else:
                matrix = cv2.imread(file_path)
                if matrix is not None:
                    self.process_and_save_stages(matrix, filename)

if __name__ == "__main__":
    preprocessor = DocumentPreprocessingLayer()
    preprocessor.process_directory("input")
    logger.info("[+] VLM-Ready Pipeline Refinement completed successfully!")
