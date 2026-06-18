import os
import cv2
import numpy as np
import logging
from PIL import Image
from pdf2image import convert_from_path

# Import the 5-stage modular processing pipeline steps belonging to our Preprocessing Layer
from steps import (
    execute_warp,
    execute_denoise,
    execute_illumination_correction,
    execute_upscale,
    execute_binarize
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DocumentPreprocessingLayer:
    def __init__(self):
        logging.info("Initializing Custom High-Visibility Preprocessing Layer (Round 2).")

    def process_and_save_stages(self, matrix: np.ndarray, filename: str, corners: list = None):
        """Executes the spec sequence and saves results to traceable stage folders."""
        try:
            name, ext = os.path.splitext(filename)
            
            # 1. Grayscale Conversion
            if len(matrix.shape) == 3:
                processed = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
            else:
                processed = matrix.copy()
            
            # 2. Illumination Normalization (CLAHE + Morph Background Divide + Gamma)
            processed = execute_illumination_correction(processed)
            cv2.imwrite(os.path.join("output/stage1_illumination", f"{name}_illumination{ext}"), processed)
            
            # 3. Geometric Warp (Straighten perspective on original density)
            processed = execute_warp(processed, corners=corners)
            cv2.imwrite(os.path.join("output/stage2_warp", f"{name}_warp{ext}"), processed)
            
            # --- FIXED: SWAP DENOISE BEFORE UPSCALE ---
            # 4. Denoise FIRST (on original resolution - faster and more effective)
            processed = execute_denoise(processed)
            cv2.imwrite(os.path.join("output/stage3_denoise", f"{name}_denoise{ext}"), processed)
            
            # 5. Upscale AFTER denoise (to avoid amplifying noise)
            processed = execute_upscale(processed)
            cv2.imwrite(os.path.join("output/stage4_upscale", f"{name}_upscale{ext}"), processed)
            
            # 6. Adaptive Gaussian Binarization LAST
            final_binary = execute_binarize(processed)
            cv2.imwrite(os.path.join("output/stage5_binarize", f"{name}_binarize{ext}"), final_binary)
            
            logging.info(f"Done: {filename} processed and saved through Round 2 stages.")
            return final_binary
        except Exception as e:
            logging.error(f"Failed to process {filename}: {str(e)}")
            return None

    def process_directory(self, input_dir: str):
        if not os.path.exists(input_dir):
            logging.error(f"Input directory does not exist: {input_dir}")
            return
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        if not files:
            logging.warning(f"No valid assets found in {input_dir}")
            return
        logging.info(f"Starting traceable batch processing for {len(files)} assets...")
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
                    logging.error(f"PDF Error ({filename}): {str(e)}")
            else:
                matrix = cv2.imread(file_path)
                if matrix is not None:
                    self.process_and_save_stages(matrix, filename)

if __name__ == "__main__":
    preprocessor = DocumentPreprocessingLayer()
    preprocessor.process_directory("input")
    logging.info("[+] Pipeline Fix Round 2 completed successfully!")
