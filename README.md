# FISCAL_DEMO: Advanced Receipt Preprocessing Engine

A high-precision, 5-stage document scanning pipeline optimized for challenging thermal receipts. This engine achieves CamScanner-grade results by correcting physical paper curl, removing complex shadows, and preparing images for professional-tier OCR.

## 🚀 Pipeline Flow Overview

```mermaid
graph LR
    A[Raw Image] --> B[Stage 1: Illumination]
    B --> C[Stage 2: Denoise]
    C --> D[Stage 3: Warp & Align]
    D --> E[Stage 4: Upscale]
    E --> F[Grayscale Conversion]
    F --> G[Stage 5: Binarize]
    G --> H[OCR-Ready Output]
```

---

## 🛠 Stage Details & Internal Logic

### 1. Illumination Normalization
Balances uneven lighting and deletes harsh shadows to ensure uniform text visibility across the entire receipt.
- **Logic**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and global gamma correction.
- **Internal Flow**:
```mermaid
graph TD
    I1[BGR Image] --> I2[Convert to LAB Color Space]
    I2 --> I3[Apply CLAHE to 'L' Channel]
    I3 --> I4[Gamma Correction for Highlights]
    I4 --> I5[Reconstruct BGR Image]
```

### 2. Fast Non-Local Means Denoising
Removes paper grain and sensor noise without losing the sharp edges of text characters.
- **Logic**: Replaces standard bilateral filters with FNLM to prevent "soft gray borders" around characters.

### 3. Precision Warp & Multi-Tier Alignment
The core engine for correcting perspective and physical paper curl. This stage uses a 3-tier fallback strategy.
- **The "Curl" Solution**: Uses **Text-Content Margin Anchoring**. By fitting linear regressions to the actual left/right text mass, it corrects for "Smile/Frown" paper bends that simple rotation cannot fix.
- **Internal Flow**:
```mermaid
graph TD
    W1[Denoised Image] --> W2{Manual Corners?}
    W2 -- Yes --> W3[Apply Perspective Warp]
    W2 -- No --> W4{Quad Detected?}
    W4 -- Yes --> W5[Apply Quad Warp]
    W4 -- No --> W6{Text Margins Found?}
    W6 -- Yes --> W7[Fit Linear Margin Quad]
    W7 --> W3
    W6 -- No --> W8[Projection Profile Deskew]
    W3 --> W9[2% Content-Safe Crop]
```

### 4. High-Fidelity Upscaling
Increases resolution using interpolation methods that preserve text stroke integrity.
- **Logic**: Uses Lanczos4 interpolation followed by an **Unsharp Mask** (Gaussian Blur subtraction) to boost character definition.

### 5. Adaptive Gaussian Binarization
Final conversion to pure Black & White, optimized for OCR engines like Tesseract or Google Vision.
- **Logic**: Uses Adaptive Gaussian Thresholding with a precision block size (41) and texture smoothing to eliminate paper grain artifacts.

---

## 📦 Installation & Usage

1. **Setup Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline**:
   Place images in the `input/` folder and execute:
   ```bash
   python main.py
   ```

Outputs will be generated in the `output/` directory, categorized by stage.
