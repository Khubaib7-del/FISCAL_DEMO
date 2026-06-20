# Stage 2: Precision Warp & Perspective Dewarping

## 1. Architectural Purpose (The "Why")
Thermal receipts and invoice documents photographed on phones are almost always skewed by 3D perspective distortion (foreshortening). Standard 2D rotation cannot correct these angled dimensions.

Stage 2 locates the four true corners of the receipt page, crops out the desk/table background, executes a Homography perspective transform to flatten the document, and runs a projection variance sweep to align text lines to a perfect $0^\circ$ horizon.

---

## 2. Multi-Tiered Corner Detection Strategy

```mermaid
graph TD
    W1[Stage 1 Output] --> W2[Bilateral Sweep & Otsu Fallback]
    W2 --> W3{Blob Found?}
    W3 -- Yes --> W4[Scale Corners to High-Res]
    W3 -- No --> W5{Canny Quad Found?}
    W5 -- Yes --> W6[Get Quad Corners]
    W5 -- No --> W7[Fit Text Margin Regression lines]
    W4 --> W8[Apply Perspective Warp]
    W6 --> W8
    W7 --> W8
    W8 --> W9[apply_line_deskew]
```

### Tier A: Bilateral Sweep & Dynamic Jump Halting (Primary)
To segment receipt paper from textured surfaces (e.g., marble, wood grain):
1. **Downscale Analysis**: Resizes the image to a $400\text{px}$ long edge and runs a bilateral filter (`d=9`, `sigmaColor=75`) to smooth wood/marble textures while maintaining hard receipt borders.
2. **Threshold Sweep**: Sweeps binary thresholds from $245$ down to $95$ in steps of 2.
3. **Dynamic Jump Halting**: At each step, evaluates the contour area coverage. If a sudden jump in the area ratio occurs ($>0.16$ when area $<60\%$, or $>0.10$ when area $\ge 60\%$), it signals background table bleed. The sweep halts immediately at the step *before* the jump, locking in the receipt boundary.
4. **Otsu Fallback**: If the sweep fails, Otsu binarization is run (triggered only if Otsu threshold value $<210$, confirming a dark background) to segment the paper blob.
5. **Direct Scale Mapping**: Extracts extreme points (Top-Left, Top-Right, Bottom-Right, Bottom-Left) on the $400\text{px}$ image and maps them to high resolution by multiplying by `1.0 / scale`. This bypasses multi-resolution smoothing mismatches.

### Tier B: Canny Document Hull (Secondary Fallback)
If the sweep fails, runs two-pass adaptive Canny edge detection. Morphes and closes edges to fit a quad convex hull approximation.

### Tier C: Text-Content Margin Anchoring (Tertiary Fallback)
If paper borders are invisible (e.g., receipt fills the frame), detects printed text blocks, runs linear regression on leftmost/rightmost coordinates to construct boundary lines, and adds an $8\%$ margin padding.

---

## 3. Base Leveling (Line-Level Deskew)
Even after perspective warping, the document may contain minor alignment tilts. Stage 2 applies a radon-like projection profile deskew:
1. Samples the $15\% - 85\%$ vertical region of the warped grayscale matrix.
2. Applies Otsu adaptive thresholding to isolate characters.
3. Iterates rotation angles from $-20^\circ$ to $+20^\circ$ in 400 steps.
4. For each angle, rotates the binary text projection profile and computes the variance of row pixel sums.
5. Applies the rotation angle ($\theta$) that maximizes row variance (aligning text lines to the horizon).

---

## 4. Algorithmic Workflow

```mermaid
flowchart TD
    INPUT(["`**INPUT**
    Stage 1 BGR Output`"])

    BLOB["`**Tier A: Paper Blob Corner Detection**
    Downsize to 400px long edge · Bilateral filter
    Sweep threshold 245 → 95 (step=2)
    Check area ratio jumps`"]

    CHECK_JUMP{"`Jump
    detected?`"}

    HALT["`Halt sweep at prev step
    Extract extreme corners`"]

    OTSU{"`Sweep failed?
    Otsu threshold < 210?`"}

    OTSU_RUN["`Run Otsu binarize
    Extract extreme corners`"]

    SCALE["`**Direct Scale Mapping**
    Project 4 corners to high-res:
    corners / scale`"]

    CANNY["`**Tier B: Canny Quad Hull**
    Adaptive edge map
    Fit convex hull → approxPolyDP`"]

    TEXT["`**Tier C: Text Margin Anchoring**
    Identify outer text blocks
    Linear regression lines + 8% padding`"]

    CHECK_BLOB{"`Blob corners
    found?`"}

    CHECK_CANNY{"`Canny Quad
    found?`"}

    WARP["`**Perspective Transform**
    cv2.getPerspectiveTransform
    cv2.warpPerspective
    Interpolation: INTER_LANCZOS4`"]

    DESKEW["`**Base Leveling (Deskew)**
    Profile vertical 15% - 85% region
    Sweep angles -20° → +20°
    Rotate to maximize row variance`"]

    OUTPUT(["`**OUTPUT**
    Warped & Leveled BGR Matrix`"])

    INPUT --> BLOB
    BLOB --> CHECK_JUMP
    CHECK_JUMP -->|"Yes"| HALT
    CHECK_JUMP -->|"No"| OTSU
    HALT --> SCALE
    OTSU -->|"Yes"| OTSU_RUN
    OTSU -->|"No"| CHECK_BLOB
    OTSU_RUN --> SCALE
    SCALE --> CHECK_BLOB
    
    CHECK_BLOB -->|"Yes"| WARP
    CHECK_BLOB -->|"No"| CHECK_CANNY
    
    CHECK_CANNY -->|"Yes"| WARP
    CHECK_CANNY -->|"No"| TEXT
    TEXT --> WARP
    
    WARP --> DESKEW
    DESKEW --> OUTPUT
```
