# Stage 1: Illumination Normalization

## 1. Architectural Purpose (The "Why")
Uneven overhead lighting, camera flash glare, and shadows cast across a document degrade contrast. When shadowed regions are fed into a standard binarization step, the character outlines compress in intensity, causing text to **shatter into unreadable fragments**.

Stage 1 mathematically separates text ink details from the background lighting grid using **Morphological Division**. It flattens lighting variations and maps shadows to a uniform white color before binarization runs.

---

## 2. Mathematical Concept & Mechanics
The normalization layer runs three mathematical operations in sequence:

### A. Dynamic LAB CLAHE (Contrast Limiting)
Standard global histogram equalization shifts intensity scales globally, washing out white backgrounds. To preserve details, the image is converted to the LAB color space, separating lightness ($L$) from chromatic values ($A$, $B$).
- **Lightness Standard Deviation ($\sigma_L$)**: Extracted from the $L$ channel.
- **Dynamic Clip Limit**: Equalization strength scales dynamically to prevent halos on bright documents while boosting low-contrast text.

### B. Background Isolation (Morphological Dilation)
To isolate shadow profiles without affecting characters, a morphological dilation is run using a wide rectangular structure element:
- A $19 \times 19$ kernel dilates bright paper pixels, painting over thin text lines completely.
- A Gaussian blur smooths out harsh edge boundaries, producing a continuous gradient representing the background illumination profile ($B$).

### C. Background Division (Shadow Neutralization)
The normalized output is computed by dividing the original pixel grid ($I$) by the background illumination profile ($B$):

$$\text{Output} = \left( \frac{I}{B} \right) \times 255$$

Since shadowed areas are dim in both $I$ and $B$, dividing them scales the pixels back to full brightness — removing shadows while keeping text ink details intact.

---

## 3. Algorithmic Workflow

```mermaid
flowchart TD
    INPUT(["`**INPUT**
    Oriented BGR Matrix`"])

    GUARD{"`Matrix null
    or empty?`"}

    EARLY(["`Return original matrix`"])

    COLOR{"`is_color?
    len(shape) == 3`"}

    TOGRAY["`cv2.cvtColor
    BGR → Grayscale`"]

    COPY["`gray_matrix = BGR.copy()`"]

    DILATE["`**Morphological Dilation**
    cv2.MORPH_RECT kernel (19×19)
    Erase text strokes
    → dilated_bg`"]

    BLUR["`**Gaussian Blur**
    cv2.GaussianBlur (19×19)
    Smooth background map
    → background_map`"]

    DIVIDE["`**Background Division**
    cv2.divide(gray, bg, scale=255)
    float32 precision
    → normalized_matrix`"]

    CPATH["`**Color Path**
    Split BGR channels
    cv2.divide each by background_map
    cv2.merge channels`"]

    YCRCB["`Convert → YCrCb
    CLAHE on Y channel
    clipLimit=2.0 · tileGrid=(8,8)
    Convert → BGR`"]

    GPATH["`**Grayscale Path**
    CLAHE directly on normalized_matrix
    clipLimit=2.0 · tileGrid=(8,8)`"]

    ERROR{"`Exception
    raised?`"}

    FALLBACK(["`Return original matrix`"])

    OUTPUT(["`**OUTPUT**
    Shadow-Free BGR Matrix`"])

    INPUT --> GUARD
    GUARD -->|"Yes"| EARLY
    GUARD -->|"No"| COLOR
    COLOR -->|"Yes"| TOGRAY
    COLOR -->|"No"| COPY
    TOGRAY --> DILATE
    COPY --> DILATE
    DILATE --> BLUR
    BLUR --> DIVIDE
    DIVIDE --> CPATH
    DIVIDE --> GPATH
    CPATH --> YCRCB
    YCRCB --> ERROR
    GPATH --> ERROR
    ERROR -->|"Yes"| FALLBACK
    ERROR -->|"No"| OUTPUT
```
