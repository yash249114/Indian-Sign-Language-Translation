# Real-World Robustness & Environmental Perturbation Audit

**Scope**: Empirical stress testing of the end-to-end vision pipeline under realistic environmental degradations.  
**Test Set**: 175 diverse image samples across all 35 sign classes + 621 held-out landmark vectors.  
**Audit Date**: August 2026  

---

## 1. Dataset Accuracy vs. Real-World Webcam Performance

A critical distinction identified during this audit:

- **Dataset Test Accuracy (99.36%)**:
  - Evaluated on *pre-extracted, perfectly localized 63D normalized landmark vectors* from static dataset images.
- **Real-World End-to-End Image Accuracy (73.7%)**:
  - Evaluated on *raw unconstrained RGB frames* passed through the live MediaPipe detection $\rightarrow$ Normalization $\rightarrow$ Classifier pipeline.
  - MediaPipe landmark jitter, subtle orientation shifts, and finger edge uncertainties introduce real-world noise not present in isolated feature arrays.

---

## 2. Quantitative Robustness Benchmark

| Perturbation Condition | Parameter / Details | MediaPipe Detection Rate (%) | End-to-End Recognition Accuracy (%) | Vulnerability Level |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Clean Test Frames)** | Unmodified original frames | **94.3%** | **73.7%** | Baseline |
| **Low Illumination** | 0.4x Gamma / Brightness reduction | 94.3% | 72.0% | **Low** (MediaPipe remains resilient in dim lighting) |
| **High Exposure** | 1.8x Brightness overexposure | 96.6% | 72.6% | **Low** (Good contrast preserved) |
| **Low Contrast** | 0.5x Dynamic range compression | 93.7% | 75.4% | **Low** (Resilient to washed-out backgrounds) |
| **In-Plane Rotation (-15°)** | Counter-clockwise tilt | 95.4% | 53.1% | **High** (Significant accuracy drop) |
| **In-Plane Rotation (+15°)** | Clockwise tilt | 95.4% | 50.3% | **High** (Significant accuracy drop) |
| **In-Plane Rotation (-30°)** | Severe counter-clockwise tilt | 94.9% | 21.1% | **Critical** (Classification fails) |
| **In-Plane Rotation (+30°)** | Severe clockwise tilt | 94.9% | 17.1% | **Critical** (Classification fails) |
| **Low Resolution** | Downscaled to 64x64 | 96.0% | 68.6% | **Moderate** (Finger landmarks blur slightly) |
| **High Resolution** | Upscaled to 240x240 | 95.4% | 73.1% | **Negligible** |
| **Landmark Noise (Jitter $\sigma=0.02$)** | Direct coordinate perturbation | 100.0% | 99.4% | **Low** |
| **Landmark Noise (Jitter $\sigma=0.05$)** | Direct coordinate perturbation | 100.0% | 98.4% | **Low** |
| **Dropped Landmarks (2 Joints Missing)**| Simulated occlusion of 2 joints | 100.0% | 94.7% | **Moderate** |

---

## 3. Key Vulnerabilities & Architectural Insights

1. **Rotation Sensitivity (Primary Vulnerability)**:
   - The current normalizer guarantees **translation invariance** (wrist centering) and **scale invariance** (max-distance normalization), but **does NOT guarantee in-plane rotation invariance**.
   - As a result, tilting the hand beyond $\pm 15^\circ$ rotates the 63D coordinate manifold outside the training distribution, reducing accuracy from 73.7% down to 21.1%.
   - *Phase 2 Recommendation*: Implement canonical hand orientation alignment (e.g. rotating wrist-to-middle-finger MCP vector to vertical axis).

2. **Hand Pose Occlusions**:
   - Signs with downward pointing orientations (`P`, `Q`) or clenched fingers (`S`) exhibit lower MediaPipe detection rates (75-77%).
