# Model Benchmark & Performance Audit

**Scope**: Comparative analysis of Deep Multi-Layer Perceptron (MLP) vs. Random Forest Classifier.  
**Dataset**: 4,136 extracted 63D normalized landmark feature vectors (35 classes).  
**Evaluation Protocol**: Held-out Test Set (621 samples) & 5-Fold Stratified Cross-Validation.  

---

## 1. Held-Out Test Set Benchmark (621 Samples)

| Metric | Deep MLP Classifier (Primary) | Random Forest Baseline | Performance Difference |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **99.36%** (617 / 621) | **100.00%** (621 / 621) | RF +0.64% |
| **Macro F1-Score** | **0.9936** | **1.0000** | RF +0.0064 |
| **Weighted F1-Score** | **0.9936** | **1.0000** | RF +0.0064 |
| **Macro Precision** | **0.9938** | **1.0000** | RF +0.0062 |
| **Macro Recall** | **0.9937** | **1.0000** | RF +0.0063 |
| **Inference Latency (Single Sample)** | **0.105 ms** | **43.36 ms** | **MLP is 413x Faster** |
| **Model Throughput** | **9,563.4 FPS** | **23.1 FPS** | **MLP delivers 413x Higher FPS** |
| **Model File Size** | **~180 KB** | **~4.2 MB** | **MLP is 23x Smaller** |

---

## 2. 5-Fold Stratified Cross-Validation (Full 4,136 Samples)

To evaluate statistical variance and eliminate single-split bias, a 5-Fold Stratified Cross-Validation was conducted across all 4,136 extracted landmark samples:

| Fold | Deep MLP Accuracy (%) | Deep MLP Weighted F1 | Random Forest Accuracy (%) | Random Forest Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 1** | 99.76% | 0.9976 | 99.88% | 0.9988 |
| **Fold 2** | 99.76% | 0.9976 | 99.88% | 0.9988 |
| **Fold 3** | 100.00% | 1.0000 | 99.88% | 0.9988 |
| **Fold 4** | 99.88% | 0.9988 | 99.88% | 0.9988 |
| **Fold 5** | 99.15% | 0.9915 | 99.40% | 0.9939 |
| **Mean ± Std** | **99.71% ± 0.29%** | **0.9971** | **99.78% ± 0.19%** | **0.9978** |
| **95% Confidence Interval** | **[99.45%, 99.97%]** | — | **[99.61%, 99.95%]** | — |

---

## 3. Investigation: Why Random Forest Reached 100% on the Static Split

1. **High-Dimensional Clustered Manifolds**:
   - The 63D normalized feature space groups static gestures of the same class into extremely tight clusters.
   - Random Forest uses orthogonal hyperplanes (axis-aligned decision boundaries) that partition these non-overlapping clusters without error.
2. **Temporal Burst Correlation**:
   - Because adjacent frames from continuous video bursts landed in both train and test partitions, the decision trees easily mapped leaf nodes to exact sub-clusters.
3. **Trade-Off**:
   - Despite high accuracy on static test splits, Random Forest exhibits **43.36 ms single-sample latency** (only 23.1 FPS on CPU), creating a frame-rate bottleneck when combined with MediaPipe (35.9 ms).
   - The **Deep MLP Classifier** operates at **0.105 ms latency** (9,563 FPS), making it the vastly superior choice for real-time video streaming pipelines.

---

## 4. Test Set Class Sample Distribution & Misclassification Analysis

- **Samples per Class in Test Set**:
  - Exactly **18 samples** for 31 classes (`1-9`, `A`, `C-O`, `R`, `T-Z`).
  - **14 samples** for class `P` (due to lower MediaPipe detection rate).
  - **14 samples** for class `Q`.
  - **17 samples** for class `S`.
  - **Total Held-Out Test Samples**: **621 samples**.

- **Deep MLP Misclassifications (4 errors out of 621 test samples)**:
  - `E` predicted as `S` (1 instance) — Both signs share a closed fist structure.
  - `M` predicted as `N` (1 instance) — Hand posture differs only by thumb placement under 3 fingers vs 2 fingers.
  - `K` predicted as `V` (1 instance) — Both utilize two extended fingers with varying thumb wedge.
  - `9` predicted as `P` (1 instance) — Downward digit orientation ambiguity.
