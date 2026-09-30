# Data Leakage & Split Independence Audit

**Scope**: Evaluation of Train (70%), Validation (15%), and Test (15%) splits.  
**Samples Audited**: 4,136 extracted 63D landmark feature vectors.  
**Audit Date**: August 2026  

---

## 1. Executive Summary: Data Leakage Findings

An investigation was performed to determine whether visually similar or duplicate samples exist across the Train, Validation, and Test partitions.

### Quantitative Leakage Metrics

| Metric | Measured Value | Percentage (%) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Exact Landmark Duplicate Vectors** ($\text{Euclidean Dist} < 10^{-4}$) | **0** | **0.00%** | No numerically identical landmark vectors exist across splits. |
| **Near-Duplicate Vectors** ($\text{Euclidean Dist} < 0.05$) | **282** / 621 | **45.41%** | Substantial geometric similarity between test and training samples. |
| **Moderately Close Vectors** ($\text{Euclidean Dist} < 0.10$) | **489** / 621 | **78.74%** | Indicates high temporal auto-correlation in source recording bursts. |
| **Mean Minimum Euclidean Distance (Test to Train)** | **0.0763 ± 0.0676** | N/A | Average proximity of a held-out test sample to its nearest training neighbor. |

---

## 2. Root Cause Analysis: Video Burst Auto-Correlation

1. **Continuous Video Capture**:
   - The source dataset files (`0.jpg` through `1199.jpg`) were generated from continuous video bursts where a signer holds a static sign while the camera records at ~30 FPS.
   - Neighboring frames in each class directory vary only by minor hand micro-tremors and sensor noise.

2. **Standard Random Stratified Split Effect**:
   - When a standard random 70/15/15 stratified split is applied across the entire pool of extracted samples, frames from the same recording session land in both the training set and the test set.
   - This creates **data leakage via temporal autocorrelation**, allowing standard classifiers to recognize static hand configurations seen during training with artificially inflated confidence.

---

## 3. Metadata & Subject Isolation Limitations

- **Signer IDs**: **Not present** in the dataset directory structure or filename metadata.
- **Session IDs**: **Not present** in the dataset directory structure or filename metadata.
- **Independence Limitation**: A strictly subject-independent (leave-one-subject-out) split **cannot be generated automatically** from this dataset without manual signer/session annotation.
- **Impact on Evaluation**:
  - The reported 99.36% (MLP) and 100.00% (Random Forest) accuracies reflect **in-distribution gesture recognition under identical recording conditions**.
  - Generalization to unseen signers and new physical environments requires real-world testing (see [`docs/robustness_test.md`](robustness_test.md)).
