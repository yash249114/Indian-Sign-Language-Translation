# Rigorous Dataset Discrepancy & Verification Audit

**Dataset Location**: `C:\Users\alany\Downloads\archive (5)\data`  
**Audit Date**: August 2026  
**Auditor**: Lead AI Engineer & Full-Stack Architect  

---

## 1. Executive Summary: 42,000 → 4,136 Discrepancy

The investigation into why **42,000 raw images** resulted in **4,136 extracted landmark samples** was conducted by analyzing the raw disk directory, cryptographic MD5 hash collisions, and the feature extraction pipeline.

### Exact Discrepancy Breakdown

| Category | Image Count | Percentage (%) | Root Cause Description |
| :--- | :--- | :--- | :--- |
| **Total Raw Dataset Images** | **42,000** | 100.00% | 35 classes (`1-9`, `A-Z`), exactly 1,200 images per class. |
| **Images Targeted for Extraction** | **4,200** | 10.00% | Script was called with an intentional sampling limit parameter: `samples_per_class = 120` ($35 \times 120 = 4,200$). |
| **Images Skipped by Sampling Limit** | **37,800** | 90.00% | Skipped because of the 120 samples/class threshold passed during extraction. |
| **Successfully Extracted Landmarks** | **4,136** | 9.85% (98.48% of processed) | Hands detected and 63D normalized vectors extracted. |
| **MediaPipe Detection Failures** | **64** | 0.15% (1.52% of processed) | Hand obscured, partially cropped, clenched, or pointing downward. |
| **Corrupted / Unreadable Files** | **0** | 0.00% | Zero byte corruption or decoding failures. |
| **Exact Duplicate Files (MD5 Collisions)**| **0** | 0.00% | All 42,000 images possess unique MD5 hashes. |

---

## 2. MediaPipe Detection Failures by Class

Out of the 4,200 processed images (120 per class), **64 images failed MediaPipe hand landmark detection**. The failures were concentrated in specific hand orientations:

| Class | Processed | Successfully Detected | Failed Detections | Detection Rate (%) | Failure Pattern Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P** | 120 | 90 | **30** | 75.00% | Hand pointed downward with palm partially occluded from camera plane. |
| **Q** | 120 | 92 | **28** | 76.67% | Downward thumb/index configuration creates self-occlusion. |
| **S** | 120 | 115 | **5** | 95.83% | Tightly clenched fist hides finger landmarks. |
| **B** | 120 | 119 | **1** | 99.17% | Edge boundary clipping. |
| **All Other 31 Classes** | 3,720 | 3,720 | **0** | **100.00%** | Perfect detection rate across all other digits and letters. |
| **Total** | **4,200** | **4,136** | **64** | **98.48%** | Overall Detection Success Rate |

---

## 3. Dataset Integrity & Storage Verification

- **Unique File Hashes**: Exactly 42,000 unique MD5 hashes across 42,000 files.
- **Cross-Class File Collisions**: 0 instances.
- **Intra-Class Exact File Duplicates**: 0 instances.
- **Directory Structure**: 35 subdirectories (`data/1/` to `data/Z/`), each containing `0.jpg` through `1199.jpg`.
- **Sampling Limit Confirmation**: The feature extraction was restricted to 120 samples per class to maintain rapid turnaround for the initial working MVP. The remaining 37,800 images remain intact in the original dataset directory and can be batch-processed for scaled training.
