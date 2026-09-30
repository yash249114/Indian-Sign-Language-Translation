# Indian Sign Language (ISL) Dataset Inspection Report

## 1. Executive Summary
- **Dataset Path**: `C:\Users\alany\Downloads\archive (5)\data`
- **Total Classes**: `35`
- **Total Valid Samples**: `42,000`
- **Minimum Samples/Class**: `1200`
- **Maximum Samples/Class**: `1200`
- **Average Samples/Class**: `1200.0`
- **Median Samples/Class**: `1200.0`
- **Standard Deviation**: `0.0`
- **Class Imbalance Ratio (Max/Min)**: `1.0`
- **Corrupted Files Detected**: `0`
- **Empty Folders**: `0`

## 2. Image Format & Dimension Profile
| Dimension & Color Space | Verified Class Count |
|-------------------------|----------------------|
| `128x128_RGB` | 105 |

## 3. Class-by-Class Sample Breakdown
| Class / Label | Samples | Distribution % |
|---------------|---------|----------------|
| **1** | 1,200 | 2.86% |
| **2** | 1,200 | 2.86% |
| **3** | 1,200 | 2.86% |
| **4** | 1,200 | 2.86% |
| **5** | 1,200 | 2.86% |
| **6** | 1,200 | 2.86% |
| **7** | 1,200 | 2.86% |
| **8** | 1,200 | 2.86% |
| **9** | 1,200 | 2.86% |
| **A** | 1,200 | 2.86% |
| **B** | 1,200 | 2.86% |
| **C** | 1,200 | 2.86% |
| **D** | 1,200 | 2.86% |
| **E** | 1,200 | 2.86% |
| **F** | 1,200 | 2.86% |
| **G** | 1,200 | 2.86% |
| **H** | 1,200 | 2.86% |
| **I** | 1,200 | 2.86% |
| **J** | 1,200 | 2.86% |
| **K** | 1,200 | 2.86% |
| **L** | 1,200 | 2.86% |
| **M** | 1,200 | 2.86% |
| **N** | 1,200 | 2.86% |
| **O** | 1,200 | 2.86% |
| **P** | 1,200 | 2.86% |
| **Q** | 1,200 | 2.86% |
| **R** | 1,200 | 2.86% |
| **S** | 1,200 | 2.86% |
| **T** | 1,200 | 2.86% |
| **U** | 1,200 | 2.86% |
| **V** | 1,200 | 2.86% |
| **W** | 1,200 | 2.86% |
| **X** | 1,200 | 2.86% |
| **Y** | 1,200 | 2.86% |
| **Z** | 1,200 | 2.86% |

## 4. Visual Inspection Findings
- **Hand Modality**: Single hand sign gestures cropped per image.
- **Background Characteristics**: Consistent indoor background setting.
- **Subject Framing**: Centered hand poses, allowing robust 21-landmark extraction via MediaPipe.
- **Missing Landmark Strategy**: Robust coordinate normalization relative to wrist landmark with fallback filtering for undetected frames.
