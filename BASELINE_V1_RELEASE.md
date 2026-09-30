# Indian Sign Language Platform — Baseline v1 Team Release Notes

**Release**: `baseline-v1`  
**Branch**: `baseline-v1-team`  
**Date**: September 2026  
**Status**: Verified Working Baseline

---

## 1. Executive Notice & Scope

> [!IMPORTANT]
> **Baseline v1 currently provides ISL fingerspelling recognition for A-Z and 1-9. Word-level temporal sign recognition and continuous signing are future development stages.**

This release provides the verified working engineering foundation for the Indian Sign Language (ISL) Character & Sign Recognition Platform. It is packaged for team onboarding, reproducible local execution, Google Colab experimentation, and academic documentation.

---

## 2. Current Verified Capabilities

1. **Camera Ingestion**: Real-time webcam frame acquisition at 30 FPS via WebSocket streaming and REST image endpoints.
2. **MediaPipe Hand Landmark Extraction**: 21 3D single-hand coordinates $(x, y, z)$ tracked locally with sub-15ms processing latency.
3. **63D Coordinate Normalization**:
   - Wrist-relative translation invariance (wrist shifted to coordinate origin $(0, 0, 0)$).
   - MCP scale invariance (Euclidean normalization by maximum hand dimension).
   - Division-by-zero protection.
4. **Static MLP Classifier**:
   - 35 classes: Digits `1`–`9` and Letters `A`–`Z` (excluding `0`).
   - Deep MLP with hidden layers `(256, 128, 64)`, ReLU activations, and Adam optimization.
   - Test accuracy: **99.36%** on held-out test split.
5. **Prediction Stabilization & State Machine**:
   - Rolling consensus buffer (`window_size=6`, `min_consensus_ratio=0.60`, `confidence_threshold=0.75`).
   - Commit-once state machine prevents repeated character flood while a sign is held.
   - States: `NO_SIGN` $\rightarrow$ `STABLE_SIGN` $\rightarrow$ `COMMIT_ONCE` $\rightarrow$ `COOLDOWN` $\rightarrow$ `WAIT_FOR_NEUTRAL`.
6. **Text Accumulator Buffer**: Append, Space, Backspace, and Clear operations.
7. **Multilingual Translation Layer**:
   - Primary: Gemini 3.5 Flash Lite asynchronous translation.
   - Supported targets: Telugu (`te`), Hindi (`hi`), Tamil (`ta`), Kannada (`kn`), Marathi (`mr`).
   - Offline fallback lexicon ensures translation succeeds even if network or API keys are unavailable.
8. **Offline Text-to-Speech (TTS)**:
   - Local pyttsx3 speech synthesis engine on port 8000.
   - Audio caching in `tts/cache/`.

---

## 3. Current Limitations

- **Isolated Character Recognition Only**: Recognizes static fingerspelling postures one character at a time.
- **No Trained Temporal Word Model**: Dynamic ISL word signs requiring temporal motion trajectories across time are not yet trained on real human signers.
- **No Continuous Sign Recognition**: The current pipeline does not perform unconstrained sentence-level continuous signing without segment boundaries.
- **Dataset Constraint**: The existing dataset (`archive (5)/data`) contains only static 2D images for alphabet and digits; it is not a video dataset and cannot be used to train temporal word signs.

---

## 4. System Architecture

```
Webcam Frame (Browser / OpenCV)
               │ [30 FPS / Base64 / Binary]
               ▼
WebSocket Stream Gateway (/ws/sign-stream)
               │
               ▼
MediaPipe Hands (Single-Hand Landmark Extraction)
               │ [21 Landmark Points: x, y, z]
               ▼
63D Landmark Normalizer (Wrist-Centric & Scale Invariant)
               │ [1D Vector of 63 Floats]
               ▼
Pretrained Scikit-Learn MLP Classifier (mlp_classifier.joblib)
               │ [Softmax Probabilities across 35 Classes]
               ▼
Temporal Stabilizer (Rolling Window Majority Vote & State Machine)
               │ [Commit Once on Stable Consensus]
               ▼
Text Accumulator Buffer (Word & Sentence Construction)
               │
               ├────────────────────────────────────────┐
               ▼                                        ▼
Gemini 3.5 Flash Lite Translation        Offline Local pyttsx3 Audio Speech
(Telugu, Hindi, Tamil, Kannada, Marathi) (Browser Playback via WAV cache)
```

---

## 5. Trained Models & Datasets

### A. Fingerspelling Classifier
- **Model File**: `models/trained/sign_classifier/mlp_classifier.joblib` (231 KB)
- **Framework**: Scikit-Learn 1.5.0 (`MLPClassifier`)
- **Input Shape**: `[63]` (21 landmarks $\times$ 3 coordinates)
- **Output Classes**: 35 classes (`['1', '2', ..., '9', 'A', 'B', ..., 'Z']`)
- **Label Mapping**: `data/processed/landmarks/label_map.json`
- **Training Samples**: Extracted from 42,000 images (`data/processed/landmarks/train.npz`, `val.npz`, `test.npz`)

### B. Dynamic Word-Sign Infrastructure (Future)
- **Word Predictor**: `models/word_temporal_bigru.pt` & `vision/temporal/word_predictor.py`
- **Classes**: 12 target glosses (`HELLO`, `THANK_YOU`, `HELP`, `YES`, `NO`, `PLEASE`, `GOOD`, `WATER`, `FOOD`, `YOU`, `ME`, `NAME`)
- **Status**: Structural code and baseline reference weights implemented; pending real human multi-signer video collection.

---

## 6. Verification and Acceptance Tests

All 49 unit and acceptance tests are passing:
```bash
python -m pytest tests/ -v
```

Verified Test Suites:
- `tests/test_acceptance.py`: Acceptance tests for Fingerspelling, Word Mode, and ISL Interpretation.
- `tests/test_api.py`: FastAPI health, stats, and buffer endpoints.
- `tests/test_model_inference.py`: Model loader, input shapes, and probability outputs.
- `tests/test_normalizer.py`: Translation invariance, scale invariance, zero-division safety.
- `tests/test_regression_state_machine.py`: 6 regression tests verifying debouncing and single emission on held signs.
- `tests/test_stabilizer.py`: Majority-vote filter and confidence thresholds.
- `tests/test_gemini_translation.py`: Gemini 3.5 Flash Lite integration and offline fallback.
- `tests/test_tts.py`: Speech synthesis and empty text handling.

---

## 7. Known Issues & Operational Guidelines

1. **Lighting & Background**: MediaPipe detection performs best in even ambient lighting against a non-cluttered background.
2. **Hand Orientation**: Fingerspelling signs require the palm or fingers to face toward the camera.
3. **Translation API Key**: If `GEMINI_API_KEY` is not provided in `.env`, the system automatically falls back to an offline rule-based dictionary without crashing.
