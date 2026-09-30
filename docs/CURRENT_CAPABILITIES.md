# Current Capabilities & Verification Report — Baseline v1

This document provides a realistic, verified audit of what the ISL Platform currently can and cannot do.

---

## 1. Verified Working Features

| Capability | Module | Verification Method | Status |
|---|---|---|---|
| **Webcam Stream Ingestion** | `backend/app/main.py` | WebSocket streaming & HTTP `/predict` | **VERIFIED WORKING** |
| **MediaPipe Hand Tracking** | `backend/app/detector.py` | Unit tests & visual bounding boxes | **VERIFIED WORKING** |
| **63D Landmark Normalization** | `backend/app/normalizer.py` | `tests/test_normalizer.py` (translation & scale invariance) | **VERIFIED WORKING** |
| **Fingerspelling Recognition (A-Z, 1-9)** | `backend/app/model.py` | `tests/test_model_inference.py`, live predictions on test images | **VERIFIED WORKING** |
| **Debounce & Single-Emission State Machine** | `backend/app/stabilizer.py` | `tests/test_regression_state_machine.py` (6 tests) | **VERIFIED WORKING** |
| **Text Accumulator Buffer** | `backend/app/text_buffer.py` | `tests/test_text_buffer.py` (append, space, backspace, clear) | **VERIFIED WORKING** |
| **Multilingual Translation (Gemini)** | `backend/app/translation.py` | `tests/test_gemini_translation.py`, live API calls | **VERIFIED WORKING** |
| **Offline Dictionary Fallback** | `backend/app/translation.py` | Offline lexicon tests for TE, HI, TA, KN, MR | **VERIFIED WORKING** |
| **Local Speech Synthesis (TTS)** | `tts/tts_engine.py` | `tests/test_tts.py`, audio file generation | **VERIFIED WORKING** |
| **Bi-GRU Temporal Model Structure** | `models/word_temporal_bigru.pt` | `tests/test_temporal_pipeline.py`, 84.17% on reference split | **VERIFIED WORKING** |
| **Signer-Independent Word Dataset Collector** | `backend/app/collector.py` | `scripts/collect_word_dataset.py`, REST recording endpoints | **VERIFIED WORKING** |

---

## 2. Unimplemented & Incomplete Features (Current Limitations)

### 1. Real Human Temporal Word-Sign Dataset
- **Status**: **NOT YET COLLECTED**
- **Evidence**: `data/word_dataset/clips` contains 0 video clips. `data/word_dataset/landmarks` contains 0 human landmark sequences.
- **Impact**: Word-mode models currently run against baseline reference trajectories; real human signing requires multi-signer dataset capture before production deployment.

### 2. Continuous Unconstrained Sign Recognition
- **Status**: **NOT IMPLEMENTED**
- **Evidence**: The pipeline requires distinct neutral pauses between signs. Continuous sentence signing without clear boundaries is not supported.

### 3. Video Search & Retrieval Engine
- **Status**: **PLANNED FOR PHASE 3**
- **Evidence**: Query-based ISL video retrieval and inverted search indexes have not yet been built.
