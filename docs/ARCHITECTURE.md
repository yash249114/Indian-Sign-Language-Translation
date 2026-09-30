# Architecture & System Design — ISL Recognition Platform

This document outlines the software engineering, machine learning pipelines, and component boundaries for the Indian Sign Language Platform.

---

## 1. High-Level System Architecture

```
                    ┌────────────────────────────────────────┐
                    │          Frontend Web Client           │
                    │   (Vanilla HTML5, CSS3, JavaScript)    │
                    └───────────────────┬────────────────────┘
                                        │
                       WebSocket /ws/sign-stream (30 FPS)
                                        │
                    ┌───────────────────▼────────────────────┐
                    │          FastAPI Server                │
                    │     (Asynchronous REST & WS)           │
                    └───────────────────┬────────────────────┘
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           │                                                        │
           ▼ [Mode: Fingerspelling]                                 ▼ [Mode: Word Sign]
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ MediaPipe Hands (Single Hand) │                       │ MultiHandTracker (Both Hands) │
│ - 21 3D Landmarks             │                       │ - 42 3D Landmarks             │
└──────────────┬────────────────┘                       └──────────────┬────────────────┘
               │                                                       │
               ▼                                                       ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ LandmarkNormalizer (63D)      │                       │ Sequence Normalizer (126D)    │
│ - Origin at wrist             │                       │ - Zero-masked missing hands   │
│ - Max Euclidean scale norm    │                       │ - Monotonic 30-frame FIFO     │
└──────────────┬────────────────┘                       └──────────────┬────────────────┘
               │                                                       │
               ▼                                                       ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ MLP Character Classifier      │                       │ Temporal Segmenter & Bi-GRU   │
│ - Scikit-Learn MLPClassifier  │                       │ - State machine lifecycle     │
│ - 35 classes (A-Z, 1-9)       │                       │ - 12 gloss vocab classifier   │
└──────────────┬────────────────┘                       └──────────────┬────────────────┘
               │                                                       │
               ▼                                                       ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ Temporal Stabilizer           │                       │ ISL Grammar & Gloss Buffer    │
│ - Rolling consensus buffer    │                       │ - Canonical OSV/SOV parser    │
│ - Commit-Once state machine   │                       │ - Uncertainty token handling  │
└──────────────┬────────────────┘                       └──────────────┬────────────────┘
               │                                                       │
               ▼                                                       ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ Text Buffer                   │                       │ Language & Speech Services    │
│ - Appends discrete character  │                       │ - Gemini 3.5 Flash Lite LLM   │
│ - Space / Backspace / Clear   │                       │ - pyttsx3 Local Offline TTS   │
└───────────────────────────────┘                       └───────────────────────────────┘
```

---

## 2. Core Components

### 2.1 MediaPipe Hand Tracking
- **File**: `backend/app/detector.py`, `vision/hand_tracking/multi_hand_tracker.py`
- Ingests standard OpenCV BGR video frames.
- Detects hand presence, bounding box, handedness, and 21 key landmarks in normalized 3D space $(x, y, z)$.

### 2.2 Feature Preprocessing & Normalization
- **File**: `backend/app/normalizer.py`
- Enforces geometric translation and scale invariance:
  $$\hat{p}_i = \frac{p_i - p_{\text{wrist}}}{\max_{j} \|p_j - p_{\text{wrist}}\|_2 + \epsilon}$$
- Guarantees exact 63-dimensional output vector for single hand, or 126-dimensional output vector for dual hands.

### 2.3 Static Classifier Model
- **File**: `backend/app/model.py`, `models/trained/sign_classifier/mlp_classifier.joblib`
- Fully connected Multi-Layer Perceptron:
  - Architecture: Input (63) $\rightarrow$ Dense(256) $\rightarrow$ Dense(128) $\rightarrow$ Dense(64) $\rightarrow$ Output(35).
  - Accuracy: 99.36% on test split.
  - Latency: $< 1.5\text{ ms}$ per sample on CPU.

### 2.4 State Machine & Temporal Debounce
- **File**: `backend/app/stabilizer.py`
- Solves the repetition bug when users hold a sign.
- Requires $60\%$ majority consensus over a 6-frame window before triggering a `COMMIT_ONCE` transition.
- Forces transition into `COOLDOWN` and `WAIT_FOR_NEUTRAL` before another character can be emitted.

### 2.5 Translation & Linguistic Layer
- **File**: `backend/app/translation.py`, `backend/app/isl_grammar.py`
- Gemini 3.5 Flash Lite translates constructed text or completed gloss sequences into 5 major Indian languages (Telugu, Hindi, Tamil, Kannada, Marathi).
- Offline fallback dictionary provides instant translation if network connectivity is interrupted.

### 2.6 Offline Speech Synthesis
- **File**: `tts/tts_engine.py`
- Local pyttsx3 text-to-speech engine synthesizes spoken audio and caches WAV clips for non-blocking browser playback.
