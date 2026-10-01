# Indian Sign Language Translation, Gloss Recognition & Query-Based Video Search Platform

[![Baseline v1](https://img.shields.io/badge/Release-baseline--v1-blue.svg)](https://github.com/yash249114/Indian-Sign-Language-Translation/releases/tag/baseline-v1)
[![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yash249114/Indian-Sign-Language-Translation/blob/baseline-v1-team/notebooks/ISL_Baseline_Colab.ipynb)
[![Project Portal](https://img.shields.io/badge/GitHub%20Pages-Project%20Portal-purple.svg)](https://yash249114.github.io/Indian-Sign-Language-Translation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Baseline v1 — Real-Time ISL Fingerspelling Recognition**  
> A-Z and 1-9 recognition using MediaPipe landmark extraction and a lightweight static classifier.  
> *Next: temporal word-sign recognition and continuous ISL understanding.*

---

## 📌 Current Baseline

**Baseline v1** delivers a verified, robust real-time Indian Sign Language fingerspelling system:
- **Alphabet**: `A` through `Z` fingerspelling recognition
- **Digits**: `1` through `9` fingerspelling recognition

The current baseline recognizes isolated fingerspelling characters and digits in real time. These recognized characters can be accumulated into text. Word-level temporal sign recognition is part of the next development stage.

Powered by:
- **MediaPipe Hands**: Local 21 3D single-hand landmark extraction with sub-15ms processing.
- **63D Geometric Normalization**: Translation-invariant (wrist origin $(0, 0, 0)$) and scale-invariant feature extraction.
- **Static MLP Classifier**: 3-layer neural network achieving **99.36%** accuracy on held-out test data with sub-2ms inference latency.
- **Prediction Stabilization & Debouncing**: Commit-once finite state machine that guarantees a held hand produces exactly one character instead of repeated characters.
- **Text Buffering**: Word and phrase builder with Space, Backspace, and Clear operations.
- **Multilingual Translation**: Asynchronous translation into Telugu, Hindi, Tamil, Kannada, and Marathi powered by **Gemini 3.5 Flash Lite** (with offline lexicon fallback).
- **Speech Synthesis**: Local offline **pyttsx3** TTS engine for instant spoken pronunciation.

> [!IMPORTANT]
> **Scope Boundaries:**
> - **Current production inference model:** Static MLP classifier.
> - **Temporal word-sign recognition is a planned future phase and is not included in Baseline v1.**
> - **Continuous ISL signing is planned future work and is not part of Baseline v1.**

---

## 🏗️ System Architecture

```
Camera (Webcam)
      │ [30 FPS Base64 / Stream]
      ▼
MediaPipe Hands (21 3D Landmarks)
      │
      ▼
63D Landmark Normalization (Wrist-Centric & Scale Invariant)
      │
      ▼
Static MLP Classifier (models/trained/sign_classifier/mlp_classifier.joblib)
      │ [35 Classes: 1-9, A-Z]
      ▼
Temporal Stabilizer & State Machine (Rolling Consensus Window)
      │ [Debounce & Commit-Once on Stable Sign]
      ▼
Text Buffer (Accumulates Characters into Words)
      │
      ├────────────────────────────────────────┬────────────────────────────────────────┐
      ▼                                        ▼                                        ▼
Gemini 3.5 Flash Lite Translation        Offline Rule-Based Lexicon              Offline Local TTS (pyttsx3)
(Telugu, Hindi, Tamil, Kannada, Marathi) (Automatic Network Fallback)            (Browser Audio Playback)
```

---

## ✨ Verified Features

- [x] **Real-Time Video Ingestion**: Native WebSocket streaming and HTTP image processing at 30 FPS.
- [x] **MediaPipe Keypoint Extraction**: 21 3D coordinates per hand with zero cloud dependencies.
- [x] **Mathematically Invariant Normalization**: Guarantees consistent 63D coordinate representations across camera distance and user position.
- [x] **High-Precision Classification**: Scikit-Learn MLP with 99.36% test accuracy and sub-2ms inference latency.
- [x] **State Machine Debouncing**: Eliminates character repetition (`NNNVVV...`) during held postures.
- [x] **Multilingual Translation**: Seamless translation to 5 Indian languages via Gemini 3.5 Flash Lite or offline dictionary.
- [x] **Local Text-to-Speech**: Speech synthesis without cloud costs or latency.
- [x] **Dataset Collection Tooling**: Dataset collection tooling is being prepared for the upcoming word-sign recognition phase. No real human word-sign training clips are included in Baseline v1.

---

## 📋 Technology Stack

### Active Production Runtime
- **Language**: Python 3.10
- **Server**: FastAPI 0.115.6, Uvicorn 0.34.0, WebSockets 14.1
- **Computer Vision & ML**: MediaPipe 0.10.14, Scikit-Learn 1.5.0 (MLP Classifier), OpenCV 4.8.1, NumPy 1.26.4
- **Translation & Speech**: Gemini 3.5 Flash Lite (REST), pyttsx3 2.90 (Offline TTS)
- **Validation**: Pytest 8.2.2 (49 Passing Tests)
- **Frontend**: Vanilla HTML5, CSS3, ES6 JavaScript (Zero external framework build requirements)

### Future Research Stack (Planned for Phase 2 & Beyond)
- **Deep Learning**: PyTorch 2.9.1 (Planned for temporal Bi-GRU / TCN word recognition)
- **Sequence Modeling**: Connectionist Temporal Classification (CTC) for continuous signing

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yash249114/Indian-Sign-Language-Translation.git
cd Indian-Sign-Language-Translation
git checkout baseline-v1-team
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example environment configuration
cp .env.example .env      # Linux/macOS
copy .env.example .env    # Windows
```
*(Optional: Add your `GEMINI_API_KEY` in `.env` for AI translation. The application works offline without it using a built-in dictionary.)*

---

## ▶️ Running the Application

Start the local server using Uvicorn:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open your browser at:
```
http://127.0.0.1:8000/
```

- Allow camera permissions.
- Perform static ISL fingerspelling signs in front of the webcam.
- Watch recognized letters form words in the text buffer.

---

## 🧪 Automated Testing

Execute the comprehensive test suite (49 passing tests):
```bash
python -m pytest tests/ -v
```

Run specific acceptance tests:
```bash
python -m pytest tests/test_acceptance.py -v
```

---

## 📊 Dataset Information

- **Alphabet Fingerspelling Dataset**: 42,000 raw images across 35 classes (`1`–`9`, `A`–`Z`) at 1,200 images per class.
- **Processed Landmarks**: 4,136 pre-extracted and normalized landmark feature vectors stored in `data/processed/landmarks/` (`train.npz`, `val.npz`, `test.npz`).
- **Data Preprocessing Script**: `scripts/preprocess_dataset.py` extracts MediaPipe coordinates and creates deterministic splits.

---

## ⚠️ Current Limitations

1. **Isolated Fingerspelling Postures Only**: Baseline v1 classifies one static gesture at a time.
2. **No Real Human Temporal Word Dataset**: Dynamic signs (e.g. `WATER`, `HELLO`, `FOOD`) require 30-frame temporal sequences from human signers; no real human word-sign training clips are included in Baseline v1.
3. **Continuous ISL Signing is Not Implemented**: The current system requires discrete pauses/neutral hand positions between signs; continuous sentence-level CTC decoding is planned future work for Phase 3.
4. **Static 2D Training Origin**: The underlying alphabet dataset consists of static images and cannot be used to train temporal video models.

---

## 🔮 Future Development Roadmap

1. **Phase 2 — Temporal Word-Sign Modeling**:
   - Collect 30-frame dynamic video clips across 9 diverse signers using `scripts/collect_word_dataset.py`.
   - Train Bi-GRU and Temporal Convolutional Networks (TCN) with signer-independent validation.
2. **Phase 3 — ISL Grammar & Continuous Signing**:
   - Integrate non-manual grammatical cues (facial expressions, head movements) using MediaPipe Face Mesh.
   - Implement CTC sequence decoding for continuous signing.
3. **Phase 4 — Query-Based Video Search**:
   - Temporal landmark embedding indexing for educational sign language video archives.
   - Natural language and sign query search interface.

---

## 👥 Documentation & Team Resources

- [Team Quick Start Guide](docs/TEAM_QUICK_START.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Current Capabilities Report](docs/CURRENT_CAPABILITIES.md)
- [Future Development Roadmap](docs/FUTURE_ROADMAP.md)
- [Baseline v1 Release Notes](BASELINE_V1_RELEASE.md)
- [GitHub Pages Project Portal](https://yash249114.github.io/Indian-Sign-Language-Translation/)
