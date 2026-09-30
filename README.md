# Indian Sign Language Translation, Gloss Recognition & Query-Based Video Search Platform

[![Baseline v1](https://img.shields.io/badge/Release-baseline--v1-blue.svg)](https://github.com/yash249114/Indian-Sign-Language-Translation/releases/tag/baseline-v1)
[![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yash249114/Indian-Sign-Language-Translation/blob/baseline-v1-team/notebooks/ISL_Baseline_Colab.ipynb)
[![Project Portal](https://img.shields.io/badge/GitHub%20Pages-Project%20Portal-purple.svg)](https://yash249114.github.io/Indian-Sign-Language-Translation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Real-Time ISL Fingerspelling Recognition • Multimodal AI • Foundation for Temporal Word-Sign Understanding

---

## 📌 Current Baseline

**Baseline v1** delivers a verified, robust real-time Indian Sign Language fingerspelling system recognizing:
- **Alphabet**: `A` through `Z`
- **Digits**: `1` through `9`

Powered by:
- **MediaPipe Hands** for sub-15ms 3D landmark extraction.
- **Geometric Normalization**: 63-dimensional translation-invariant and scale-invariant feature extraction.
- **Deep MLP Classifier**: 3-layer neural network achieving **99.36%** accuracy on held-out test data.
- **Prediction Stabilization & Debouncing**: Commit-once finite state machine that guarantees a held hand produces exactly one character instead of repeated characters.
- **Text Accumulation**: Word and phrase builder with Space, Backspace, and Clear operations.
- **Translation Layer**: Asynchronous translation into Telugu, Hindi, Tamil, Kannada, and Marathi powered by **Gemini 3.5 Flash Lite** (with offline lexicon fallback).
- **Speech Synthesis**: Local offline **pyttsx3** TTS engine for instant spoken pronunciation.

> [!IMPORTANT]
> **Word-level temporal recognition is not included in Baseline v1.**  
> Baseline v1 provides static fingerspelling character recognition. Temporal sequence modeling for dynamic word signs is currently in active development for Phase 2.

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
- [x] **Mathematically Invariant Normalization**: Guarantees consistent coordinate representations across camera distance and user position.
- [x] **High-Precision Classification**: Scikit-Learn MLP with 99.36% test accuracy and $<1.5\text{ms}$ inference latency.
- [x] **State Machine Debouncing**: Eliminates character repetition (`NNNVVV...`) during held postures.
- [x] **Multilingual Translation**: Seamless translation to 5 Indian languages via Gemini 3.5 Flash Lite or offline dictionary.
- [x] **Local Text-to-Speech**: Speech synthesis without cloud costs or latency.
- [x] **Zero-Leakage Signer Dataset Tooling**: Built-in scripts for recording and validating future temporal datasets.

---

## 📋 System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: Version 3.10.x recommended (3.9 to 3.11 supported)
- **Web Browser**: Google Chrome, Microsoft Edge, or Mozilla Firefox with camera access
- **Hardware**: Standard 720p/1080p webcam; CPU-only execution is fully supported (no GPU required for inference)

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

1. **Isolated Character Postures Only**: Baseline v1 classifies one static gesture at a time.
2. **No Real Human Temporal Word Dataset**: Dynamic signs (e.g. `WATER`, `HELLO`, `FOOD`) require 30-frame temporal sequences from human signers, which have not yet been recorded.
3. **No Continuous Sentence Signing**: Continuous unconstrained signing without neutral pauses is not supported in this baseline.
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
