# Staged Implementation Plan: Indian Sign Language Word-Level & Temporal Recognition Upgrade

**Author**: Senior Computer-Vision Engineer, Sign-Language NLP Researcher & AI Systems Architect  
**Workspace**: `Y:\CAPSTONE PROJECT`  
**Target Milestone**: Real-Time ISL Word/Sign Recognition, Temporal Sequence Segmentation, ISL Grammar/Gloss Interpretation, and Multimodal Translation Pipeline  

---

## 1. System Limits & Architectural Reality Check

| Dimension | Current State (Baseline) | Upgraded Target State |
| :--- | :--- | :--- |
| **Domain** | Isolated fingerspelling (alphabets A–Z, numbers 1–9) | Dual Mode: Fingerspelling Mode + Temporal Word-Sign Recognition |
| **Spatial / Temporal** | Single static frame, 63D 1-hand landmarks | Multi-Hand / Body Pose temporal sequences ($T=30$ frames, $126\text{D}$ or $144\text{D}$) |
| **Model** | Scikit-learn MLP Classifier (`mlp_classifier.joblib`) | Temporal Model (Lightweight 1D-CNN / Bi-GRU / TCN) + Static MLP |
| **Segmentation** | Hold-debounce state machine on single characters | Temporal Window Segmenter: `IDLE` $\rightarrow$ `SIGN_START` $\rightarrow$ `COLLECT_WINDOW` $\rightarrow$ `PREDICT` $\rightarrow$ `CONFIRM` $\rightarrow$ `COOLDOWN` |
| **Language Layer** | Direct character concatenation (English spelling) | Formal ISL Gloss Sequence Representation with ISL Grammar/Context Mapping (Topic-Comment, SOV, Time-first) |
| **Gemini Role** | Ad-hoc text translation on space commit | Strictly structured gloss sequence $\rightarrow$ Natural Language Translator with validated JSON Schema, rate-limit protection, and local offline fallback |
| **TTS Engine** | Windows SAPI5 (pyttsx3) English-only | Asynchronous, non-blocking playback with verified language voice detection and browser Web Speech API fallback |
| **Dataset Pipeline**| Unlabeled burst images with 45.41% leakage | Local video collection and annotation tool with metadata manifest, pseudonymized signer IDs, and signer-independent splits |

---

## 2. Staged Implementation Roadmap (Phases 0 – 8)

### Phase 0: Baseline Audit & Frozen Dependencies
- **Audit Findings**:
  - Raw dataset `archive (5)/data` contains only 35 static character classes (1–9, A–Z) across 42,000 images.
  - Zero word-level or continuous sign video datasets exist in local workspace or downloads.
  - Missing `requirements.txt` despite references in documentation.
  - Concurrency flaw: WebSocket translation call on space/translate is synchronous, causing ~1.2s video freeze.
- **Deliverables**:
  - Baseline audit artifact created.
  - Frozen `requirements.txt` generated with verified installed package versions.

### Phase 1: Fingerspelling Hardening & WebSocket Concurrency
- **Actions**:
  - Decouple synchronous translation in `backend/app/main.py` using `asyncio.to_thread` or non-blocking background task.
  - Preserve working fingerspelling state machine and verify all 39 existing unit tests pass.
  - Add rotation invariance or tilt tolerance in `vision/preprocessing/normalizer.py`.

### Phase 2: Word-Level Dataset Pipeline & In-Browser Collection Tool
- **Actions**:
  - Define an authoritative everyday ISL vocabulary (10–12 essential signs: `HELLO`, `THANK_YOU`, `HELP`, `YES`, `NO`, `PLEASE`, `GOOD`, `WATER`, `FOOD`, `YOU`, `ME`, `NAME`) based on ISLRTC/NCERT dictionaries.
  - Build `backend/app/collector.py` and API endpoints:
    - POST `/api/dataset/record-clip` (receives recorded video/landmark trajectory, metadata: `signer_id`, `gloss`, `handedness`, `lighting`, `split`).
    - GET `/api/dataset/manifest` (reads and validates `data/word_dataset/manifest.json`).
    - GET `/api/dataset/vocab` (returns verified sign descriptions, reference motions, and grammatical roles).
  - Implement signer-independent train/val/test split logic (grouping by `signer_id`).
  - Provide CLI collection utility `scripts/collect_word_dataset.py`.

### Phase 3: Real-Time Multi-Hand/Pose Pipeline & Temporal Model
- **Actions**:
  - Create `vision/hand_tracking/multi_hand_tracker.py` using MediaPipe Hands (supporting 2 hands: left + right, 126D normalized features, velocity/delta features, handedness tracking, missing-hand zero-masking).
  - Create temporal sequence buffer: window length $T=30$ frames (~1 second at 30 FPS).
  - Build and benchmark temporal models in PyTorch (CPU-optimized):
    - Model A: Lightweight 1D Temporal Convolutional Network (TCN / Conv1D).
    - Model B: Bidirectional GRU (Bi-GRU, 2 layers, hidden dimension 64).
  - Measure per-frame feature extraction latency, temporal inference latency, and memory footprint.

### Phase 4: Sign Segmentation & Stability State Machine
- **Actions**:
  - Implement temporal segmenter in `vision/temporal/segmenter.py`:
    - States: `IDLE` $\rightarrow$ `SIGN_START` (motion energy threshold exceeded) $\rightarrow$ `COLLECTING` (sliding window buffer) $\rightarrow$ `PREDICTING` $\rightarrow$ `CONFIDENCE_CHECK` $\rightarrow$ `COMMIT` $\rightarrow$ `COOLDOWN` $\rightarrow$ `WAIT_NEUTRAL` $\rightarrow$ `IDLE`.
  - Calibrate dynamic thresholding to prevent duplicate commits while holding a sign.
  - Expose boundary timings, top-k candidates, and uncertainty score.

### Phase 5: ISL Grammar & Gloss Interpretation Layer
- **Actions**:
  - Build `backend/app/isl_grammar.py`:
    - Distinct representation: Gloss tokens (e.g. `[TIME] [TOPIC/SUBJECT] [OBJECT] [ACTION/VERB] [QUESTION/NEGATION]`).
    - ISL linguistic structure: Topic-Comment, SOV (Subject-Object-Verb), Time-first ordering, Negation-at-end (e.g., `ME WATER WANT` $\rightarrow$ "I want water"; `YOU NAME WHAT` $\rightarrow$ "What is your name?").
    - Rule-based parser + uncertainty flagging for non-canonical sequences.
    - Separate raw gloss emission from interpreted natural text.

### Phase 6: Gemini Integration for Completed Gloss Sequences
- **Actions**:
  - Upgrade `backend/app/translation.py`:
    - Strict structured prompting with JSON schema validation.
    - Schema fields: `gloss_sequence`, `natural_english_sentence`, `target_language_translation`, `grammatical_notes`, `confidence_level`, `uncertainty_warning`.
    - Enforce rate-limiting, debouncing, SHA-256 caching, and offline fallback (lexicon + grammar mapping).
    - Guarantee zero transmission of video frames or landmarks to the cloud.

### Phase 7: Multilingual Output & Non-Blocking Offline TTS
- **Actions**:
  - Verify installed pyttsx3 voices and language capabilities.
  - Implement async audio dispatch and audio replay/stop controls.
  - Graceful fallback to client-side Web Speech API when native offline voice is missing for specific Indic languages.

### Phase 8: UI Upgrades & Comprehensive Signer-Independent Evaluation
- **Actions**:
  - Update `frontend/index.html` and `frontend/app.js`:
    - Mode Switcher: `[Fingerspelling Mode]` vs `[Word / Sign Mode]` vs `[Continuous Signing]`.
    - Live Gloss Timeline: Displays recognized glosses with timestamps and confidence tags.
    - ISL Structure Panel: Shows Gloss $\rightarrow$ English Interpretation $\rightarrow$ Indic Translation.
    - Annotation & Dataset Collection Modal.
    - Action buttons: Speak, Replay, Edit, Undo, Clear.
  - Create evaluation suite `scripts/evaluate_temporal_model.py`:
    - Top-1 accuracy, Macro F1, Confusion Matrix.
    - Signer-independent evaluation.
    - End-to-end component latency breakdown.
  - Run regression test suite and verify 100% test pass rate.
