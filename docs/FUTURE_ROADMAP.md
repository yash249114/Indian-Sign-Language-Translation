# Future Development Roadmap — Indian Sign Language Platform

Following the successful delivery and verification of **Baseline v1**, this roadmap defines the planned phases for future development.

---

## Phase 2: Human Word-Sign Dataset Collection & Temporal Modeling

- [ ] **Multi-Signer Data Collection**:
  - Record 10+ repetitions per sign across 9 distinct signers (`S01`–`S09`).
  - Target Initial 12-Word Vocabulary: `HELLO`, `THANK_YOU`, `HELP`, `YES`, `NO`, `PLEASE`, `GOOD`, `WATER`, `FOOD`, `YOU`, `ME`, `NAME`.
  - Capture varied lighting, background conditions, and signing speeds using `scripts/collect_word_dataset.py`.
- [ ] **Signer-Independent Model Training**:
  - Train Bi-GRU and Temporal Convolutional Network (TCN) on real human landmark sequences.
  - Strict signer splitting: `S01`–`S06` (train), `S07` (validation), `S08`–`S09` (test).
  - Target test accuracy: $\ge 90\%$ on unseen signers.

---

## Phase 3: Continuous Signing & Advanced ISL Linguistics

- [ ] **Continuous Temporal Boundary Segmentation**:
  - Connectionist Temporal Classification (CTC) loss or Transformer encoder-decoder for continuous sequence transcription.
  - End-to-end gloss sequence generation without requiring neutral-pause delimiters.
- [ ] **Non-Manual Feature Extraction**:
  - Integrate MediaPipe Face Mesh to capture facial expressions, head tilts, and eyebrow movements (grammatical markers in ISL).
- [ ] **Expanded ISL Grammar Engine**:
  - Contextual handling of classifiers, directional verbs, and spatial reference points.

---

## Phase 4: Query-Based ISL Video Search & Retrieval

- [ ] **Video Ingestion & Indexing Pipeline**:
  - Extract temporal landmark embeddings from educational ISL video archives.
  - Index embeddings into a vector database (e.g. Qdrant or FAISS).
- [ ] **Multimodal Search Interface**:
  - Search ISL videos by English text queries, spoken voice, or webcam signing queries.
  - Frame-accurate video playback jumping directly to the searched sign occurrence.
