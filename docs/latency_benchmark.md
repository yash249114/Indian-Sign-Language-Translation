# Latency, Throughput & Hardware Telemetry Benchmark

**Scope**: Empirical profiling of isolated components, end-to-end pipeline latency, and system resource utilization.  
**Hardware Environment**: Windows x64 (CPU Execution).  
**Profiling Runs**: 100 benchmark iterations per component on real video frames.  
**Audit Date**: August 2026  

---

## 1. Component Latency Breakdown

| Component | Mean Latency (ms) | Std Dev (ms) | P95 Latency (ms) | % of Total Frame Time | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. MediaPipe Hands Tracking** | **35.93 ms** | ±1.82 ms | 38.45 ms | **98.3%** | Palm detection & 21 3D landmark localization (Primary Bottleneck). |
| **2. Landmark Normalization** | **0.080 ms** | ±0.012 ms | 0.095 ms | **0.2%** | Translation & max-distance scale normalization. |
| **3. MLP Model Inference** | **0.476 ms** | ±0.065 ms | 0.580 ms | **1.3%** | Forward pass through 128x64 Deep Neural Classifier. |
| **4. Temporal Stabilization** | **0.055 ms** | ±0.008 ms | 0.068 ms | **0.2%** | 6-frame rolling majority voting and debouncing filter. |
| **Total Frame Latency** | **36.55 ms** | **±1.85 ms** | **39.20 ms** | **100.0%** | **Camera Frame Input → Stabilized Sign Output** |

---

## 2. Realized Pipeline Throughput & Webcam FPS

- **Realized Pipeline Throughput**: **27.4 FPS** ($1000 \text{ ms} / 36.55 \text{ ms}$).
- **Target Standard Webcam Stream**: 30.0 FPS.
- **Frame Drop Rate**: Less than 8.6% under un-accelerated CPU execution.
- **Bottleneck Analysis**:
  - The MLP classifier inference takes only **0.476 ms** (allowing over 2,000 inferences/sec).
  - The single dominant compute cost is **MediaPipe Hands Landmark Tracking (35.93 ms)**.

---

## 3. Multilingual Translation & Speech Synthesis Latency

| Operation | Mean Latency (ms) | Engine Type | Zero-Cloud Status |
| :--- | :--- | :--- | :--- |
| **Translation (Telugu)** | **0.0027 ms** | Local Indic Integration Interface / Offline Lexicon | 100% Local / Offline |
| **Translation (Hindi)** | **0.0025 ms** | Local Indic Integration Interface / Offline Lexicon | 100% Local / Offline |
| **Translation (Tamil)** | **0.0026 ms** | Local Indic Integration Interface / Offline Lexicon | 100% Local / Offline |
| **Translation (Kannada)** | **0.0026 ms** | Local Indic Integration Interface / Offline Lexicon | 100% Local / Offline |
| **Translation (Malayalam)** | **0.0028 ms** | Local Indic Integration Interface / Offline Lexicon | 100% Local / Offline |
| **TTS Speech Synthesis** | **313.24 ms** | Local pyttsx3 Offline Audio Engine | 100% Local / Offline |

---

## 4. Hardware Resource Utilization

| Resource | Measured Utilization | Notes |
| :--- | :--- | :--- |
| **RAM (Resident Set Size)** | **240.7 MB** | Extremely lightweight memory footprint. |
| **CPU Utilization** | **~100.0%** (Single Active Core) | MediaPipe pipeline utilizes available CPU thread capacity. |
| **GPU Utilization** | **Not Applicable** | Runs on CPU without requiring CUDA hardware. |
