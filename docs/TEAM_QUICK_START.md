# Team Quick Start Guide — Indian Sign Language Platform

Welcome to the ISL Translation, Gloss Recognition & Query-Based Video Search Platform. Follow these exact steps to clone, configure, run, and test the project on your local machine.

---

## Prerequisites

- **Python**: Version 3.10.x recommended (3.9–3.11 compatible)
- **Git**: Installed and configured
- **Webcam**: Standard USB or built-in webcam
- **Web Browser**: Chrome, Edge, or Firefox with WebRTC support

---

## 1. Clone Repository & Checkout Team Branch

```bash
git clone https://github.com/yash249114/Indian-Sign-Language-Translation.git
cd Indian-Sign-Language-Translation
git checkout baseline-v1-team
```

---

## 2. Set Up Virtual Environment

### Windows (PowerShell / Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Runtime Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Copy the template to create your local `.env`:

### Windows:
```powershell
copy .env.example .env
```

### Linux / macOS:
```bash
cp .env.example .env
```

Edit `.env` to include your Google AI Studio API key (optional for offline mode):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_TRANSLATION_MODEL=gemini-3.5-flash-lite
```

*(Note: If no API key is specified, the application will run with an offline translation fallback.)*

---

## 5. Verify the Installation with Tests

Run the complete test suite to confirm everything is operational:
```bash
python -m pytest tests/ -v
```
All 49 unit and acceptance tests should pass.

---

## 6. Launch the Local Application Server

Start the FastAPI backend with Uvicorn:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

---

## 7. Open the Web Application

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

1. Grant webcam permissions when prompted.
2. In **Fingerspelling Mode**, perform any static ISL sign for `A`–`Z` or `1`–`9`.
3. Observe real-time bounding box, 21 hand landmarks, stabilized prediction, and character accumulation.
4. Select a target Indian language (e.g. Telugu, Hindi, Tamil) and click **Translate** to test Gemini translation and local TTS.

---

## 8. Run in Google Colab (Cloud Experimentation)

No local GPU or webcam required! Open the baseline Colab notebook directly in your browser:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yash249114/Indian-Sign-Language-Translation/blob/baseline-v1-team/notebooks/ISL_Baseline_Colab.ipynb)

---

## 9. API Reference Quick-Check

| Route | Method | Description |
|---|---|---|
| `http://127.0.0.1:8000/health` | GET | Server health check and model loading status |
| `http://127.0.0.1:8000/classes` | GET | List of 35 fingerspelling classes |
| `http://127.0.0.1:8000/api/stats` | GET | Telemetry, benchmarks, and dataset summary |
| `http://127.0.0.1:8000/docs` | GET | Interactive Swagger UI API documentation |
