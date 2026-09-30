"""
Temporal Sign Predictor Engine.
Runs forward inference on a 30-frame temporal window of 126D multi-hand features.
Supports both PyTorch TemporalCNN and BiGRU models with low-latency CPU execution.
"""

import os
import sys
import json
import time
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temporal.architectures import TemporalCNN, BiGRUClassifier
from backend.app.collector import load_vocabulary

VOCAB_LIST = [item["gloss"] for item in load_vocabulary()]
if not VOCAB_LIST:
    VOCAB_LIST = ["HELLO", "THANK_YOU", "HELP", "YES", "NO", "PLEASE", "GOOD", "WATER", "FOOD", "YOU", "ME", "NAME"]

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained", "temporal_word_model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "temporal_cnn.pt")


class WordSignPredictor:
    """Inference engine for temporal sign sequence classification."""

    def __init__(self, weights_path: Optional[str] = None, architecture: str = "tcn", confidence_threshold: float = 0.70):
        self.classes = VOCAB_LIST
        self.num_classes = len(self.classes)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cpu")
        self.architecture = architecture

        if architecture == "bigru":
            self.model = BiGRUClassifier(in_features=126, hidden_dim=64, num_classes=self.num_classes)
        else:
            self.model = TemporalCNN(in_features=126, num_classes=self.num_classes)

        self.weights_path = weights_path or WEIGHTS_PATH
        self.is_trained = False

        if os.path.exists(self.weights_path):
            try:
                ckpt = torch.load(self.weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                    self.classes = ckpt.get("classes", self.classes)
                else:
                    self.model.load_state_dict(ckpt)
                self.is_trained = True
                print(f"[WordSignPredictor] Successfully loaded temporal model from {self.weights_path}")
            except Exception as e:
                print(f"[WordSignPredictor] Warning: Could not load weights: {e}")

        self.model.to(self.device)
        self.model.eval()

    def predict_window(self, window_features: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Infers sign label from a temporal window of shape (30, 126) or (SeqLen, 126).

        Returns:
            Tuple of (top_gloss, confidence, top3_list)
        """
        t0 = time.perf_counter()
        if window_features.shape[0] < 5:
            return "UNKNOWN", 0.0, []

        # Check for zero or neutral hand presence (no hands in frame)
        if np.sum(np.abs(window_features)) < 0.5:
            return "NEUTRAL", 0.0, [{"gloss": "NEUTRAL", "confidence": 1.0}]

        # Pad or interpolate to exactly 30 frames if necessary
        if window_features.shape[0] != 30:
            indices = np.linspace(0, window_features.shape[0] - 1, 30).astype(int)
            seq = window_features[indices]
        else:
            seq = window_features

        tensor_in = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 30, 126)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:3]
        top_gloss = self.classes[top_indices[0]] if top_indices[0] < len(self.classes) else "UNKNOWN"
        top_conf = float(probs[top_indices[0]])

        top3 = []
        for idx in top_indices:
            name = self.classes[idx] if idx < len(self.classes) else f"Class_{idx}"
            top3.append({
                "gloss": name,
                "confidence": round(float(probs[idx]), 3)
            })

        t1 = time.perf_counter()
        inference_time_ms = round((t1 - t0) * 1000.0, 2)

        return top_gloss, top_conf, top3
