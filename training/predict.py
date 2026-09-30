"""
Standalone Sign Recognition Predictor.
Loads trained MLP model and provides normalized feature vector to class prediction & confidence scores.
"""

import os
import sys
import json
import joblib
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.preprocessing.normalizer import LandmarkNormalizer

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "sign_classifier", "mlp_classifier.joblib")
LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks", "label_map.json")


class SignPredictor:
    """Inference wrapper for Indian Sign Language landmark-based classifier."""

    def __init__(self, model_path=MODEL_PATH, label_path=LABEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label map not found: {label_path}.")

        self.model = joblib.load(model_path)
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        self.class_names = label_data["class_names"]
        self.idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
        self.normalizer = LandmarkNormalizer()

    def predict_from_features(self, feature_vector):
        """
        Predicts sign class and confidence from a 63D normalized feature vector.

        Args:
            feature_vector: np.ndarray of shape (63,) or (1, 63)

        Returns:
            dict with 'sign', 'confidence', 'class_index', 'top3'
        """
        feat = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if feat.shape[1] != LandmarkNormalizer.FEATURE_DIM:
            raise ValueError(f"Expected feature vector of length {LandmarkNormalizer.FEATURE_DIM}, got {feat.shape[1]}")

        probs = self.model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.idx_to_label[pred_idx]

        # Top 3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"sign": self.idx_to_label[int(idx)], "confidence": round(float(probs[idx]), 4)}
            for idx in top3_indices
        ]

        return {
            "sign": pred_label,
            "confidence": round(confidence, 4),
            "class_index": pred_idx,
            "top3": top3
        }

    def predict_from_landmarks(self, landmarks):
        """Predicts sign class directly from raw MediaPipe hand landmarks."""
        feat = self.normalizer.extract_and_normalize_mediapipe_landmarks(landmarks)
        return self.predict_from_features(feat)


if __name__ == "__main__":
    predictor = SignPredictor()
    # Dummy feature test
    dummy = np.zeros(63, dtype=np.float32)
    res = predictor.predict_from_features(dummy)
    print("SignPredictor test output:", res)
