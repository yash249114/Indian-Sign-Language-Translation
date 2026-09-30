"""
Model Training Engine for Indian Sign Language Classifier.
Trains both an optimized Deep Multi-Layer Perceptron (MLP) Classifier and a
Random Forest Classifier baseline on the extracted 63D normalized hand landmark dataset.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained", "sign_classifier")


def load_dataset():
    """Loads the train, validation, and test splits along with the label mapping."""
    train_path = os.path.join(DATA_DIR, "train.npz")
    val_path = os.path.join(DATA_DIR, "val.npz")
    test_path = os.path.join(DATA_DIR, "test.npz")
    label_path = os.path.join(DATA_DIR, "label_map.json")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path, label_path]):
        raise FileNotFoundError("Extracted dataset files missing. Please run extract_landmarks.py first.")

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return (
        (train_data["X"], train_data["y"]),
        (val_data["X"], val_data["y"]),
        (test_data["X"], test_data["y"]),
        label_map
    )


def train_models():
    """Trains both MLP and Random Forest classifiers and saves models."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map = load_dataset()
    
    num_classes = len(label_map["class_names"])
    print(f"[Training] Dataset loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"[Training] Total classes: {num_classes}")

    # =========================================================================
    # 1. Train MLP Classifier (Primary Neural Model)
    # Architecture: Input (63) -> Hidden 1 (128) -> Hidden 2 (64) -> Output (35)
    # Regularization: Early stopping, L2 penalty alpha=1e-4, Adam optimizer
    # =========================================================================
    print("\n--- Training Deep MLP Classifier ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
        random_state=42,
        verbose=False
    )

    t0 = time.time()
    mlp.fit(X_train, y_train)
    mlp_train_time = time.time() - t0

    mlp_val_preds = mlp.predict(X_val)
    mlp_val_acc = accuracy_score(y_val, mlp_val_preds)
    mlp_val_f1 = f1_score(y_val, mlp_val_preds, average="weighted")
    print(f"[MLP] Training time: {mlp_train_time:.2f}s | Val Acc: {mlp_val_acc*100:.2f}% | Val F1: {mlp_val_f1:.4f}")

    # =========================================================================
    # 2. Train Random Forest Classifier Baseline
    # =========================================================================
    print("\n--- Training Random Forest Classifier Baseline ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    rf_val_preds = rf.predict(X_val)
    rf_val_acc = accuracy_score(y_val, rf_val_preds)
    rf_val_f1 = f1_score(y_val, rf_val_preds, average="weighted")
    print(f"[Random Forest] Training time: {rf_train_time:.2f}s | Val Acc: {rf_val_acc*100:.2f}% | Val F1: {rf_val_f1:.4f}")

    # Save models
    mlp_model_path = os.path.join(MODEL_DIR, "mlp_classifier.joblib")
    rf_model_path = os.path.join(MODEL_DIR, "rf_baseline.joblib")

    joblib.dump(mlp, mlp_model_path)
    joblib.dump(rf, rf_model_path)

    # Save metadata
    training_summary = {
        "mlp": {
            "model_path": "models/trained/sign_classifier/mlp_classifier.joblib",
            "training_time_seconds": round(mlp_train_time, 3),
            "val_accuracy": round(float(mlp_val_acc), 4),
            "val_weighted_f1": round(float(mlp_val_f1), 4),
            "num_layers": len(mlp.hidden_layer_sizes) + 2,
            "hidden_layers": list(mlp.hidden_layer_sizes),
            "iterations_trained": int(mlp.n_iter_)
        },
        "random_forest": {
            "model_path": "models/trained/sign_classifier/rf_baseline.joblib",
            "training_time_seconds": round(rf_train_time, 3),
            "val_accuracy": round(float(rf_val_acc), 4),
            "val_weighted_f1": round(float(rf_val_f1), 4),
            "n_estimators": 100,
            "max_depth": 20
        }
    }

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4)

    print(f"\n[Training] Models successfully saved to '{MODEL_DIR}'")
    return training_summary


if __name__ == "__main__":
    train_models()
