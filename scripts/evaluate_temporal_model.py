"""
Evaluation script for temporal word-sign recognition models.
Evaluates Bi-GRU and Temporal CNN models on held-out signer-independent test sets.
Reports Accuracy, Macro F1, Per-class Precision/Recall, and Confusion Matrix.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temporal.architectures import BiGRUClassifier, TemporalCNN
from backend.app.collector import load_vocabulary
from training.train_temporal import generate_reference_sign_trajectories


def evaluate_temporal_checkpoint(model_path: str):
    print("=" * 65)
    print("         ISL TEMPORAL SEQUENCE MODEL INDEPENDENT EVALUATION       ")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found: {model_path}")
        return

    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device)

    classes = ckpt.get("classes", [])
    if not classes:
        vocab = load_vocabulary()
        classes = [v["gloss"] for v in vocab]

    num_classes = len(classes)
    model_type = ckpt.get("model_type", "BiGRU" if "bigru" in model_path.lower() else "TCN")

    if model_type.lower() == "bigru":
        model = BiGRUClassifier(in_features=126, hidden_dim=64, num_layers=2, num_classes=num_classes)
    else:
        model = TemporalCNN(in_features=126, num_classes=num_classes)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded Model: {model_type} from {model_path}")
    print(f"Vocabulary ({num_classes} classes): {classes}")

    # Generate test set strictly from unseen test signer (Signer 04)
    _, _, _, _, X_test, y_test = generate_reference_sign_trajectories(classes, num_signers=4, reps_per_signer=20)
    print(f"Signer-Independent Test Partition: {len(X_test)} samples (Signer 04 only)")

    t0 = time.perf_counter()
    with torch.no_grad():
        bx = torch.tensor(X_test, dtype=torch.float32)
        logits = model(bx)
        preds = logits.argmax(1).numpy()
    eval_time = round((time.perf_counter() - t0) * 1000.0, 2)
    per_seq_ms = round(eval_time / len(X_test), 3)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))

    print("\n[PERFORMANCE METRICS]")
    print(f"  Top-1 Test Accuracy: {acc * 100:.2f}%")
    print(f"  Macro F1 Score:     {macro_f1:.4f}")
    print(f"  Mean Latency:       {per_seq_ms} ms/sequence ({round(1000.0/per_seq_ms, 1)} sequences/sec)")

    print("\n[PER-CLASS CLASSIFICATION REPORT]")
    print(classification_report(y_test, preds, target_names=classes))

    print("[CONFUSION MATRIX]")
    cm = confusion_matrix(y_test, preds)
    print(cm)

    return {
        "model_path": model_path,
        "model_type": model_type,
        "classes": classes,
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_sequence_latency_ms": per_seq_ms
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.join(PROJECT_ROOT, "models", "word_temporal_bigru.pt"))
    args = parser.parse_args()
    evaluate_temporal_checkpoint(args.model_path)
