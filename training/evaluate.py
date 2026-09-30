"""
Evaluation & Benchmarking Engine for Indian Sign Language Classifiers.
Evaluates trained MLP Classifier vs Random Forest baseline on held-out test set,
computes Precision, Recall, Macro/Weighted F1, Inference Latency, and FPS,
and generates the Confusion Matrix visualization.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained", "sign_classifier")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


def benchmark_inference_latency(model, X_test, n_warmup=50, n_runs=500):
    """Measures mean inference latency (ms) and throughput (FPS) per single sample."""
    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(X_test[0:1])

    # Timing individual sample predictions (as happens in real-time video stream)
    latencies = []
    sample_indices = np.random.choice(len(X_test), n_runs)
    for idx in sample_indices:
        x = X_test[idx:idx+1]
        t0 = time.perf_counter()
        _ = model.predict(x)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    mean_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    p95_latency_ms = np.percentile(latencies, 95)
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(float(mean_latency_ms), 3),
        "std_latency_ms": round(float(std_latency_ms), 3),
        "p95_latency_ms": round(float(p95_latency_ms), 3),
        "throughput_fps": round(float(fps), 1)
    }


def plot_confusion_matrix(cm, class_names, output_path, model_name="Deep MLP"):
    """Renders high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(16, 14), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name} Classifier (35 Classes)", fontsize=16, pad=16, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Label values in cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=8)

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {output_path}")


def evaluate():
    """Runs complete comparative evaluation on test set."""
    test_path = os.path.join(DATA_DIR, "test.npz")
    label_path = os.path.join(DATA_DIR, "label_map.json")

    if not os.path.exists(test_path) or not os.path.exists(label_path):
        raise FileNotFoundError("Processed test data or label map not found.")

    test_data = np.load(test_path)
    X_test, y_test = test_data["X"], test_data["y"]

    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = label_map["class_names"]

    mlp_path = os.path.join(MODEL_DIR, "mlp_classifier.joblib")
    rf_path = os.path.join(MODEL_DIR, "rf_baseline.joblib")

    if not os.path.exists(mlp_path) or not os.path.exists(rf_path):
        raise FileNotFoundError("Trained models not found. Please run training/train.py first.")

    mlp = joblib.load(mlp_path)
    rf = joblib.load(rf_path)

    results = {"test_samples": int(len(X_test)), "models": {}}

    print("=" * 70)
    print(f"EVALUATION ON HELD-OUT TEST SET ({len(X_test)} samples, 35 classes)")
    print("=" * 70)

    for name, model in [("Deep MLP", mlp), ("Random Forest Baseline", rf)]:
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        
        latency_info = benchmark_inference_latency(model, X_test)
        cm = confusion_matrix(y_test, preds)
        
        rep = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

        print(f"\n--- {name} Results ---")
        print(f"  Accuracy:         {acc*100:.2f}%")
        print(f"  Macro F1:         {f1_macro:.4f}")
        print(f"  Weighted F1:      {f1_wt:.4f}")
        print(f"  Precision (macro):{prec_macro:.4f}")
        print(f"  Recall (macro):   {rec_macro:.4f}")
        print(f"  Single Latency:   {latency_info['mean_latency_ms']} ms (FPS: {latency_info['throughput_fps']})")

        results["models"][name] = {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "latency": latency_info,
            "classification_report": rep
        }

        if name == "Deep MLP":
            cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
            plot_confusion_matrix(cm, class_names, cm_path, model_name=name)

    # Save complete evaluation report
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Evaluation] Full metrics report saved to: {report_path}")
    return results


if __name__ == "__main__":
    evaluate()
