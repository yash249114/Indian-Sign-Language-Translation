"""
Temporal Model Training Pipeline for ISL Word Recognition.
Trains baseline Bi-GRU and Temporal CNN architectures on real/reference temporal sequences.
Saves model checkpoints and formal metadata JSON artifacts to models/.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.temporal.architectures import BiGRUClassifier, TemporalCNN
from backend.app.collector import load_vocabulary, collector
from training.train_temporal import generate_reference_sign_trajectories, SignSequenceDataset


def train_temporal_model(model_type: str = "BiGRU", epochs: int = 35, lr: float = 0.001):
    print("=" * 65)
    print(f"      TRAINING ISL TEMPORAL SEQUENCE CLASSIFIER [{model_type.upper()}]      ")
    print("=" * 65)

    vocab = load_vocabulary()
    classes = [v["gloss"] for v in vocab] if vocab else ["HELLO", "THANK_YOU", "HELP", "YES", "NO", "PLEASE", "GOOD", "WATER", "FOOD", "YOU", "ME", "NAME"]
    num_classes = len(classes)

    # Check if real recorded samples exist in manifest
    manifest_data = collector.get_manifest()
    clips = manifest_data.get("clips", [])
    has_real_data = len(clips) >= 24

    if has_real_data:
        print(f"[DATA] Loading {len(clips)} real human-recorded sequence files...")
        # Load real landmarks from disk
        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
        for c in clips:
            p = os.path.join(PROJECT_ROOT, c.get("landmarks_file", ""))
            if os.path.exists(p) and c.get("gloss") in classes:
                data = np.load(p)
                feat = data["features"]
                cls_idx = classes.index(c["gloss"])
                split = c.get("split", "train")
                if split == "train":
                    X_train.append(feat)
                    y_train.append(cls_idx)
                elif split == "val":
                    X_val.append(feat)
                    y_val.append(cls_idx)
                else:
                    X_test.append(feat)
                    y_test.append(cls_idx)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)
    else:
        print("[DATA NOTICE] No real multi-signer clips collected yet.")
        print("              Using verified linguistic kinematic trajectory baselines.")
        X_train, y_train, X_val, y_val, X_test, y_test = generate_reference_sign_trajectories(classes)

    print(f"Dataset Partitions (Signer-Independent): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_loader = DataLoader(SignSequenceDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SignSequenceDataset(X_val, y_val), batch_size=32, shuffle=False)

    device = torch.device("cpu")
    if model_type.lower() == "bigru":
        model = BiGRUClassifier(in_features=126, hidden_dim=64, num_layers=2, num_classes=num_classes)
    else:
        model = TemporalCNN(in_features=126, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights = None

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        t_loss, corr, tot = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            corr += (out.argmax(1) == by).sum().item()
            tot += len(by)
        train_acc = corr / tot

        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_corr += (model(bx).argmax(1) == by).sum().item()
                v_tot += len(by)
        val_acc = v_corr / v_tot if v_tot > 0 else 0.0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict()

        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs:2d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    train_time = round(time.perf_counter() - t0, 2)
    if best_weights:
        model.load_state_dict(best_weights)

    # Save to models/word_temporal_bigru.pt or models/word_temporal_tcn.pt
    target_pt = os.path.join(PROJECT_ROOT, "models", f"word_temporal_{model_type.lower()}.pt")
    target_json = os.path.join(PROJECT_ROOT, "models", f"word_temporal_{model_type.lower()}.json")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "model_type": model_type,
        "feature_dim": 126,
        "sequence_length": 30,
        "best_val_acc": round(best_val_acc, 4)
    }, target_pt)

    metadata = {
        "classes": classes,
        "feature_dim": 126,
        "sequence_length": 30,
        "normalization": "wrist_centered_max_distance_scale_invariant",
        "model_type": model_type,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_accuracy": round(best_val_acc, 4),
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "is_real_human_dataset": has_real_data
    }

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Model Checkpoint saved: {target_pt}")
    print(f"[OK] Model Metadata saved:   {target_json}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BiGRU", choices=["BiGRU", "TCN"])
    args = parser.parse_args()
    train_temporal_model(model_type=args.model)
