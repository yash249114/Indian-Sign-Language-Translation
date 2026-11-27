"""
Temporal Model Training Pipeline for ISL Word Recognition.

Default: trains ONLY on real human clips from the dataset manifest whose landmark
files exist and are valid, split by signer ID. Exits with an error when real data
is insufficient. Pass --allow-synthetic to train a clearly labelled reference model
on synthetic trajectories instead (written to a separate *_reference checkpoint).
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
from training.temporal_data import (
    FEATURE_DIM, SEQUENCE_LENGTH, DATA_SOURCE_REAL, DATA_SOURCE_SYNTHETIC,
    InsufficientRealDataError, build_real_dataset, checkpoint_status,
)

DEFAULT_CLASSES = ["HELLO", "THANK_YOU", "HELP", "YES", "NO", "PLEASE", "GOOD", "WATER", "FOOD", "YOU", "ME", "NAME"]


def checkpoint_paths(model_type: str, synthetic: bool, out_dir: str = None):
    out_dir = out_dir or os.path.join(PROJECT_ROOT, "models")
    stem = f"word_temporal_{model_type.lower()}" + ("_reference" if synthetic else "")
    return os.path.join(out_dir, stem + ".pt"), os.path.join(out_dir, stem + ".json")


def train_temporal_model(model_type: str = "BiGRU", epochs: int = 35, lr: float = 0.001,
                         allow_synthetic: bool = False, val_signers=None, test_signers=None,
                         manifest_clips=None, out_dir: str = None):
    print("=" * 65)
    print(f"      TRAINING ISL TEMPORAL SEQUENCE CLASSIFIER [{model_type.upper()}]      ")
    print("=" * 65)

    vocab = load_vocabulary()
    classes = [v["gloss"] for v in vocab] if vocab else DEFAULT_CLASSES
    num_classes = len(classes)
    clips = manifest_clips if manifest_clips is not None else collector.get_manifest().get("clips", [])

    signer_splits, skipped, valid_clips = None, {}, 0
    try:
        ds = build_real_dataset(clips, classes, val_signers=val_signers, test_signers=test_signers)
        has_real_data = True
        X_train, y_train = ds["X_train"], ds["y_train"]
        X_val, y_val = ds["X_val"], ds["y_val"]
        X_test, y_test = ds["X_test"], ds["y_test"]
        signer_splits, skipped, valid_clips = ds["signer_splits"], ds["skipped"], ds["valid_clips"]
        print(f"[DATA] {valid_clips} valid real clips (of {ds['manifest_entries']} manifest entries).")
        print(f"[DATA] Signer splits: {signer_splits}")
        if skipped:
            print(f"[DATA] Skipped entries: {skipped}")
    except InsufficientRealDataError as e:
        if not allow_synthetic:
            print(f"[ERROR] {e}")
            print("[ERROR] Refusing to train. Use --allow-synthetic to build a labelled reference model.")
            raise
        has_real_data = False
        print("[DATA NOTICE] Real data insufficient; training SYNTHETIC REFERENCE model (not real-world validated).")
        X_train, y_train, X_val, y_val, X_test, y_test = generate_reference_sign_trajectories(classes)
        signer_splits = {"train": ["synthetic_signer_01", "synthetic_signer_02"],
                         "val": ["synthetic_signer_03"], "test": ["synthetic_signer_04"]}

    print(f"Partitions (signer-independent): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    torch.manual_seed(42)
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
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs:2d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    train_time = round(time.perf_counter() - t0, 2)
    if best_weights:
        model.load_state_dict(best_weights)

    metadata = {
        "classes": classes,
        "num_classes": num_classes,
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "normalization": "wrist_centered_max_distance_scale_invariant",
        "model_type": model_type,
        "is_real_human_dataset": has_real_data,
        "data_source": DATA_SOURCE_REAL if has_real_data else DATA_SOURCE_SYNTHETIC,
        "validated_on_held_out_signers": False,
        "signer_splits": signer_splits,
        "valid_real_clips": valid_clips,
        "skipped_manifest_entries": skipped,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_accuracy_note": "real held-out val signers" if has_real_data else "SYNTHETIC trajectories only; not real-world performance",
        "training_time_seconds": train_time,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    st = checkpoint_status(metadata)
    metadata["model_status"], metadata["model_status_label"] = st["status"], st["label"]

    target_pt, target_json = checkpoint_paths(model_type, synthetic=not has_real_data, out_dir=out_dir)
    os.makedirs(os.path.dirname(target_pt), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata, **metadata}, target_pt)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Checkpoint saved: {target_pt}")
    print(f"[OK] Metadata saved:   {target_json}")
    print(f"[STATUS] {metadata['model_status_label']}")
    metadata["checkpoint_path"] = target_pt
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BiGRU", choices=["BiGRU", "TCN"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="If real data is insufficient, train a labelled synthetic reference model instead of failing.")
    parser.add_argument("--val-signers", nargs="*", default=None)
    parser.add_argument("--test-signers", nargs="*", default=None)
    args = parser.parse_args()
    try:
        train_temporal_model(model_type=args.model, epochs=args.epochs, allow_synthetic=args.allow_synthetic,
                             val_signers=args.val_signers, test_signers=args.test_signers)
    except InsufficientRealDataError:
        sys.exit(2)
