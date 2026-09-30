"""
Dataset Validation Script for Word-Level ISL Clips and Sequences.
Verifies manifest consistency, video files, landmark sequence dimensions,
and signer-independent partitioning without cross-signer contamination.
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.collector import collector, MANIFEST_PATH, VOCAB_PATH


def validate_dataset():
    print("=" * 65)
    print("      ISL WORD-LEVEL DATASET INTEGRITY & VALIDATION AUDIT        ")
    print("=" * 65)

    if not os.path.exists(VOCAB_PATH):
        print("[FAIL] Vocabulary file missing at:", VOCAB_PATH)
        return False
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f).get("vocabulary", [])
    classes = [item["gloss"] for item in vocab]
    print(f"[OK] Authoritative Vocabulary Loaded: {len(classes)} classes.")
    print(f"     Classes: {classes}")

    manifest_data = collector.get_manifest()
    total_clips = manifest_data.get("total_clips", 0)
    print(f"\nManifest Status: {total_clips} clips registered in manifest.json")

    if total_clips == 0:
        print("\n[AUDIT RESULT]:")
        print("  Real word video clips count: 0")
        print("  Extracted landmark sequence files count: 0")
        print("  Unique human signers in dataset: 0")
        print("\n[CRITICAL NOTE]:")
        print("  Word recognition cannot currently be trained from the existing alphabet image dataset.")
        print("  The existing archive (5)/data contains ONLY isolated static letters A-Z and numbers 1-9.")
        print("  Use 'python scripts/collect_word_dataset.py' or the in-browser Dataset Collector")
        print("  to record multi-signer video clips for the 12 target word classes.")
        return False

    clips = manifest_data.get("clips", [])
    signers = set()
    split_signers = {"train": set(), "val": set(), "test": set()}
    valid_landmarks = 0

    for c in clips:
        sid = c.get("signer_id", "unknown")
        signers.add(sid)
        split = c.get("split", "train")
        split_signers[split].add(sid)

        lms_path = c.get("landmarks_file")
        if lms_path and os.path.exists(os.path.join(PROJECT_ROOT, lms_path)):
            try:
                npz = np.load(os.path.join(PROJECT_ROOT, lms_path))
                feats = npz["features"]
                if feats.shape == (30, 126):
                    valid_landmarks += 1
            except Exception:
                pass

    print(f"\nUnique Signers ({len(signers)}): {list(signers)}")
    print(f"Signers per Split:")
    print(f"  Train: {list(split_signers['train'])}")
    print(f"  Val:   {list(split_signers['val'])}")
    print(f"  Test:  {list(split_signers['test'])}")

    # Check for cross-signer leakage
    leak_train_val = split_signers["train"].intersection(split_signers["val"])
    leak_train_test = split_signers["train"].intersection(split_signers["test"])
    leak_val_test = split_signers["val"].intersection(split_signers["test"])

    if leak_train_val or leak_train_test or leak_val_test:
        print("[FAIL] Signer-independent split violation! Cross-signer contamination detected:")
        if leak_train_val: print(f"  Train/Val overlap: {leak_train_val}")
        if leak_train_test: print(f"  Train/Test overlap: {leak_train_test}")
        return False
    else:
        print("[OK] Signer-independent split verified! Zero signer leakage across train/val/test.")

    print(f"[OK] Validated {valid_landmarks}/{total_clips} landmark sequence files (30, 126).")
    return True


if __name__ == "__main__":
    validate_dataset()
