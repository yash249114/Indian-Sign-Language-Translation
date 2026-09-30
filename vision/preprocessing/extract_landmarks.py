"""
Batch MediaPipe Landmark Extraction & Dataset Splitter.
Processes raw images through MediaPipe Hands, extracts 21 hand landmarks (63D),
applies wrist-relative scale normalization, and performs stratified Train/Val/Test splitting.
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import cv2
import numpy as np
try:
    import mediapipe.python.solutions.hands as mp_hands
except ImportError:
    import mediapipe.solutions.hands as mp_hands

from sklearn.model_selection import train_test_split
from vision.preprocessing.normalizer import LandmarkNormalizer
from vision.preprocessing.dataset_discovery import find_dataset_root, DEFAULT_CANDIDATE_PATHS

OUTPUT_DIR = os.path.join("data", "processed", "landmarks")


def extract_landmarks_from_dataset(dataset_root=None, samples_per_class=120, random_seed=42):
    """
    Extracts MediaPipe landmarks from dataset images across all discovered classes.

    Args:
        dataset_root: Path to the dataset directory containing class subfolders.
        samples_per_class: Maximum number of samples to process per class.
        random_seed: Seed for reproducibility.
    """
    if dataset_root is None:
        for p in DEFAULT_CANDIDATE_PATHS:
            if os.path.exists(p):
                root = find_dataset_root(p)
                if root:
                    dataset_root = root
                    break

    if not dataset_root or not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"[Extraction] Loading dataset from: {dataset_root}")

    class_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_label = {i: c for i, c in enumerate(class_names)}

    print(f"[Extraction] Found {len(class_names)} classes: {class_names}")

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4
    )

    X_list = []
    y_list = []
    stats_per_class = {}

    np.random.seed(random_seed)

    for cidx, cname in enumerate(class_names):
        cpath = os.path.join(dataset_root, cname)
        all_files = [f for f in os.listdir(cpath) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Subsample if more than samples_per_class
        if len(all_files) > samples_per_class:
            selected_files = list(np.random.choice(all_files, samples_per_class, replace=False))
        else:
            selected_files = all_files

        detected_count = 0
        
        for fname in selected_files:
            fpath = os.path.join(cpath, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                feature_vec = LandmarkNormalizer.extract_and_normalize_mediapipe_landmarks(hand_lms)
                X_list.append(feature_vec)
                y_list.append(cidx)
                detected_count += 1

        detection_rate = (detected_count / len(selected_files) * 100) if selected_files else 0
        stats_per_class[cname] = {
            "processed": len(selected_files),
            "detected": detected_count,
            "detection_rate_pct": round(detection_rate, 2)
        }
        print(f"  [{cidx+1:02d}/{len(class_names):02d}] Class '{cname}': {detected_count}/{len(selected_files)} hands detected ({detection_rate:.1f}%)")

    hands.close()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\n[Extraction Summary] Total extracted samples: {X.shape[0]} (Feature dimension: {X.shape[1]})")

    if X.shape[0] == 0:
        raise RuntimeError("No hand landmarks were detected in the dataset! Check image formats and hand visibility.")

    # Stratified Train (70%), Val (15%), Test (15%) split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed, stratify=y
    )

    val_relative_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, random_state=random_seed, stratify=y_train_val
    )

    print(f"[Split] Train samples: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"[Split] Validation samples: {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"[Split] Test samples: {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"), X=X_test, y=y_test)

    label_map_data = {
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "num_classes": len(class_names),
        "feature_dim": LandmarkNormalizer.FEATURE_DIM,
        "total_extracted_samples": int(X.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "detection_stats": stats_per_class
    }

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, indent=4)

    print(f"[Extraction] Landmark dataset saved successfully in '{OUTPUT_DIR}'")
    return label_map_data


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    extract_landmarks_from_dataset(samples_per_class=samples)
