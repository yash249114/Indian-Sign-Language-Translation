"""
Diagnostic script to trace the exact runtime behavior of the word-level pipeline.
"""
import time
import numpy as np
import torch
import json

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.hand_tracking.multi_hand_tracker import MultiHandTracker
from vision.temporal.segmenter import TemporalSegmenter, TemporalState
from vision.temporal.word_predictor import WordSignPredictor
from backend.app.collector import collector

print("=== 1. CHECKING LOADED WORD MODEL ===")
predictor = WordSignPredictor(architecture="tcn")
print(f"Predictor is_trained: {predictor.is_trained}")
print(f"Predictor weights path: {predictor.weights_path}")
print(f"Predictor classes ({len(predictor.classes)}): {predictor.classes}")
print(f"Predictor device: {predictor.device}")

print("\n=== 2. TESTING TEMPORAL PREDICTOR ON SYNTHETIC VS REAL-WORLD RANDOM FEATURES ===")
# Test 1: Random noise (simulating untrained / unseen real inputs)
rand_win = np.random.uniform(-0.5, 0.5, (30, 126)).astype(np.float32)
gloss, conf, top3 = predictor.predict_window(rand_win)
print(f"Random noise prediction: Gloss='{gloss}', Confidence={conf:.4f}")
print(f"Top 3 on random input: {top3}")

# Test 2: Flat zero features (no hands detected)
zero_win = np.zeros((30, 126), dtype=np.float32)
gloss_z, conf_z, top3_z = predictor.predict_window(zero_win)
print(f"Zero window prediction: Gloss='{gloss_z}', Confidence={conf_z:.4f}")
print(f"Top 3 on zero input: {top3_z}")

print("\n=== 3. TESTING TEMPORAL SEGMENTER TRANSITIONS ===")
segmenter = TemporalSegmenter(window_size=30, confidence_threshold=0.70)
print(f"Initial State: {segmenter.state.value}")

# Feed 35 frames of non-zero features
reached_predicting = False
reached_commit = False
for f_idx in range(1, 45):
    # Simulate a person moving hands
    t = f_idx / 30.0
    feat = np.sin(t * np.arange(126)).astype(np.float32) * 0.3
    res = segmenter.update(feat, detected=True, predictor_fn=predictor.predict_window)
    if res["state"] in ("PREDICTING", "CONFIDENCE_CHECK"):
        reached_predicting = True
    if res["is_new_commit"]:
        reached_commit = True
    if f_idx in (1, 10, 20, 29, 30, 31, 35, 40):
        print(f"  Frame {f_idx:2d} | State: {res['state']:16s} | Buffer: {res['buffer_fill']:2d}/30 | Conf: {res['confidence']:.3f} | Commit: {res['is_new_commit']} | Gloss: {res['committed_gloss']}")

print(f"\nReached PREDICTING: {reached_predicting}")
print(f"Reached COMMIT: {reached_commit}")
