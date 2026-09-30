"""
End-to-End Real-Time Sign Detection Pipeline.
Combines hand tracking, coordinate normalization, MLP classification, and visual rendering.
Provides separated latency telemetry for MediaPipe tracking vs Model inference.
"""

import time
import cv2
import numpy as np

from vision.hand_tracking.hand_tracker import HandTracker
from training.predict import SignPredictor


class SignDetector:
    """End-to-end real-time sign detector pipeline."""

    def __init__(self, confidence_threshold=0.75):
        self.tracker = HandTracker()
        self.predictor = SignPredictor()
        self.confidence_threshold = confidence_threshold

    def process_frame(self, frame_bgr):
        """
        Executes complete inference cycle on a single camera frame.

        Args:
            frame_bgr: np.ndarray (H, W, 3)

        Returns:
            dict containing detection results, bounding boxes, annotated frame,
            and separated component timing latencies.
        """
        t0 = time.perf_counter()
        
        # 1. MediaPipe Hand Tracking
        t_mp0 = time.perf_counter()
        track_res = self.tracker.process_frame(frame_bgr)
        t_mp1 = time.perf_counter()
        mediapipe_ms = (t_mp1 - t_mp0) * 1000.0

        if not track_res["detected"]:
            t_end = time.perf_counter()
            return {
                "detected": False,
                "sign": None,
                "raw_sign": None,
                "confidence": 0.0,
                "top3": [],
                "bbox": None,
                "landmarks": None,
                "annotated_frame": frame_bgr,
                "mediapipe_time_ms": round(mediapipe_ms, 2),
                "model_time_ms": 0.0,
                "inference_time_ms": round((t_end - t0) * 1000.0, 2)
            }

        # 2. MLP Model Inference
        t_model0 = time.perf_counter()
        pred_res = self.predictor.predict_from_features(track_res["normalized_features"])
        t_model1 = time.perf_counter()
        model_ms = (t_model1 - t_model0) * 1000.0

        annotated = self.tracker.draw_landmarks(frame_bgr, track_res["hand_landmarks_obj"])

        sign = pred_res["sign"] if pred_res["confidence"] >= self.confidence_threshold else None

        # Draw bounding box and label on annotated frame
        bbox = track_res["bbox"]
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 200, 80) if pred_res["confidence"] >= self.confidence_threshold else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{pred_res['sign']} ({pred_res['confidence']*100:.1f}%)"
            
            # Label background banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 10)), (x1 + lw + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        t_end = time.perf_counter()
        return {
            "detected": True,
            "sign": sign,
            "raw_sign": pred_res["sign"],
            "confidence": pred_res["confidence"],
            "top3": pred_res["top3"],
            "bbox": bbox,
            "landmarks": track_res["raw_landmarks"],
            "annotated_frame": annotated,
            "mediapipe_time_ms": round(mediapipe_ms, 2),
            "model_time_ms": round(model_ms, 3),
            "inference_time_ms": round((t_end - t0) * 1000.0, 2)
        }

    def close(self):
        self.tracker.close()
