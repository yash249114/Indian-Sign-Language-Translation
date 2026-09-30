"""
Multi-Hand Landmark Tracking and Feature Extraction Module.
Tracks up to 2 hands simultaneously using MediaPipe Hands,
classifies Left vs Right hand, normalizes landmarks relative to wrist,
and produces a standardized 126-dimensional feature vector per frame.
"""

import cv2
import numpy as np
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles

from vision.preprocessing.normalizer import LandmarkNormalizer


class MultiHandTracker:
    """
    Simultaneously tracks left and right hands, extracts wrist-normalized coordinates,
    and outputs a structured 126D multi-hand feature vector (63D Left + 63D Right).
    """

    HAND_FEATURE_DIM = 63
    TOTAL_FEATURE_DIM = 126  # 63 Left + 63 Right

    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.normalizer = LandmarkNormalizer()

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Processes a BGR image frame and extracts normalized multi-hand features.

        Args:
            frame_bgr: np.ndarray (H, W, 3)

        Returns:
            dict containing:
                - 'detected': bool (True if at least one hand is detected)
                - 'num_hands': int (0, 1, or 2)
                - 'left_detected': bool
                - 'right_detected': bool
                - 'feature_vector': np.ndarray of shape (126,)
                - 'left_features': np.ndarray of shape (63,)
                - 'right_features': np.ndarray of shape (63,)
                - 'left_hand_obj': MediaPipe hand landmarks object or None
                - 'right_hand_obj': MediaPipe hand landmarks object or None
                - 'bboxes': list of bounding boxes
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_features = np.zeros(self.HAND_FEATURE_DIM, dtype=np.float32)
        right_features = np.zeros(self.HAND_FEATURE_DIM, dtype=np.float32)
        left_obj = None
        right_obj = None
        left_detected = False
        right_detected = False
        bboxes = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                # Calculate bounding box
                xs = [lm.x * w for lm in hand_lms.landmark]
                ys = [lm.y * h for lm in hand_lms.landmark]
                pad = 15
                bbox = [
                    max(0, int(min(xs)) - pad),
                    max(0, int(min(ys)) - pad),
                    min(w, int(max(xs)) + pad),
                    min(h, int(max(ys)) + pad)
                ]
                bboxes.append({"label": label, "bbox": bbox, "score": score})

                # Normalized coordinates relative to wrist
                norm_feat = self.normalizer.extract_and_normalize_mediapipe_landmarks(hand_lms)

                if label == "Left":
                    left_features = norm_feat
                    left_obj = hand_lms
                    left_detected = True
                else:
                    right_features = norm_feat
                    right_obj = hand_lms
                    right_detected = True

        combined_feature = np.concatenate([left_features, right_features], axis=0)
        detected = left_detected or right_detected
        num_hands = int(left_detected) + int(right_detected)

        return {
            "detected": detected,
            "num_hands": num_hands,
            "left_detected": left_detected,
            "right_detected": right_detected,
            "feature_vector": combined_feature,
            "left_features": left_features,
            "right_features": right_features,
            "left_hand_obj": left_obj,
            "right_hand_obj": right_obj,
            "bboxes": bboxes
        }

    def draw_landmarks(self, frame_bgr: np.ndarray, left_obj, right_obj) -> np.ndarray:
        """Visualizes left and right hand landmarks with distinct color styling."""
        annotated = frame_bgr.copy()
        if left_obj is not None:
            mp_drawing.draw_landmarks(
                annotated,
                left_obj,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        if right_obj is not None:
            mp_drawing.draw_landmarks(
                annotated,
                right_obj,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return annotated

    def close(self):
        """Releases MediaPipe hand tracking pipeline."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
