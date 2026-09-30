"""
Real-time Hand Tracking Module using MediaPipe Hands.
Extracts 21 hand landmarks from video frames and provides drawing utilities.
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


class HandTracker:
    """Encapsulates MediaPipe Hands detection and landmark extraction."""

    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.normalizer = LandmarkNormalizer()

    def process_frame(self, frame_bgr):
        """
        Processes a single BGR OpenCV frame.

        Args:
            frame_bgr: np.ndarray (H, W, 3)

        Returns:
            dict containing:
                - 'detected': bool
                - 'raw_landmarks': list of (x,y,z) normalized coords
                - 'normalized_features': 63D np.ndarray or None
                - 'hand_landmarks_obj': MediaPipe landmarks object or None
                - 'bbox': [x_min, y_min, x_max, y_max] in pixel coords or None
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return {
                "detected": False,
                "raw_landmarks": None,
                "normalized_features": None,
                "hand_landmarks_obj": None,
                "bbox": None
            }

        hand_lms = results.multi_hand_landmarks[0]
        
        # Extract pixel bounding box
        xs = [lm.x * w for lm in hand_lms.landmark]
        ys = [lm.y * h for lm in hand_lms.landmark]
        padding = 20
        bbox = [
            max(0, int(min(xs)) - padding),
            max(0, int(min(ys)) - padding),
            min(w, int(max(xs)) + padding),
            min(h, int(max(ys)) + padding)
        ]

        normalized_features = self.normalizer.extract_and_normalize_mediapipe_landmarks(hand_lms)
        raw_lms = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]

        return {
            "detected": True,
            "raw_landmarks": raw_lms,
            "normalized_features": normalized_features,
            "hand_landmarks_obj": hand_lms,
            "bbox": bbox
        }

    def draw_landmarks(self, frame_bgr, hand_landmarks_obj):
        """Draws aesthetic hand landmark connections on frame."""
        if hand_landmarks_obj is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks_obj,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        return annotated

    def close(self):
        """Releases MediaPipe resources."""
        self.hands.close()
