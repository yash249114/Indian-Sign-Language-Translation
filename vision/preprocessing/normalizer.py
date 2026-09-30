"""
Landmark Coordinate Normalization Engine.
Implements translation normalization relative to wrist and scale normalization
to produce a 63-dimensional scale- and position-invariant feature vector.
"""

import numpy as np


class LandmarkNormalizer:
    """
    Normalizes 21 3D hand landmarks (x, y, z) into a 63D invariant feature vector.
    """

    NUM_LANDMARKS = 21
    FEATURE_DIM = 63

    @staticmethod
    def normalize_landmarks(landmarks):
        """
        Normalizes a list or array of 21 (x, y, z) landmarks.

        Args:
            landmarks: array-like of shape (21, 3) or flat array of length 63.

        Returns:
            np.ndarray: 1D array of shape (63,) with wrist-centered, scale-normalized coordinates.
        """
        coords = np.array(landmarks, dtype=np.float32)
        if coords.ndim == 1:
            if coords.shape[0] != LandmarkNormalizer.FEATURE_DIM:
                raise ValueError(f"Expected flat landmark array of length 63, got {coords.shape[0]}")
            coords = coords.reshape((LandmarkNormalizer.NUM_LANDMARKS, 3))
        elif coords.shape != (LandmarkNormalizer.NUM_LANDMARKS, 3):
            raise ValueError(f"Expected landmarks of shape (21, 3), got {coords.shape}")

        # 1. Translation Normalization: Shift relative to wrist (landmark 0)
        wrist = coords[0].copy()
        centered = coords - wrist

        # 2. Scale Normalization: Normalize by maximum distance from wrist
        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)

        if max_dist > 1e-6:
            normalized = centered / max_dist
        else:
            normalized = centered

        return normalized.flatten()

    @staticmethod
    def extract_and_normalize_mediapipe_landmarks(hand_landmarks):
        """
        Extracts coordinates from a MediaPipe NormalizedLandmarkList protobuf and normalizes them.

        Args:
            hand_landmarks: mediapipe hand_landmarks object with .landmark list

        Returns:
            np.ndarray: 1D array of shape (63,)
        """
        raw_coords = []
        for lm in hand_landmarks.landmark:
            raw_coords.append([lm.x, lm.y, lm.z])
        return LandmarkNormalizer.normalize_landmarks(raw_coords)
