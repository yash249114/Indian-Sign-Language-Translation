"""
Temporal Sign Segmentation & Boundary Detection State Machine.
Implements the formal 8-stage sequence lifecycle:
  IDLE -> SIGN_START -> COLLECTING -> PREDICTING -> CONFIDENCE_CHECK -> COMMIT -> COOLDOWN -> WAIT_NEUTRAL -> IDLE

Features:
- Kinetic motion energy estimation to detect gesture onset and boundary endings
- Window accumulator of T=30 frames
- Confidence calibration check to prevent premature or uncertain commits
- Single-commit guarantee per sign execution
- Explicit boundary timing telemetry
"""

import time
import collections
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np


class TemporalState(Enum):
    IDLE = "IDLE"
    SIGN_START = "SIGN_START"
    COLLECTING = "COLLECTING"
    PREDICTING = "PREDICTING"
    CONFIDENCE_CHECK = "CONFIDENCE_CHECK"
    COMMIT = "COMMIT"
    COOLDOWN = "COOLDOWN"
    WAIT_NEUTRAL = "WAIT_NEUTRAL"


class TemporalSegmenter:
    """
    Segments continuous multi-hand feature streams into isolated word signs,
    manages the 30-frame temporal buffer, and enforces debouncing.
    """

    def __init__(
        self,
        window_size: int = 30,
        motion_threshold: float = 0.035,
        confidence_threshold: float = 0.70,
        cooldown_frames: int = 15,
        neutral_reset_frames: int = 8
    ):
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        self.neutral_reset_frames = neutral_reset_frames

        self.state = TemporalState.IDLE
        self.buffer = collections.deque(maxlen=window_size)
        self.prev_feature = None

        self.consecutive_neutral = 0
        self.cooldown_counter = 0
        self.last_committed_gloss = None
        self.last_commit_time = 0.0

        self.sign_start_time = 0.0
        self.sign_end_time = 0.0

    def compute_motion_energy(self, current_feature: np.ndarray) -> float:
        """Computes kinetic delta across hand coordinate landmarks."""
        if self.prev_feature is None:
            self.prev_feature = current_feature.copy()
            return 0.0
        delta = np.linalg.norm(current_feature - self.prev_feature)
        self.prev_feature = current_feature.copy()
        return float(delta)

    def update(
        self,
        feature_vector: np.ndarray,
        detected: bool,
        predictor_fn=None
    ) -> Dict[str, Any]:
        """
        Processes a single incoming frame feature vector through the state machine.

        Args:
            feature_vector: 126D np.ndarray (or zeros if undetected)
            detected: bool whether at least one hand is detected
            predictor_fn: callable taking (window_size, 126) np.ndarray and returning (gloss, conf, top3)

        Returns:
            dict containing state, committed gloss, confidence, telemetry, and boundary timings.
        """
        now = time.perf_counter()
        motion = self.compute_motion_energy(feature_vector) if detected else 0.0

        is_new_commit = False
        committed_gloss = None
        commit_confidence = 0.0
        top_candidates = []
        is_uncertain = False

        # Track consecutive neutral frames
        if not detected:
            self.consecutive_neutral += 1
        else:
            self.consecutive_neutral = 0

        # State Transitions
        if self.state == TemporalState.IDLE:
            if detected and motion > self.motion_threshold:
                self.state = TemporalState.SIGN_START
                self.sign_start_time = now
                self.buffer.clear()
                self.buffer.append(feature_vector)
            else:
                self.buffer.append(feature_vector)

        elif self.state == TemporalState.SIGN_START:
            self.buffer.append(feature_vector)
            self.state = TemporalState.COLLECTING

        elif self.state == TemporalState.COLLECTING:
            self.buffer.append(feature_vector)
            if len(self.buffer) >= self.window_size:
                self.state = TemporalState.PREDICTING

        elif self.state == TemporalState.PREDICTING:
            if predictor_fn is not None and len(self.buffer) >= self.window_size:
                window_data = np.array(list(self.buffer), dtype=np.float32)
                pred_gloss, pred_conf, top_candidates = predictor_fn(window_data)
                
                self.state = TemporalState.CONFIDENCE_CHECK
                if pred_conf >= self.confidence_threshold:
                    # Confidence verified
                    self.state = TemporalState.COMMIT
                    committed_gloss = pred_gloss
                    commit_confidence = pred_conf
                    self.last_committed_gloss = pred_gloss
                    self.last_commit_time = now
                    self.sign_end_time = now
                    is_new_commit = True
                    self.cooldown_counter = self.cooldown_frames
                else:
                    # Low confidence / uncertain sign
                    is_uncertain = True
                    self.state = TemporalState.WAIT_NEUTRAL
            else:
                self.state = TemporalState.IDLE

        elif self.state == TemporalState.COMMIT:
            # Transition to cooldown
            self.state = TemporalState.COOLDOWN

        elif self.state == TemporalState.COOLDOWN:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = TemporalState.WAIT_NEUTRAL

        elif self.state == TemporalState.WAIT_NEUTRAL:
            # Wait until hand returns to rest or leaves frame
            if self.consecutive_neutral >= self.neutral_reset_frames or motion < self.motion_threshold:
                self.state = TemporalState.IDLE
                self.last_committed_gloss = None

        duration_ms = round((now - self.sign_start_time) * 1000.0, 1) if self.sign_start_time > 0 else 0.0

        return {
            "state": self.state.value,
            "is_new_commit": is_new_commit,
            "committed_gloss": committed_gloss,
            "confidence": round(commit_confidence, 3),
            "top3": top_candidates,
            "is_uncertain": is_uncertain,
            "motion_energy": round(motion, 4),
            "buffer_fill": len(self.buffer),
            "buffer_target": self.window_size,
            "sign_duration_ms": duration_ms
        }

    def reset(self):
        """Forces immediate reset back to IDLE state."""
        self.state = TemporalState.IDLE
        self.buffer.clear()
        self.prev_feature = None
        self.consecutive_neutral = 0
        self.cooldown_counter = 0
        self.last_committed_gloss = None
