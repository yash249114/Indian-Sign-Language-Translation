"""
Temporal Stabilization & Gesture Commit State Machine.
Implements a strict finite state machine for live sign recognition:
  NO_SIGN -> DETECTING -> STABLE_SIGN -> COMMIT_ONCE -> LOCKED_HOLD -> WAIT_FOR_NEUTRAL -> NO_SIGN

Guarantees:
1. Holding a sign for any duration emits EXACTLY ONE character.
2. Only final stabilized predictions can commit to the text buffer.
3. Returning to neutral (no hand or below threshold) unlocks the next gesture emission.
"""

import time
import collections
from enum import Enum


class RecognitionState(Enum):
    NO_SIGN = "NO_SIGN"
    DETECTING = "DETECTING"
    STABLE_SIGN = "STABLE_SIGN"
    COMMIT_ONCE = "COMMIT_ONCE"
    LOCKED_HOLD = "LOCKED_HOLD"
    WAIT_FOR_NEUTRAL = "WAIT_FOR_NEUTRAL"


class TemporalStabilizer:
    """
    Stabilizes real-time sign predictions over a sliding temporal window
    and enforces an explicit state machine for single-emission gesture commits.
    """

    def __init__(
        self,
        window_size: int = 6,
        confidence_threshold: float = 0.75,
        min_consensus_ratio: float = 0.60,
        neutral_reset_frames: int = 3,
        auto_repeat_seconds: float = None
    ):
        """
        Args:
            window_size: Number of recent frames to consider for majority vote.
            confidence_threshold: Minimum confidence score to accept a prediction.
            min_consensus_ratio: Ratio of matching frames in window required for consensus.
            neutral_reset_frames: Consecutive neutral (no hand) frames required to reset lock.
            auto_repeat_seconds: Seconds of continuous hold before allowing a repeat (None = disabled).
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_consensus_ratio = min_consensus_ratio
        self.neutral_reset_frames = neutral_reset_frames
        self.auto_repeat_seconds = auto_repeat_seconds

        self.history = collections.deque(maxlen=window_size)
        self.state = RecognitionState.NO_SIGN
        self.locked_sign = None
        self.last_commit_time = 0.0
        self.consecutive_neutral_count = 0
        self.current_stable_sign = None

    def update(self, raw_sign: str, confidence: float) -> dict:
        """
        Processes a single frame's raw prediction through the state machine.

        Args:
            raw_sign: Predicted sign string or None (if no hand / detection failed).
            confidence: Float confidence score [0.0, 1.0].

        Returns:
            dict containing:
                - 'stable_sign': Currently stabilized sign string or None.
                - 'is_new_emission': True ONLY on the single frame where a sign is newly committed.
                - 'state': Current state machine string.
                - 'majority_confidence': Majority vote agreement ratio.
        """
        now = time.time()
        is_valid_prediction = (raw_sign is not None) and (confidence >= self.confidence_threshold)

        # 1. Update temporal history window
        if is_valid_prediction:
            self.history.append(raw_sign)
            self.consecutive_neutral_count = 0
        else:
            self.history.append(None)
            self.consecutive_neutral_count += 1

        # 2. Check for Neutral State Reset (hand removed / below threshold)
        if self.consecutive_neutral_count >= self.neutral_reset_frames:
            self.state = RecognitionState.NO_SIGN
            self.locked_sign = None
            self.current_stable_sign = None
            return {
                "stable_sign": None,
                "is_new_emission": False,
                "state": self.state.value,
                "majority_confidence": 0.0
            }

        # 3. Compute majority vote in sliding window
        valid_items = [s for s in self.history if s is not None]
        if not valid_items:
            self.current_stable_sign = None
            return {
                "stable_sign": None,
                "is_new_emission": False,
                "state": self.state.value,
                "majority_confidence": 0.0
            }

        counts = collections.Counter(valid_items)
        top_sign, top_count = counts.most_common(1)[0]
        consensus_ratio = top_count / self.window_size

        # Require consensus threshold
        if consensus_ratio < self.min_consensus_ratio:
            self.current_stable_sign = None
            self.state = RecognitionState.DETECTING
            return {
                "stable_sign": None,
                "is_new_emission": False,
                "state": self.state.value,
                "majority_confidence": round(consensus_ratio, 2)
            }

        # Consensus reached
        self.current_stable_sign = top_sign

        # 4. State Machine Evaluation
        is_new_emission = False

        # If a distinctly different sign is stabilized, release lock on old sign
        if self.locked_sign is not None and self.current_stable_sign != self.locked_sign:
            self.locked_sign = None
            self.state = RecognitionState.STABLE_SIGN

        if self.state in (RecognitionState.NO_SIGN, RecognitionState.DETECTING):
            self.state = RecognitionState.STABLE_SIGN

        if self.state == RecognitionState.STABLE_SIGN:
            if self.locked_sign != self.current_stable_sign:
                # COMMIT ONCE
                self.state = RecognitionState.COMMIT_ONCE
                self.locked_sign = self.current_stable_sign
                self.last_commit_time = now
                is_new_emission = True
            else:
                self.state = RecognitionState.LOCKED_HOLD

        elif self.state == RecognitionState.COMMIT_ONCE:
            # Immediately transition to LOCKED_HOLD
            self.state = RecognitionState.LOCKED_HOLD
            is_new_emission = False

        elif self.state == RecognitionState.LOCKED_HOLD:
            # Holding the same sign: NEVER emit again unless auto_repeat is explicitly configured
            if self.auto_repeat_seconds and (now - self.last_commit_time >= self.auto_repeat_seconds):
                self.last_commit_time = now
                is_new_emission = True
            else:
                is_new_emission = False

        return {
            "stable_sign": self.current_stable_sign,
            "is_new_emission": is_new_emission,
            "state": self.state.value,
            "majority_confidence": round(consensus_ratio, 2)
        }

    def reset(self):
        """Full reset of stabilizer and state machine."""
        self.history.clear()
        self.state = RecognitionState.NO_SIGN
        self.locked_sign = None
        self.last_commit_time = 0.0
        self.consecutive_neutral_count = 0
        self.current_stable_sign = None
