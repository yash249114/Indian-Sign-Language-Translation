"""
Dataset Collection and Annotation Management Module.
Persists word-level ISL samples: a verified 30x126 MediaPipe landmark sequence (.npz),
an optional consented .webm clip, pseudonymous signer metadata, and a
signer-independent train/val/test split. A sample is only added to the manifest
after its files have been written and read back successfully.
"""

import os
import re
import json
import uuid
import hashlib
import threading
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "word_dataset")
CLIPS_DIR = os.path.join(DATASET_DIR, "clips")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
VOCAB_PATH = os.path.join(DATASET_DIR, "vocabulary.json")

SEQ_LEN = 30
FEATURE_DIM = 126  # 63 left-hand + 63 right-hand (21 landmarks x xyz each)
FEATURE_SCHEMA = "mediapipe_hands_v1:left21xyz+right21xyz"
MIN_HAND_FRAMES = 20  # frames (of 30) that must contain at least one detected hand
MAX_VIDEO_BYTES = 15 * 1024 * 1024
WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
CONSENT_VERSION = "isl-collector-consent-v1"
SIGNER_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")

ALLOWED_METADATA = {
    "handedness": {"one-handed", "two-handed"},
    "lighting": {"normal", "bright", "dim"},
    "distance": {"near", "normal", "far"},
    "background": {"plain", "cluttered"},
    "signing_speed": {"slow", "normal", "fast"},
}

# Default Signer-to-Split Mapping (Signer-Independent Partitioning)
# Signers 1 and 2: Train; Signer 3: Validation; Signer 4: Test
DEFAULT_SPLIT_MAP = {
    "signer_01": "train",
    "signer_02": "train",
    "signer_03": "val",
    "signer_04": "test"
}


class ClipValidationError(ValueError):
    """Raised when a submitted sample is rejected before anything is persisted."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_manifest_path(abs_path: str) -> str:
    """Project-relative POSIX path when possible, else absolute (e.g. temp dirs on another drive)."""
    try:
        rel = os.path.relpath(abs_path, PROJECT_ROOT)
    except ValueError:
        return abs_path
    return abs_path if rel.startswith("..") else rel.replace(os.sep, "/")


def resolve_manifest_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, *p.split("/"))


def load_vocabulary(vocab_path: str = VOCAB_PATH) -> List[Dict[str, Any]]:
    """Loads authoritative ISL vocabulary list."""
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f).get("vocabulary", [])
        except Exception as e:
            print(f"[Collector] Error loading vocabulary: {e}")
    return []


def assign_split(signer_id: str) -> str:
    """Assigns partition split strictly based on signer ID."""
    signer_clean = signer_id.strip().lower()
    if signer_clean in DEFAULT_SPLIT_MAP:
        return DEFAULT_SPLIT_MAP[signer_clean]
    # Deterministic partition for unknown signer IDs (70% train, 15% val, 15% test)
    hash_val = sum(ord(c) for c in signer_clean) % 100
    if hash_val < 70:
        return "train"
    elif hash_val < 85:
        return "val"
    return "test"


def validate_features(features: Any) -> np.ndarray:
    """Returns a float32 (SEQ_LEN, FEATURE_DIM) array or raises ClipValidationError."""
    try:
        arr = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError):
        raise ClipValidationError("landmark features must be a numeric 2-D array")
    if arr.shape != (SEQ_LEN, FEATURE_DIM):
        raise ClipValidationError(f"landmark features must have shape ({SEQ_LEN}, {FEATURE_DIM}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ClipValidationError("landmark features contain NaN or infinite values")
    hand_frames = int(np.any(arr != 0, axis=1).sum())
    if hand_frames < MIN_HAND_FRAMES:
        raise ClipValidationError(
            f"hands detected in only {hand_frames}/{SEQ_LEN} frames (need >= {MIN_HAND_FRAMES}); re-record with hands in view"
        )
    return arr


class DatasetCollector:
    """Manages recording, storing, and indexing of word-level ISL samples."""

    def __init__(self, manifest_file: str = MANIFEST_PATH, dataset_dir: Optional[str] = None,
                 vocab_path: str = VOCAB_PATH):
        self.manifest_file = manifest_file
        self.dataset_dir = dataset_dir or os.path.dirname(os.path.abspath(manifest_file))
        self.clips_dir = os.path.join(self.dataset_dir, "clips")
        self.landmarks_dir = os.path.join(self.dataset_dir, "landmarks")
        self.vocab_path = vocab_path
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads dataset manifest from disk or initializes empty structure."""
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)  # fail loudly: never silently overwrite a corrupt manifest
        return {
            "dataset_name": "ISL-Word-Level-Everyday-Lexicon",
            "version": "1.1",
            "last_updated": _utcnow(),
            "license": "CC-BY-4.0 Academic Use Only",
            "consent_policy": (
                "Every sample requires explicit signer consent recorded per clip. Landmark sequences "
                "contain hand keypoints only. Video (.webm, shows the face) is saved only with separate video consent."
            ),
            "split_strategy": "Signer-independent partition (Grouping by pseudonymized signer ID)",
            "feature_schema": FEATURE_SCHEMA,
            "sequence_length": SEQ_LEN,
            "feature_dim": FEATURE_DIM,
            "total_clips": 0,
            "clips": []
        }

    def _save_manifest(self):
        """Persists manifest atomically."""
        self.manifest["last_updated"] = _utcnow()
        self.manifest["total_clips"] = len(self.manifest["clips"])
        tmp = f"{self.manifest_file}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.manifest_file)

    def assign_split(self, signer_id: str) -> str:
        return assign_split(signer_id)

    def allowed_glosses(self) -> List[str]:
        return [v["gloss"] for v in load_vocabulary(self.vocab_path) if "gloss" in v]

    def record_clip(
        self,
        gloss: str,
        signer_id: str,
        features: Any,
        consent: bool,
        clip_bytes: Optional[bytes] = None,
        video_consent: bool = False,
        handedness: str = "one-handed",
        lighting: str = "normal",
        distance: str = "normal",
        background: str = "plain",
        signing_speed: str = "normal",
        fps: float = 15.0,
        frame_timestamps_ms: Optional[Sequence[float]] = None,
        source: str = "unknown",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Validates a sample, writes .npz (and optional .webm), reads them back to verify,
        then appends the manifest entry. Raises ClipValidationError on bad input; on any
        failure no partial files or manifest entries are left behind.
        """
        # ---- validation (nothing touches disk before this passes) ----
        if consent is not True:
            raise ClipValidationError("explicit signer consent is required")
        gloss_upper = (gloss or "").strip().upper()
        allowed = self.allowed_glosses()
        if not allowed:
            raise ClipValidationError(f"vocabulary missing or empty at {self.vocab_path}")
        if gloss_upper not in allowed:
            raise ClipValidationError(f"unknown gloss '{gloss_upper}'")
        signer_clean = (signer_id or "").strip().lower()
        if not SIGNER_ID_RE.match(signer_clean):
            raise ClipValidationError("signer_id must be 1-32 chars of a-z, 0-9, '_' or '-' (pseudonymous, no real names)")
        meta = {"handedness": handedness, "lighting": lighting, "distance": distance,
                "background": background, "signing_speed": signing_speed}
        for k, v in meta.items():
            if v not in ALLOWED_METADATA[k]:
                raise ClipValidationError(f"{k} must be one of {sorted(ALLOWED_METADATA[k])}")
        feats = validate_features(features)
        hand_mask = np.any(feats != 0, axis=1)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            raise ClipValidationError("fps must be a number")
        if not (1.0 <= fps <= 120.0):
            raise ClipValidationError("fps must be between 1 and 120")
        ts = None
        if frame_timestamps_ms is not None:
            ts = np.asarray(frame_timestamps_ms, dtype=np.float64)
            if ts.shape != (SEQ_LEN,) or not np.isfinite(ts).all() or np.any(np.diff(ts) < 0):
                raise ClipValidationError(f"frame_timestamps_ms must be {SEQ_LEN} non-decreasing numbers")
        if clip_bytes is not None:
            if not video_consent:
                raise ClipValidationError("video consent is required to save the video clip")
            if len(clip_bytes) == 0 or len(clip_bytes) > MAX_VIDEO_BYTES:
                raise ClipValidationError(f"video must be 1 byte to {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
            if not clip_bytes.startswith(WEBM_MAGIC):
                raise ClipValidationError("video is not a WebM file")
        notes = (notes or "")[:500]

        with self._lock:
            repetition = sum(1 for c in self.manifest["clips"]
                             if c["gloss"] == gloss_upper and c["signer_id"] == signer_clean) + 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            clip_id = f"{signer_clean}_{gloss_upper}_rep{repetition:03d}_{stamp}_{uuid.uuid4().hex[:6]}"
            lms_abs = os.path.join(self.landmarks_dir, f"{clip_id}.npz")
            clip_abs = os.path.join(self.clips_dir, f"{clip_id}.webm") if clip_bytes is not None else None
            written: List[str] = []
            try:
                tmp = lms_abs + ".part"
                with open(tmp, "wb") as f:
                    np.savez_compressed(
                        f, features=feats, hand_mask=hand_mask,
                        frame_timestamps_ms=ts if ts is not None else np.array([], dtype=np.float64),
                        gloss=np.array(gloss_upper), signer_id=np.array(signer_clean),
                        fps=np.array(fps), feature_schema=np.array(FEATURE_SCHEMA))
                os.replace(tmp, lms_abs)
                written.append(lms_abs)

                with np.load(lms_abs, allow_pickle=False) as z:
                    back = z["features"]
                    if back.shape != (SEQ_LEN, FEATURE_DIM) or not np.array_equal(back, feats) \
                            or str(z["gloss"]) != gloss_upper or str(z["signer_id"]) != signer_clean:
                        raise IOError("landmark file verification failed after write")

                if clip_abs:
                    tmp = clip_abs + ".part"
                    with open(tmp, "wb") as f:
                        f.write(clip_bytes)
                    os.replace(tmp, clip_abs)
                    written.append(clip_abs)
                    if os.path.getsize(clip_abs) != len(clip_bytes):
                        raise IOError("video file verification failed after write")

                now = _utcnow()
                entry = {
                    "clip_id": clip_id,
                    "gloss": gloss_upper,
                    "signer_id": signer_clean,
                    "repetition": repetition,
                    "split": assign_split(signer_clean),
                    "handedness": handedness,
                    "camera_conditions": {"lighting": lighting, "distance": distance, "background": background},
                    "signing_speed": signing_speed,
                    "fps": round(fps, 2),
                    "frame_count": SEQ_LEN,
                    "feature_dim": FEATURE_DIM,
                    "hand_frames": int(hand_mask.sum()),
                    "duration_sec": round(SEQ_LEN / fps, 2),
                    "landmarks_file": to_manifest_path(lms_abs),
                    "landmarks_sha256": _sha256(lms_abs),
                    "clip_file": to_manifest_path(clip_abs) if clip_abs else None,
                    "clip_sha256": _sha256(clip_abs) if clip_abs else None,
                    "consent": {"given": True, "version": CONSENT_VERSION, "timestamp": now,
                                "video_consent": bool(clip_abs)},
                    "source": source,
                    "data_origin": "real_human",
                    "notes": notes,
                    "timestamp": now
                }
                self.manifest["clips"].append(entry)
                try:
                    self._save_manifest()
                except Exception:
                    self.manifest["clips"].pop()
                    raise
                return entry
            except Exception:
                for p in written + [lms_abs + ".part"] + ([clip_abs + ".part"] if clip_abs else []):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                raise

    def get_manifest(self) -> Dict[str, Any]:
        """Returns the current manifest with summary statistics."""
        clips = self.manifest.get("clips", [])
        splits_count = {"train": 0, "val": 0, "test": 0}
        gloss_counts: Dict[str, int] = {}
        signer_counts: Dict[str, int] = {}

        for c in clips:
            s = c.get("split", "train")
            splits_count[s] = splits_count.get(s, 0) + 1
            g = c.get("gloss", "UNKNOWN")
            gloss_counts[g] = gloss_counts.get(g, 0) + 1
            sid = c.get("signer_id", "UNKNOWN")
            signer_counts[sid] = signer_counts.get(sid, 0) + 1

        return {
            "metadata": {
                "dataset_name": self.manifest.get("dataset_name"),
                "version": self.manifest.get("version"),
                "license": self.manifest.get("license"),
                "split_strategy": self.manifest.get("split_strategy"),
                "last_updated": self.manifest.get("last_updated")
            },
            "total_clips": len(clips),
            "splits_distribution": splits_count,
            "class_distribution": gloss_counts,
            "signers": list(signer_counts.keys()),
            "clips": clips
        }

    def delete_clip(self, clip_id: str) -> bool:
        """Removes a clip, its files, and its manifest entry."""
        with self._lock:
            target = next((c for c in self.manifest["clips"] if c["clip_id"] == clip_id), None)
            if not target:
                return False
            for key in ("clip_file", "landmarks_file"):
                if target.get(key):
                    p = resolve_manifest_path(target[key])
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            self.manifest["clips"] = [c for c in self.manifest["clips"] if c["clip_id"] != clip_id]
            self._save_manifest()
            return True


collector = DatasetCollector()
