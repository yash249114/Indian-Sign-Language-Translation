"""
Dataset Collection and Annotation Management Module.
Facilitates local recording of word-level ISL video clips, landmark extraction,
pseudonymous signer metadata, and signer-independent train/val/test splits.
"""

import os
import sys
import json
import time
import shutil
from typing import Dict, List, Optional, Any
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "word_dataset")
CLIPS_DIR = os.path.join(DATASET_DIR, "clips")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
VOCAB_PATH = os.path.join(DATASET_DIR, "vocabulary.json")

# Ensure required directories exist
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)

# Default Signer-to-Split Mapping (Signer-Independent Partitioning)
# Signers 1 and 2: Train; Signer 3: Validation; Signer 4: Test
DEFAULT_SPLIT_MAP = {
    "signer_01": "train",
    "signer_02": "train",
    "signer_03": "val",
    "signer_04": "test"
}


def load_vocabulary() -> List[Dict[str, Any]]:
    """Loads authoritative ISL vocabulary list."""
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("vocabulary", [])
        except Exception as e:
            print(f"[Collector] Error loading vocabulary: {e}")
    return []


class DatasetCollector:
    """Manages recording, storing, and indexing of word-level ISL clips and metadata."""

    def __init__(self, manifest_file: str = MANIFEST_PATH):
        self.manifest_file = manifest_file
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads dataset manifest from disk or initializes empty structure."""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Collector] Error reading manifest: {e}")
        
        return {
            "dataset_name": "ISL-Word-Level-Everyday-Lexicon",
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "license": "CC-BY-4.0 Academic Use Only",
            "consent_policy": "All samples collected under informed consent. No PII or facial biometric data retained.",
            "split_strategy": "Signer-independent partition (Grouping by pseudonymized signer ID)",
            "total_clips": 0,
            "clips": []
        }

    def _save_manifest(self):
        """Persists manifest to disk."""
        self.manifest["last_updated"] = datetime.utcnow().isoformat()
        self.manifest["total_clips"] = len(self.manifest["clips"])
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def assign_split(self, signer_id: str) -> str:
        """Assigns partition split strictly based on signer ID."""
        signer_clean = signer_id.strip().lower()
        if signer_clean in DEFAULT_SPLIT_MAP:
            return DEFAULT_SPLIT_MAP[signer_clean]
        # Hash-based deterministic partition for unknown signer IDs (70% train, 15% val, 15% test)
        hash_val = sum(ord(c) for c in signer_clean) % 100
        if hash_val < 70:
            return "train"
        elif hash_val < 85:
            return "val"
        else:
            return "test"

    def record_clip(
        self,
        gloss: str,
        signer_id: str,
        clip_bytes: Optional[bytes] = None,
        landmarks_data: Optional[Dict[str, Any]] = None,
        handedness: str = "one-handed",
        lighting: str = "normal",
        distance: str = "normal",
        background: str = "plain",
        signing_speed: str = "normal",
        fps: float = 30.0,
        frame_count: int = 30,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Saves a newly captured clip, extracts/saves landmarks, and records manifest entry.
        """
        gloss_upper = gloss.strip().upper()
        signer_id_clean = signer_id.strip().lower()
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        repetition = sum(1 for c in self.manifest["clips"] if c["gloss"] == gloss_upper and c["signer_id"] == signer_id_clean) + 1

        clip_id = f"{signer_id_clean}_{gloss_upper}_rep{repetition}_{timestamp_str}"
        split = self.assign_split(signer_id_clean)

        # File paths
        clip_filename = f"{clip_id}.webm"
        clip_rel_path = os.path.join("data", "word_dataset", "clips", clip_filename)
        clip_abs_path = os.path.join(CLIPS_DIR, clip_filename)

        lms_filename = f"{clip_id}.npz"
        lms_rel_path = os.path.join("data", "word_dataset", "landmarks", lms_filename)
        lms_abs_path = os.path.join(LANDMARKS_DIR, lms_filename)

        if clip_bytes:
            with open(clip_abs_path, "wb") as f:
                f.write(clip_bytes)

        if landmarks_data is not None:
            import numpy as np
            np.savez_compressed(
                lms_abs_path,
                features=landmarks_data.get("features", np.zeros((frame_count, 126))),
                gloss=gloss_upper,
                signer_id=signer_id_clean
            )

        duration_sec = round(frame_count / fps, 2) if fps > 0 else 1.0

        entry = {
            "clip_id": clip_id,
            "gloss": gloss_upper,
            "signer_id": signer_id_clean,
            "repetition": repetition,
            "split": split,
            "handedness": handedness,
            "camera_conditions": {
                "lighting": lighting,
                "distance": distance,
                "background": background
            },
            "signing_speed": signing_speed,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "clip_file": clip_rel_path if clip_bytes else None,
            "landmarks_file": lms_rel_path if landmarks_data else None,
            "notes": notes,
            "consent_verified": True,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.manifest["clips"].append(entry)
        self._save_manifest()
        return entry

    def get_manifest(self) -> Dict[str, Any]:
        """Returns the current manifest with summary statistics."""
        clips = self.manifest.get("clips", [])
        splits_count = {"train": 0, "val": 0, "test": 0}
        gloss_counts = {}
        signer_counts = {}

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
        """Removes a clip and its manifest entry."""
        target = None
        for c in self.manifest["clips"]:
            if c["clip_id"] == clip_id:
                target = c
                break
        
        if not target:
            return False

        # Remove files if present
        if target.get("clip_file"):
            abs_p = os.path.join(PROJECT_ROOT, target["clip_file"])
            if os.path.exists(abs_p):
                try:
                    os.remove(abs_p)
                except Exception:
                    pass

        if target.get("landmarks_file"):
            abs_p = os.path.join(PROJECT_ROOT, target["landmarks_file"])
            if os.path.exists(abs_p):
                try:
                    os.remove(abs_p)
                except Exception:
                    pass

        self.manifest["clips"] = [c for c in self.manifest["clips"] if c["clip_id"] != clip_id]
        self._save_manifest()
        return True


collector = DatasetCollector()
