"""
Real-data loading and signer-independent splitting for temporal word models.

Only manifest entries whose landmark file exists, loads, and has a valid
(T, 126) finite feature array are counted. Splits are made by signer ID so
no signer appears in more than one partition.
"""

import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURE_DIM = 126
SEQUENCE_LENGTH = 30
MIN_FRAMES = 5

MIN_SIGNERS = 3
MIN_TRAIN_CLIPS_PER_CLASS = 5
MIN_TEST_CLIPS_PER_CLASS = 1

DATA_SOURCE_REAL = "real_human"
DATA_SOURCE_SYNTHETIC = "synthetic_reference"

STATUS_REFERENCE = "reference_model_not_validated"
STATUS_REAL_UNVALIDATED = "real_data_model_not_validated"
STATUS_REAL_VALIDATED = "real_data_model_validated_on_held_out_signers"

STATUS_LABELS = {
    STATUS_REFERENCE: "Reference model, not validated (trained on synthetic trajectories)",
    STATUS_REAL_UNVALIDATED: "Trained on real clips, not yet validated on held-out signers",
    STATUS_REAL_VALIDATED: "Trained on real clips, evaluated on held-out signers",
}


class InsufficientRealDataError(RuntimeError):
    """Raised when the manifest does not hold enough valid real clips to train or evaluate."""


def resample_sequence(feat: np.ndarray, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    if feat.shape[0] == seq_len:
        return feat.astype(np.float32)
    idx = np.linspace(0, feat.shape[0] - 1, seq_len).round().astype(int)
    return feat[idx].astype(np.float32)


def _validate_clip(clip: Dict[str, Any], classes: Sequence[str]) -> (Optional[np.ndarray], Optional[str]):
    gloss = clip.get("gloss")
    if gloss not in classes:
        return None, "gloss_not_in_vocabulary"
    if not clip.get("signer_id"):
        return None, "missing_signer_id"
    rel = clip.get("landmarks_file")
    if not rel:
        return None, "no_landmarks_file"
    path = rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)
    if not os.path.isfile(path):
        return None, "landmarks_file_missing"
    try:
        with np.load(path) as data:
            feat = np.asarray(data["features"], dtype=np.float32)
    except Exception:
        return None, "landmarks_file_unreadable"
    if feat.ndim != 2 or feat.shape[1] != FEATURE_DIM:
        return None, "bad_feature_shape"
    if feat.shape[0] < MIN_FRAMES:
        return None, "too_few_frames"
    if not np.all(np.isfinite(feat)):
        return None, "non_finite_values"
    if float(np.abs(feat).sum()) == 0.0:
        return None, "all_zero_features"
    return resample_sequence(feat), None


def load_real_clips(clips: List[Dict[str, Any]], classes: Sequence[str]) -> Dict[str, Any]:
    """Returns valid samples and a count of skipped entries by reason."""
    samples, skipped = [], Counter()
    for c in clips:
        feat, reason = _validate_clip(c, classes)
        if reason:
            skipped[reason] += 1
            continue
        samples.append({
            "x": feat,
            "y": classes.index(c["gloss"]),
            "gloss": c["gloss"],
            "signer_id": str(c["signer_id"]).strip().lower(),
            "clip_id": c.get("clip_id"),
        })
    return {"samples": samples, "skipped": dict(skipped), "manifest_entries": len(clips)}


def assign_signer_splits(
    signers: Sequence[str],
    val_signers: Optional[Sequence[str]] = None,
    test_signers: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """Deterministic signer-level split. Defaults: last sorted signer -> test, second-to-last -> val."""
    signers = sorted(set(signers))
    test = sorted(set(s.lower() for s in test_signers)) if test_signers else signers[-1:]
    rest = [s for s in signers if s not in test]
    val = sorted(set(s.lower() for s in val_signers)) if val_signers else rest[-1:]
    train = [s for s in rest if s not in val]
    unknown = (set(test) | set(val)) - set(signers)
    if unknown:
        raise InsufficientRealDataError(f"Requested signers not in valid data: {sorted(unknown)}")
    if set(test) & set(val):
        raise InsufficientRealDataError(f"Signers in both val and test: {sorted(set(test) & set(val))}")
    return {"train": train, "val": val, "test": test}


def build_real_dataset(
    clips: List[Dict[str, Any]],
    classes: Sequence[str],
    val_signers: Optional[Sequence[str]] = None,
    test_signers: Optional[Sequence[str]] = None,
    min_signers: int = MIN_SIGNERS,
    min_train_per_class: int = MIN_TRAIN_CLIPS_PER_CLASS,
    min_test_per_class: int = MIN_TEST_CLIPS_PER_CLASS,
) -> Dict[str, Any]:
    """Loads valid real clips, splits by signer, and enforces minimum data requirements."""
    loaded = load_real_clips(clips, classes)
    samples = loaded["samples"]
    signers = sorted({s["signer_id"] for s in samples})

    problems = []
    if len(signers) < min_signers:
        problems.append(f"need >= {min_signers} distinct signers with valid clips, found {len(signers)} {signers}")

    split_map, parts = None, None
    if not problems:
        split_map = assign_signer_splits(signers, val_signers, test_signers)
        if not split_map["train"]:
            problems.append("no signers left for training after val/test hold-out")
        sid_to_split = {s: k for k, v in split_map.items() for s in v}
        parts = {"train": [], "val": [], "test": []}
        for s in samples:
            parts[sid_to_split[s["signer_id"]]].append(s)
        train_counts = Counter(s["gloss"] for s in parts["train"])
        test_counts = Counter(s["gloss"] for s in parts["test"])
        low_train = {g: train_counts.get(g, 0) for g in classes if train_counts.get(g, 0) < min_train_per_class}
        low_test = {g: test_counts.get(g, 0) for g in classes if test_counts.get(g, 0) < min_test_per_class}
        if low_train:
            problems.append(f"classes below {min_train_per_class} train clips: {low_train}")
        if low_test:
            problems.append(f"classes below {min_test_per_class} test clips (held-out signers {split_map['test']}): {low_test}")
        if not parts["val"]:
            problems.append(f"no validation clips (val signers {split_map['val']})")

    if problems:
        raise InsufficientRealDataError(
            "Insufficient real human data for temporal training/evaluation:\n  - "
            + "\n  - ".join(problems)
            + f"\n  Manifest entries: {loaded['manifest_entries']}, valid clips: {len(samples)}, "
            + f"skipped: {loaded['skipped'] or 'none'}"
            + "\n  Record clips with landmarks via the Dataset Collector, then re-run."
        )

    def stack(part):
        if not part:
            return np.zeros((0, SEQUENCE_LENGTH, FEATURE_DIM), np.float32), np.zeros((0,), np.int64)
        return np.stack([s["x"] for s in part]), np.array([s["y"] for s in part], dtype=np.int64)

    out = {"signer_splits": split_map, "skipped": loaded["skipped"],
           "manifest_entries": loaded["manifest_entries"], "valid_clips": len(samples)}
    for k in ("train", "val", "test"):
        out[f"X_{k}"], out[f"y_{k}"] = stack(parts[k])
    return out


def checkpoint_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Derives the user-facing status from checkpoint metadata. Missing metadata = reference model."""
    is_real = bool(meta.get("is_real_human_dataset", False))
    if not is_real:
        code = STATUS_REFERENCE
    elif meta.get("validated_on_held_out_signers"):
        code = STATUS_REAL_VALIDATED
    else:
        code = STATUS_REAL_UNVALIDATED
    return {
        "status": code,
        "label": STATUS_LABELS[code],
        "is_real_human_dataset": is_real,
        "data_source": meta.get("data_source", DATA_SOURCE_REAL if is_real else DATA_SOURCE_SYNTHETIC),
        "validated": code == STATUS_REAL_VALIDATED,
    }
