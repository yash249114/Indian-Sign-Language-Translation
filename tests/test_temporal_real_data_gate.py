"""Tests for real-data gating, signer splits and checkpoint status of temporal word models."""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.temporal_data import (
    InsufficientRealDataError, build_real_dataset, load_real_clips, checkpoint_status,
    STATUS_REFERENCE, STATUS_REAL_UNVALIDATED,
)

CLASSES = ["HELLO", "WATER"]


def _clip(tmp_path, signer, gloss, rep, feat=None, write=True):
    rel = None
    if write:
        p = tmp_path / f"{signer}_{gloss}_{rep}.npz"
        if feat is None:
            rng = np.random.default_rng(hash((signer, gloss, rep)) % 2**32)
            feat = rng.normal(CLASSES.index(gloss), 0.1, (30, 126)).astype(np.float32)
        np.savez_compressed(p, features=feat)
        rel = str(p)
    return {"clip_id": f"{signer}_{gloss}_{rep}", "gloss": gloss, "signer_id": signer, "landmarks_file": rel}


def _manifest(tmp_path, signers=("s1", "s2", "s3", "s4"), reps=5):
    return [_clip(tmp_path, s, g, r) for s in signers for g in CLASSES for r in range(reps)]


def test_metadata_only_entries_are_not_counted(tmp_path):
    clips = [_clip(tmp_path, f"s{i}", g, r, write=False) for i in range(4) for g in CLASSES for r in range(10)]
    loaded = load_real_clips(clips, CLASSES)
    assert loaded["samples"] == []
    assert loaded["skipped"] == {"no_landmarks_file": len(clips)}
    with pytest.raises(InsufficientRealDataError, match="distinct signers"):
        build_real_dataset(clips, CLASSES)


def test_invalid_landmark_files_are_skipped(tmp_path):
    clips = [
        _clip(tmp_path, "s1", "HELLO", 0, feat=np.zeros((30, 63), np.float32)),
        _clip(tmp_path, "s1", "HELLO", 1, feat=np.full((30, 126), np.nan, np.float32)),
        _clip(tmp_path, "s1", "HELLO", 2, feat=np.ones((3, 126), np.float32)),
        {"gloss": "HELLO", "signer_id": "s1", "landmarks_file": str(tmp_path / "missing.npz")},
        _clip(tmp_path, "s1", "HELLO", 3, feat=np.ones((45, 126), np.float32)),
    ]
    loaded = load_real_clips(clips, CLASSES)
    assert len(loaded["samples"]) == 1
    assert loaded["samples"][0]["x"].shape == (30, 126)
    assert set(loaded["skipped"]) == {"bad_feature_shape", "non_finite_values", "too_few_frames", "landmarks_file_missing"}


def test_split_is_signer_independent(tmp_path):
    ds = build_real_dataset(_manifest(tmp_path), CLASSES)
    sp = ds["signer_splits"]
    assert sp == {"train": ["s1", "s2"], "val": ["s3"], "test": ["s4"]}
    assert not (set(sp["train"]) & set(sp["test"])) and not (set(sp["train"]) & set(sp["val"]))
    assert len(ds["X_train"]) == 20 and len(ds["X_val"]) == 10 and len(ds["X_test"]) == 10


def test_explicit_test_signer(tmp_path):
    ds = build_real_dataset(_manifest(tmp_path), CLASSES, test_signers=["s1"])
    assert ds["signer_splits"]["test"] == ["s1"]
    assert "s1" not in ds["signer_splits"]["train"]


def test_insufficient_clips_per_class_fails(tmp_path):
    with pytest.raises(InsufficientRealDataError, match="below 5 train clips"):
        build_real_dataset(_manifest(tmp_path, reps=2), CLASSES)


def test_checkpoint_status_defaults_to_reference():
    assert checkpoint_status({})["status"] == STATUS_REFERENCE
    assert checkpoint_status({"is_real_human_dataset": False})["validated"] is False
    assert checkpoint_status({"is_real_human_dataset": True})["status"] == STATUS_REAL_UNVALIDATED


def test_training_refuses_without_real_data(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.train_temporal_model import train_temporal_model
    with pytest.raises(InsufficientRealDataError):
        train_temporal_model(epochs=1, manifest_clips=[], out_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.pt"))


def test_synthetic_training_is_labelled(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.train_temporal_model import train_temporal_model
    meta = train_temporal_model(epochs=1, manifest_clips=[], out_dir=str(tmp_path), allow_synthetic=True)
    assert meta["checkpoint_path"].endswith("word_temporal_bigru_reference.pt")
    ckpt = torch.load(meta["checkpoint_path"], map_location="cpu")
    m = ckpt["metadata"]
    assert m["is_real_human_dataset"] is False
    assert m["data_source"] == "synthetic_reference"
    assert m["model_status"] == STATUS_REFERENCE
    assert m["feature_dim"] == 126 and m["sequence_length"] == 30 and len(m["classes"]) == m["num_classes"]


def test_real_training_records_signers(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import scripts.train_temporal_model as ttm
    monkeypatch.setattr(ttm, "load_vocabulary", lambda: [{"gloss": g} for g in CLASSES])
    meta = ttm.train_temporal_model(epochs=1, manifest_clips=_manifest(tmp_path), out_dir=str(tmp_path / "out"))
    assert meta["is_real_human_dataset"] is True
    assert meta["signer_splits"]["test"] == ["s4"]
    assert meta["model_status"] == STATUS_REAL_UNVALIDATED
    assert meta["checkpoint_path"].endswith("word_temporal_bigru.pt")


def test_predictor_reports_reference_status_for_legacy_checkpoint():
    pytest.importorskip("torch")
    from vision.temporal.word_predictor import WordSignPredictor
    st = WordSignPredictor(architecture="tcn").model_status()
    assert st["status"] in (STATUS_REFERENCE, "not_loaded")
    assert st["validated"] is False
