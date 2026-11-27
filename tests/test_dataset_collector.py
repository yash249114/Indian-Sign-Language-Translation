"""Tests for the word-level dataset collector persistence path and manifest validator."""

import json
import os

import numpy as np
import pytest

from backend.app.collector import DatasetCollector, ClipValidationError, SEQ_LEN, FEATURE_DIM
from scripts.validate_word_dataset import validate


@pytest.fixture
def col(tmp_path):
    vocab = tmp_path / "vocabulary.json"
    vocab.write_text(json.dumps({"vocabulary": [{"gloss": "HELLO"}, {"gloss": "WATER"}]}))
    return DatasetCollector(manifest_file=str(tmp_path / "manifest.json"), vocab_path=str(vocab))


def feats(hand_frames=SEQ_LEN):
    rng = np.random.default_rng(0)
    f = rng.normal(size=(SEQ_LEN, FEATURE_DIM)).astype(np.float32)
    f[hand_frames:] = 0
    return f


def test_saves_verified_npz_and_manifest(col):
    f = feats()
    e = col.record_clip("hello", "signer_01", f, consent=True, fps=15, frame_timestamps_ms=list(range(SEQ_LEN)))
    path = e["landmarks_file"]
    path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), path)
    with np.load(path) as z:
        assert z["features"].shape == (SEQ_LEN, FEATURE_DIM)
        assert np.array_equal(z["features"], f)
        assert str(z["gloss"]) == "HELLO"
    assert e["split"] == "train" and e["consent"]["given"] is True and e["clip_file"] is None
    on_disk = json.load(open(col.manifest_file))
    assert on_disk["total_clips"] == 1 and on_disk["clips"][0]["clip_id"] == e["clip_id"]


@pytest.mark.parametrize("kwargs, msg", [
    (dict(consent=False), "consent"),
    (dict(gloss="UNKNOWN_SIGN"), "unknown gloss"),
    (dict(signer_id="../etc"), "signer_id"),
    (dict(signer_id="Jane Doe"), "signer_id"),
    (dict(features=np.zeros((29, FEATURE_DIM))), "shape"),
    (dict(features=np.zeros((SEQ_LEN, 63))), "shape"),
    (dict(features=feats(hand_frames=5)), "hands detected"),
    (dict(lighting="disco"), "lighting"),
    (dict(clip_bytes=b"\x1a\x45\xdf\xa3data"), "video consent"),
    (dict(clip_bytes=b"notwebm", video_consent=True), "WebM"),
])
def test_rejects_invalid_and_writes_nothing(col, kwargs, msg):
    args = dict(gloss="HELLO", signer_id="signer_01", features=feats(), consent=True)
    args.update(kwargs)
    with pytest.raises(ClipValidationError, match=msg):
        col.record_clip(**args)
    assert not os.listdir(col.landmarks_dir) and not os.listdir(col.clips_dir)
    assert not os.path.exists(col.manifest_file)


def test_nan_rejected(col):
    f = feats()
    f[3, 4] = np.nan
    with pytest.raises(ClipValidationError, match="NaN"):
        col.record_clip("HELLO", "signer_01", f, consent=True)


def test_saves_webm_with_video_consent(col):
    vid = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
    e = col.record_clip("WATER", "signer_03", feats(), consent=True, clip_bytes=vid, video_consent=True)
    assert e["clip_file"] and e["consent"]["video_consent"] is True and e["split"] == "val"
    assert len(os.listdir(col.clips_dir)) == 1


def test_validator_pass_and_detects_tampering(col, tmp_path):
    col.record_clip("HELLO", "signer_01", feats(), consent=True)
    col.record_clip("WATER", "signer_04", feats(), consent=True)
    vocab = str(tmp_path / "vocabulary.json")
    r = validate(col.manifest_file, vocab)
    assert r["ok"], r["errors"]
    assert r["valid_clips"] == 2 and r["per_split"] == {"train": 1, "test": 1}

    # corrupt one npz, delete nothing else
    m = json.load(open(col.manifest_file))
    p = m["clips"][0]["landmarks_file"]
    p = p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), p)
    np.savez_compressed(p, features=np.zeros((10, FEATURE_DIM), dtype=np.float32))
    open(os.path.join(col.landmarks_dir, "stray.npz"), "wb").write(b"x")
    r = validate(col.manifest_file, vocab)
    assert not r["ok"]
    assert any("shape" in e for e in r["errors"])
    assert any("orphan" in w for w in r["warnings"])


def test_validator_flags_missing_consent_and_leakage(col, tmp_path):
    col.record_clip("HELLO", "signer_01", feats(), consent=True)
    m = json.load(open(col.manifest_file))
    m["clips"][0]["consent"] = None
    m["clips"][0]["split"] = "test"
    json.dump(m, open(col.manifest_file, "w"))
    r = validate(col.manifest_file, str(tmp_path / "vocabulary.json"))
    assert any("consent" in e for e in r["errors"])
    assert any("split" in e for e in r["errors"])


def test_validator_empty_dataset(tmp_path):
    r = validate(str(tmp_path / "none.json"), str(tmp_path / "v.json"))
    assert r["ok"] and r["empty"] and r["total_clips"] == 0
