"""
Strict validator for the word-level ISL dataset manifest.

Checks every manifest entry against the files on disk: landmark .npz exists, loads,
has shape (30, 126), is finite, matches the entry's gloss/signer and checksum, has
enough hand frames; optional .webm exists with matching checksum; consent recorded;
gloss in vocabulary; split matches the signer-independent rule; no duplicate ids;
no signer appears in more than one split. Also reports orphan files on disk.

Usage: python scripts/validate_word_dataset.py [--manifest PATH] [--json]
Exit code 0 = valid (may be empty), 1 = errors found, 2 = manifest unreadable.
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.collector import (  # noqa: E402
    MANIFEST_PATH, VOCAB_PATH, SEQ_LEN, FEATURE_DIM, MIN_HAND_FRAMES, SIGNER_ID_RE,
    assign_split, load_vocabulary, resolve_manifest_path, _sha256,
)


def validate(manifest_path: str = MANIFEST_PATH, vocab_path: str = VOCAB_PATH) -> dict:
    errors, warnings = [], []
    if not os.path.exists(manifest_path):
        return {"ok": True, "empty": True, "errors": [], "warnings": [f"no manifest at {manifest_path} (0 clips)"],
                "total_clips": 0, "valid_clips": 0, "per_gloss": {}, "per_signer": {}, "per_split": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    clips = manifest.get("clips", [])
    vocab = {v["gloss"] for v in load_vocabulary(vocab_path)}
    if not vocab:
        errors.append(f"vocabulary missing or empty at {vocab_path}")
    if manifest.get("total_clips", len(clips)) != len(clips):
        errors.append(f"total_clips={manifest.get('total_clips')} but {len(clips)} entries listed")

    ids = Counter(c.get("clip_id") for c in clips)
    for cid, n in ids.items():
        if n > 1:
            errors.append(f"duplicate clip_id {cid} ({n}x)")

    referenced = set()
    split_signers = defaultdict(set)
    per_gloss, per_signer, per_split = Counter(), Counter(), Counter()
    valid = 0

    for c in clips:
        cid = c.get("clip_id", "<missing id>")
        errs = []
        gloss, sid, split = c.get("gloss"), c.get("signer_id", ""), c.get("split")
        if vocab and gloss not in vocab:
            errs.append(f"gloss '{gloss}' not in vocabulary")
        if not SIGNER_ID_RE.match(sid or ""):
            errs.append(f"invalid signer_id '{sid}'")
        elif split != assign_split(sid):
            errs.append(f"split '{split}' != expected '{assign_split(sid)}' for signer {sid}")
        consent = c.get("consent") or {}
        if consent.get("given") is not True:
            errs.append("no recorded consent")

        lp = c.get("landmarks_file")
        if not lp:
            errs.append("no landmarks_file")
        else:
            ap = resolve_manifest_path(lp)
            referenced.add(os.path.normcase(os.path.abspath(ap)))
            if not os.path.exists(ap):
                errs.append(f"landmarks file missing: {lp}")
            else:
                try:
                    with np.load(ap, allow_pickle=False) as z:
                        feats = z["features"]
                        if feats.shape != (SEQ_LEN, FEATURE_DIM):
                            errs.append(f"features shape {feats.shape} != ({SEQ_LEN}, {FEATURE_DIM})")
                        elif not np.isfinite(feats).all():
                            errs.append("features contain NaN/inf")
                        else:
                            hf = int(np.any(feats != 0, axis=1).sum())
                            if hf < MIN_HAND_FRAMES:
                                errs.append(f"only {hf}/{SEQ_LEN} frames with hands")
                        if "gloss" in z.files and str(z["gloss"]) != gloss:
                            errs.append(f"npz gloss '{z['gloss']}' != manifest '{gloss}'")
                        if "signer_id" in z.files and str(z["signer_id"]) != sid:
                            errs.append(f"npz signer '{z['signer_id']}' != manifest '{sid}'")
                    if c.get("landmarks_sha256") and _sha256(ap) != c["landmarks_sha256"]:
                        errs.append("landmarks checksum mismatch")
                    elif not c.get("landmarks_sha256"):
                        warnings.append(f"{cid}: no landmarks_sha256 (legacy entry)")
                except Exception as e:
                    errs.append(f"landmarks unreadable: {e}")

        vp = c.get("clip_file")
        if vp:
            ap = resolve_manifest_path(vp)
            referenced.add(os.path.normcase(os.path.abspath(ap)))
            if not os.path.exists(ap):
                errs.append(f"video file missing: {vp}")
            elif c.get("clip_sha256") and _sha256(ap) != c["clip_sha256"]:
                errs.append("video checksum mismatch")
            if not consent.get("video_consent"):
                errs.append("video saved without video consent")

        if errs:
            errors.extend(f"{cid}: {e}" for e in errs)
        else:
            valid += 1
            per_gloss[gloss] += 1
            per_signer[sid] += 1
            per_split[split] += 1
            split_signers[split].add(sid)

    splits = list(split_signers)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = split_signers[a] & split_signers[b]
            if overlap:
                errors.append(f"signer leakage between {a} and {b}: {sorted(overlap)}")

    ds_dir = os.path.dirname(os.path.abspath(manifest_path))
    for sub, ext in (("landmarks", ".npz"), ("clips", ".webm")):
        d = os.path.join(ds_dir, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                p = os.path.normcase(os.path.abspath(os.path.join(d, fn)))
                if fn.endswith(".part"):
                    warnings.append(f"leftover partial write: {sub}/{fn}")
                elif fn.endswith(ext) and p not in referenced:
                    warnings.append(f"orphan file not in manifest: {sub}/{fn}")

    if vocab:
        missing = sorted(vocab - set(per_gloss))
        if missing:
            warnings.append(f"glosses with 0 valid clips: {missing}")

    return {"ok": not errors, "empty": len(clips) == 0, "errors": errors, "warnings": warnings,
            "total_clips": len(clips), "valid_clips": valid, "per_gloss": dict(per_gloss),
            "per_signer": dict(per_signer), "per_split": dict(per_split),
            "signers_per_split": {k: sorted(v) for k, v in split_signers.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--json", action="store_true", help="print machine-readable report")
    args = ap.parse_args()
    try:
        r = validate(args.manifest, args.vocab)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[FAIL] cannot read manifest: {e}")
        return 2
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    print("=" * 60)
    print(" ISL WORD DATASET VALIDATION")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Clips: {r['valid_clips']} valid / {r['total_clips']} listed")
    print(f"Per split: {r['per_split']}")
    print(f"Signers per split: {r.get('signers_per_split', {})}")
    print(f"Per gloss: {r['per_gloss']}")
    for w in r["warnings"]:
        print(f"[WARN] {w}")
    for e in r["errors"]:
        print(f"[FAIL] {e}")
    if r["empty"]:
        print("\n[RESULT] 0 real clips collected. Word recognition has no real training data yet.")
    print(f"\n[RESULT] {'PASS' if r['ok'] else 'FAIL'}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
