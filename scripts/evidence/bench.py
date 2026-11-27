"""Fresh end-to-end evaluation on raw dataset images (sampled), plus latency stats and demo-frame picks."""
import os, sys, json, time, random
import numpy as np, cv2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from vision.inference.sign_detector import SignDetector
OUT = sys.argv[1]
random.seed(7)
det = SignDetector()
data = os.path.join(ROOT, "archive (5)", "data")
classes = sorted(os.listdir(data))
PER = 20
rows, mp_t, mdl_t, e2e_t = [], [], [], []
best = {}
for c in classes:
    files = sorted(os.listdir(os.path.join(data, c)))
    pick = random.sample(files, PER)
    for f in pick:
        img = cv2.imread(os.path.join(data, c, f))
        # place the 128x128 sample, upscaled, on a 640x480 canvas (webcam-like frame)
        big = cv2.resize(img, (360, 360), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((480, 640, 3), 235, np.uint8)
        canvas[60:420, 140:500] = big
        r = det.process_frame(canvas)
        mp_t.append(r["mediapipe_time_ms"]); e2e_t.append(r["inference_time_ms"])
        if r["detected"]: mdl_t.append(r["model_time_ms"])
        ok = r["detected"] and r["raw_sign"] == c
        rows.append({"cls": c, "file": f, "detected": r["detected"], "pred": r["raw_sign"], "conf": r["confidence"], "ok": ok})
        if ok and r["confidence"] > best.get(c, (0, None))[0]:
            best[c] = (r["confidence"], f)
n = len(rows)
det_rate = sum(r["detected"] for r in rows) / n
acc = sum(r["ok"] for r in rows) / n
acc_det = sum(r["ok"] for r in rows) / max(1, sum(r["detected"] for r in rows))
per = {c: sum(r["ok"] for r in rows if r["cls"] == c) / PER for c in classes}
def st(a):
    a = np.array(a); return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2), "p95": round(float(np.percentile(a, 95)), 2), "n": int(len(a))}
res = {"samples": n, "per_class": PER, "detection_rate": round(det_rate, 4), "e2e_accuracy": round(acc, 4),
       "accuracy_given_detection": round(acc_det, 4), "per_class_accuracy": per,
       "latency_ms": {"mediapipe": st(mp_t), "model": st(mdl_t), "end_to_end": st(e2e_t)},
       "best_frames": {c: v[1] for c, v in best.items()}, "date": time.strftime("%Y-%m-%d %H:%M")}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_class_accuracy", "best_frames")}, indent=1))
print("weak:", {c: a for c, a in per.items() if a < 0.6})
