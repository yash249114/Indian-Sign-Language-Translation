"""Pick demo frames per letter that survive the browser path (YUV420 camera + JPEG q70) with high confidence."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, ROOT); os.chdir(ROOT)
from vision.inference.sign_detector import SignDetector
S = os.environ.get("EVIDENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evidence_out")); os.makedirs(S, exist_ok=True)
det = SignDetector()
letters = sys.argv[1]
def canvas(img):
    c = np.full((480, 640, 3), 235, np.uint8)
    c[60:420, 140:500] = cv2.resize(img, (360, 360), interpolation=cv2.INTER_CUBIC); return c
def browser_path(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420); bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70]); return cv2.imdecode(enc, 1)
blank = np.full((480, 640, 3), 235, np.uint8)
picks = {}
for c in sorted(set(letters)):
    d = os.path.join(ROOT, "archive (5)", "data", c)
    for f in sorted(os.listdir(d), key=lambda x: int(x.split(".")[0]))[::3]:
        fr = browser_path(canvas(cv2.imread(os.path.join(d, f))))
        for _ in range(3): det.process_frame(blank)
        rs = [det.process_frame(fr) for _ in range(8)]
        if all(r["detected"] and r["raw_sign"] == c and r["confidence"] > 0.8 for r in rs):
            picks[c] = f; break
    print(c, picks.get(c))
json.dump(picks, open(os.path.join(S, "picks.json"), "w"))
