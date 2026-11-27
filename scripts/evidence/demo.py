"""Drive the running app with a synthetic webcam feed built from dataset frames and capture evidence screenshots."""
import os, sys, json, time
import numpy as np, cv2
from playwright.sync_api import sync_playwright

S = os.environ.get("EVIDENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evidence_out")); os.makedirs(S, exist_ok=True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(S, "img"); os.makedirs(IMG, exist_ok=True)
best = json.load(open(os.path.join(S, "picks.json")))
WORD = sys.argv[1] if len(sys.argv) > 1 else "HELP"

def frame_for(c):
    canvas = np.full((480, 640, 3), 235, np.uint8)
    if c is None:
        return canvas
    img = cv2.imread(os.path.join(ROOT, "archive (5)", "data", c, best[c]))
    canvas[60:420, 140:500] = cv2.resize(img, (360, 360), interpolation=cv2.INTER_CUBIC)
    return canvas

# Sequence: blank, then each letter held 2.5 s separated by 1 s of empty frame (neutral reset); last letter held long.
seq = [(None, 30)]
for i, c in enumerate(WORD):
    seq.append((c, 75 if i < len(WORD) - 1 else 300))
    if i < len(WORD) - 1:
        seq.append((None, 30))
y4m = os.path.join(S, "feed.y4m")
with open(y4m, "wb") as f:
    f.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
    for c, n in seq:
        yuv = cv2.cvtColor(frame_for(c), cv2.COLOR_BGR2YUV_I420).tobytes()
        for _ in range(n):
            f.write(b"FRAME\n" + yuv)

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=[
        "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-video-capture={y4m}"])
    ctx = b.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5, permissions=["camera"])
    pg = ctx.new_page()
    pg.request.post("http://127.0.0.1:8000/api/buffer/control?action=clear")
    pg.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    time.sleep(1.5)
    pg.screenshot(path=os.path.join(IMG, "01_dashboard_idle.png"))
    pg.click("#camToggleBtn")
    deadline = time.time() + 40
    txt = ""
    while time.time() < deadline:
        txt = pg.inner_text("#fullTextDisplay").replace(" ", "")
        if WORD in txt:
            break
        time.sleep(0.3)
    print("buffer:", txt)
    time.sleep(1.0)
    pg.screenshot(path=os.path.join(IMG, "02_live_recognition.png"))
    pg.locator("#videoContainer").screenshot(path=os.path.join(IMG, "03_landmark_overlay.png"))
    pg.locator(".detection-bar").screenshot(path=os.path.join(IMG, "04_confidence_state.png"))
    # translation to Telugu then Hindi
    for lang, name in (("te", "telugu"), ("hi", "hindi")):
        try:
            pg.select_option("#targetLangSelect", lang)
        except Exception:
            pg.select_option("#targetLangSelect", index=0)
        pg.click("#translateBtn")
        time.sleep(0.5)
        for _ in range(80):
            if "complete" in pg.inner_text("#transStatusBadge").lower() or "cache" in pg.inner_text("#transStatusBadge").lower():
                break
            time.sleep(0.25)
        time.sleep(0.5)
        print(name, "->", pg.inner_text("#translatedDisplay"), pg.inner_text("#transStatusBadge"), pg.inner_text("#transLatencyTxt"))
        pg.screenshot(path=os.path.join(IMG, f"05_translation_{name}.png"))
    pg.locator(".telemetry-card, #telemetryCard, .telemetry-grid").first.screenshot(path=os.path.join(IMG, "06_telemetry.png"))
    # word mode with truthful model-status banner
    pg.click("#modeWordBtn"); time.sleep(3)
    pg.screenshot(path=os.path.join(IMG, "07_word_mode.png"))
    pg.click("#modeFingerspellingBtn")
    # dataset collector
    pg.click("#openCollectorBtn"); time.sleep(1)
    pg.locator("#collectorModal .modal-content, #collectorModal > div").first.screenshot(path=os.path.join(IMG, "08_collector.png"))
    pg.click("#closeModalBtn")
    pg.goto("http://127.0.0.1:8000/docs", wait_until="networkidle"); time.sleep(1.5)
    pg.screenshot(path=os.path.join(IMG, "09_api_docs.png"))
    b.close()
