"""Report figures: architecture, state machine, landmark normalisation, results charts."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

S = os.environ.get("EVIDENCE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evidence_out")); os.makedirs(S, exist_ok=True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(S, "fig"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "savefig.dpi": 300})
BLUE, TEAL, ORANGE, GREY, RED = "#1f4e79", "#2a9d8f", "#e07a1f", "#6b7280", "#c0392b"
bench = json.load(open(os.path.join(S, "bench.json")))


def box(ax, x, y, w, h, text, fc="#eef3f9", ec=BLUE, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom", fontsize=8, color="#333")


# ---------------- Fig: system architecture ----------------
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.15, 3.35), 2.3, 2.6, boxstyle="round,pad=0.02", fc="#fafafa", ec=GREY, ls="--"))
ax.text(1.3, 5.75, "Browser client (HTML/JS)", ha="center", fontsize=9, color=GREY)
box(ax, 0.35, 4.6, 1.9, 0.8, "USB/laptop webcam\n640×480 @ 30 fps")
box(ax, 0.35, 3.55, 1.9, 0.8, "Canvas capture\nJPEG q=0.7, base64")
arrow(ax, 1.3, 4.6, 1.3, 4.35)
ax.add_patch(FancyBboxPatch((2.85, 0.25), 7.95, 5.7, boxstyle="round,pad=0.02", fc="#fbfdff", ec=BLUE, ls="--"))
ax.text(6.8, 5.75, "FastAPI server (Python 3.10, CPU only)", ha="center", fontsize=9, color=BLUE)
box(ax, 3.05, 3.55, 1.6, 0.8, "WebSocket\n/ws/sign-stream")
arrow(ax, 2.25, 3.95, 3.05, 3.95)
ax.text(2.65, 4.45, "JSON frames\n~15 /s", ha="center", fontsize=8, color="#333")
box(ax, 5.0, 3.55, 1.6, 0.8, "MediaPipe Hands\n21 landmarks (x,y,z)", fc="#e8f6f3", ec=TEAL)
arrow(ax, 4.65, 3.95, 5.0, 3.95)
box(ax, 6.95, 3.55, 1.6, 0.8, "Wrist-centred,\nscale-normalised 63-D", fc="#e8f6f3", ec=TEAL)
arrow(ax, 6.6, 3.95, 6.95, 3.95)
box(ax, 8.9, 3.55, 1.7, 0.8, "MLP 63-128-64-35\n(scikit-learn)", fc="#e8f6f3", ec=TEAL, bold=True)
arrow(ax, 8.55, 3.95, 8.9, 3.95)
box(ax, 8.9, 2.25, 1.7, 0.85, "Stabiliser FSM\n6-frame vote, ≥60 %\ncommit-once", fc="#fff4e8", ec=ORANGE)
arrow(ax, 9.75, 3.55, 9.75, 3.1)
box(ax, 6.95, 2.25, 1.6, 0.85, "Text buffer\n(chars, space,\nbackspace)")
arrow(ax, 8.9, 2.67, 8.55, 2.67)
box(ax, 5.0, 2.25, 1.6, 0.85, "Translation\nGemini REST +\noffline lexicon")
arrow(ax, 6.95, 2.67, 6.6, 2.67)
box(ax, 3.05, 2.25, 1.6, 0.85, "Offline TTS\npyttsx3 → WAV")
arrow(ax, 5.0, 2.67, 4.65, 2.67)
box(ax, 5.0, 0.55, 1.6, 0.95, "Word/continuous mode\n2-hand 126-D × 30\nTemporal CNN\n(reference only)", fc="#f3f4f6", ec=GREY, fs=8.5)
box(ax, 6.95, 0.55, 1.6, 0.95, "Dataset collector\nconsent + checksum\nmanifest", fc="#f3f4f6", ec=GREY, fs=8.5)
box(ax, 3.05, 0.55, 1.6, 0.95, "JSON telemetry\nlatency, FPS,\nstate, top-3", fc="#f3f4f6", ec=GREY, fs=8.5)
ax.plot([3.05, 1.3], [2.67, 2.67], color=BLUE, lw=1.2)
arrow(ax, 1.3, 2.67, 1.3, 3.55, color=BLUE)
ax.text(0.3, 2.05, "responses: annotated frame,\nsign, top-3, text, audio", fontsize=8, color=BLUE)
fig.savefig(os.path.join(OUT, "f_architecture.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: state machine ----------------
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")
W, G = 1.35, 0.55
states = [(n, 0.1 + i * (W + G)) for i, n in enumerate(["NO_SIGN", "DETECTING", "STABLE_SIGN", "COMMIT_ONCE", "LOCKED_HOLD", "WAIT_FOR\nNEUTRAL"])]
for n, x in states:
    box(ax, x, 1.9, W, 0.75, n, fc="#fff4e8" if n == "COMMIT_ONCE" else "#eef3f9", ec=ORANGE if n == "COMMIT_ONCE" else BLUE, fs=8.5, bold=True)
labels = ["valid pred.\n(conf ≥ 0.75)", "≥ 4 of 6\nframes agree", "first stable\nframe", "emit exactly\n1 character", "hand drops or\nconf < 0.75"]
for i in range(5):
    x1 = states[i][1] + W; x2 = states[i + 1][1]
    arrow(ax, x1, 2.27, x2, 2.27)
    ax.text((x1 + x2) / 2, 2.75, labels[i], ha="center", va="bottom", fontsize=7.5, color="#333")
ax.add_patch(FancyArrowPatch((states[-1][1] + W / 2, 1.9), (states[0][1] + W / 2, 1.9), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.2))
ax.text(5.5, 0.35, "3 consecutive neutral frames (no hand or confidence < 0.75) reset to NO_SIGN; a different stable sign releases the lock", ha="center", fontsize=9, color=RED)
fig.savefig(os.path.join(OUT, "f_state_machine.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: landmark normalisation ----------------
sys.path.insert(0, ROOT)
d = np.load(os.path.join(ROOT, "data/processed/landmarks/test.npz"))
lm = json.load(open(os.path.join(ROOT, "data/processed/landmarks/label_map.json")))
names = lm.get("class_names") or lm
X, y = d["X"], d["y"]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
for ax, cls in zip(axes, ["A", "L", "V", "Y"]):
    idx = names.index(cls) if isinstance(names, list) else int([k for k, v in names.items() if v == cls][0])
    p = X[np.where(y == idx)[0][0]].reshape(21, 3)
    for a, b in EDGES:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=GREY, lw=1.2)
    ax.scatter(p[:, 0], p[:, 1], s=16, c=[RED] + [BLUE] * 20, zorder=3)
    ax.set_title(f"Class '{cls}'", fontsize=11); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.axhline(0, color="#ddd", lw=0.6); ax.axvline(0, color="#ddd", lw=0.6)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(1.1, -1.1); ax.tick_params(labelsize=8)
axes[0].set_ylabel("normalised y (unitless)")
fig.text(0.5, -0.02, "normalised x (unitless); wrist (red) at origin, farthest landmark at distance 1", ha="center", fontsize=10)
fig.savefig(os.path.join(OUT, "f_landmarks.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: per-class end-to-end accuracy ----------------
pc = bench["per_class_accuracy"]
fig, ax = plt.subplots(figsize=(11, 3.6))
ks = list(pc.keys()); vs = [pc[k] * 100 for k in ks]
ax.bar(ks, vs, color=[TEAL if v >= 80 else (ORANGE if v >= 50 else RED) for v in vs])
ax.axhline(bench["e2e_accuracy"] * 100, color=BLUE, ls="--", lw=1.2)
ax.text(34.6, bench["e2e_accuracy"] * 100 + 2, f"mean {bench['e2e_accuracy']*100:.1f} %", color=BLUE, ha="right", fontsize=9)
ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 105); ax.set_xlabel("Class (n = 20 raw images per class)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(OUT, "f_per_class.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: latency breakdown ----------------
L = bench["latency_ms"]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
comp = ["MediaPipe Hands", "MLP inference", "End-to-end (server)"]
means = [L["mediapipe"]["mean"], L["model"]["mean"], L["end_to_end"]["mean"]]
p95 = [L["mediapipe"]["p95"], L["model"]["p95"], L["end_to_end"]["p95"]]
yy = np.arange(3)
ax.barh(yy, means, color=[TEAL, ORANGE, BLUE], height=0.5, label="mean")
ax.scatter(p95, yy, color="black", marker="|", s=300, label="95th percentile", zorder=3)
for i, (m, q) in enumerate(zip(means, p95)):
    ax.text(max(m, q) + 0.5, i, f"{m:.2f} ms (p95 {q:.2f})", va="center", fontsize=9)
ax.axvline(33.3, color=RED, ls=":", lw=1.2); ax.text(33.0, -0.42, "33.3 ms budget (30 fps)", color=RED, fontsize=8, ha="right")
ax.set_yticks(yy); ax.set_yticklabels(comp); ax.set_xlabel("Latency per frame (ms), n = 700 frames"); ax.set_xlim(0, 42)
ax.invert_yaxis(); ax.legend(loc="center right", fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(OUT, "f_latency.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: robustness (team audit, Aug 2026) ----------------
rob = [("Clean", 73.7), ("Low light 0.4×", 72.0), ("Over-exposed 1.8×", 72.6), ("Low contrast 0.5×", 75.4),
       ("Rotate −15°", 53.1), ("Rotate +15°", 50.3), ("Rotate −30°", 21.1), ("Rotate +30°", 17.1), ("Down-scaled 64 px", 68.6)]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar([r[0] for r in rob], [r[1] for r in rob], color=[BLUE] + [TEAL] * 3 + [ORANGE] * 2 + [RED] * 2 + [TEAL])
for i, r in enumerate(rob): ax.text(i, r[1] + 1.5, f"{r[1]:.1f}", ha="center", fontsize=8)
ax.set_ylabel("End-to-end accuracy (%)"); ax.set_ylim(0, 90); plt.xticks(rotation=25, ha="right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(OUT, "f_robustness.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: accuracy gap ----------------
fig, ax = plt.subplots(figsize=(7.5, 3.2))
lab = ["MLP on held-out\nlandmark vectors", "Random forest on\nheld-out vectors", "Full pipeline on\nraw images (this run)", "Full pipeline,\nAug 2026 audit"]
val = [99.36, 100.0, bench["e2e_accuracy"] * 100, 73.7]
ax.bar(lab, val, color=[BLUE, GREY, TEAL, TEAL], width=0.55)
for i, v in enumerate(val): ax.text(i, v + 1.2, f"{v:.1f} %", ha="center", fontsize=9)
ax.set_ylim(0, 110); ax.set_ylabel("Top-1 accuracy (%)"); ax.spines[["top", "right"]].set_visible(False); plt.xticks(fontsize=9)
fig.savefig(os.path.join(OUT, "f_accuracy_gap.png"), bbox_inches="tight"); plt.close(fig)
print(os.listdir(OUT))
