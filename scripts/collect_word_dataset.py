"""
Interactive CLI Video Clip & Landmark Collection Utility for Word-Level ISL.
Allows recording isolated word signs via webcam with start/end triggers,
immediate MediaPipe multi-hand landmark extraction, and signer-independent metadata registration.
"""

import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.collector import collector, load_vocabulary
from vision.hand_tracking.multi_hand_tracker import MultiHandTracker


def run_interactive_collector():
    vocab = load_vocabulary()
    if not vocab:
        print("[Error] No vocabulary found in data/word_dataset/vocabulary.json")
        return

    gloss_list = [item["gloss"] for item in vocab]
    print("=" * 60)
    print("ISL Word-Level Dataset Collector")
    print("=" * 60)
    print("Available Glosses:")
    for idx, g in enumerate(gloss_list, 1):
        print(f"  {idx:2d}. {g}")
    print("=" * 60)

    signer_id = input("Enter pseudonymous Signer ID (e.g., signer_01, signer_02): ").strip()
    if not signer_id:
        signer_id = "signer_01"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam index 0.")
        return

    tracker = MultiHandTracker()
    print("\n[Controls]")
    print("  SPACE: Start / Stop recording 30-frame clip")
    print("  N:     Next gloss")
    print("  Q:     Quit")

    current_idx = 0
    recording = False
    recorded_frames = []
    recorded_features = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            track_res = tracker.process_frame(frame)
            annotated = tracker.draw_landmarks(frame, track_res["left_hand_obj"], track_res["right_hand_obj"])

            current_gloss = gloss_list[current_idx]

            # Status Banner
            status_text = f"Gloss: {current_gloss} | Signer: {signer_id}"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(annotated, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if recording:
                recorded_frames.append(frame.copy())
                recorded_features.append(track_res["feature_vector"])
                cv2.circle(annotated, (w - 40, 40), 15, (0, 0, 255), -1)
                cv2.putText(annotated, f"REC {len(recorded_frames)}/30", (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if len(recorded_frames) >= 30:
                    recording = False
                    print(f"Recorded 30 frames for {current_gloss}. Saving...")
                    feat_array = np.array(recorded_features, dtype=np.float32)
                    collector.record_clip(
                        gloss=current_gloss,
                        signer_id=signer_id,
                        clip_bytes=None,
                        landmarks_data={"features": feat_array},
                        handedness="two-handed" if "HELP" in current_gloss or "NAME" in current_gloss else "one-handed",
                        frame_count=30,
                        fps=30.0
                    )
                    print(f"Saved {current_gloss} clip metadata and landmark sequence.")
                    recorded_frames.clear()
                    recorded_features.clear()

            cv2.imshow("ISL Word Dataset Collector", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if not recording:
                    recording = True
                    recorded_frames.clear()
                    recorded_features.clear()
                    print(f"Started recording for {current_gloss}...")
                else:
                    recording = False
                    recorded_frames.clear()
                    recorded_features.clear()
                    print("Recording cancelled.")
            elif key == ord('n'):
                current_idx = (current_idx + 1) % len(gloss_list)
                print(f"Switched to gloss: {gloss_list[current_idx]}")

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_interactive_collector()
